from abc import ABCMeta, abstractmethod
import logging
from sys import stdout
from time import time
import tensorflow as tf

from sklearn.base import BaseEstimator, RegressorMixin
from .constraints import *
from .losses import *
from .train import *


class KnowledgeGraphEmbeddingModel(BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    """An abstract class for the knowledge graph embedding models.

    Attributes
    ----------
    em_size: int, optional
            embedding vector size
    batch_size: int
        batch size
    nb_epochs: int
        number of epoch i.e training iterations
    initialiser: str
        initialiser name e.g. xavier_uniform or he_normal
    nb_negs: int
        number of negative instance per each positive training instance
    optimiser: str
        optimiser name
    lr: float
        optimiser learning rate
    loss: str
        loss type e.g. pt_logistic or pt_se.
    nb_ents: int
        total number of knowledge graph entities
    nb_rels: int
        total number of knowldege graph relations
    reg_wt: float
        regularisation parameter weight
    seed: int
        random seed
    verbose: int
        verbosity level. options are {0, 1, 2}

    """

    def __init__(self, em_size=100, batch_size=128, nb_epochs=100, initialiser="xavier_uniform", nb_negs=2,
                 optimiser="amsgrad", lr=0.01, loss="pt_logistic", nb_ents=-1, nb_rels=-1, reg_wt=0.01,
                 predict_batch_size=40000, seed=1234, verbose=1, log_interval=5):
        """ Initialize a :code:`KnowledgeGraphEmbeddingModel` instance.

        Parameters
        ----------
        em_size: int
            embedding vector size
        batch_size: int
            batch size
        nb_epochs: int
            number of epoch i.e training iterations
        initialiser: str
            initialiser name e.g. xavier_uniform or he_normal
        nb_negs: int
            number of negative instance per each positive training instance
        optimiser: str
            optimiser name
        lr: float
            optimiser learning rate
        loss: str
            loss type e.g. pt_logistic or pt_se.
        nb_ents: int
            total number of knowledge graph entities
        nb_rels: int
            total number of knowledge graph relations
        reg_wt: float
            regularisation parameter weight
        predict_batch_size : int
            batch size in prediction mode
        seed: int
            random seed
        verbose: int
            verbosity level. options are {0, 1, 2}
        log_interval: int
            the number of epochs to wait until reporting the next training loss. (loss logging frequency)
        """
        super().__init__()

        self.em_size = em_size
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self.initialiser = initialiser
        self.nb_negs = nb_negs
        self.optimiser = optimiser
        self.lr = lr
        self.loss = loss
        self.nb_ents = nb_ents
        self.nb_rels = nb_rels
        self.reg_wt = reg_wt
        self.predict_batch_size = predict_batch_size
        self.log_interval = log_interval

        # init tf related vars
        self._embeddings = dict()
        self._tf_vars = dict()
        self._predict_pipeline_on = False

        # logging - initialise multiple verbosity logger
        self.log = None
        self.verbose = verbose

        # random states - initialise numpy and tensorflow random seeds
        self.seed = seed
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.rs = np.random.RandomState(seed=seed)

    def __getstate__(self):
        """ This is called before pickling. """
        _blacklist = ['tf_vars_', 'trainable_vars_', '_tf_session', '_tf_session_config', '_logger', '_random_state']
        accepted_types = {int, float, str, bool}
        state = {k: v for k, v in self.__dict__.items() if type(k) in accepted_types}
        # print('GET STATE: %s' % state)
        return state

    def __setstate__(self, state):
        """ This is called while unpickling.
        """
        # print('SET STATE: %s' % state)
        self.__dict__.update(state)
        if 'verbose' in state.keys():
            self.init_logging(state['verbose'])
        if 'seed' in state.keys():
            np.random.seed(self.seed)
            tf.set_random_seed(self.seed)
            self.rs = np.random.RandomState(seed=self.seed)
        # init some variables for tensorflow
        self._init_tf_session()

    def init_logging(self, verbose=0):
        """ Initialise class logger with specified verbosity.

        Parameters
        ----------
        verbose : int
            verbosity level.

        Note
        ----------
        Based on an answer on Stackoverflow question:
        https://stackoverflow.com/questions/11927278/how-to-configure-logging-in-python/11927374
        """
        self.log = logging.getLogger(self.__class__.__name__)
        if verbose == 0:
            self.log.setLevel(logging.WARN)
        elif verbose == 1:
            self.log.setLevel(logging.INFO)
        elif verbose == 2:
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.critical('Unknown verbosity level (%d). Options are [0, 1, 2]' % verbose)

        # important for parallel processing
        self.log.propagate = False

        if not self.log.handlers:
            console_handler = logging.StreamHandler(stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self.log.level)
            self.log.addHandler(console_handler)

    def init_embeddings(self):
        """ Initialise model _embeddings for both entities and relations

        Returns
        -------
        tf.Variable
            _embeddings of knowledge graph entities
        tf.Variable
            _embeddings of knowledge graph relations
        """
        # get initialiser variable and initialise tensorflow variables for each component _embeddings
        var_init = get_initializer(self.initialiser, self.seed)
        em_ents = tf.get_variable("em_ents", shape=[self.nb_ents+1, self.em_size], initializer=var_init)
        em_rels = tf.get_variable("em_rels", shape=[self.nb_rels+1, self.em_size], initializer=var_init)

        # add component embedding to the embedding vars dictionary
        self._embeddings["ents"] = em_ents
        self._embeddings["rels"] = em_rels

        return em_ents, em_rels

    def lookup_triples_embeddings(self, triples):
        """ Lookup triple _embeddings

        Parameters
        ----------
        triples: tf.tensor
            tensorflow tensor of size [?, 3]

        Returns
        -------
        tf.tensor
            Embeddings of the subject entities
        tf.tensor
            Embeddings of the relations
        tf.tensor
            Embeddings of the object entities
        """
        subs_em = tf.nn.embedding_lookup(self._embeddings["ents"], triples[:, 0])
        rels_em = tf.nn.embedding_lookup(self._embeddings["rels"], triples[:, 1])
        objs_em = tf.nn.embedding_lookup(self._embeddings["ents"], triples[:, 2])

        return subs_em, rels_em, objs_em

    def generate_negatives(self, triples, *args, **kwargs):
        """ Generate negative triple corruptions.

        Parameters
        ----------
        triples: tf.tensor
            (N, 3) tensorflow tensor of original true triples.
        args : list
            Non-key arguments
        kwargs : dict
            Key arguments

        Returns
        -------
        tf.tensor
            (N, 3) tensorflow tensor with negative triples.
        """
        return generate_rand_negs(triples, self.nb_negs, self.nb_ents, self.seed, *args, **kwargs)

    def embedding_normalisation(self, *args, **kwargs):
        """ Execute post optimisation embedding normalisation.

        Parameters
        ----------
        args : list
            Non-key arguments.
        kwargs : dict
            Key arguments.

        Returns
        -------
        list
            list of embedding normalisation operations.
        """
        return []

    @abstractmethod
    def score_triples(self, sub_em, rel_em, obj_em):
        """ Compute scores of triples using the _embeddings of their components

        Parameters
        ----------
        sub_em: tf.tensor
            Embeddings of the subject entities
        rel_em: tf.tensor
            Embeddings of the relations
        obj_em: tf.tensor
            Embeddings of the object entities

        Returns
        -------
        tf.tensor
            Scores of the original triples of the given components _embeddings
        """
        raise NotImplementedError("Not implemented model dependant function")

    @abstractmethod
    def compute_loss(self, scores, *args, **kwargs):
        """ Model dependant loss function.

        Parameters
        ----------
        scores: tf.tensor
            (N,) tensorflow tensor with all batch triples scores.
        args : list
            Non-key arguments.
        kwargs : dict
            Key arguments.

        Returns
        -------
        tf.float32
            Model loss.
        """
        # set pairwise hinge loss to be the default loss
        return compute_kge_loss(scores, loss_type="pr_hinge", *args, **kwargs)

    def embedding_regularisation(self, sub_em, rel_em, obj_em, *args, **kwargs):
        """ Compute embedding regularisation term

        Parameters
        ----------
        sub_em: tf.tensor
            Embeddings of the subject entities
        rel_em: tf.tensor
            Embeddings of the relations
        obj_em: tf.tensor
            Embeddings of the object entities
        args: list
            Non-key arguments
        kwargs: dict
            Key arguments

        Returns
        -------
        tf.float32
            regularisation value
        """
        # set the regularisation function to zero by default. (disabled)
        reg_val = 0
        return reg_val * self.reg_wt

    def fit(self, X, y=None, *args, **kwargs):
        """ Train the model on the given input triples

        Parameters
        ----------
        X: ndarray
            input triples array of size [?, 3]
        y: ndarray
            input triplets labels.
        args: list
            unnamed arguments
        kwargs: dict
            named arguments
        """

        # initialise the model's logger
        self.init_logging(self.verbose)

        self._predict_pipeline_on = False
        self.log.debug("Logging model parameters ...")
        # log model parameters in debug mode
        log_model_params(self)
        self.log.debug("Model training started ...")
        # compute the number of entities and relations if not given
        train_size = len(X)
        if self.nb_ents is None or self.nb_ents <= 0:
            entities_vocab = np.unique(np.concatenate([X[:, 0], X[:, 2]], axis=0))
            self.nb_ents = int(max(entities_vocab) + 1)

        if self.nb_rels is None or self.nb_rels <= 0:
            relations_vocab = np.unique(X[:, 1])
            self.nb_rels = int(max(relations_vocab) + 1)

        self.log.debug("Training model [ %d #Instances - %d #Entities - %d #Relations ]"
                       % (train_size, self.nb_ents, self.nb_rels))

        # ================================================================================================
        # tensorflow graph for the embedding model
        tf.reset_default_graph()

        # initialise model _embeddings
        self.init_embeddings()

        # define input placeholder
        self._tf_vars["xin_pos"] = xin_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])

        # generate negative corruption from the input triples
        xin_neg = self.generate_negatives(xin_pos)

        # tile positive triples and join them with negatives to make the batch triples classes balanced and divisible
        # from the middle into positive and negative classes
        xin_all = tf.concat([tf.tile(xin_pos, [self.nb_negs, 1]), xin_neg], axis=0)

        # lookup embedding of the triples components
        em_subs, em_rels, em_objs = self.lookup_triples_embeddings(xin_all)

        # compute triples' scores
        self._tf_vars["scores"] = scores = self.score_triples(em_subs, em_rels, em_objs)

        # compute regularisation for components _embeddings
        reg_term = self.embedding_regularisation(em_subs, em_rels, em_objs)

        # compute model loss and objective cost
        self._tf_vars["loss"] = model_loss = self.compute_loss(scores)
        model_train_error = model_loss + reg_term

        # initialise optimiser and minimise training error
        optimiser = init_tf_optimiser(self.optimiser, self.lr)
        optimisation = optimiser.minimize(model_train_error)

        # execute embedding normalisation procedure
        exec_norm = self.embedding_normalisation()
        # ================================================================================================

        self.log.debug("Initialising tensorflow session")
        session = self._init_tf_session()
        self.log.debug("Executing tensorflow global variable initialiser")
        session.run(tf.global_variables_initializer())

        tr_loss_list = []
        tr_speed_list = []
        for epoch in range(self.nb_epochs):
            train_batches = generate_batches(X, batch_size=self.batch_size, shuffle=True)
            epoch_loss_list = []
            epoch_tr_start_time = time()
            for batch_idx, batch_data in enumerate(train_batches):
                arg_dict = {xin_pos: batch_data}
                batch_tr_loss, _ = session.run([model_loss, optimisation], feed_dict=arg_dict)
                session.run(exec_norm)
                epoch_loss_list.append(batch_tr_loss)
                tr_loss_list.append(batch_tr_loss)
            epoch_tr_end_time = time()
            epoch_tr_time = train_size / (epoch_tr_end_time - epoch_tr_start_time)
            epoch_tr_time /= 1000.0
            tr_speed_list.append(epoch_tr_time)
            epoch_loss_avg = np.mean(epoch_loss_list)
            if epoch == 0 or (epoch+1) % self.log_interval == 0:
                self.log.debug("[Training] Epoch # %-4d - Speed: %1.3f (k. record/sec) - Loss: %-4.4f "
                               "- Avg(Loss): %-4.4f - Std(Loss): %-4.4f" %
                               (epoch+1, epoch_tr_time, epoch_loss_avg, np.mean(tr_loss_list), np.std(tr_loss_list)))
        self.log.debug("[Reporting] Finished (%d Epochs) - Avg(Speed): %1.3f (k. record/sec) "
                       "- Avg(Loss): %-4.4f - Std(Loss): %-4.4f" %
                       (self.nb_epochs, np.mean(tr_speed_list), np.mean(tr_loss_list), np.std(tr_loss_list)))

    def _init_tf_session(self):
        tf_session_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        tf_session_config.gpu_options.allow_growth = True
        self._tf_vars["session"] = session = tf.Session(config=tf_session_config)
        return session

    def _init_prediction_flow(self):
        """ Initialise the tensorflow graph for predicting new triples
        Returns
        ----------
        tf.Placeholder
            the input triplets placeholder
        tf.Tensor
            the scores tensorflow tensor
        """
        if not ("session" in self._tf_vars and type(self._tf_vars["session"]) == tf.Session):
            self._init_tf_session()
        self._tf_vars["xin_predict"] = xin_predict = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        # lookup embedding of the triples components
        em_subs, em_rels, em_objs = self.lookup_triples_embeddings(xin_predict)
        # compute triples' scores
        self._tf_vars["scores_predict"] = scores = self.score_triples(em_subs, em_rels, em_objs)
        self._predict_pipeline_on = True
        return xin_predict, scores

    def predict(self, X, *args, **kwargs):
        """ Predict scores of a set of knowledge triples.

        Parameters
        ----------
        X: ndarray
            input triples array of size [?, 3]
        args: list
            unnamed arguments
        kwargs: dict
            named arguments

        Returns
        -------
        ndarray
            outcome scores of input triples
        """

        # initialise the prediction flow if not initialised
        if not self._predict_pipeline_on:
            self._init_prediction_flow()
        xin, scores_tf, session = self._tf_vars["xin_predict"], self._tf_vars["scores_predict"], self._tf_vars["session"]

        predict_batches = generate_batches(X, batch_size=self.predict_batch_size, shuffle=False)
        output_scores = []
        for batch_data in predict_batches:
            batch_scores = session.run(scores_tf, feed_dict={xin: batch_data})
            output_scores.extend(batch_scores.tolist())
        return np.array(output_scores)

    def get_embeddings(self):
        """ Get model learnt embeddings

        Returns
        -------
        dict
            dictionary of embeddings
        """
        em_dict = dict()
        for k, v in self._embeddings.items():
            em_dict[k] = self._tf_vars["session"].run(v)
        return em_dict


class KnowledgeGraphEmbeddingModelMCL(KnowledgeGraphEmbeddingModel):
    """An abstract class for the knowledge graph embedding models with multi-class loss.

    Attributes
    ----------
    em_size: int, optional
            embedding vector size
    batch_size: int
        batch size
    nb_epochs: int
        number of epoch i.e training iterations
    initialiser: str
        initialiser name e.g. xavier_uniform or he_normal
    nb_negs: int
        number of negative instance per each positive training instance
    optimiser: str
        optimiser name
    lr: float
        optimiser learning rate
    loss: str
        loss type e.g. pt_logistic or pt_se.
    nb_ents: int
        total number of knowledge graph entities
    nb_rels: int
        total number of knowldege graph relations
    reg_wt: float
        regularisation parameter weight
    seed: int
        random seed
    verbose: int
        verbosity level. options are {0, 1, 2}
    """

    def __init__(self, em_size=100, batch_size=128, nb_epochs=100, nb_negs=-1, dropout=0.02, optimiser="amsgrad",
                 lr=0.01, loss="nll", nb_ents=0, nb_rels=0, reg_wt=0.03, predict_batch_size=40000, seed=1234,
                 verbose=1, log_interval=5, initialiser="xavier_uniform"):
        """ Initialize a :code:`KnowledgeGraphEmbeddingModelMCL` instance.

        Parameters
        ----------
        em_size: int
            embedding vector size
        batch_size: int
            batch size
        nb_epochs: int
            number of epoch i.e training iterations
        initialiser: str
            initialiser name e.g. xavier_uniform or he_normal
        nb_negs: int
            number of negative instance per each positive training instance
        dropout : float
            dropout rate
        optimiser: str
            optimiser name
        lr: float
            optimiser learning rate
        loss: str
            loss type e.g. pt_logistic or pt_se.
        nb_ents: int
            total number of knowledge graph entities
        nb_rels: int
            total number of knowledge graph relations
        reg_wt: float
            regularisation parameter weight
        predict_batch_size : int
            batch size in prediction mode
        seed: int
            random seed
        verbose: int
            verbosity level. options are {0, 1, 2}
        log_interval: int
            the number of epochs to wait until reporting the next training loss. (loss logging frequency)
        """
        super().__init__(em_size=em_size, batch_size=batch_size, nb_epochs=nb_epochs, initialiser=initialiser,
                         nb_negs=nb_negs, optimiser=optimiser, lr=lr, loss=loss, nb_ents=nb_ents, nb_rels=nb_rels,
                         reg_wt=reg_wt, predict_batch_size=predict_batch_size, seed=seed, verbose=verbose,
                         log_interval=log_interval)
        self.dropout = dropout

    def sample_ent_corrs(self, mode='s', *args, **kwargs):
        """ Generate random entity corruptions

        Parameters
        ----------
        mode: str
            corruption mode (subjects, objects). Options: 's' and 'o'.
        args: list
            unnamed arguments
        kwargs: dict
            named arguments

        Returns
        ----------
        tf.tensor
            sub corruptions
        tf.tensor
            obj corruptions
        """
        # TODO: finish this part
        all_corrs = tf.range(0, self.nb_ents)
        if self.nb_negs <= 0:
            sub_corrs = all_corrs
            obj_corrs = all_corrs
        else:
            sub_corrs = tf.nn.uniform_candidate_sampler(all_corrs, 1, self.nb_negs, unique=False,
                                                        range_max=self.nb_ents, seed=self.seed)
            sub_corrs = tf.nn.uniform_candidate_sampler(all_corrs, 1, self.nb_negs, unique=False,
                                                        range_max=self.nb_ents, seed=self.seed)

    def fit(self, X, y=None, *args, **kwargs):
        """ Train the model on the given input triples

        Parameters
        ----------
        X: ndarray
            input triples array of size [?, 3]
        y: ndarray
            input triplets labels.
        args: list
            unnamed arguments
        kwargs: dict
            named arguments
        """
        self.init_logging(self.verbose)

        self.log.debug("Logging model parameters ...")
        # log model parameters in debug mode
        log_model_params(self)
        self.log.debug("Model training started ...")
        # compute the number of entities and relations if not given
        train_size = len(X)
        if self.nb_ents is None or self.nb_ents <= 0:
            entities_vocab = np.unique(np.concatenate([X[:, 0], X[:, 2]], axis=0))
            self.nb_ents = int(max(entities_vocab) + 1)

        if self.nb_rels is None or self.nb_rels <= 0:
            relations_vocab = np.unique(X[:, 1])
            self.nb_rels = int(max(relations_vocab) + 1)

        self.log.debug("Training model [ %d #Instances - %d #Entities - %d #Relations ]"
                       % (train_size, self.nb_ents, self.nb_rels))

        # ================================================================================================
        # tensorflow graph for the embedding model
        tf.reset_default_graph()

        # initialise model _embeddings
        self.init_embeddings()

        # define input placeholder
        self._tf_vars["xin_pos"] = xin_pos = tf.placeholder(dtype=tf.int32, shape=[None, 3])

        # lookup embedding of the triples components
        em_subs, em_rels, em_objs = self.lookup_triples_embeddings(xin_pos)

        # compute triples' scores
        self._tf_vars["scores"] = scores = self.score_triples_mc(em_subs, em_rels, em_objs)
        scores_mc = self.score_triples_mc(em_subs, em_rels, em_objs)

        # compute regularisation for components _embeddings
        reg_term = self.embedding_regularisation(em_subs, em_rels, em_objs)

        # compute model loss and objective cost
        self._tf_vars["loss"] = model_loss = self.compute_loss(scores_mc,
                                                               subs=xin_pos[:, 0],
                                                               objs=xin_pos[:, 2])
        model_train_error = model_loss + reg_term

        # initialise optimiser and minimise training error
        optimiser = init_tf_optimiser(self.optimiser, self.lr)
        optimisation = optimiser.minimize(model_train_error)

        # execute embedding normalisation procedure
        exec_norm = self.embedding_normalisation()
        # ================================================================================================

        self.log.debug("Initialising tensorflow session")
        session = self._init_tf_session()
        self.log.debug("Executing tensorflow global variable initialiser")
        session.run(tf.global_variables_initializer())

        tr_loss_list = []
        tr_speed_list = []
        for epoch in range(self.nb_epochs):
            train_batches = generate_batches(X, batch_size=self.batch_size, shuffle=True)
            epoch_loss_list = []
            epoch_tr_start_time = time()
            for batch_idx, batch_data in enumerate(train_batches):
                arg_dict = {xin_pos: batch_data}
                batch_tr_loss, _ = session.run([model_loss, optimisation], feed_dict=arg_dict)
                session.run(exec_norm)
                epoch_loss_list.append(batch_tr_loss)
                tr_loss_list.append(batch_tr_loss)
            epoch_tr_end_time = time()
            epoch_tr_time = train_size / (epoch_tr_end_time - epoch_tr_start_time)
            epoch_tr_time /= 1000.0
            tr_speed_list.append(epoch_tr_time)
            epoch_loss_avg = np.mean(epoch_loss_list)
            if epoch == 0 or (epoch + 1) % self.log_interval == 0:
                self.log.debug("[Training] Epoch # %-4d - Speed: %1.3f (k. record/sec) - Loss: %-4.4f "
                               "- Avg(Loss): %-4.4f - Std(Loss): %-4.4f" %
                               (epoch + 1, epoch_tr_time, epoch_loss_avg, np.mean(tr_loss_list), np.std(tr_loss_list)))
        self.log.debug("[Reporting] Finished (%d Epochs) - Avg(Speed): %1.3f (k. record/sec) "
                       "- Avg(Loss): %-4.4f - Std(Loss): %-4.4f" %
                       (self.nb_epochs, np.mean(tr_speed_list), np.mean(tr_loss_list), np.std(tr_loss_list)))

    @abstractmethod
    def score_triples(self, sub_em, rel_em, obj_em):
        """ Model dependent scoring function

        Parameters
        ----------
        sub_em: tf.tensor
            subject embeddings
        rel_em: tf.tensor
            predicate embeddings
        obj_em: tf.tensor
            object embeddings

        Returns
        -------

        """
        raise NotImplementedError("Not implemented model dependant function")

    @abstractmethod
    def score_triples_mc(self, sub_em, rel_em, obj_em):
        """ Model dependent scoring function

        Parameters
        ----------
        sub_em: tf.tensor
            subject embeddings
        rel_em: tf.tensor
            predicate embeddings
        obj_em: tf.tensor
            object embeddings

        Returns
        -------

        """
        raise NotImplementedError("Not implemented model dependant function")

    def compute_loss(self, scores_mc, *args, **kwargs):
        """ Compute model training loss

        Parameters
        ----------
        scores_mc : tuple
            scores of subject and object corruptions
        args : list
            Non-key arguments.
        kwargs : dict
            Key arguments.

        Returns
        -------
        tf.tensor
            Model loss.
        """

        eps = 1e-10
        subs = kwargs["subs"]
        objs = kwargs["objs"]

        sub_scores, obj_scores = scores_mc
        sub_scores = tf.clip_by_value(sub_scores, eps, 1 - eps)
        obj_scores = tf.clip_by_value(obj_scores, eps, 1 - eps)

        rows = tf.range(tf.shape(subs)[0])
        sub_pos_idx = tf.transpose(tf.stack([rows, subs]))
        obj_pos_idx = tf.transpose(tf.stack([rows, objs]))

        sub_pos = tf.gather_nd(sub_scores, sub_pos_idx)
        obj_pos = tf.gather_nd(obj_scores, obj_pos_idx)

        sub_loss = - tf.log(sub_pos)
        obj_loss = - tf.log(obj_pos)

        return tf.reduce_sum(sub_loss + obj_loss)

    def embedding_regularisation(self, sub_em, rel_em, obj_em, *args, **kwargs):
        """ Embedding regularisation function

        Parameters
        ----------
        sub_em: tf.tensor
            subject embeddings
        rel_em: tf.tensor
            predicate embeddings
        obj_em: tf.tensor
            object embeddings
        args: list
            unnamed arguments
        kwargs: dict
            named arguments

        Returns
        -------
        regularisation value
        """
        # the nuclear 3-norm regularisation
        reg = tf.reduce_sum(tf.pow(tf.abs(sub_em), 3) + tf.pow(tf.abs(rel_em), 3) + tf.pow(tf.abs(obj_em), 3))
        return self.reg_wt/3 * reg
