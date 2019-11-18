from .losses import *
from .base_model import KnowledgeGraphEmbeddingModel, get_initializer


class TransE(KnowledgeGraphEmbeddingModel):
    """
    The Translating Embedding model (TransE)
    """

    def __init__(self, em_size=100, batch_size=128, nb_epochs=100, initialiser="xavier_uniform", nb_negs=2, margin=1.0,
                 optimiser="amsgrad", lr=0.01, similarity="l1", nb_ents=0, nb_rels=0, reg_wt=0.01, loss="default",
                 seed=1234, verbose=1, log_interval=5):
        """ Initialise new instance of the class TranslatingEmbeddingModel

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
        margin: float
            hinge loss margin
        similarity: str
            embedding similarity function
        optimiser: str
            optimiser name
        lr: float
            optimiser learning rate
        nb_ents: int
            total number of knowledge graph entities
        nb_rels: int
            total number of knowledge graph relations
        reg_wt: float
            regularisation parameter weight
        seed: int
            random seed
        verbose: int
            verbosity level. options are {0, 1, 2}
        log_interval: int
            the number of epochs to wait until reporting the next training loss. (loss logging frequency)
        """
        super().__init__(em_size=em_size, batch_size=batch_size, nb_epochs=nb_epochs, initialiser=initialiser,
                         nb_negs=nb_negs, optimiser=optimiser, lr=lr, loss=loss, nb_ents=nb_ents, nb_rels=nb_rels,
                         reg_wt=reg_wt, seed=seed, verbose=verbose, log_interval=log_interval)

        self.margin = margin
        self.similarity = similarity

    def score_triples(self, sub_em, rel_em, obj_em):
        """ Compute TransE scores for a set of triples given their component _embeddings

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
            model scores for the original triples of the given _embeddings
        """

        em_interactions = sub_em + rel_em - obj_em
        if self.similarity.lower() == "l1":
            scores = tf.norm(em_interactions, ord=1, axis=1)
        elif self.similarity.lower() == "l2":
            scores = tf.norm(em_interactions, ord=2, axis=1)
        else:
            raise ValueError("Unknown similarity type (%s)." % self.similarity)

        # the use of negative score complies with loss objective
        return -scores

    def compute_loss(self, scores, *args, **kwargs):
        """ Compute TransE training loss using the pairwise hinge loss

        Parameters
        ----------
        scores: tf.Tenor
            scores tensor
        args: list
            Non-Key arguments
        kwargs: dict
            Key arguments

        Returns
        -------
        tf.float32
            model loss value
        """
        if self.loss == "default":
            pos_scores, neg_scores = tf.split(scores, num_or_size_splits=2)
            return pairwise_logistic_loss(pos_scores, neg_scores, reduction_type="avg")
        else:
            return compute_kge_loss(scores, self.loss, reduction_type="avg")


class DistMult(KnowledgeGraphEmbeddingModel):
    """
    The DistMult Embedding model (DistMult)
    """

    def __init__(self, em_size=100, batch_size=128, nb_epochs=100, initialiser="xavier_uniform", nb_negs=2, margin=1.0,
                 optimiser="amsgrad", loss="default", lr=0.01, nb_ents=0, nb_rels=0, reg_wt=0.01, seed=1234, verbose=1,
                 log_interval=5):
        """ Initialise new instance of the class DistMult

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
        margin: float
            hinge loss margin
        optimiser: str
            optimiser name
        lr: float
            optimiser learning rate
        nb_ents: int
            total number of knowledge graph entities
        nb_rels: int
            total number of knowledge graph relations
        reg_wt: float
            regularisation parameter weight
        seed: int
            random seed
        verbose: int
            verbosity level. options are {0, 1, 2}
        log_interval: int
            the number of epochs to wait until reporting the next training loss. (loss logging frequency)
        """
        super().__init__(em_size=em_size, batch_size=batch_size, nb_epochs=nb_epochs, initialiser=initialiser,
                         nb_negs=nb_negs, optimiser=optimiser, lr=lr, loss=loss, nb_ents=nb_ents, nb_rels=nb_rels,
                         reg_wt=reg_wt, seed=seed, verbose=verbose, log_interval=log_interval)

        self.margin = margin

    def score_triples(self, sub_em, rel_em, obj_em):
        """ Compute DistMult scores for a set of triples given their component _embeddings

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
            model scores for the original triples of the given _embeddings
        """
        em_interactions = sub_em * rel_em * obj_em
        scores = tf.reduce_sum(em_interactions, axis=1)
        return scores

    def compute_loss(self, scores, *args, **kwargs):
        """ Compute TransE training loss using the pairwise hinge loss

        Parameters
        ----------
        scores: tf.Tenor
            scores tensor
        args: list
            Non-Key arguments
        kwargs: dict
            Key arguments

        Returns
        -------
        tf.float32
            model loss value
        """
        # run the pointwise hinge loss as a default loss
        if self.loss == "default":
            pos_scores, neg_scores = tf.split(scores, num_or_size_splits=2)
            targets = tf.concat((tf.ones(tf.shape(pos_scores)), -1 * tf.ones(tf.shape(neg_scores))), axis=0)
            return pointwise_square_error_loss(scores, targets=targets, margin=self.margin, reduction_type="avg")
        else:
            return compute_kge_loss(scores, self.loss, reduction_type="avg")


class ComplEx(KnowledgeGraphEmbeddingModel):
    """
    The ComplEx embedding model (ComplEx)
    """

    def __init__(self, em_size=100, batch_size=128, nb_epochs=100, initialiser="xavier_uniform", nb_negs=2,
                 optimiser="amsgrad", loss="default", lr=0.01, nb_ents=0, nb_rels=0, reg_wt=0.01, seed=1234, verbose=1,
                 log_interval=5):
        """ Initialise new instance of the ComplEx model

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
        nb_ents: int
            total number of knowledge graph entities
        nb_rels: int
            total number of knowledge graph relations
        reg_wt: float
            regularisation parameter weight
        seed: int
            random seed
        verbose: int
            verbosity level. options are {0, 1, 2}
        log_interval: int
            the number of epochs to wait until reporting the next training loss. (loss logging frequency)
        """
        super().__init__(em_size=em_size, batch_size=batch_size, nb_epochs=nb_epochs, initialiser=initialiser,
                         nb_negs=nb_negs, optimiser=optimiser, lr=lr, loss=loss, nb_ents=nb_ents, nb_rels=nb_rels,
                         reg_wt=reg_wt, seed=seed, verbose=verbose, log_interval=log_interval)

    def init_embeddings(self):
        """ Initialise the ComplEx _embeddings for both entities and relations. ComplEx _embeddings are twice the size
        of normal _embeddings as they use two embedding vectors: real and imaginary.

        Returns
        -------
        tf.Variable
            _embeddings of knowledge graph entities
        tf.Variable
            _embeddings of knowledge graph relations
        """
        # get initialiser variable and initialise tensorflow variables for each component _embeddings
        var_init = get_initializer(self.initialiser, self.seed)
        em_ents = tf.get_variable("em_ents", shape=[self.nb_ents+1, self.em_size*2], initializer=var_init)
        em_rels = tf.get_variable("em_rels", shape=[self.nb_rels+1, self.em_size*2], initializer=var_init)

        # add component embedding to the embedding vars dictionary
        self._embeddings["ents"] = em_ents
        self._embeddings["rels"] = em_rels

        return em_ents, em_rels

    def score_triples(self, sub_em, rel_em, obj_em):
        """ Compute TransE scores for a set of triples given their component _embeddings

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
            model scores for the original triples of the given _embeddings
        """
        # extract complex (real and imaginary) _embeddings
        sub_em_real, sub_em_imag = tf.split(value=sub_em, num_or_size_splits=2, axis=1)
        rel_em_real, rel_em_imag = tf.split(value=rel_em, num_or_size_splits=2, axis=1)
        obj_em_real, obj_em_imag = tf.split(value=obj_em, num_or_size_splits=2, axis=1)

        # compute the real part of the complex embedding product
        em_prod_real = rel_em_real * sub_em_real * obj_em_real \
                       + rel_em_real * sub_em_imag * obj_em_imag \
                       + rel_em_imag * sub_em_real * obj_em_imag \
                       + rel_em_imag * sub_em_imag * obj_em_real

        # compute scores (tf.nn.sigmoid wrapper is an option)
        scores = tf.reduce_sum(em_prod_real, axis=1)
        return scores

    def compute_loss(self, scores, *args, **kwargs):
        """ Compute TransE training loss using the pairwise hinge loss

        Parameters
        ----------
        scores: tf.Tenor
            scores tensor
        args: list
            Non-Key arguments
        kwargs: dict
            Key arguments

        Returns
        -------
        tf.float32
            model loss value
        """
        if self.loss == "default":
            pos_scores, neg_scores = tf.split(scores, num_or_size_splits=2)
            targets = tf.concat((tf.ones(tf.shape(pos_scores)), -1 * tf.ones(tf.shape(neg_scores))), axis=0)
            return pointwise_logistic_loss(scores, targets=targets, reduction_type="avg")
        else:
            return compute_kge_loss(scores, self.loss, reduction_type="avg")


class TriModel(KnowledgeGraphEmbeddingModel):
    """
    The TriModel embedding model (TriModel)
    """

    def __init__(self, em_size=100, batch_size=128, nb_epochs=100, initialiser="xavier_uniform", nb_negs=2,
                 optimiser="amsgrad", loss="default", lr=0.01, nb_ents=0, nb_rels=0, reg_wt=0.01, seed=1234, verbose=1,
                 log_interval=5):
        """ Initialise new instance of the TriModel model

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
        nb_ents: int
            total number of knowledge graph entities
        nb_rels: int
            total number of knowledge graph relations
        reg_wt: float
            regularisation parameter weight
        seed: int
            random seed
        verbose: int
            verbosity level. options are {0, 1, 2}
        log_interval: int
            the number of epochs to wait until reporting the next training loss. (loss logging frequency)
        """
        super().__init__(em_size=em_size, batch_size=batch_size, nb_epochs=nb_epochs, initialiser=initialiser,
                         nb_negs=nb_negs, optimiser=optimiser, lr=lr, loss=loss, nb_ents=nb_ents, nb_rels=nb_rels,
                         reg_wt=reg_wt, seed=seed, verbose=verbose, log_interval=log_interval)

    def init_embeddings(self):
        """ Initialise the TriModel _embeddings for both entities and relations. Tri _embeddings are three times the size
        of normal _embeddings as they use three embedding vectors for each entity and relation.

        Returns
        -------
        tf.Variable
            _embeddings of knowledge graph entities
        tf.Variable
            _embeddings of knowledge graph relations
        """
        # get initialiser variable and initialise tensorflow variables for each component _embeddings
        var_init = get_initializer(self.initialiser, self.seed)
        em_ents = tf.get_variable("em_ents", shape=[self.nb_ents+1, self.em_size*3], initializer=var_init)
        em_rels = tf.get_variable("em_rels", shape=[self.nb_rels+1, self.em_size*3], initializer=var_init)

        # add component embedding to the embedding vars dictionary
        self._embeddings["ents"] = em_ents
        self._embeddings["rels"] = em_rels

        return em_ents, em_rels

    def score_triples(self, sub_em, rel_em, obj_em):
        """ Compute TriModel scores for a set of triples given their component _embeddings

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
            model scores for the original triples of the given _embeddings
        """
        sub_em_v1, sub_em_v2, sub_em_v3 = tf.split(value=sub_em, num_or_size_splits=3, axis=1)
        rel_em_v1, rel_em_v2, rel_em_v3 = tf.split(value=rel_em, num_or_size_splits=3, axis=1)
        obj_em_v1, obj_em_v2, obj_em_v3 = tf.split(value=obj_em, num_or_size_splits=3, axis=1)

        # compute the interaction vectors of the tri-vector _embeddings
        em_interaction = sub_em_v1*rel_em_v1*obj_em_v3 \
                         + sub_em_v2*rel_em_v2*obj_em_v2 \
                         + sub_em_v3*rel_em_v3*obj_em_v1

        scores = tf.reduce_sum(em_interaction, axis=1)
        return scores

    def compute_loss(self, scores, *args, **kwargs):
        """ Compute TransE training loss using the pairwise hinge loss

        Parameters
        ----------
        scores: tf.Tenor
            scores tensor
        args: list
            Non-Key arguments
        kwargs: dict
            Key arguments

        Returns
        -------
        tf.float32
            model loss value
        """
        if self.loss == "default":
            pos_scores, neg_scores = tf.split(scores, num_or_size_splits=2)
            targets = tf.concat((tf.ones(tf.shape(pos_scores)), -1 * tf.ones(tf.shape(neg_scores))), axis=0)
            return pointwise_logistic_loss(scores, targets=targets, reduction_type="avg")
        else:
            return compute_kge_loss(scores, self.loss, reduction_type="avg")
