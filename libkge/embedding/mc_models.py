# -*- coding: utf-8 -*-
import tensorflow as tf
from .base_model import KnowledgeGraphEmbeddingModelMCL, get_initializer


class DistMult_MCL(KnowledgeGraphEmbeddingModelMCL):
    """
    The DistMult embedding model (DistMult) with multi class loss
    """

    def __init__(self, em_size=100, batch_size=128, nb_epochs=100, nb_negs=-1, dropout=0.02, optimiser="amsgrad",
                 lr=0.01, loss="nll", nb_ents=0, nb_rels=0, reg_wt=0.03, predict_batch_size=40000, seed=1234,
                 verbose=1, log_interval=5, initialiser="xavier_uniform"):
        """ Initialise new instance of the DistMult model

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
            dropout probability
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
        super().__init__(em_size=em_size, batch_size=batch_size, nb_epochs=nb_epochs, dropout=dropout,
                         optimiser=optimiser, lr=lr, loss=loss, nb_ents=nb_ents, nb_rels=nb_rels, reg_wt=reg_wt,
                         predict_batch_size=predict_batch_size, seed=seed, verbose=verbose, log_interval=log_interval,
                         initialiser=initialiser)

    def init_embeddings(self, initializer='xavier_uniform', *args, **kwargs):
        ent_embed_shape = [self.nb_ents, self.em_size]
        rel_embed_shape = [self.nb_ents, self.em_size]
        var_init = get_initializer(initializer=initializer, seed=self.seed)

        self._embeddings["Ee"] = Ee = tf.get_variable('Ee', initializer=var_init, shape=ent_embed_shape)

        self._embeddings["Er"] = Er = tf.get_variable('Er', initializer=var_init, shape=rel_embed_shape)

        return Ee, Er

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
        subs_em = tf.nn.embedding_lookup(self._embeddings["Ee"], triples[:, 0])

        rels_em = tf.nn.embedding_lookup(self._embeddings["Er"], triples[:, 1])

        objs_em = tf.nn.embedding_lookup(self._embeddings["Ee"], triples[:, 2])

        return subs_em, rels_em, objs_em

    def score_triples_mc(self, sub_em, rel_em, obj_em):
        """ score triplets suing the ComplEx scoring function in a multi-class fashion

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
        tuple:
            scores of all possible subject and object corruptions
        """
        ent_em = self._embeddings["Ee"]

        sub_em = tf.nn.dropout(sub_em, keep_prob=1 - self.dropout, seed=self.seed)
        rel_em = tf.nn.dropout(rel_em, keep_prob=1 - self.dropout, seed=self.seed)
        obj_em = tf.nn.dropout(obj_em, keep_prob=1 - self.dropout, seed=self.seed)

        scores_subs_p1 = tf.matmul(rel_em * obj_em, ent_em, transpose_b=True)
        scores_objs_p1 = tf.matmul(rel_em * sub_em, ent_em, transpose_b=True)

        scores_subs = scores_subs_p1
        scores_objs = scores_objs_p1

        return tf.nn.softmax(scores_subs), tf.nn.softmax(scores_objs)

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


class ComplEx_MCL(KnowledgeGraphEmbeddingModelMCL):
    """
    The complex embedding model (ComplEx) with multi class loss
    """

    def __init__(self, em_size=100, batch_size=128, nb_epochs=100, nb_negs=-1, dropout=0.02, optimiser="amsgrad",
                 lr=0.01, loss="nll", nb_ents=0, nb_rels=0, reg_wt=0.03, predict_batch_size=40000, seed=1234,
                 verbose=1, log_interval=5, initialiser="xavier_uniform"):
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
        dropout : float
            dropout probability
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
        super().__init__(em_size=em_size, batch_size=batch_size, nb_epochs=nb_epochs, dropout=dropout,
                         optimiser=optimiser, lr=lr, loss=loss, nb_ents=nb_ents, nb_rels=nb_rels, reg_wt=reg_wt,
                         predict_batch_size=predict_batch_size, seed=seed, verbose=verbose, log_interval=log_interval,
                         initialiser=initialiser)

    def init_embeddings(self, initializer='xavier_uniform', *args, **kwargs):
        ent_embed_shape = [self.nb_ents, self.em_size]
        rel_embed_shape = [self.nb_ents, self.em_size]
        var_init = get_initializer(initializer=initializer, seed=self.seed)

        self._embeddings["Ee1"] = Ee1 = tf.get_variable('Ee1', initializer=var_init, shape=ent_embed_shape)
        self._embeddings["Ee2"] = Ee2 = tf.get_variable('Ee2', initializer=var_init, shape=ent_embed_shape)

        self._embeddings["Er1"] = Er1 = tf.get_variable('Er1', initializer=var_init, shape=rel_embed_shape)
        self._embeddings["Er2"] = Er2 = tf.get_variable('Er2', initializer=var_init, shape=rel_embed_shape)

        return (Ee1, Ee2), (Er1, Er2)

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
        subs_em1 = tf.nn.embedding_lookup(self._embeddings["Ee1"], triples[:, 0])
        subs_em2 = tf.nn.embedding_lookup(self._embeddings["Ee2"], triples[:, 0])

        rels_em1 = tf.nn.embedding_lookup(self._embeddings["Er1"], triples[:, 1])
        rels_em2 = tf.nn.embedding_lookup(self._embeddings["Er2"], triples[:, 1])

        objs_em1 = tf.nn.embedding_lookup(self._embeddings["Ee1"], triples[:, 2])
        objs_em2 = tf.nn.embedding_lookup(self._embeddings["Ee2"], triples[:, 2])

        return (subs_em1, subs_em2), (rels_em1, rels_em2), (objs_em1, objs_em2)

    def score_triples_mc(self, sub_em, rel_em, obj_em):
        """ score triplets suing the ComplEx scoring function in a multi-class fashion

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
        tuple:
            scores of all possible subject and object corruptions
        """
        ee_real, ee_img, = self._embeddings["Ee1"], self._embeddings["Ee2"]

        sub_em1, sub_em2 = sub_em
        rel_em1, rel_em2 = rel_em
        obj_em1, obj_em2 = obj_em

        sub_em_real = tf.nn.dropout(sub_em1, keep_prob=1 - self.dropout, seed=self.seed)
        sub_em_img = tf.nn.dropout(sub_em2, keep_prob=1 - self.dropout, seed=self.seed)

        rel_em_real = tf.nn.dropout(rel_em1, keep_prob=1 - self.dropout, seed=self.seed)
        rel_em_img = tf.nn.dropout(rel_em2, keep_prob=1 - self.dropout, seed=self.seed)

        obj_em_real = tf.nn.dropout(obj_em1, keep_prob=1 - self.dropout, seed=self.seed)
        obj_em_img = tf.nn.dropout(obj_em2, keep_prob=1 - self.dropout, seed=self.seed)

        scores_subs_p1 = tf.matmul(rel_em_real * obj_em_real, ee_real, transpose_b=True)
        scores_subs_p2 = tf.matmul(rel_em_real * obj_em_img, ee_img, transpose_b=True)
        scores_subs_p3 = tf.matmul(rel_em_img * obj_em_img, ee_real, transpose_b=True)
        scores_subs_p4 = tf.matmul(rel_em_img * obj_em_real, ee_img, transpose_b=True)

        scores_objs_p1 = tf.matmul(rel_em_real * sub_em_real, ee_real, transpose_b=True)
        scores_objs_p2 = tf.matmul(rel_em_real * sub_em_img, ee_img, transpose_b=True)
        scores_objs_p3 = tf.matmul(rel_em_img * sub_em_real, ee_img, transpose_b=True)
        scores_objs_p4 = tf.matmul(rel_em_img * sub_em_img, ee_real, transpose_b=True)

        scores_subs = scores_subs_p1 + scores_subs_p2 + scores_subs_p3 - scores_subs_p4
        scores_objs = scores_objs_p1 + scores_objs_p2 + scores_objs_p3 - scores_objs_p4

        return tf.nn.softmax(scores_subs), tf.nn.softmax(scores_objs)

    def score_triples(self, sub_em, rel_em, obj_em):
        """ Compute ComplEx scores for a set of triples given their component _embeddings

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
        sub_em_real, sub_em_imag = sub_em
        rel_em_real, rel_em_imag = rel_em
        obj_em_real, obj_em_imag = obj_em

        # compute the real part of the complex embedding product
        em_prod_real = rel_em_real * sub_em_real * obj_em_real \
                       + rel_em_real * sub_em_imag * obj_em_imag \
                       + rel_em_imag * sub_em_real * obj_em_imag \
                       + rel_em_imag * sub_em_imag * obj_em_real

        # compute scores (tf.nn.sigmoid wrapper is an option)
        scores = tf.reduce_sum(em_prod_real, axis=1)
        return scores

    def embedding_regularisation(self, sub_em, rel_em, obj_em, *args, **kwargs):
        """ Model regularisation function

        Parameters
        ----------
        Parameters
        ----------
        sub_em: tf.tensor
            Embeddings of the subject entities
        rel_em: tf.tensor
            Embeddings of the relations
        obj_em: tf.tensor
            Embeddings of the object entities
        args
        kwargs

        Returns
        -------
        tf.tensor:
            regularisation value
        """
        sub_em1, sub_em2 = sub_em
        rel_em1, rel_em2 = rel_em
        obj_em1, obj_em2 = obj_em

        # the nuclear 3-norm regularisation
        reg = tf.reduce_sum(
            tf.pow(tf.abs(sub_em1), 3) +
            tf.pow(tf.abs(sub_em2), 3) +
            tf.pow(tf.abs(rel_em1), 3) +
            tf.pow(tf.abs(rel_em2), 3) +
            tf.pow(tf.abs(obj_em1), 3) +
            tf.pow(tf.abs(obj_em2), 3))

        return self.reg_wt/3 * reg


class TriModel_MCL(KnowledgeGraphEmbeddingModelMCL):
    """
    The TriModel embedding model (TriModel) with multi class loss
    """

    def __init__(self, em_size=100, batch_size=128, nb_epochs=100, nb_negs=-1, dropout=0.02, optimiser="amsgrad",
                 lr=0.01, loss="nll", nb_ents=0, nb_rels=0, reg_wt=0.03, predict_batch_size=40000, seed=1234,
                 verbose=1, log_interval=5, initialiser="xavier_uniform"):
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
        dropout : float
            dropout probability
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
        super().__init__(em_size=em_size, batch_size=batch_size, nb_epochs=nb_epochs, dropout=dropout,
                         optimiser=optimiser, lr=lr, loss=loss, nb_ents=nb_ents, nb_rels=nb_rels, reg_wt=reg_wt,
                         predict_batch_size=predict_batch_size, seed=seed, verbose=verbose, log_interval=log_interval,
                         initialiser=initialiser)

    def init_embeddings(self, initializer='xavier_uniform', *args, **kwargs):
        ent_embed_shape = [self.nb_ents, self.em_size]
        rel_embed_shape = [self.nb_ents, self.em_size]
        var_init = get_initializer(initializer=initializer, seed=self.seed)

        self._embeddings["Ee1"] = Ee1 = tf.get_variable('Ee1', initializer=var_init, shape=ent_embed_shape)
        self._embeddings["Ee2"] = Ee2 = tf.get_variable('Ee2', initializer=var_init, shape=ent_embed_shape)
        self._embeddings["Ee3"] = Ee3 = tf.get_variable('Ee3', initializer=var_init, shape=ent_embed_shape)

        self._embeddings["Er1"] = Er1 = tf.get_variable('Er1', initializer=var_init, shape=rel_embed_shape)
        self._embeddings["Er2"] = Er2 = tf.get_variable('Er2', initializer=var_init, shape=rel_embed_shape)
        self._embeddings["Er3"] = Er3 = tf.get_variable('Er3', initializer=var_init, shape=rel_embed_shape)

        return (Ee1, Ee2, Ee3), (Er1, Er2, Er3)

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
        subs_em1 = tf.nn.embedding_lookup(self._embeddings["Ee1"], triples[:, 0])
        subs_em2 = tf.nn.embedding_lookup(self._embeddings["Ee2"], triples[:, 0])
        subs_em3 = tf.nn.embedding_lookup(self._embeddings["Ee3"], triples[:, 0])

        rels_em1 = tf.nn.embedding_lookup(self._embeddings["Er1"], triples[:, 1])
        rels_em2 = tf.nn.embedding_lookup(self._embeddings["Er2"], triples[:, 1])
        rels_em3 = tf.nn.embedding_lookup(self._embeddings["Er3"], triples[:, 1])

        objs_em1 = tf.nn.embedding_lookup(self._embeddings["Ee1"], triples[:, 2])
        objs_em2 = tf.nn.embedding_lookup(self._embeddings["Ee2"], triples[:, 2])
        objs_em3 = tf.nn.embedding_lookup(self._embeddings["Ee3"], triples[:, 2])

        return (subs_em1, subs_em2, subs_em3), (rels_em1, rels_em2, rels_em3), (objs_em1, objs_em2, objs_em3)

    def score_triples_mc(self, sub_em, rel_em, obj_em):
        """ score triplets suing the TriModel scoring function in a multi-class fashion

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
        tuple:
            scores of all possible subject and object corruptions
        """
        ee1, ee2, ee3 = self._embeddings["Ee1"], self._embeddings["Ee2"], self._embeddings["Ee3"]

        sub_em1, sub_em2, sub_em3 = sub_em
        rel_em1, rel_em2, rel_em3 = rel_em
        obj_em1, obj_em2, obj_em3 = obj_em

        sub_em1 = tf.nn.dropout(sub_em1, keep_prob=1 - self.dropout, seed=self.seed)
        sub_em2 = tf.nn.dropout(sub_em2, keep_prob=1 - self.dropout, seed=self.seed)
        sub_em3 = tf.nn.dropout(sub_em3, keep_prob=1 - self.dropout, seed=self.seed)

        rel_em1 = tf.nn.dropout(rel_em1, keep_prob=1 - self.dropout, seed=self.seed)
        rel_em2 = tf.nn.dropout(rel_em2, keep_prob=1 - self.dropout, seed=self.seed)
        rel_em3 = tf.nn.dropout(rel_em3, keep_prob=1 - self.dropout, seed=self.seed)

        obj_em1 = tf.nn.dropout(obj_em1, keep_prob=1 - self.dropout, seed=self.seed)
        obj_em2 = tf.nn.dropout(obj_em2, keep_prob=1 - self.dropout, seed=self.seed)
        obj_em3 = tf.nn.dropout(obj_em3, keep_prob=1 - self.dropout, seed=self.seed)

        scores_subs_i1 = tf.matmul(obj_em3 * rel_em1, ee1, transpose_b=True)
        scores_subs_i2 = tf.matmul(obj_em2 * rel_em2, ee2, transpose_b=True)
        scores_subs_i3 = tf.matmul(obj_em1 * rel_em3, ee3, transpose_b=True)

        scores_objs_i1 = tf.matmul(sub_em1 * rel_em1, ee3, transpose_b=True)
        scores_objs_i2 = tf.matmul(sub_em2 * rel_em2, ee2, transpose_b=True)
        scores_objs_i3 = tf.matmul(sub_em3 * rel_em3, ee1, transpose_b=True)

        scores_subs = scores_subs_i1 + scores_subs_i2 + scores_subs_i3
        scores_objs = scores_objs_i1 + scores_objs_i2 + scores_objs_i3

        return tf.nn.softmax(scores_subs), tf.nn.softmax(scores_objs)

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
        sub_em1, sub_em2, sub_em3 = sub_em
        rel_em1, rel_em2, rel_em3 = rel_em
        obj_em1, obj_em2, obj_em3 = obj_em

        # compute the interaction vectors of the tri-vector _embeddings
        em_interaction = sub_em1*rel_em1*obj_em3 + sub_em2*rel_em2*obj_em2 + sub_em3*rel_em3*obj_em1
        scores = tf.reduce_sum(em_interaction, axis=1)
        return scores

    def embedding_regularisation(self, sub_em, rel_em, obj_em, *args, **kwargs):
        """ Model regularisation function

        Parameters
        ----------
        Parameters
        ----------
        sub_em: tf.tensor
            Embeddings of the subject entities
        rel_em: tf.tensor
            Embeddings of the relations
        obj_em: tf.tensor
            Embeddings of the object entities
        args
        kwargs

        Returns
        -------
        tf.tensor:
            regularisation value
        """
        sub_em1, sub_em2, sub_em3 = sub_em
        rel_em1, rel_em2, rel_em3 = rel_em
        obj_em1, obj_em2, obj_em3 = obj_em

        # the nuclear 3-norm regularisation
        reg = tf.reduce_sum(
            tf.pow(tf.abs(sub_em1), 3) +
            tf.pow(tf.abs(sub_em2), 3) +
            tf.pow(tf.abs(sub_em3), 3) +
            tf.pow(tf.abs(rel_em1), 3) +
            tf.pow(tf.abs(rel_em2), 3) +
            tf.pow(tf.abs(rel_em3), 3) +
            tf.pow(tf.abs(obj_em1), 3) +
            tf.pow(tf.abs(obj_em2), 3) +
            tf.pow(tf.abs(obj_em3), 3))

        return self.reg_wt / 3 * reg
