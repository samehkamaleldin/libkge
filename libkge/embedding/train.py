import numpy as np
from math import ceil, floor
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer


def generate_batches(data, batch_size=128, shuffle=False):
    """ Generate batches for a data array.

    Parameters
    ----------
    data: ndarray
        Input data array
    batch_size: int
        Batch size
    shuffle: bool
        Flag to shuffle the input data before generating batches if true.

    Yields
    -------
    iterator
        The next batch in the sequence of batches of the specified size
    """
    data_size = len(data)
    data_idxs = np.array(range(data_size))
    if shuffle:
        np.random.shuffle(data_idxs)
    nb_batches = int(ceil(data_size/batch_size))
    for idx in range(nb_batches):
        batch_indices = data_idxs[idx * batch_size: min((idx + 1) * batch_size, len(data))]
        yield data[batch_indices]


def generate_rand_negs(triples, nb_corrs, nb_ents, seed, *args, **kwargs):
    """ Generate random negatives for some positive triples.

    Parameters
    ----------
    triples : tf.Tensor
        tensorflow tensor for positive triples with size [?, 3].
    nb_corrs : int
        Number of corruptions to generate per triple.
    nb_ents : int
        Total number of entities.
    seed : int
        Random seed.

    Returns
    ---------
    tf.Tensor
        tensorflow tensor for negative triples of size [?, 3].

    Note
    ---------
    The passed `nb_corrs` is evenly distributed between head and tail corruptions.

    Warning
    ---------
    This corruption heuristic might generate original true triples as corruptions.
    """
    nb_corrs /= 2
    neg_sub_rel, objs = tf.split(tf.tile(triples, [ceil(nb_corrs), 1]), [2, 1], axis=1)
    subs, neg_rel_obj = tf.split(tf.tile(triples, [floor(nb_corrs), 1]), [1, 2], axis=1)

    neg_objs = tf.random_uniform(tf.shape(objs), dtype=tf.int32, minval=0, maxval=nb_ents, seed=seed)
    neg_subs = tf.random_uniform(tf.shape(subs), dtype=tf.int32, minval=0, maxval=nb_ents, seed=seed)

    return tf.concat([tf.concat([neg_subs, neg_rel_obj], axis=1), tf.concat([neg_sub_rel, neg_objs], axis=1)], axis=0)


def init_tf_optimiser(optimiser, lr=0.01, *args, **kwargs):
    """ Initialise tensorflow optimiser object

    Parameters
    ----------
    optimiser: str
        optimiser name
    lr: float
        learning rate
    args: list
        Non-key arguments
    kwargs: dict
        Key arguments

    Returns
    -------
    Optimizer
        tensorflow optimiser
    """

    if optimiser.lower() == "sgd":
        opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif optimiser.lower() == "adagrad":
        opt = tf.train.AdagradOptimizer(learning_rate=lr)
    elif optimiser.lower() == "adam":
        opt = tf.train.AdamOptimizer(learning_rate=lr)
    elif optimiser.lower() == "amsgrad":
        opt = AMSGrad(learning_rate=lr)
    elif optimiser.lower() == "adadelta":
        opt = tf.train.AdadeltaOptimizer(learning_rate=lr)
    else:
        raise ValueError("Unknown optimiser type (%s)." % optimiser)

    return opt


class AMSGrad(optimizer.Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8, use_locking=False, name="AMSGrad"):
        super(AMSGrad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new:
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)
        # Create slots for the first and second moments.
        for v in var_list :
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._beta1_t = ops.convert_to_tensor(self._beta1)
        self._beta2_t = ops.convert_to_tensor(self._beta2)
        self._epsilon_t = ops.convert_to_tensor(self._epsilon)

    def _apply_dense(self, grad, var):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)

        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _resource_apply_dense(self, grad, var):
        var = var.handle
        beta1_power = math_ops.cast(self._beta1_power, grad.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, grad.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, grad.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m").handle
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v").handle
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat").handle
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)

        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)
        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(
                    self._beta1_power * self._beta1_t,
                    use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(
                    self._beta2_power * self._beta2_t,
                    use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                      name=name_scope)


def log_model_params(model_obj):
    """ Log model parameters

    Parameters
    ----------
    model_obj: KnowledgeGraphEmbeddingModel
        KGE model object

    Returns
    -------

    """
    model_attrs = model_obj.__dict__
    for attr in sorted(list(model_attrs.keys())):
        if type(model_attrs[attr]) == str or type(model_attrs[attr]) == int or type(model_attrs[attr]) == float or \
                type(model_attrs[attr]) == bool:
            attr_val = model_attrs[attr]
            model_obj.log.debug("[Parameter] %-20s: %s" % (attr, attr_val))
