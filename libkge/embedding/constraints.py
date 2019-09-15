import tensorflow as tf


def tensor_norm(tensor_var, norm_rank=2., norm_val=1.):
    """ Normalization of a tensor to a specific value with a specified rank.

    Parameters
    ----------
    tensor_var: tf.Tensor
        Input tensor
    norm_rank: int
        Norm rank i.e. order
    norm_val: int
        Norm value

    Returns
    -------
    tf.Tensor:
        Normalised tensor
    """
    rows_rank_norms = tf.norm(tensor_var, ord=norm_rank, axis=1, keep_dims=True)
    scaled_tensor = tensor_var * (norm_val / rows_rank_norms)
    return tf.assign(tensor_var, scaled_tensor)


def tensor_unit_norm(tensor_var, norm_rank=2.):
    """ Unit normalization of a tensor with a specific norm rank.

    Parameters
    ----------
    tensor_var: tf.Tensor
        Input tensor
    norm_rank: int
        Unit norm order

    Returns
    -------
    tf.Tensor:
        normalised tensor
    """
    return tensor_norm(tensor_var=tensor_var, norm_rank=norm_rank, norm_val=1)


def unit_sphere_projection(tensor_var):
    """ Unit sphere normalisation of a tensor.

    Parameters
    ----------
    tensor_var : tf.Tensor
        Embeddings tensor.

    Returns
    -------
    tf.Tensor
        Embedding matrix with the unit sphere normalisation applied.
    """
    return tensor_unit_norm(tensor_var, norm_rank=2)


def unit_cube_projection(tensor_var):
    """ Unit cube normalisation of a tensor.

    Parameters
    ----------
    tensor_var : tf.Tensor
        Embeddings tensor.

    Returns
    -------
    tf.Tensor
        Embedding matrix with the unit cube normalisation applied.
    """
    return tensor_unit_norm(tensor_var, norm_rank=4)


def tensor_bounded_update(tensor_var, max_boundary=1., min_boundary=0.):
    """ Bounded normalization of a tensor.

    Parameters
    ----------
    tensor_var : tf.Tensor
        Embeddings tensor.
    max_boundary: int
        max value boundary
    min_boundary: int
        min value boundary


    Returns
    -------
    tf.Tensor
        Embedding matrix with the boundaries applied.
    """
    boundaries_projection = tf.minimum(max_boundary, tf.maximum(tensor_var, min_boundary))
    return tf.assign(tensor_var, boundaries_projection)


def tensor_sigmoid_update(tensor_var):
    """ Sigmoid normalisation of a tensor.

    Parameters
    ----------
    tensor_var : tf.Tensor
        Embeddings tensor.

    Returns
    -------
    tf.Tensor
        Embedding matrix with the sigmoid applied.
    """
    tensor_sigmoid = tf.nn.sigmoid(tensor_var)
    return tf.assign(tensor_var, tensor_sigmoid)


def tensor_tanh_update(tensor_var):
    """ Tanh normalisation of a tensor.

    Parameters
    ----------
    tensor_var : tf.Tensor
        Embeddings tensor.

    Returns
    -------
    tf.Tensor
        Embedding matrix with the tanh applied.
    """
    tensor_tanh = tf.nn.tanh(tensor_var)
    return tf.assign(tensor_var, tensor_tanh)


def get_initializer(initializer, seed=1234):
    """ Get tensorflow initialiser

    Parameters
    ----------
    initializer: str
        initialiser type
    seed: int
        random seed

    Returns
    -------
    init
        initialiser object
    """

    if initializer == 'xavier_uniform':
        var_init = tf.contrib.layers.xavier_initializer(uniform=True, seed=seed)

    elif initializer == 'xavier_normal':
        var_init = tf.contrib.layers.xavier_initializer(uniform=False, seed=seed)
    else:
        raise ValueError("Unknown initialiser type (%s)" % initializer)

    return var_init
