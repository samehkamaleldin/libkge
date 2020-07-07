import tensorflow as tf


def reduce_loss(loss_vector, reduction_type):
    """ Reduce loss vector

    Parameters
    ----------
    loss_vector
    reduction_type

    Returns
    -------

    """
    if reduction_type == "sum":
        return tf.reduce_sum(loss_vector)
    elif reduction_type == "avg" or reduction_type == "average":
        return tf.reduce_mean(loss_vector)
    elif reduction_type == "none" or reduction_type == "raw":
        return loss_vector
    else:
        raise ValueError("Unknown reduction type (%s). options are ['sum', 'avg', 'none']" % reduction_type)


def pointwise_hinge_loss(scores, targets, margin=1.0, reduction_type="sum", *args, **kwargs):
    """ Pairwise hinge loss

    Parameters
    ----------
    scores : tf.tensor
        Tensor containing (N,) scores one for each example.
    targets : tf.tensor
        Tensor containing (N,) labels one for each example.
    margin: float
        Margin value.
    reduction_type: str
        loss reduction technique. options ['sum', 'avg']
    args: list
            Non-key arguments
    kwargs: dict
        Key arguments

    Returns
    -------
    tf.float32
        Loss value.
    """
    hinge_losses = tf.nn.relu(margin - targets * scores)
    return reduce_loss(hinge_losses, reduction_type)


def pairwise_hinge_loss(positive_scores, negative_scores, margin=1.0, reduction_type="sum", *args, **kwargs):
    """ Pairwise hinge loss

    Parameters
    ----------
    positive_scores : tf.tensor
        Tensor containing (N,) scores of positive examples.
    negative_scores : tf.tensor
        Tensor containing (N,) scores of negative examples.
    margin: float
        Margin value.
    reduction_type: str
        loss reduction technique. options ['sum', 'avg']
    args: list
            Non-key arguments
    kwargs: dict
        Key arguments

    Returns
    -------
    tf.float32
        Loss value.
    """
    hinge_losses = tf.nn.relu(margin + negative_scores - positive_scores)
    return reduce_loss(hinge_losses, reduction_type)


def pointwise_logistic_loss(scores, targets, reduction_type="sum", *args, **kwargs):
    """ Negative log-likelihood loss.

    Parameters
    ----------
    scores : tf.tensor
        Tensor containing (N,) scores one for each example.
    targets : tf.tensor
        Tensor containing (N,) labels one for each example.
    reduction_type: str
        loss reduction technique. options ['sum', 'avg']
    args: list
            Non-key arguments
    kwargs: dict
        Key arguments

    Returns
    -------
    tf.float32
        Loss value.
    """
    logistic_losses = tf.nn.softplus(-targets * scores)
    return reduce_loss(logistic_losses, reduction_type)


def pairwise_logistic_loss(positive_scores, negative_scores, reduction_type="sum", *args, **kwargs):
    """ Negative log-likelihood loss.

    Parameters
    ----------
    positive_scores : tf.tensor
        Tensor containing (N,) positive scores one for each example.
    negative_scores : tf.tensor
        Tensor containing (N,) negative scores one for each example.
    reduction_type: str
        loss reduction technique. options ['sum', 'avg']
    args: list
            Non-key arguments
    kwargs: dict
        Key arguments

    Returns
    -------
    tf.float32
        Loss value.
    """
    logistic_losses = tf.nn.softplus(negative_scores - positive_scores)
    return reduce_loss(logistic_losses, reduction_type)


def pointwise_square_error_loss(scores, targets, reduction_type="sum", *args, **kwargs):
    """Compute pairwise square loss

    Parameters
    ----------
    scores : tf.tensor
        Tensor containing (N,) scores one for each example.
    targets : tf.tensor
        Tensor containing (N,) labels one for each example.
    reduction_type: str
        loss reduction technique. options ['sum', 'avg']
    args: list
        Non-key arguments
    kwargs: dict
        Key arguments

    Returns
    -------
    tf.float32
        Loss value.

    """
    squared_error_losses = tf.square(scores - targets)
    return reduce_loss(squared_error_losses, reduction_type)


def compute_kge_loss(scores, loss_type, reduction_type="sum", *args, **kwargs):
    """ Compute loss function.

    Parameters
    ----------
    scores: tf.tensor
        (N,) tensorflow tensor with all batch triples score.
    loss_type: str
        loss function type
    reduction_type: str
        loss reduction technique. options ['sum', 'avg']
    args: list
        Non-key arguments
    kwargs: dict
        Key arguments

    Returns
    -------
    tf.float32
        Model loss.
    """
    positive_scores, negative_scores = tf.split(value=scores, num_or_size_splits=2, axis=0)
    targets = tf.concat((tf.ones(tf.shape(positive_scores)), -1 * tf.ones(tf.shape(negative_scores))), axis=0)

    if loss_type == "pt_sel" or loss_type == "pointwise_square_error_loss" or loss_type == "pt_se":
        targets = (targets + 1) / 2
        loss = pointwise_square_error_loss(scores, targets, reduction_type=reduction_type, *args, **kwargs)

    elif loss_type == "pt_log" or loss_type == "pointwise_log_loss":
        loss = pointwise_logistic_loss(scores, targets, *args, **kwargs)

    elif loss_type == "pt_hinge" or loss_type == "pointwise_hinge_loss":
        loss = pointwise_hinge_loss(scores, targets, *args, **kwargs)

    elif loss_type == "pr_hinge" or loss_type == "pairwise_hinge_loss":
        loss = pairwise_hinge_loss(positive_scores, negative_scores, *args, **kwargs)

    elif loss_type == "pr_log" or loss_type == "pairwise_logistic_loss":
        loss = pairwise_logistic_loss(positive_scores, negative_scores, *args, **kwargs)

    else:
        raise ValueError("Unknown loss type (%s)" % loss_type)

    return loss


def mc_softmax_negative_log_loss(score_matrix, true_indices, reduction_type="sum", *args, **kwargs):
    """ Compute the softmax negative-log loss for a score matrix of a data batch

    Parameters
    ----------
    score_matrix : tf.tensor
        data scores matrix of size [N, M] where N is the data size and M is the number of scores per instance
    true_indices : tf.tensor
        indices of the true triples of the scoring matrix where each index represent the index of the only true
        instance of a matrix row of scores.
    reduction_type: str
        loss reduction technique. options ['sum', 'avg']
    args: list
        Non-key arguments
    kwargs: dict
        Key arguments
    Returns
    -------
    tf.float
        the loss value
    """
    # apply softmax on scores
    score_matrix = tf.nn.softmax(score_matrix)

    # clip score values to fit the neg-log loss constraints
    eps = 1e-15
    score_matrix = tf.clip_by_value(score_matrix, eps, 1 - eps)

    # get positive scores
    rows = tf.range(tf.shape(score_matrix)[0])
    data_pos_idx = tf.transpose(tf.stack([rows, true_indices]))
    data_pos_scores = tf.gather_nd(score_matrix, data_pos_idx)

    # compute the loss
    data_loss = -tf.log(data_pos_scores)
    return reduce_loss(data_loss, reduction_type)
