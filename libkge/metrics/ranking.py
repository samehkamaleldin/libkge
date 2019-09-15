import numpy as np


def ranks(y_true, y_pred, pos_label=1.0):
    """ Compute ranks of the true labels in a rank

    Parameters
    ----------
    y_true: np.ndarray
        true labels
    y_pred: np.ndarray
        predicted scores
    pos_label: float
        label of the positive true instances

    Returns
    -------
    np.ndarray
        ranks of the true labels in the rank
    """
    rank_order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[rank_order]
    pos_label_mask = y_true_sorted == pos_label
    return np.nonzero(pos_label_mask)[0] + 1


def mean_rank(y_true, y_pred, pos_label=1.0):
    """ Compute the mean rank of the true labels in a rank

    Parameters
    ----------
    y_true: np.ndarray
        true labels
    y_pred: np.ndarray
        predicted scores
    pos_label: float
        label of the positive true instances

    Returns
    -------
    float
        the mean rank of the true labels in the rank
    """
    return np.mean(ranks(y_true, y_pred, pos_label=pos_label))


def reciprocal_ranks(y_true, y_pred, pos_label=1.0):
    """ Compute reciprocal ranks of the true labels in a rank

    Parameters
    ----------
    y_true: np.ndarray
        true labels
    y_pred: np.ndarray
        predicted scores
    pos_label: float
        label of the positive true instances

    Returns
    -------
    np.ndarray
        reciprocal ranks of the true labels in the rank
    """
    return 1/ranks(y_true, y_pred, pos_label=pos_label)


def mean_reciprocal_ranks(y_true, y_pred, pos_label=1.0):
    """ Compute the mean reciprocal rank of the true labels in a rank

    Parameters
    ----------
    y_true: np.ndarray
        true labels
    y_pred: np.ndarray
        predicted scores
    pos_label: float
        label of the positive true instances

    Returns
    -------
    float
        the mean reciprocal rank of the true labels in the rank
    """
    return np.mean(reciprocal_ranks(y_true, y_pred, pos_label=pos_label))


def precision_at_k(y_true, y_pred, k, pos_label=1.0):
    """ Compute the mean precision at k of a rank of predicted scores

        Parameters
        ----------
        y_true: np.ndarray
            true labels
        y_pred: np.ndarray
            predicted scores
        k: int
            the position `k`
        pos_label: float
            label of the positive true instances

        Returns
        -------
        float
            the mean reciprocal rank of the true labels in the rank
    """
    if k < 1 or k > len(y_true):
        raise ValueError('Invalid k value: %s' % k)

    rank_order = np.argsort(y_pred)[::-1]
    y_true_k_sorted = y_true[rank_order[:k]]
    return np.count_nonzero(y_true_k_sorted == pos_label) / k


def average_precision(y_true, y_pred, pos_label=1.0):
    """ Compute the average precision of a rank of predicted scores

        Parameters
        ----------
        y_true: np.ndarray
            true labels
        y_pred: np.ndarray
            predicted scores
        pos_label: float
            label of the positive true instances

        Returns
        -------
        float
            the mean reciprocal rank of the true labels in the rank
    """
    ranks_array = ranks(y_true, y_pred, pos_label=1.0)

    pk_list = []
    for k in ranks_array:
        pk_list.append(precision_at_k(y_true, y_pred, k, pos_label=pos_label))
    return np.mean(pk_list)
