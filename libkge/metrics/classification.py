from sklearn.metrics import roc_auc_score, average_precision_score


def auc_roc(y_true, y_pred):
    """ Compute the area under the ROC curve

    Parameters
    ----------
    y_true: array
        true scores
    y_pred: array
        target scores

    Returns
    -------
    float
        the area under the ROC curve score
    """
    return roc_auc_score(y_true, y_pred)


def auc_pr(y_true, y_pred):
    """ Compute the area under the precision-recall curve. The outcome summarizes a precision-recall curve as the
    weighted mean of precisions achieved at each threshold.

    Parameters
    ----------
    y_true: array
        true scores
    y_pred: array
        target scores

    Returns
    -------
    float
        the area under the ROC curve score
    """
    return average_precision_score(y_true, y_pred)