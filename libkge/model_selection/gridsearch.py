import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._search import BaseSearchCV, _check_param_grid
from .eval import evaluate_kge_model_lp


# define the model eval scoring part
def lp_scorer(estimator, x_data, y_data=None):
    """ A KGE scorer for the link prediction task

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
    x_data : np.ndarray
        evaluation data array
    y_data : np.ndarray
        evaluation labels array

    Returns
    -------
    float
        evaluation output MRR filtered score

    """
    model_obj = estimator
    if type(estimator) == Pipeline:
        for step in estimator.steps:
            if callable(getattr(step[1], "score_triples", None)):
                model_obj = step[1]
                break
    results = evaluate_kge_model_lp(model_obj, x_data, known_triples=x_data, verbose=0)
    return results['filtered']['avg']['mrr']


class KGEGridSearch(BaseSearchCV):
    """Brute-force hyper-parameters gridsearch for knowledge graph embeddings models
    """
    def __init__(self, estimator, param_grid, known_triples=None, cv=None, n_jobs=1, refit=True, verbose=0):
        """ Initialise an object of the KGEGridSearch class

        Parameters
        ----------
        estimator : estimator object.
            This is assumed to implement the scikit-learn estimator interface.
        param_grid : dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.
        known_triples : np.ndarray
            array of known triples
        cv : int, cross-validation generator or an iterable, optional
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:
              - None, to use the default KGE train/valid gridsearch,
              - integer, to specify the number of folds in a scikit-learn `(Stratified)KFold`,
              - An object to be used as a cross-validation generator.
              - An iterable yielding train, test splits.
        n_jobs : int
            number of parallel jobs
        verbose : int
            level of logging verbosity
        """

        super(KGEGridSearch, self).__init__(estimator=estimator,
                                            fit_params=None,
                                            n_jobs=n_jobs,
                                            iid=True,
                                            refit=refit,
                                            cv=cv,
                                            verbose=verbose,
                                            pre_dispatch='2*n_jobs',
                                            error_score='raise',
                                            return_train_score=False)

        self.cv = cv
        self.param_grid = param_grid
        self.valid_data = None
        self.known_triples = known_triples
        _check_param_grid(param_grid)

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""
        return ParameterGrid(self.param_grid)

    def fit(self, X, X_valid=None, y=None, y_valid=None, **fit_params):
        """ Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        X_valid : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        y_valid : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator

        """
        all_data = X
        if X_valid is not None:
            all_data = np.concatenate([X, X_valid], axis=0)
            self.valid_data = X_valid
        elif self.cv is None:
            raise ValueError("Cross validation config (cv) is None, either specify cv value .eg, 3 or 5, "
                             "or specify validation data (valid_data)")

        if self.cv is None:
            all_indices = np.arange(len(all_data))
            self.cv = [[all_indices[:len(X)], all_indices[len(X):]]]
        train_triples = X

        self.scoring = lp_scorer
        return super().fit(X=all_data, y=None, groups=None, **fit_params)
