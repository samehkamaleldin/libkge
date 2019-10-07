import numpy as np
from sklearn.pipeline import Pipeline
from libkge.datasets import load_dataset_from_dir
from libkge.model_selection import KGEGridSearch, evaluate_kge_model_lp, print_lp_eval_result
from libkge.embedding.models import *


if __name__ == '__main__':

    n_jobs = 1  # for now: values more than (1) will cause a problem
    seed = 1234

    # load kg dataset
    # load kg dataset
    dataset_dir = "../data/kinship/kinship"
    dataset = load_dataset_from_dir(dataset_dir)
    train_data = dataset.data["train"]
    valid_data = dataset.data["valid"]
    test_data = dataset.data["test"]
    all_data = np.concatenate([train_data, valid_data, test_data])

    nb_ents = dataset.get_ents_count()
    nb_rels = dataset.get_rels_count()

    kge_model = DistMult(verbose=0, seed=seed, nb_ents=nb_ents, nb_rels=nb_rels)
    model_pipeline = Pipeline([
        ('kge_model', kge_model)
    ])

    # set model parameters grid
    hyperparams_grid = {
        'kge_model__em_size': [50, 20],
        'kge_model__lr': [0.01, 0.2],
        'kge_model__optimiser': ["amsgrad"],
        'kge_model__loss': ["default"],
        'kge_model__nb_negs': [2],
        'kge_model__batch_size': [128],
        'kge_model__nb_epochs': [10]
    }

    gridsearch = KGEGridSearch(model_pipeline, param_grid=hyperparams_grid, n_jobs=n_jobs, refit=True, verbose=2)

    gridsearch.fit(train_data, valid_data)
    best_params = gridsearch.best_params_
    model_obj = gridsearch.best_estimator_.named_steps["kge_model"]
    print("")
    print("==========================================================")
    print("=                    GRID SCORES                         =")
    print("==========================================================")
    res = gridsearch.cv_results_
    candidates_size = len(res['params'])
    for item_idx in range(candidates_size):
        print("score: %1.3f - params: %s" % (res['mean_test_score'][item_idx], res['params'][item_idx]))

    print("")
    print("==========================================================")
    print("=                   BEST PARAMETERS                      =")
    print("==========================================================")
    for k, v in best_params.items():
        print("[Parameter] %-30s: %s" % (k, v))
    print("==========================================================")
    print(" [BEST SCORE: %1.3f]" % gridsearch.best_score_)
    print("==========================================================")

    print("= Evaluate on test data")
    results = evaluate_kge_model_lp(model_obj, test_data, known_triples=all_data)
    print_lp_eval_result(results)
