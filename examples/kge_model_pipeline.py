import numpy as np
from sklearn.pipeline import Pipeline
from libkge.embedding.models import *
from libkge.embedding.mc_models import *
from libkge.datasets import load_dataset_from_dir
from libkge.model_selection import evaluate_kge_model_lp, print_lp_eval_result


if __name__ == '__main__':

    # load kg dataset
    dataset_dir = "../data/kinship/kinship"
    dataset = load_dataset_from_dir(dataset_dir)
    train_data = dataset.data["train"]
    valid_data = dataset.data["valid"]
    test_data = dataset.data["test"]
    all_data = np.concatenate([train_data, valid_data, test_data])

    nb_ents = dataset.get_ents_count()
    nb_rels = dataset.get_rels_count()

    # model pipeline definition - model's verbosity value must be assigned in the model object initiation
    kge_model = DistMult_MCL(verbose=2)
    model_pipeline = Pipeline([
        ('kge_model', kge_model)
    ])

    # set model parameters
    model_params = {
        'kge_model__em_size': 20,
        'kge_model__lr': 0.01,
        'kge_model__optimiser': "amsgrad",
        'kge_model__loss': "default",
        'kge_model__nb_negs': 2,
        'kge_model__batch_size': 1024,
        'kge_model__nb_epochs': 10,
        'kge_model__initialiser': 'xavier_uniform',
        'kge_model__nb_ents': nb_ents,
        'kge_model__nb_rels': nb_rels
    }

    # assign parameters to the pipeline's base model
    model_pipeline.set_params(**model_params)

    # start the training procedure
    model_pipeline.fit(X=train_data, y=None)
    model_obj = model_pipeline.named_steps["kge_model"]
    results = evaluate_kge_model_lp(model_obj, test_data, known_triples=all_data)
    print_lp_eval_result(results)
