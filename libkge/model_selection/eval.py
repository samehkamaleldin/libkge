import numpy as np
from tqdm import tqdm
from ..metrics.classification import auc_roc, auc_pr


def evaluate_kge_model_lp(model, test_triples, known_triples=None, verbose=0):
    """ Evaluate a knowledge graph embedding model using the standard link prediction evaluation protocol.

    Parameters
    ----------
    model : KnowledgeGraphEmbeddingModel
        Model object.
    test_triples : np.ndarray
        evaluation test triples.
    known_triples : np.ndarray or None
        array with all the true known triples. If None, no filtered metrics are computed, only raw ones.
    verbose : int
        level of verbosity.

    Returns
    -------
    dict
        results dictionary with the raw and filtered average metrics.

    Notes
    -------
    This evaluation technique is inspired by the code at:
    https://github.com/ttrouill/complex/blob/master/efe/evaluation.py

    """
    # build indices for (sub, pred) and (pred, obj) pairs
    known_sub_triples = {}
    known_obj_triples = {}
    if known_triples is not None:
        for sub_id, rel_id, obj_id in known_triples:
            if (sub_id, rel_id) not in known_obj_triples:
                known_obj_triples[(sub_id, rel_id)] = [obj_id]
            elif obj_id not in known_obj_triples[(sub_id, rel_id)]:
                known_obj_triples[(sub_id, rel_id)].append(obj_id)
            if (rel_id, obj_id) not in known_sub_triples:
                known_sub_triples[(rel_id, obj_id)] = [sub_id]
            elif sub_id not in known_sub_triples[(rel_id, obj_id)]:
                known_sub_triples[(rel_id, obj_id)].append(sub_id)

    nb_test = len(test_triples)
    sub_ranks = np.zeros(nb_test, dtype=np.int32)
    sub_ranks_fl = np.zeros(nb_test, dtype=np.int32)
    obj_ranks = np.zeros(nb_test, dtype=np.int32)
    obj_ranks_fl = np.zeros(nb_test, dtype=np.int32)
    data_hash = hash(test_triples.data.tobytes())

    test_instances = enumerate(test_triples) if verbose == 0 else enumerate(tqdm(test_triples))
    for idx, (sub_id, rel_id, obj_id) in test_instances:

        # generate all possible subject corruption
        sub_corr = np.concatenate(
            [np.arange(model.nb_ents).reshape([-1, 1]), np.tile([rel_id, obj_id], [model.nb_ents, 1])], axis=1)

        # generate all possible object corruption
        obj_corr = np.concatenate(
            [np.tile([sub_id, rel_id], [model.nb_ents, 1]), np.arange(model.nb_ents).reshape([-1, 1])], axis=1)

        # evaluate the object corruptions
        sub_corr_scores = model.predict(sub_corr)
        sub_ranks[idx] = 1 + np.sum(sub_corr_scores > sub_corr_scores[sub_id])
        if known_triples is not None:
            sub_ranks_fl[idx] = sub_ranks[idx] - np.sum(
                sub_corr_scores[known_sub_triples[(rel_id, obj_id)]] > sub_corr_scores[sub_id])

        # evaluate the object corruptions
        obj_corr_scores = model.predict(obj_corr)
        obj_ranks[idx] = 1 + np.sum(obj_corr_scores > obj_corr_scores[obj_id])
        if known_triples is not None:
            obj_ranks_fl[idx] = obj_ranks[idx] - np.sum(
                obj_corr_scores[known_obj_triples[(sub_id, rel_id)]] > obj_corr_scores[obj_id])

    ranks_raw = np.concatenate([sub_ranks, obj_ranks], axis=0)
    mrr_raw = np.mean(1.0 / ranks_raw)
    ranks_fil = np.concatenate([sub_ranks_fl, obj_ranks_fl], axis=0)
    mrr_fil = np.mean(1.0 / ranks_fil)

    hits_at1 = (np.sum(ranks_fil <= 1) + 1e-10) / float(len(ranks_fil))
    hits_at3 = (np.sum(ranks_fil <= 3) + 1e-10) / float(len(ranks_fil))
    hits_at10 = (np.sum(ranks_fil <= 10) + 1e-10) / float(len(ranks_fil))

    if known_triples is None:
        result = {'hash': data_hash, 'raw': {'avg': dict()}}
    else:
        result = {'hash': data_hash, 'raw': {'avg': dict()}, 'filtered': {'avg': dict()}}

    result['raw']['avg']['mr'] = np.mean(ranks_raw)
    result['raw']['avg']['mrr'] = mrr_raw
    if known_triples is not None:
        result['filtered']['avg']['mr'] = np.mean(ranks_fil)
        result['filtered']['avg']['mrr'] = mrr_fil
        result['filtered']['avg']['hits@1'] = hits_at1
        result['filtered']['avg']['hits@3'] = hits_at3
        result['filtered']['avg']['hits@10'] = hits_at10
    return result


def evaluate_kge_model_auc(model, test_triples, test_triples_labels, random_state=None):
    """ Evaluate a knowledge graph embedding model using the area under the roc and precision recall curves

    Parameters
    ----------
    model : KnowledgeGraphEmbeddingModel
    test_triples : ndarray
        2D array containing test triples
    test_triples_labels : ndarray
        1D array containing true labels of the test triples (1 if true, 0 otherwise)
    random_state : np.random.RandomState
        random state

    Returns
    -------
    float
        model's area under the roc curve
    float
        model's area under the precision recall curve
    float
        random baseline area under the roc curve
    float
        random baseline area under the precision recall curve
    """
    if random_state is None:
        random_state = np.random.RandomState(1234)

    scores = model.predict(test_triples)
    scores_rnd = random_state.uniform(0.0, 1.0, scores.shape)

    roc_score = auc_roc(test_triples_labels, scores)
    rnd_roc_score = auc_roc(test_triples_labels, scores_rnd)

    pr_score = auc_pr(test_triples_labels, scores)
    rnd_pr_score = auc_pr(test_triples_labels, scores_rnd)
    return roc_score, pr_score, rnd_roc_score, rnd_pr_score


def evaluate_kge_model_rel_auc(model, relation_idx, nb_ents, entity_pairs, sub_corrs=None, obj_corrs=None,
                               known_pairs=None, neg2pos_ratio=1.0, seed=1234):
    """ Evaluate a knowledge graph embedding model using the area under the roc and precision recall curves
    on a specific relation type

    Parameters
    ----------
    model : KnowledgeGraphEmbeddingModel
        Model object.
    relation_idx : int
        relation index
    nb_ents: int
        number of entities
    entity_pairs : ndarray
        2D array containing true entity pairs of the given relation
    sub_corrs : ndarray
        1D array containing possible subject corruptions
    obj_corrs : list
        1D array containing possible object corruptions
    known_pairs : set
        the set of known true entity pairs for the specified relation
    neg2pos_ratio : float
        negative to positive ratio for sampling negative instances
    seed : int
        random seed

    Returns
    -------
    float
        area under the roc curve
    float
        area under the precision recall curve
    """
    rs = np.random.RandomState(seed=seed)

    test_size = len(entity_pairs)
    nb_corrs = int(test_size * neg2pos_ratio)

    test_triples = np.concatenate([entity_pairs[:, 0].reshape([-1, 1]),
                                   np.zeros([test_size, 1])+relation_idx,
                                   entity_pairs[:, 1].reshape([-1, 1])], axis=1)
    if sub_corrs is None:
        sub_corrs = np.arange(0, nb_ents)
    if obj_corrs is None:
        obj_corrs = np.arange(0, nb_ents)

    sub_corrs = rs.choice(sub_corrs, [nb_corrs*3, 1])
    obj_corrs = rs.choice(obj_corrs, [nb_corrs*3, 1])
    corr_pairs = np.array([[n1, n2] for n1, n2 in np.concatenate([sub_corrs, obj_corrs], axis=1)
                           if (n1, n2) not in known_pairs])
    corr_pairs_indices = np.random.choice(np.arange(corr_pairs.shape[0]), [nb_corrs])
    sub_corrs = corr_pairs[corr_pairs_indices, 0].reshape(-1, 1)
    obj_corrs = corr_pairs[corr_pairs_indices, 1].reshape(-1, 1)
    rel_corrs = np.zeros([nb_corrs, 1], dtype=np.int) + relation_idx
    corrs_triples = np.concatenate([sub_corrs, rel_corrs, obj_corrs], axis=1)

    test_all_triples = np.concatenate([test_triples, corrs_triples], axis=0)
    test_all_labels = np.concatenate([np.ones([test_size]), np.zeros([nb_corrs])])
    roc, pr, rnd_roc, rnd_pr = evaluate_kge_model_auc(model, test_all_triples, test_all_labels, random_state=rs)
    return roc, pr, rnd_roc, rnd_pr


def print_lp_eval_result(results):
    """ Formatting the results of an evaluation.

    Parameters
    ----------
    results : dict
        Dictionary returned by :code:`evaluation_cpu`.
    """
    raw_mrr = results["raw"]["avg"]["mrr"]
    raw_mrank = str(round(results["raw"]["avg"]["mr"]))

    mr_raw_len = max(len(raw_mrank) + 2, 8)
    val_len = 9
    mr_txt_margin = mr_raw_len - 8
    mr_raw_val_margin = max(0, 6-len(raw_mrank))
    print(" ")
    table_hline = "+%s+%s+%s+%s+%s+%s+%s+" % ("-" * mr_raw_len, "-" * mr_raw_len, "-" * val_len, "-" * val_len,
                                              "-" * val_len, "-" * val_len, "-" * val_len)

    if 'filtered' in results:
        filtered_mrr = results["filtered"]["avg"]["mrr"]
        filtered_mrank = str(round(results["filtered"]["avg"]["mr"]))
        filtered_h1 = results["filtered"]["avg"]["hits@1"]
        filtered_h3 = results["filtered"]["avg"]["hits@3"]
        filtered_h10 = results["filtered"]["avg"]["hits@10"]

        mr_fil_val_margin = max(0, 6 - len(filtered_mrank))
        print(table_hline)
        print("| Link prediction evaluation protocol results%s|" % (" " * (len(table_hline)-46)))
        print(table_hline)
        print("| MR_RAW%s | MR_FIL%s | MRR_RAW | MRR_FIL | Hits@1  | Hits@3  | Hits@10 |"
              % (" "*mr_txt_margin, " "*mr_txt_margin))
        print(table_hline)
        print("| %s%s | %s%s | %-1.5f | %-1.5f | %-1.5f | %-1.5f | %-1.5f |"
              % (raw_mrank, " "*mr_raw_val_margin, filtered_mrank, " "*mr_fil_val_margin, raw_mrr, filtered_mrr,
                 filtered_h1, filtered_h3, filtered_h10), flush=True)
        print(table_hline)
    else:
        print("MRANK_RAW\tMRR_RAW")
        print("%s\t%s" % (raw_mrank, raw_mrr), flush=True)


def print_model_roc_rel_results(results_list):
    """ Print a list result of result outcomes from the `evaluate_kge_model_rel_auc` function for different relations

    Parameters
    ----------
    results_list : list
        results list where each item is formatted a tuple as follows:
        [relation_name, model_roc, model_pr, random_roc, random_pr]
    """
    print("")
    print("+--------------------------------+-----------+-----------++-----------+-----------+")
    print("| %-30s | MODEL-ROC | RAND-ROC  || MODEL-PR  | RAND-PR   |" % "Relation")
    print("+--------------------------------+-----------+-----------++-----------+-----------+")

    for rel, roc, pr, rnd_roc, rnd_pr in results_list:
        print("| %-30s |   %1.3f   |   %1.3f   ||   %1.3f   |    %1.3f  |" % (rel, roc, rnd_roc, pr, rnd_pr))
        print("+--------------------------------+-----------+-----------++-----------+-----------+")
    print("")