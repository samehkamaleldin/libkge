from libkge.datasets import load_dataset


def print_dataset_stats(dataset_obj, dataset_name):
    """ Print stats of a dataset

    Parameters
    ----------
    dataset_obj: KgDataset
        Dataset object
    dataset_name: str
        Dataset name
    """
    nb_ents = len(dataset_obj.ent_mappings.keys())
    nb_rels = len(dataset_obj.rel_mappings.keys())
    print("= Dataset: %-10s - nb_ents: %-10d - nb_rels: %-10d - nb_splits: %-10d" % (dataset_name, nb_ents, nb_rels, len(dataset_obj.data.keys())))


wn18_dataset = load_dataset("wordnet", "wn18")
print_dataset_stats(wn18_dataset, "wn18")

wn18rr_dataset = load_dataset("wordnet", "wn18rr")
print_dataset_stats(wn18rr_dataset, "wn18rr")

nell_dataset = load_dataset("nell", "nell239")
print_dataset_stats(nell_dataset, "nell239")