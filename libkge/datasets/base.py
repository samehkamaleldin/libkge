import os
from libkge.io import load_kg_file
from libkge.util.kg import KgDataset

THIS_DIR, _ = os.path.split(__file__)
DATA_DIR = os.path.join(THIS_DIR, "data")


def print_data_dir():
    print(DATA_DIR)


def load_dataset(name, version):
    """ Load a benchmarking dataset.

    Parameters
    ----------
    name: str
        Dataset name (e.g., 'freebase', 'wordnet', 'nell').
    version: str
        Dataset version tag (e.g., 'fb15k-aaai11').

    Returns
    -------
    KgDataset
        Dataset object for request dataset.

    Example
    -------
    >>> dataset_name = "wordnet"
    >>> dataset_version = "wn18"
    >>> dataset = load_dataset(dataset_name, dataset_version)

    """
    dataset_dir = os.path.join(DATA_DIR, name)
    dataset_ver_dir = os.path.join(dataset_dir, version)

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError("Dataset dir not found")
    if not os.path.isdir(dataset_ver_dir):
        raise FileNotFoundError("Dataset version dir not found")

    train_data = load_kg_file(os.path.join(dataset_ver_dir, "train.txt.gz"))
    valid_data = load_kg_file(os.path.join(dataset_ver_dir, "valid.txt.gz"))
    test_data = load_kg_file(os.path.join(dataset_ver_dir, "test.txt.gz"))

    dataset = KgDataset()
    dataset.load_triples(train_data, tag="train")
    dataset.load_triples(valid_data, tag="valid")
    dataset.load_triples(test_data, tag="test")
    return dataset


def load_dataset_from_dir(dataset_dir):
    """ Load a benchmarking dataset.

    Parameters
    ----------
    dataset_dir: str
        path to dataset directory

    Returns
    -------
    KgDataset
        Dataset object for request dataset.

    Example
    -------
    >>> dataset_dir = "./data/dataset"
    >>> dataset = load_dataset_from_dir(dataset_dir)

    """

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError("Dataset directory (%s) not found" % dataset_dir)

    train_data = load_kg_file(os.path.join(dataset_dir, "train.txt.gz"))
    valid_data = load_kg_file(os.path.join(dataset_dir, "valid.txt.gz"))
    test_data = load_kg_file(os.path.join(dataset_dir, "test.txt.gz"))

    dataset = KgDataset()
    dataset.load_triples(train_data, tag="train")
    dataset.load_triples(valid_data, tag="valid")
    dataset.load_triples(test_data, tag="test")
    return dataset
