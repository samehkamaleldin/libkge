# -*- coding: utf-8 -*-

from libkge.util import KgDataset
import numpy as np

triples = np.array([
    ['a', 'friend-of', 'b'],
    ['a', 'parent-of', 'c'],
    ['k', 'brother-of', 'c'],
    ['a', 'parent-of', 'k'],
    ['k', 'friend-of', 'k'],
    ['k', 'friend-of', 'c'],
    ['b', 'friend-of', 'c'],
])


def test_load_triplets():
    """

    """
    dataset = KgDataset()
    dataset.load_triples(triples)

    assert dataset.data["default"].shape == triples.shape

    dataset.load_triples(triples, "train")
    assert dataset.data["train"].shape == triples.shape


def test_conversion():
    """

    """
    dataset = KgDataset()
    dataset.load_triples(triples)

    triples_idx = dataset.data["default"]
    triples_labels = dataset.indices2labels(triples_idx)
    triples_labels2idx = dataset.labels2indices(triples_labels)
    assert (triples_labels == triples).all()
    assert (triples_labels2idx == triples_idx).all()


def test_ent_rel_counts():
    """

    """
    dataset = KgDataset()
    dataset.load_triples(triples)

    ent_count = dataset.get_ents_count()
    rel_count = dataset.get_rels_count()

    assert ent_count == 4
    assert rel_count == 3

