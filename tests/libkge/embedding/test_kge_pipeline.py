# -*- coding: utf-8 -*-

from libkge.util import KgDataset
from libkge.embedding import TransE, DistMult, ComplEx, TriModel, DistMult_MCL, ComplEx_MCL, TriModel_MCL
import numpy as np

triples = np.array([
    ['a', 'friend-of', 'b'],
    ['a', 'parent-of', 'c'],
    ['k', 'brother-of', 'c'],
    ['a', 'parent-of', 'k'],
    ['k', 'friend-of', 'r'],
    ['m', 'friend-of', 'c'],
    ['b', 'friend-of', 'p'],
    ['f', 'brother-of', 'c'],
    ['j', 'parent-of', 't'],
    ['b', 'friend-of', 'k'],
    ['n', 'friend-of', 'd'],
    ['b', 'friend-of', 'c']
])


def test_transe_pipeline():
    """

    """
    dataset = KgDataset()
    dataset.load_triples(triples)
    train_data = dataset.data["default"]
    kge_model = TransE()
    kge_model.fit(train_data)
    train_data_scores = kge_model.predict(train_data)


def test_distmult_pipeline():
    """

    """
    dataset = KgDataset()
    dataset.load_triples(triples)
    train_data = dataset.data["default"]
    kge_model = DistMult()
    kge_model.fit(train_data)
    train_data_scores = kge_model.predict(train_data)


def test_complex_pipeline():
    """

    """
    dataset = KgDataset()
    dataset.load_triples(triples)
    train_data = dataset.data["default"]
    kge_model = ComplEx()
    kge_model.fit(train_data)
    train_data_scores = kge_model.predict(train_data)


def test_trimodel_pipeline():
    """

    """
    dataset = KgDataset()
    dataset.load_triples(triples)
    train_data = dataset.data["default"]
    kge_model = TriModel()
    kge_model.fit(train_data)
    train_data_scores = kge_model.predict(train_data)


def test_distmult_mcl_pipeline():
    """

    """
    dataset = KgDataset()
    dataset.load_triples(triples)
    train_data = dataset.data["default"]
    kge_model = DistMult_MCL()
    kge_model.fit(train_data)
    train_data_scores = kge_model.predict(train_data)


def test_complex_mcl_pipeline():
    """

    """
    dataset = KgDataset()
    dataset.load_triples(triples)
    train_data = dataset.data["default"]
    kge_model = ComplEx_MCL()
    kge_model.fit(train_data)
    train_data_scores = kge_model.predict(train_data)


def test_trimodel_mcl_pipeline():
    """

    """
    dataset = KgDataset()
    dataset.load_triples(triples)
    train_data = dataset.data["default"]
    kge_model = TriModel_MCL()
    kge_model.fit(train_data)
    train_data_scores = kge_model.predict(train_data)
