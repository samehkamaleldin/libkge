"""
:code:`libkge` is a Python library for knowledge graph embedding models.
"""
from .util.kg import KgDataset
from .datasets import load_dataset_from_dir, load_kg_file
from .embedding import KnowledgeGraphEmbeddingModel

ver_build = 1
ver_min = 0
ver_maj = 0
ver_target = "dev"

__version__ = '%d.%d.%d-%s' % (ver_maj, ver_min, ver_build, ver_target)

__all__ = ['KgDataset', 'KnowledgeGraphEmbeddingModel', 'load_kg_file', 'load_dataset_from_dir']
