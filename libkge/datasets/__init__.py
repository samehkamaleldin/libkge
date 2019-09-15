"""
The :mod:`libkge.datasets` includes dataset loading related functions.
"""

from .base import load_dataset, load_dataset_from_dir, load_kg_file

__all__ = ['load_kg_file', 'load_dataset', 'load_dataset_from_dir']