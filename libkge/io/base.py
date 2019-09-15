# -*- coding: utf-8 -*-

import gzip
import bz2
import numpy as np


def advanced_open(filepath, *args, **kwargs):
    """ Open function interface for files with different extensions.

    Parameters
    ----------
    filepath: str
        File path with extension.
    args: list
        Non-key arguments
    kwargs: dict
        Key arguments

    Returns
    -------

    """
    open_fn = open
    if filepath.endswith('.gz'):
        open_fn = gzip.open
    elif filepath.endswith('.bz2'):
        open_fn = bz2.open

    return open_fn(filepath, mode="rt", *args, **kwargs)


def load_kg_file(filepath, separator="\t", as_stream=False):
    """ Import knowledge graph from file

    Parameters
    ----------
    filepath: str
        File path
    separator: str
        File column separator

    Returns
    -------
    iterator
        The knowledge graph triplets obtained from the files with size [?, 3]
    """

    kg_triples = []
    with advanced_open(filepath) as file_content:
        for line in file_content:
            kg_triples.append(line.strip().split(separator))
    return np.array(kg_triples)


def load_kg_file_as_stream(filepath, separator="\t"):
    """ Import knowledge graph from file as a stream

    Parameters
    ----------
    filepath: str
        File path
    separator: str
        File column separator

    Returns
    -------
    generator
        The knowledge graph triplets obtained from the files with size [?, 3]
    """

    with advanced_open(filepath) as file_content:
        for line in file_content:
            yield line.strip().split(separator)