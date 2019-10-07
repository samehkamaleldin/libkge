# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import libkge

VERSION = libkge.__version__
NAME = 'libkge'
DESCRIPTION = 'A library for knowledge graph embedding models'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

AUTHOR = 'Sameh K. Mohamed'
URL = 'http://samehkamaleldin.github.io/'

with open('LICENSE') as f:
    LICENSE = f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    url=URL,
    install_requires=['numpy',
                      'bidict',
                      'tqdm',
                      'umap-learn',
                      'sklearn',
                      ],
    license=LICENSE,
    packages=find_packages(exclude=('tests', 'docs')),
    extras_require={
        'tf': ['tensorflow>=1.13.0'],
        'tf_gpu': ['tensorflow-gpu>=1.2.0']
    }
)
