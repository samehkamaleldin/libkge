#!/usr/bin/env bash

export PATH=$PATH:$HOME/miniconda/bin
source activate libkge

# ---------------------------------------------------------------------------
# update conda
conda update -n base conda

# ---------------------------------------------------------------------------
# this is to avoid issues with installing tensorflow in some linux settings
# ---------------------------------------------------------------------------
conda install -c conda-forge libprotobuf -y

# ---------------------------------------------------------------------------
# install required packages
# ---------------------------------------------------------------------------
conda install -c conda-forge bidict  -y
conda install numpy tqdm scikit-klearn -y


# ---------------------------------------------------------------------------
# install tensorflow: choose the relevant version
# ---------------------------------------------------------------------------
conda install -y tensorflow=1.14
# conda install -y tensorflow-gpu
