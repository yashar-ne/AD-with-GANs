#!/bin/bash

conda create -y -n thesis python=3.10
conda activate thesis
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y matplotlib -c conda-forge
conda install -y numpy -c conda-forge
conda install -y jupyterlab -c conda-forge
conda install -y scikit-learn -c conda-forge
conda install -y pandas -c conda-forge
conda install -y uvicorn -c conda-forge
conda install -y fastapi -c conda-forge
conda install -y aiofiles -c conda-forge
conda install -y pymongo -c conda-forge
conda install -y seaborn -c conda-forge
