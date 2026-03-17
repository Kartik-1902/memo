#!/usr/bin/env bash
# MEMO-MODIFICATION: glibc 2.17 compatible environment setup (Python 3.8)

# Create environment
conda create -n memo python=3.8 -y
conda activate memo

# Torch (CUDA 11.6) - glibc 2.17 compatible
conda install -y -c pytorch -c conda-forge pytorch==1.13.1 torchvision==0.14.1 cudatoolkit=11.6

# CPU-only alternative
# conda install -y -c pytorch -c conda-forge pytorch==1.13.1 torchvision==0.14.1 cpuonly

# Python deps
pip install -r imagenet-exps/requirements.txt
