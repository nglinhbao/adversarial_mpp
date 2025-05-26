#!/bin/bash

# Create and activate a Conda environment
conda create -n adversarial_mpp_env python=3.10 -y
conda activate adversarial_mpp_env

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Install pomegranate using conda
conda install -c conda-forge pomegranate -y

# Install molsets without dependencies
pip install molsets --no-deps