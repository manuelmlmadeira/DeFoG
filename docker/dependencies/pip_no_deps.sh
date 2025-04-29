#!/bin/bash

# Activate the conda environment so that the installation happens within the environment
source /conda/etc/profile.d/conda.sh
conda activate defog

# Install fcd within the activated environment
pip install fcd==1.2 --no-deps