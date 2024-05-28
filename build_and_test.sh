#!/bin/bash
# Initialize Conda in the sub-shell
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate the environment
conda create -n reach-for-the-arcs-temp-test-env python=3.9 -y
conda activate reach-for-the-arcs-temp-test-env

# Compile
mkdir build
cd build
cmake ..
make -j8
cd ..

# Install dependencies
python -m pip install -r requirements.txt

# Run tests
python -m unittest discover -s test/ -t .

# Conda cleanup
conda deactivate
sleep 2
conda env remove -n reach-for-the-arcs-temp-test-env
