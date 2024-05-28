#!/bin/bash

# Check if two arguments are given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <filename> <resolution>"
    exit 1
fi

# Assign arguments to variables
FILENAME=$1
RESOLUTION=$2

# Define the remote host
REMOTE_HOST="belle"

# Copy the file to the remote host
scp "$FILENAME" "${REMOTE_HOST}:~/"

# Run the SDFGen command on the remote host
ssh "$REMOTE_HOST" "~/NDC/data_preprocessing/get_groundtruth_NDC/SDFGen $FILENAME $RESOLUTION 0"

# Copy the resulting .sdf file back to the local machine
scp "${REMOTE_HOST}:~/${FILENAME%.obj}.sdf" ./

# SSH to the remote host, change directory, activate Conda environment, and run the Python script
ssh "$REMOTE_HOST" "bash -c '\
    cd ~/NDC/; \
    source activate neural-stochastic-psr; \
    python main.py --test_input ~/${FILENAME%.obj}.sdf --input_type sdf --method ndcx \
'"

# Copy the generated file back to the local machine
scp "${REMOTE_HOST}:~/NDC/samples/quicktest_ndcx_sdf.obj" "${FILENAME%.obj}_reconstruction.obj"

# Optional: Clean up by removing intermediate files on the remote host
ssh "$REMOTE_HOST" "rm ~/${FILENAME} ~/${FILENAME%.obj}.sdf ~/NDC/samples/quicktest_ndcx_sdf.obj"
# also remove the .sdf file
rm "${FILENAME%.obj}.sdf"
