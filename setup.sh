#!/bin/sh
# Download dataset
DATA_DIR="data"
if [[ ! -e $DATA_DIR ]]; then
    mkdir $DATA_DIR
    cd $DATA_DIR
    wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    mv aclImdb_v1.tar.gz aclImdb_v1.tar.gz
    tar -xf aclImdb_v1.tar.gz
    cd ..
    pwd
fi

# Create log directory
LOG_DIR="log"
if [[ ! -e $LOG_DIR ]]; then
    mkdir $LOG_DIR
fi

# Create output directory
OUTPUT_DIR="output"
if [[ ! -e $OUTPUT_DIR ]]; then
    mkdir $OUTPUT_DIR
fi