#!/usr/bin/env bash

DATASET_NAME="Delicious-200K"
FILES_PREFIX="deliciousLarge"

# K-means tree params
PARAMS="-lr 0.1 -epoch 25 -arity 2 -dim 500 -l2 0.001 -treeType kmeans"

# Huffman tree params
PARAMS="-lr 0.1 -epoch 25 -arity 2 -dim 500 -l2 0.001 -treeType huffman"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
