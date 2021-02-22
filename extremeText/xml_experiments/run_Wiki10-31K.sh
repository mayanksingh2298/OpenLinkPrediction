#!/usr/bin/env bash

DATASET_NAME="Wiki10-31K"
FILES_PREFIX="wiki10"

# K-means tree params
PARAMS="-lr 0.5 -epoch 25 -arity 2 -dim 500 -l2 0.002 -wordsWeights -treeType kmeans"

# Complete tree params
PARAMS="-lr 0.1 -epoch 25 -arity 2 -dim 500 -l2 0.002 -wordsWeights -treeType complete"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
