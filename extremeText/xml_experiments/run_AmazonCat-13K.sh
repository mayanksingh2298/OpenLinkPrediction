#!/usr/bin/env bash

DATASET_NAME="AmazonCat-13K"
FILES_PREFIX="amazonCat"
PARAMS="-lr 0.2 -epoch 15 -arity 2 -dim 500 -l2 0.001 -wordsWeights -treeType kmeans"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
