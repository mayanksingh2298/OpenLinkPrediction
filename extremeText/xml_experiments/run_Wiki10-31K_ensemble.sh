#!/usr/bin/env bash

DATASET_NAME="Wiki10-31K"
FILES_PREFIX="wiki10"
PARAMS="-lr 0.5 -epoch 25 -arity 2 -dim 500 -l2 0.002 -wordsWeights -treeType kmeans -ensemble 3"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
