#!/usr/bin/env bash

DATASET_NAME="RCV1-2K"
FILES_PREFIX="rcv1x"

echo "$(basename "$0") needs update..."
exit 1

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
