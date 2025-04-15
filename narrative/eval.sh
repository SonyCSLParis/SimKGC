#!/usr/bin/env bash

set -x
set -e

model_path="bert"
task="FB15k237"

# Check if input folder argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <data_folder>"
    echo "The <data_folder> should contain the test set "
    exit 1
fi

DATA_DIR=$1
MODEL_PATH=$2

test_path="${DATA_DIR}/test.txt.json"

neighbor_weight=0.
rerank_n_hop=5

python3 -u evaluate.py \
--task "${task}" \
--is-test \
--eval-model-path "${MODEL_PATH}" \
--neighbor-weight "${neighbor_weight}" \
--rerank-n-hop "${rerank_n_hop}" \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${test_path}" "$@"
