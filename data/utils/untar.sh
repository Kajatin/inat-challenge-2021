#!/bin/bash

declare -a tarballs=(
    "data/download/train.tar.gz"
    "data/download/train.json.tar.gz"
    "data/download/val.tar.gz"
    "data/download/val.json.tar.gz"
    "data/download/public_test.tar.gz"
    "data/download/public_test.json.tar.gz"
    "data/download/train_mini.tar.gz"
    "data/download/train_mini.json.tar.gz"
)

for i in {0..7};
do
    echo "${tarballs[i]}"
    tar -xvzf "${tarballs[i]}" -C data/
done