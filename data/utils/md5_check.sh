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

declare -a true_md5s=(
    "e0526d53c7f7b2e3167b2b43bb2690ed"
    "38a7bb733f7a09214d44293460ec0021"
    "f6f6e0e242e3d4c9569ba56400938afc"
    "4d761e0f6a86cc63e8f7afc91f6a8f0b"
    "7124b949fe79bfa7f7019a15ef3dbd06"
    "7a9413db55c6fa452824469cc7dd9d3d"
    "db6ed8330e634445efc8fec83ae81442"
    "395a35be3651d86dc3b0d365b8ea5f92"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

for i in {0..7};
do
    md5_val_json=$(md5sum "${tarballs[i]}" | awk '{print $1}')
    if [ "$md5_val_json" = "${true_md5s[i]}" ]; then
        echo -e "${GREEN}Valid:${NC} "${tarballs[i]}""
    else
        echo -e "${RED}Invalid:${NC} "${tarballs[i]}""
    fi
done