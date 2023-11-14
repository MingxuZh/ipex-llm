#!/usr/bin/env bash
set -ex

cd /bhavanis
git clone -b main https://github.com/jianan-gu/text-generation-inference
cd text-generation-inference

sudo docker build --build-arg http_proxy=http://proxy-chain.intel.com:911 --build-arg https_proxy=http://proxy-chain.intel.com:911 --build-arg no_proxy='*.intel.com' -t tgi_xeon .
mkdir $PWD/data
export volume=$PWD/data

cd $PWD/data
wget https://raw.githubusercontent.com/MingxuZh/ipex-llm/main/tgi_indocker.sh
cd $PWD

sudo docker run -it --shm-size 1g -p 8088:80 --cap-add=sys_nice --ipc=host -v $volume:/data tgi_xeon bash /data/tgi_indocker.sh || true




