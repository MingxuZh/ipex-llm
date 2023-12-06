#!/usr/bin/env bash
set -ex

export https_proxy=http://proxy-chain.intel.com:911
export http_proxy=http://proxy-chain.intel.com:911

apt-get update
apt-get install -y numactl
# apt-get install tee
nowpath=$PWD

apt-get install -y wget

wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz
tar -xzf gperftools-2.7.90.tar.gz 
cd gperftools-2.7.90
./configure --prefix=$HOME/.local
make
make install

cp -r $HOME/.local/lib/libtcmalloc.so ${CONDA_PREFIX}/lib/libtcmalloc.so


pip uninstall -y torch
pip install torch==2.1
pip install http://mlpc.intel.com/downloads/cpu/ipex-2.1/ipex_release/rc2/intel_extension_for_pytorch-2.1.0+cpu-cp39-cp39-linux_x86_64.whl

export TRANSFORMERS_OFFLINE=0
pip install --upgrade huggingface_hub
huggingface-cli login --token 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

# nohup numactl -C 0-55 -m 0 text-generation-launcher -p 80 --model-id EleutherAI/gpt-j-6B huggyllama/llama-7b --dtype bfloat16 >> /data/serve.log 2>&1 || true &
export KMP_BLOCKTIME=INF
export KMP_TPAUSE=0
export KMP_SETTINGS=1
export KMP_FORKJOIN_BARRIER_PATTERN=dist,dist
export KMP_PLAIN_BARRIER_PATTERN=dist,dist
export KMP_REDUCTION_BARRIER_PATTERN=dist,dist

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so

export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6

nohup numactl -C 0-55 -m 0 text-generation-launcher -p 80 --model-id meta-llama/Llama-2-7b-hf --dtype bfloat16 2>&1 | tee -a /data/serve.log
