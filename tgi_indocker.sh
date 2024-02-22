#!/usr/bin/env bash
set -ex

export http_proxy=http://proxy-dmz.intel.com:912
export https_proxy=http://proxy-dmz.intel.com:912

model=${1:-"meta-llama/Llama-2-7b-hf"}
dtype=${2:-bfloat16}

apt-get update
apt-get install -y vim
apt-get install -y numactl
# apt-get install tee
nowpath=$PWD

apt-get install -y wget
apt-get install -y git

conda install -y mkl
conda install -y gperftools -c conda-forge

wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7.90/gperftools-2.7.90.tar.gz
tar -xzf gperftools-2.7.90.tar.gz 
cd gperftools-2.7.90
./configure --prefix=$HOME/.local
make
make install

cp -r $HOME/.local/lib/libtcmalloc.so ${CONDA_PREFIX}/lib/libtcmalloc.so

pip install cmake lit
pip uninstall -y torch

# python -m pip install torch --index-url https://download.pytorch.org/whl/nightly/cpu
# pip install https://download.pytorch.org/whl/nightly/cpu/torch-2.2.0.dev20231007%2Bcpu-cp39-cp39-linux_x86_64.whl
python -m pip install torch==2.3.0.dev20240107+cpu --index-url https://download.pytorch.org/whl/nightly/cpu

# git clone   https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu -b llm_feature_branch
# cd frameworks.ai.pytorch.ipex-cpu
# git submodule update --init --recursive
git config --global --add safe.directory /data/frameworks.ai.pytorch.ipex-cpu
cd /data/frameworks.ai.pytorch.ipex-cpu
python setup.py install
# pip install https://download.pytorch.org/whl/nightly/pytorch_triton-2.1.0%2B6e4932cda8-cp39-cp39-linux_x86_64.whl
cd ..

export TRANSFORMERS_OFFLINE=0
export HF_HOME=/root/.cache/huggingface
# pip install --upgrade huggingface_hub
pip install huggingface-hub==0.16.4
huggingface-cli login --token 

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
export LOG_LEVEL=info,text_generation_router=debug,text_generation_launcher=debug
export OMP_NUM_THREADS=56 

nohup numactl -C 0-55 -m 0 text-generation-launcher -p 80 --model-id ${model} --dtype ${dtype} 2>&1 | tee -a /data/serve.log
