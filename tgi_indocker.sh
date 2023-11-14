#!/usr/bin/env bash
set -ex

export https_proxy=http://proxy-chain.intel.com:911
export http_proxy=http://proxy-chain.intel.com:911

apt-get update
apt-get install -y numactl
# apt-get install tee


pip uninstall -y torch
pip install torch==2.1
pip install http://mlpc.intel.com/downloads/cpu/ipex-2.1/ipex_release/rc2/intel_extension_for_pytorch-2.1.0+cpu-cp39-cp39-linux_x86_64.whl

nohup numactl -C 0-55 -m 0 text-generation-launcher -p 80 --model-id huggyllama/llama-7b --dtype bfloat16 >> /data/serve.log 2>&1 || true &

sleep 480s

curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":128, "do_sample": true}}' -H 'Content-Type: application/json' >> /data/client.log 2>&1 || true &
sleep 10s
curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":128, "do_sample": true}}' -H 'Content-Type: application/json' >> /data/client.log 2>&1 || true &
sleep 10s
curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":128, "do_sample": true}}' -H 'Content-Type: application/json' >> /data/client.log 2>&1 || true &
sleep 10s
curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":128, "do_sample": true}}' -H 'Content-Type: application/json' >> data/client.log 2>&1 || true &
sleep 10s
curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":128, "do_sample": true}}' -H 'Content-Type: application/json' >> /data/client.log 2>&1 || true &
sleep 10s