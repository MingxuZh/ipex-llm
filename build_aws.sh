#!/bin/bash

mkdir /home/ubuntu/workspace
WORKSPACE=/home/ubuntu/workspace
set -x
sudo chmod 777  ${WORKSPACE} -R
rm -rf ${WORKSPACE}/log/
lscpu

cd /home/ubuntu/workspace

# export http_proxy=http://proxy-dmz.intel.com:912
# export https_proxy=http://proxy-dmz.intel.com:912

git clone https://github.com/MingxuZh/ipex-llm
cd /home/ubuntu/workspace/ipex-llm

sh /home/ubuntu/workspace/ipex-llm/build_awsllm.sh

docker run -it --privileged --rm forllmaws

source activate llm

git clone https://github.com/MingxuZh/ipex-llm
cd ipex-llm

bash ipex_cpu_llm_aws_yaml.sh aws