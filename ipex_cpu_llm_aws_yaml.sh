#!/bin/bash
# set -x

set -x
pwd

regular=aws
log_dir=/root/workspace/ipex-llm/log
rm -rf ${log_dir}
mkdir -p $log_dir
script_dir=/root/workspace/ipex-llm
cd ${script_dir}

# source $HOME/oneCCL_install/env/setvars.sh

source activate llm
pip install pyyaml
pip install SentencePiece
export TRANSFORMERS_OFFLINE=1
source /root/oneCCL_install/env/setvars.sh
# collec result
function run_benchmark {
    # generate cmd
    if [[ ${regular} == "aws" ]]; then
        echo "Running Large Language Models for aws..."
        python generate_run_script_aws.py --aws
    fi
    # run benchmark
    for item in `ls | grep run_ | grep .sh`; do
        bash ${item}
    done
}

run_benchmark
