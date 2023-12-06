NODE_LABEL = 'LLM_SPR_aws06'
if ('NODE_LABEL' in params) {
    echo "NODE_LABEL in params"
    if (params.NODE_LABEL != '') {
        NODE_LABEL = params.NODE_LABEL
    }
}

env.NODE_PROXY = 'http://proxy-dmz.intel.com:912'
if ('NODE_PROXY' in params) {
    echo "NODE_PROXY in params"
    if (params.NODE_PROXY != '') {
        env.NODE_PROXY = params.NODE_PROXY
    }
}
echo "NODE_PROXY: $NODE_PROXY"


echo "NODE_LABEL: $NODE_LABEL"
env.REGULAR = ''
if ('REGULAR' in params) {
    echo "REGULAR in params"
    if (params.REGULAR != '') {
        env.REGULAR = params.REGULAR
    }
}
echo "REGULAR: $REGULAR"

env.modelid = 'meta-llama/Llama-2-7b-hf'
if ('modelid' in params) {
    echo "modelid in params"
    if (params.modelid != '') {
        env.modelid = params.modelid
    }
}
echo "modelid: $modelid"

env.dtype = 'bfloat16'
if ('dtype' in params) {
    echo "dtype in params"
    if (params.dtype != '') {
        env.dtype = params.dtype
    }
}
echo "dtype: $dtype"

env.BUILD_IMAGE = ''
if ('BUILD_IMAGE' in params) {
    echo "BUILD_IMAGE in params"
    if (params.BUILD_IMAGE != '') {
        env.BUILD_IMAGE = params.BUILD_IMAGE
    }
}
echo "BUILD_IMAGE: $BUILD_IMAGE"

env.docker_name = 'BuildDocker'
if ('docker_name' in params) {
    echo "docker_name in params"
    if (params.docker_name != '') {
        env.docker_name = params.docker_name
    }
}
echo "docker_name: $docker_name"

env.conda_name = 'tgi'
if ('conda_name' in params) {
    echo "conda_name in params"
    if (params.conda_name != '') {
        env.conda_name = params.conda_name
    }
}
echo "conda_name: $conda_name"


env.OOB_NUM_WARMUP=5
env.OOB_NUM_ITER=200
env.OOB_CHANNELS_LAST=1

// env.docker_name = 'ccr-registry.caas.intel.com/pytorch/ipex_build_env/centos8.3.2011:ww09'
//                    wget --no-proxy -O IPEX_Dockerfile/prompt.json http://mlpc.intel.com/downloads/cpu/llm/prompt.json
def cleanup(){
    try {
        sh '''#!/bin/bash 
        set -x
        sudo chmod 777  ${WORKSPACE} -R
        rm -rf ${WORKSPACE}/log/
        lscpu
        '''
    } catch(e) {
        echo "==============================================="
        echo "ERROR: Exception caught in cleanup()           "
        echo "ERROR: ${e}"
        echo "==============================================="
        echo "Error while doing cleanup"
    }
}

def get_dockername(){
    cur_date="`date +%Y_%m_%d`"
    docker_name=${conda_name}_${cur_date}_llm
}

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

def get_currentdate(){
    // currentDate = `echo $(TZ=America/Vancouver date +%Y-%m-%d)`
    def dateFormat = DateTimeFormatter.ofPattern("yyyy-MM-dd")
    def currentDate = LocalDateTime.now().minusHours(16).format(dateFormat)
    return currentDate
}

node(NODE_LABEL){
    properties(
        [disableConcurrentBuilds(),]
    )
    cleanup()
    deleteDir()
    checkout scm
    // wget -O IPEX_Dockerfile/prompt.json https://raw.githubusercontent.com/MingxuZh/ipex-llm/main/prompt.json
                        // git pull origin ${ipex_branch}

    stage("clone repo"){
        withEnv(["modelid=${modelid}", "dtype=${dtype}", "docker_name=${docker_name}","conda_name=${conda_name}"]){
            sh '''
                cd ${WORKSPACE}
                mkdir ${WORKSPACE}/TGI
                export http_proxy=$NODE_PROXY
                export https_proxy=$NODE_PROXY
                
                # cd ${WORKSPACE}
                # if [ ! -d "IPEX_Dockerfile" ];then
                #     git clone -b back --depth 1 --recursive https://github.com/WeizhuoZhang-intel/IPEX_Dockerfile.git

                # fi
                
                export LC_ALL=C.UTF-8
                export LANG=C.UTF-8
                

                chmod +x launchserve.sh

                cp -r launchserve.sh ${WORKSPACE}/TGI

                cd ${WORKSPACE}/TGI
                sudo rm -rf text-generation-inference 
                git clone -b main https://github.com/jianan-gu/text-generation-inference 
                cd text-generation-inference



            '''
            if (params.BUILD_IMAGE == 'True') {
                withEnv(["docker_name=${docker_name}"]){
                sh '''
                    echo "docker_name: ${docker_name}"
                    if  [ ${docker_name} = "BuildDocker" ]; then
                        cur_date="`date +%Y-%m-%d`"
                        docker_name=${cur_date}_tgi
                    fi
                    sudo docker build --build-arg http_proxy=http://proxy-chain.intel.com:911 --build-arg https_proxy=http://proxy-chain.intel.com:911 --build-arg no_proxy='*.intel.com' -t ccr-registry.caas.intel.com/pytorch/pytorch-ipex-spr:${docker_name} .
                    sudo docker push ccr-registry.caas.intel.com/pytorch/pytorch-ipex-spr:${docker_name}
                '''
                }


            }
            sh '''

                mkdir ${WORKSPACE}/TGI/text-generation-inference/data
                export volume=${WORKSPACE}/TGI/text-generation-inference/data
                export REPOPATH=${WORKSPACE}/TGI/text-generation-inference
                echo $REPOPATH

                cp -r ipex_cpu_tgi_yaml.sh ${WORKSPACE}/TGI/text-generation-inference/data
                cd ${WORKSPACE}/TGI/text-generation-inference/data
                export https_proxy=http://proxy-chain.intel.com:911
                export http_proxy=http://proxy-chain.intel.com:911
                wget https://raw.githubusercontent.com/MingxuZh/ipex-llm/main/tgi_indocker.sh
                touch serve.log

                bash ipex_cpu_tgi_yaml.sh debug $REPOPATH/data $conda_name

                cd $REPOPATH/data

                git clone https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu -b llm_feature_branch
                cd frameworks.ai.pytorch.ipex-cpu
                git submodule update --init --recursive
                cd ..
                # python setup.py install
                # pip install https://download.pytorch.org/whl/nightly/pytorch_triton-2.1.0%2B6e4932cda8-cp39-cp39-linux_x86_64.whl


                container_id=$(docker ps -q --format "{{.ID}}" --filter "publish=8088")
                if [ -n "$container_id" ]; then
                    echo "Stopping Docker container $container_id"
                    docker stop $container_id
                else
                    echo "No container found with publish=8088"
                fi 
                # docker stop $(docker ps -q --format "{{.ID}}" --filter "publish=8088")
                sudo docker run -dt --shm-size 1g -p 8088:80 -d --cap-add=sys_nice --ipc=host -v $volume:/data ccr-registry.caas.intel.com/pytorch/pytorch-ipex-spr:${docker_name} bash /data/tgi_indocker.sh "meta-llama/Llama-2-7b-hf" "bfloat16" || true

                for item in `ls | grep run_ | grep .sh`; do
                    bash ${item}
                done

                # time_per_tokens=$(awk '/time_per_token/ {gsub(/"/, "", $43); print $43}' $REPOPATH/data/serve.log)
                # time_per_tokens_num=$(echo $time_per_tokens | awk -F'=' '{gsub(/ms/, "", $2); print $2}')
                # time_per_tokens_num=$(echo $time_per_tokens_num | tr -d '"')
                # total_time_num=$(awk '/total_time/ {match($0, /"([0-9]+\\.[0-9]+)s"/, arr); print arr[1]}' $REPOPATH/data/serve.log)
                total_time_num=($(awk '/total_time/ {match($0, /"([0-9]+\\.[0-9]+)s"/, arr); print arr[1]}' $REPOPATH/data/serve.log))
                validation_time_num=($(awk '/validation_time/ {match($0, /"([0-9]+\\.[0-9]+)µs"/, arr); print arr[1]}' $REPOPATH/data/serve.log))
                inference_time_num=($(awk '/inference_time/ {match($0, /"([0-9]+\\.[0-9]+)s"/, arr); print arr[1]}' $REPOPATH/data/serve.log))
                time_per_tokens_num=($(awk '/time_per_token/ {match($0, /"([0-9]+\\.[0-9]+)ms"/, arr); print arr[1]}' $REPOPATH/data/serve.log))


                output_file="$REPOPATH/data/merged_results.log"
                echo "'model_id' 'dtype' 'input-output-bs' 'numiter' 'total_time' 'validation_time' 'inference_time' 'time_per_tokens'" >> "$output_file"
                
                for item in `ls | grep run_ | grep .sh`; do

                    # Remove existing merged_results.log file
                    # rm -f "$output_file"

                    # Extract export filen values from run_llama7b_bf16.sh
                    
                    filen_values=($(grep -oP 'export filen=\\K[^\\n]+' "$item"))

                    # Extract validation_time values from serve.log
                    # validation_times=($(awk '/validation_time/ {match($0, /"([0-9]+\\.[0-9]+)µs"/, arr); print arr[1]}' $REPOPATH/data/serve.log))

                    # Combine filen values, validation_time values, modelid, dtype, inputoutputbs, and numiter
                    for ((i = 0; i < ${#filen_values[@]}; i++)); do
                        # Extract inputoutputbs and numiter from filen_values[i]

                        echo "Filename: ${filen_values[i]}"
                        inputoutputbs=$(echo "${filen_values[i]}" | sed -E 's/([0-9]+-[0-9]+-[0-9]+)file.*/\\1/')
                        numiter=$(echo "${filen_values[i]}" | sed -E 's/.*file([0-9]+)\\.log/\\1/')

                        echo "intputoutputbs: $inputoutputbs"
                        echo "numiter: $numiter"

                        total_time="${total_time_num[i]}"
                        validation_time="${validation_time_num[i]}"
                        inference_time="${inference_time_num[i]}"
                        time_per_tokens="${time_per_tokens_num[i]}"
                        echo $total_time $validation_time $inference_time $time_per_tokens
                        # modelid="meta-llama/Llama-2-7b-hf"
                        # dtype="bfloat16"

                        # Write the result to merged_results.log
                        echo "$modelid $dtype $inputoutputbs $numiter $total_time $validation_time $inference_time $time_per_tokens" >> "$output_file"
                    done
                
                    # filenames=$(grep -oP 'export filen=\\K[^ ]+' $REPOPATH/data/${item})

                    
                    # inputoutputbs=$(echo grep -oP 'export filen=\\K[^ ]+' run_llama7b_bf16.sh | awk -F 'file' '{print $1}')
                    # inputoutputbs=$(echo $filenames | awk -F'file' '{print $1}')
                    # numiter=$(echo $filenames | awk -F'file' '{print $2}' | awk -F'.' '{print $1}')
                    # numiter=$(echo grep -oP 'export filen=\\K[^ ]+' run_llama7b_bf16.sh | awk -F 'file|\\.log' '{print $1, $2}')
                    # while read -r filename; do
                        # inputoutputbs=$(echo "$filename" | awk -F 'file|\\.log' '{print $1}')
                        # numiter=$(echo "$filename" | awk -F 'file|\\.log' '{print $2}')
                        # echo $inputoutputbs
                        # echo $numiter
                        # echo "$modelid $dtype $inputoutputbs $numiter $total_time_num $validation_time_num $inference_time_num $time_per_tokens_num" >> merged_results.log
                        # paste -d" " <(echo "$modelid") <(echo "$dtype") <(echo "$inputoutputbs") <(echo "$numiter") <(echo "$total_time_num") <(echo "$validation_time_num") <(echo "$inference_time_num") <(echo "$time_per_tokens_num") > merged_results.log
                    # done <<< "$filenames"
                done

                # filenames=$(grep -oP 'export filen=\\K[^ ]+' $REPOPATH/script.sh)

                # paste -d" " <(echo "$modelid") <(echo "$dtype") <(echo "$filenames") <(echo "$time_per_tokens") > merged_results.txt

                pwd
                cd ..
                pwd
                path=$PWD

            '''
            
            archiveArtifacts artifacts: 'TGI/text-generation-inference/data/*.log, TGI/text-generation-inference/data/*.txt, TGI/text-generation-inference/data/*.sh', fileCharset: 'UTF-8', excludes: null 
            fingerprint: true
        }

    }

    stage("run workloads"){

        sh '''

        '''
        
    }
    
}
