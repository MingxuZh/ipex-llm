#!/usr/bin/env bash
set -ex

# cd /bhavanis
# git clone -b main https://github.com/jianan-gu/text-generation-inference
cd text-generation-inference

# sudo docker build --build-arg http_proxy=http://proxy-chain.intel.com:911 --build-arg https_proxy=http://proxy-chain.intel.com:911 --build-arg no_proxy='*.intel.com' -t tgi_xeon .
# mkdir $PWD/data
export volume=$PWD/data

cd $PWD/data
wget https://raw.githubusercontent.com/MingxuZh/ipex-llm/main/tgi_indocker.sh
cd $PWD

sudo docker run -dt --shm-size 1g -p 8088:80 -d --cap-add=sys_nice --ipc=host -v $volume:/data ccr-registry.caas.intel.com/pytorch/pytorch-ipex-spr:tgi_xeon bash /data/tgi_indocker.sh || true

sleep 200s
while [ -f /localdisk2/mint/text-generation-inference/data/serve.log ]
do
        echo "exist"
        if [ `grep -c "Invalid hostname, defaulting to 0.0.0.0" $PWD/data/serve.log` -ne '0' ];then
                echo "Yes"
                curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":32, "do_sample": false}}' -H 'Content-Type: application/json' | tee -a /localdisk2/mint/text-generation-inference/data/client1.log
                while [ -f /localdisk2/mint/text-generation-inference/data/client1.log ]
                do 
                        if [ `grep -c "generated_text" client1.log` -ne '0' ];then
                                echo "client1.log"
                                curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":32, "do_sample": false}}' -H 'Content-Type: application/json' | tee -a /localdisk2/mint/text-generation-inference/data/client2.log
                                while [ -f /localdisk2/mint/text-generation-inference/data/client2.log ]
                                do
                                        if [ `grep -c "generated_text" client2.log` -ne '0' ];then
                                        echo "client2.log"
                                        fi
                                        break
                                done
                        fi
                        break
                done
                break
        else
                echo "No"
        fi
done

# sleep 200s

# curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":32, "do_sample": false}}' -H 'Content-Type: application/json' | tee -a /localdisk2/mint/text-generation-inference/data/serve1.log
# sleep 20s
# curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":32, "do_sample": false}}' -H 'Content-Type: application/json' | tee -a /localdisk2/mint/text-generation-inference/data/serve2.log
# sleep 20s
# curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":128, "do_sample": false}}' -H 'Content-Type: application/json' | tee -a /localdisk2/mint/text-generation-inference/data/serve3.log
# sleep 20s
# curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":128, "do_sample": false}}' -H 'Content-Type: application/json' | tee -a /localdisk2/mint/text-generation-inference/data/serve4.log
# wait
# curl 127.0.0.1:8088/generate -X POST -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":128, "do_sample": false}}' -H 'Content-Type: application/json' | tee -a /localdisk2/mint/text-generation-inference/data/serve5.log
# sleep 20s

