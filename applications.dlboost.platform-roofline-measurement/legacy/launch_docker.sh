dockername=aiwl
docker stop $dockername
docker rm $dockername
docker run -it --name $dockername -v /root/workspace/:/workspace --privileged dcsorepo.jf.intel.com/ai-inference/tensorflow-spr:latest /bin/bash


