#!/bin/bash

model=$(lscpu | awk '/Model:/{print $NF}')
family=$(lscpu | awk '/CPU family:/{print $NF}')
stepping=$(lscpu | awk '/Stepping:/{print $NF}')
total_sockets=$(lscpu | awk '/Socket\(s\):/{print $NF}')
total_cores=$(lscpu | awk '/CPU\(s\):/{print $NF;exit}')
cores_per_socket=$(lscpu | awk '/Core\(s\) per socket:/{print $NF}')

edp_dir="/root/edp-v4.23"
sep_dir="/opt/intel/sep"
new_dir="results/ai_inference_$(date +%m%d%Y%H%M%S)" && mkdir -p $new_dir
new_dir="results/$1" && mkdir -p $new_dir

docker_image="dcsorepo.jf.intel.com/pnpwls/tf_inference-spr:mlp_ww09_exp"
container_name="ai_inference"
working_dir="/home/workspace/benchmark"
collect_emon=0
docker_token="eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NzEyMjIzMjIsImlhdCI6MTYxMzU0MjMyMiwiaXNzIjoiaGFyYm9yLXRva2VuLWRlZmF1bHRJc3N1ZXIiLCJpZCI6NiwicGlkIjo1LCJhY2Nlc3MiOlt7IlJlc291cmNlIjoiL3Byb2plY3QvNS9yZXBvc2l0b3J5IiwiQWN0aW9uIjoicHVsbCIsIkVmZmVjdCI6IiJ9XX0.ISgM3cYcxAM9JVQ5CH5N4O3R2Aif0yDIc11mo29-tj4kog5J0965ueoND6cvhLKbbOx1En4ln3WYnzIvaoHxh3MrY4Lqecsod9i8DqkBewqUvjkG8PjfI2W826v9724uYrx5M-fIty_Pu3QSMCwY5lxq2QmToAyUJ69BVGOQXGoMsN5E6Htw-9evsCkUF0xNdce1nzxFKL-h6rcRvPmlOkdkNklgEZwfMi1Vp256wQP8HcnrFNjUEBJkLVFtSnvx_DEnzUEySzcbAFL9SpmcFq_XAytFDcgrmURiApJ-T2qePt8vLQC-58Jvk7nwcq0v-uVhtCP21f-qemdRw065uQyIUP6q2GR2HDOs0riRIcTH6C4jvzuQjcgbWQrUcCSJBLrfhtagatNDYJ7ciCKeSTbE7TgIgau9xyFrCsDmEDKrxZJVsPDvkyjHulZ80JsZ6SuRVsfJPXgh4EU0t1OB8ZmGgNOY-3uU1hA9wYiadRv_dPEH8ep0ZM42I7ee2apcFUYqDyEQf9reb5CtSV6HAQdJ8ELeffJYCrldP2ix0eEF90S_VoGVBPVZDuMH7DVazk0xjSHh3dYdS9z9_riBwO5ti3eJyP9PFOzyfGkYJylt9kmFmf30e0UnshPqPTUm1PaCBWhT1dA2edOrDbe1rz4lCUL-rRQhaC3xWuLvlyo"
docker stop ${container_name} && docker rm ${container_name}
echo $docker_token | docker login --username "robot\$automation"  --password-stdin dcsorepo.jf.intel.com
docker run -dit --name ${container_name} --privileged --net host -v $(pwd)/${new_dir}:${working_dir}/log/ ${docker_image}

#run_cmd="python runall.py"
run_cmd="python main.py --topology ssd_mobilenet_v1 --precision fp32"

filename=$1
if [ -z $filename ]; then
    filename=$(date +%m%d%Y%H%M%S)
fi

testname=$2
if [ -z $testname ]; then
    testname='benchmarking'
fi

echo "performance" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

supported_platforms(){
    local family=$1
    local model=$2
    local stepping=$3

    if [[ $family -ne 6 ]];then
        echo -e "\n\n Test not impletemented for the CPU \n\n"
        exit 1
    elif [[ $model -eq 85 ]];then
        if [[ $stepping -ge 0 && $stepping -le 4 ]];then
            echo "Skylake"
        elif [[ $stepping -ge 5 && $stepping -le 7 ]];then
            echo "CascadeLake"
        elif [[ $stepping -ge 10 && $stepping -le 11 ]];then
            echo "CooperLake"
        fi
    elif [[ $model -eq 106 ]];then
        if [[ $stepping -ge 0 && $stepping -le 7 ]];then
            echo "Icelake"
        fi
    elif [[ $model -eq 108 ]];then
        if [[ $stepping -eq 0 ]];then
            echo "Icelake"
        fi
    elif [[ $model -eq 143 ]];then
        if [[ $stepping -eq 0 ]];then
            echo "SapphireRapids"    
        fi
    else
        echo -e "\n\n Test not impletemented for the CPU \n\n"
        exit 1
    fi
}

platform=$(supported_platforms $family $model $stepping)
if [[ $platform != "SapphireRapids" ]];then
    events_file="${edp_dir}/Architecture_Specific/Server/${platform}/*/*-${total_sockets}s-events.txt"
else
    events_file="${edp_dir}/Architecture_Specific/Server/${platform}/*-events.txt"
fi

if [ ${collect_emon} -eq 1 ]; then
    if [ ! -f ${events_file} ];then
        echo -e "\n\n Error: Not able find the events file. Please check/correct EDP installation path \n\n"
        exit 1
    elif [ ! -d "${sep_dir}" ];then
        echo -e "\n\n Error: Not able find the EMON installation path. Please check/correct EMON installation path \n\n"
        exit 1
    else
        SEP_DRIVER=$(lsmod | grep -c sepint)
        if [[ "$SEP_DRIVER" != "2" ]]; then
            pushd ${sep_dir}/sepdk/src || exit
            ./build-driver -ni --no-udev
            ./insmod-sep
            popd || exit
        fi
    fi
    source ${sep_dir}/sep_vars.sh
    emon -i ${events_file} > ${new_dir}/emon_${fl_name}.dat &
fi

model="resnet50_v15"
batchsize=128
wltype="amx_int8"
GROUP="PVPRUN_TF"
RUNS=3
for (( run=1; run<=$RUNS; run++ )); do
    run_cmd="python3 runexp.py --platform spr --mode throughput --topology $model  --function inference --precision $wltype --test ${testname}_${run}"
    basecmd="docker exec -it ${container_name} bash -c \"cd ${working_dir}; $run_cmd\""

    cmd="emc -c '$basecmd' -a \"TF_${model}_BS${batchsize}_${FREQ}Freq_${wltype}_${testname}_${run}\" -i \"SPR TF ${model} batchsize $wltype $testname ${run}\" -G $GROUP -u -n -x chinnai"

    if [ "$3" == "DRYRUN" ] ; then
        echo $cmd
    else
        eval $cmd
    fi
done

if [ ${collect_emon} -eq 1 ]; then
    killall emon
fi
docker stop ${container_name} && docker rm ${container_name}

echo -e  "\n******************************************************************************************************"
echo "Completed running AI Inference. Results are copied under: " "$(pwd)/${new_dir}"
echo -e  "******************************************************************************************************\n"

cat $(pwd)/${new_dir}/results.csv
