#!/bin/bash
set -e
set -o pipefail
EMON=1
family=$(lscpu  | grep "CPU family:" | awk '{ print $3}')
model=$(lscpu  | grep "Model:" | awk '{ print $2}')
stepping=$(lscpu  | grep "Stepping:" | awk '{ print $2}')
l3cache=$(lscpu  | grep "L3 cache:" | awk '{ print $3}')
cores=$(lscpu  | grep "Core(s) per socket:" | awk '{ print $4}')
threads=$(lscpu  | grep "Thread(s) per core:" | awk '{ print $4}')
baseconfig="${family}_${model}_${stepping}_with_LLC_${l3cache}_${cores}_cores_with_${threads}_threads_each"

total_cpus=$(lscpu | sed -n 's/^CPU(s):[ \t]*//p')
threads_per_cores=$(lscpu | sed -n 's/^Thread(s) per core:[ \t]*//p')
total_cores=$(echo $(($total_cpus / $threads_per_cores)))
total_numas=$(lscpu | sed -n 's/^NUMA node(s):[ \t]*//p')
cores_per_numa=$(echo $(($total_cores / $total_numas)))

SDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATE_WITH_TIME=$(date "+%Y%m%d%H%M%S")
GROUP="TMULPVP_RUN2"
log_dir="/workspace/logs/$GROUP/benchdnn_${DATE_WITH_TIME}"
if [ "$1" == "run" ] ; then
    mkdir -p $log_dir
fi
model="benchdnn"
TYPES="AVX3_FP32 AMX_INT8 AVX3_INT8 AMX_BF16"
TYPES="AMX_FP16"
FREQ="OOB"
RUNS=1
MEMORY="4000"
HT="HT"
sufix=""
if [ -z "$2" ]; then
    COMMENT="${baseconfig}"
else
    COMMENT="${baseconfig}_$2"
    sufix="_$2"
fi

UPLOAD="-u -G $GROUP"
export max_duration=120000
onednn_rn50_kernels="ic3ih224oc64oh112kh7sh2ph3n ic64ih56oc256oh56kh1ph0n ic64ih56oc64oh56kh1ph0n ic64ih56oc64oh56kh3ph1n ic256ih56oc64oh56kh1ph0n ic256ih56oc512oh28kh1sh2ph0n ic256ih56oc128oh56kh1ph0n ic128ih56oc128oh28kh3sh2ph1n ic128ih28oc512oh28kh1ph0n ic512ih28oc128oh28kh1ph0n ic128ih28oc128oh28kh3ph1n ic512ih28oc1024oh14kh1sh2ph0n ic512ih28oc256oh28kh1ph0n ic256ih28oc256oh14kh3sh2ph1n ic256ih14oc1024oh14kh1ph0n ic1024ih14oc256oh14kh1ph0n ic256ih14oc256oh14kh3ph1n ic1024ih14oc2048oh7kh1sh2ph0n ic1024ih14oc512oh14kh1ph0n ic512ih14oc512oh7kh3sh2ph1n ic512ih7oc2048oh7kh1ph0n ic2048ih7oc512oh7kh1ph0n ic512ih7oc512oh7kh3ph1n"
#onednn_rn50_kernels="ic3ih224oc64oh112kh7sh2ph3n"
#onednn_rn50_kernels="ic3ih224oc64oh112kh7sh2ph3n"
#onednn_rn50_kernels="all_rn50"
datatype="synthetic"
accuracy='a'
week="guojian_AMX_FP16_E9I.0093.D22"
count=0
export dryrun=1

CORES_PER_SOCKET=$(lscpu | grep -i "core(s) per" | awk '{print  $NF }')
batches="$cores_per_numa"
#batches="128"

for bs in $batches; do
    if [ "$bs" == "1" ] ; then
        export mode='1'
    else
        export mode='x'
    fi
    export batchsize=$bs
    for wltype in $TYPES; do
        export wlmode=$wltype
        for (( RUN=1; RUN<=$RUNS; RUN++ )); do
	        knumber=1
            for kernel in ${onednn_rn50_kernels}; do
                if [ $kernel == "all_rn50" ] ; then
                    export convkernel="--batch=./inputs/conv/shapes_resnet_50_v1_5"
                else
                    export convkernel=$kernel
                fi
                wl_logs="$log_dir/internals"
                basecmd="bash ${SDIR}/${model}_runner.sh"
                if [ "$1" == "run" ] ; then
                    mkdir -p $wl_logs
                    internal_log="${wl_logs}/BENCHDNN_LOG_${max_duration}s_${wltype}_${knumber}_MB${batchsize}_${kernel}_${FREQ}Freq_${MEMORY}MTs_run${RUN}_${HT}_${COMMENT}.txt"
                    basecmd="$basecmd 2>&1 | tee ${internal_log} "
                fi
                #cmd="emc -c \"$basecmd\" -D \"BDNN_${wltype}_mb${batchsize}_${knumber}_${kernel}${sufix}\" -i \"BENCHDNN ${wltype} MB${batchsize} ${kernel} ${FREQ}Freq ${MEMORY}MTs run${RUN} ${HT} ${COMMENT}\" $UPLOAD -n -x chinnai -o ${max_duration}"
                cmd="tmc -c \"$basecmd\" -D \"BDNN_${wltype}_${sufix}_${week}\" -i \"BENCHDNN ${wltype} MB${batchsize} ${kernel} ${FREQ}Freq ${MEMORY}MTs run${RUN} ${HT} ${COMMENT}\" $UPLOAD -n -x jiguojix -o ${max_duration}"
                if [ "$EMON" == "1" ] ; then
                    echo "emon will be collected"
                else
                    cmd=$basecmd
                fi
                if [ "$1" == "run" ] ; then
                    log_file="$log_dir/BENCHDNN_LOG_${max_duration}s_${wltype}_${knumber}_MB${batchsize}_${kernel}_${FREQ}Freq_${MEMORY}MTs_run${RUN}_${HT}_${COMMENT}.txt"
                    cmd="$cmd 2>&1 | tee ${log_file} "
                    export dryrun=0
                    eval $cmd
                else 
                    if [ "$EMON" == "1" ] ; then
                        echo $cmd
                        eval $basecmd 
                    else
                        eval $cmd
                    fi
                fi
		        knumber=$((knumber + 1))
            done
        done
    done
done
echo $log_dir
