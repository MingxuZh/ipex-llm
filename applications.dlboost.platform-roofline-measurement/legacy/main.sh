#!/bin/bash
set -e
set -o pipefail
SDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATE_WITH_TIME=$(date "+%Y%m%d%H%M%S")
log_dir="/workspace/logs/a2_stress_amx_checks_${DATE_WITH_TIME}"
if [ "$1" == "run" ] ; then
    mkdir -p $log_dir
fi
MODEL="ssdrn34"
MODEL="rn50"
ALLTYPES="amx"
FREQ="OOB"
RUNS=1
MEMORY="4000"
HT="HT"
COMMENT="QWNW_PERFRECIPE"
UPLOAD="-u"
EMON=0

datatype="synthetic"
accuracy='a'

export dryrun=1
for model in $MODEL; do
    if [ "$model" == "rn50" ] ; then
        batches="1 128 $accuracy"
        TYPES=$ALLTYPES
    elif [ "$model" == "mnv1" ] ; then
        batches="1 112 $accuracy"
        TYPES=$ALLTYPES
    elif [ "$model" == "ssdmnv1" ] ; then
        batches="1 448 $accuracy"
        TYPES=$ALLTYPES
    elif [ "$model" == "ssdrn34" ] ; then
        batches="x $accuracy"
        TYPES=$ALLTYPES
    elif [ "$model" == "bert" ] ; then
        batches="1 32 $accuracy"
        TYPES="fp32"
    fi
    batches='128'
    for bs in $batches; do
        if [ "$bs" == "a" ] ; then
            export mode='a'
        elif [ "$bs" == "1" ] ; then
            export mode='1'
        else
            if [ "$model" == "ssdrn34" ] ; then
                bs=1
            fi
            export mode='x'
        fi
        export batchsize=$bs
        for wltype in $TYPES; do
            export wlmode=$wltype
            for (( RUN=1; RUN<=$RUNS; RUN++ )); do
                basecmd="bash ${SDIR}/${model}_runner.sh"
                cmd="emc -c \"$basecmd\" -a \"SPR_B0_TF_${model}_BS${batchsize}_${FREQ}Freq_${MEMORY}MTs_${wltype}_run${RUN}_${HT}_${COMMENT}\" -i \"SPR TF ${model} BS128 ${FREQ} Freq ${MEMORY}MTs $wltype run${RUN}\" $UPLOAD -n -x chinnai"
                if [ "$EMON" == "1" ] ; then
                    echo "emon will be collected"
                else
                    cmd=$basecmd
                fi
                if [ "$1" == "run" ] ; then
                    cmd="$cmd 2>&1 | tee $log_dir/SPR_B0_TF_${model}_BS${batchsize}_${FREQ}Freq_${MEMORY}MTs_${wltype}_run${RUN}_${HT}_${COMMENT}.txt "
                    export dryrun=0
                fi
                if [ "$EMON" == "1" ] ; then
                echo $cmd
                eval $basecmd 
            else	    
                    eval $cmd
                fi
            done
        done
    done
done    

if [ "$1" == "run" ] ; then
    FILES=${log_dir}/*
    for f in $FILES
    do
        result=$(cat $f | grep "Throughput" | awk ' { sum += $3 } END { if (NR > 0) printf "= %s",sum } {printf "%s+",$3;} END{print "\n"}')
        filename="${f##*/}"
        echo "$filename = $result"
    done
fi
