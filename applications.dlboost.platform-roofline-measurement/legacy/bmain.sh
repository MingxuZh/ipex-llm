#!/bin/bash
set -e
set -o pipefail

DATE_WITH_TIME=$(date "+%Y%m%d%H%M%S")
log_dir="benchdnn_amx_checks_${DATE_WITH_TIME}"
if [ "$1" == "run" ] ; then
    mkdir -p $log_dir
fi
TYPES="bamx bnon-amx bfp32"
FREQ="OOB"
RUNS=3
MEMORY="4000"
HT="HT"
COMMENT="QW21_WITH_PERFRECEIPE"
UPLOAD="-u"
EMON=1

for wltype in $TYPES; do
    for (( RUN=1; RUN<=$RUNS; RUN++ )); do
        basecmd="bash run-${wltype}.sh"
        cmd="emc -c \"$basecmd\" -a \"SPR_A2_TF_RN50_BS128_${FREQ}Freq_${MEMORY}MTs_${wltype}_run${RUN}_${HT}_${COMMENT}\" -i \"SPR TF RN50 BS128 ${FREQ} Freq ${MEMORY}MTs $wltype run${RUN}\" $UPLOAD -n -x chinnai"
        if [ "$EMON" == "1" ] ; then
            cmd="$cmd 2>&1 | tee $log_dir/SPR_A2_TF_RN50_BS128_${FREQ}Freq_${MEMORY}MTs_${wltype}_run${RUN}_${HT}_${COMMENT}.txt "
        else
            cmd="$basecmd 2>&1 | tee $log_dir/SPR_A2_TF_RN50_BS128_${FREQ}Freq_${MEMORY}MTs_${wltype}_run${RUN}_${HT}_${COMMENT}.txt "
        fi
        echo $cmd
        if [ "$1" == "run" ] ; then
            eval $cmd
        fi
    done
done
