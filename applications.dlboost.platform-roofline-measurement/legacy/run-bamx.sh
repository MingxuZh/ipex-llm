cd /root/raj/oneDNN/build/tests/benchdnn

source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

export CORES_PER_SOCKET=$(lscpu | grep -i "core(s) per" | awk '{print  $NF }')

export LD_LIBRARY_PATH=$PWD/../../src:$LD_LIBRARY_PATH

export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX

time OMP_NUM_THREADS=$CORES_PER_SOCKET KMP_AFFINITY=compact,1,0,granularity=fine ./benchdnn --conv --mode=p --cfg=u8s8u8  --fix-times-per-prb=900000 mb24ic128ih14oc128oh14kh3ph1n


