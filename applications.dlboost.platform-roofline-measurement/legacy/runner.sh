total_cpus=$(lscpu | sed -n 's/^CPU(s):[ \t]*//p')
total_sockets=$(lscpu | sed -n 's/^Socket(s):[ \t]*//p')
cores_per_socket=$(lscpu | sed -n 's/^Core(s) per socket:[ \t]*//p')
total_numas=$(lscpu | sed -n 's/^NUMA node(s):[ \t]*//p')
threads_per_cores=$(lscpu | sed -n 's/^Thread(s) per core:[ \t]*//p')
ht_range_start=$(echo $(($total_cpus / $threads_per_cores)))
total_cores=$(echo $(($total_cpus / $threads_per_cores)))

USE_SMT=1
BS1_WARMUP_STEPS=500
BS1_STEPS=5000
BSX_WARMUP_STEPS=50
BSX_STEPS=2000

total_instances=1
cd /workspace/models/benchmarks
if [ "$wlmode" == "amx" ] ; then
    precision=int8
    model_file=freezed_s8s8_resnetv1.5.pb
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
elif [ "$wlmode" == "fp32" ] ; then
    precision=fp32
    model_file=resnet50_v15_fp32.pb
    export DNNL_MAX_CPU_ISA=""
else
    precision=int8
    model_file=freezed_s8s8_resnetv1.5.pb
    export DNNL_MAX_CPU_ISA=""
fi

if [ "$batchsize" == "1" ] ; then
    total_instances=14
fi


arraycontains() {
    local e matching="$1"
    shift
    for e; do [[ "$e" == "$matching" ]] && return 0; done
    return 1
}

getnuma() {
    for i in $(seq 0 $(($total_numas - 1))); do
        read -a arr <<<$(numactl --hardware | sed -n "s/node $i cpus:*//p")
        arraycontains $1 "${arr[@]}"
        if [ $? = "0" ]; then
            echo $i
            return
        fi
    done
    echo "-1"
}

findnuma() {
    start_numa=$(getnuma $1)
    end_numa=$(getnuma $2)
    if [ $start_numa = $end_numa ]; then
        echo $start_numa
    elif (((($start_numa + 1)) < $end_numa)); then
        numa=""
        for i in $(seq $start_numa $end_numa); do
            if [ "$numa" = "" ]; then
                numa="$i"
            else
                numa="$numa,$i"
            fi
        done
        echo $numa
    else
        echo $start_numa,$end_numa
    fi
}

for num_instances in "${total_instances}"; do
    avail_cores=$total_cores
    sockets_run=2
    if [ "$num_instances" = "0.5" ]; then
        num_instances=1
        sockets_run=1
        avail_cores=$cores_per_socket
        if (($total_sockets > 2)); then
            avail_cores=$(($cores_per_socket * 2))
        fi
    fi
    if (($num_instances > $avail_cores)); then
        continue
    fi
    sync
    echo 3 >/proc/sys/vm/drop_caches
    per_instance_cores=$(echo $(($avail_cores / $num_instances)))
    logical_cores=$(echo $(($per_instance_cores * $threads_per_cores)))

    echo -e "\nStarting tests for $num_instances instances using $per_instance_cores physical cores & $logical_cores logical cores per instance"
    echo -e "Running $wlmode batch size $batchsize tests"
    cpu_limit=$(echo $(($num_instances - 1)))
    start_cpu=0
    end_cpu=$(echo $(($per_instance_cores - 1)))

    for instance in $(seq 1 $num_instances); do
        ht_start_cpu=$(echo $(($start_cpu + $ht_range_start)))
        ht_end_cpu=$(echo $(($end_cpu + $ht_range_start)))
        numa_c="numactl --physcpubind $start_cpu-$end_cpu"
        if [ $USE_SMT = "0" ]; then
            numa_c="numactl --physcpubind $start_cpu-$end_cpu,$ht_start_cpu-$ht_end_cpu"
        fi
        numa=$(findnuma $start_cpu $end_cpu)
        numa_c="$numa_c -m $numa"
        nbatches=$BSX_STEPS
        warmup_batches=$BSX_WARMUP_STEPS
        if [ "$batchsize" == "1" ] ; then
            warmup_batches=$BS1_WARMUP_STEPS
            nbatches=$BS1_STEPS
        fi

        omp_threads=$per_instance_cores

        #export OMP_NUM_THREADS=$omp_threads
        export KMP_AFFINITY=granularity=fine,verbose,compact,1,0

        cmd="python launch_benchmark.py \
            --in-graph $model_file \
            --model-name resnet50v1_5 \
            --framework tensorflow \
            --precision $precision \
            --mode inference \
            --num-intra-threads $omp_threads \
            --num-inter-threads 1 \
            --batch-size=$batchsize \
            --benchmark-only \
            -- warmup_steps=${warmup_batches} steps=${nbatches}"

        run_cmd="$numa_c $cmd"
        if [ "$instance" = "$num_instances" ]; then
            echo -e "Instance $instance will be running in foreground with $omp_threads OMP Threads, $data_threads data thread with follow command"
        else
            echo -e "Instance $instance will be running in background with $omp_threads OMP Threads, $data_threads data thread with follow command"
            run_cmd="$run_cmd &"
        fi

        echo $run_cmd
        if [ "$dryrun" == "0" ] ; then
            eval $run_cmd
        fi
        start_cpu=$(echo $(($end_cpu + 1)))
        end_cpu=$(echo $(($end_cpu + $per_instance_cores)))

    done
done
