source /root/training/non-amx/bin/activate
cd /root/raj/models/benchmarks

python launch_benchmark.py \
    --in-graph resnet50_v1.pb \
    --model-name resnet50v1_5 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --batch-size=128 \
    --benchmark-only \
    -- warmup_steps=50 steps=100


