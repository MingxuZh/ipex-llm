# Script to Collect BenchDNN performance for Platform Roofline Measurement

## How to do

### Clone and Setup
```
git clone https://github.com/intel-sandbox/applications.dlboost.platform-roofline-measurement.git
cd applications.dlboost.platform-roofline-measurement
bash setup_benchdnn.sh
# change config.json to specify resnet50 kernels/precisions/emon arguments...
# By Default, run 23 resnet50 kernels with precision avx_fp32,amx_bf16,amx_int8,avx_int8,amx_fp16.
```

### Dry Run to check BenchDNN commands
```
python3 main.py
```

### Kick off all cases specified in config.json
```
python3 main.py --run
```

### Specify multiple precision split in comma(,) in terminal argument. This has higher priority than precision specified in config.json
```
python3 main.py --run precision avx_fp32,avx_int8
```
### Post Process Only
BenchDNN and emon results saves at path results/\<timestamp\>/. You can use the following commands to only post process.
```
python3 src/post_process.py --log_folder_name <timestamp>
```