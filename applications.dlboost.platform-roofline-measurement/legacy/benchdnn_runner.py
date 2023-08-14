import os 
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from utils import get_system_info

def run(args, configs):
    logger = logging.getLogger("run")
    # Kick off rn50 conv kernels with emon.
    df = pd.DataFrame()

    for precision in precisions:
        env = set_env(precision)
        kernels = onednn_rn50_kernels.split(" ")
        for i, kernel in enumerate(kernels):
            logger.info(f"precision is {precision}, kernel is {kernel}")
            log_path = os.path.join(log_dir, f"{precision}_{kernel}_log.txt")
            
            test_cfg = set_cfg(precision)
            exec_cmd = set_cmd(cfg, kernel, log_path)
            
            if not args.no_tmc:
                tmc_arg = set_tmc_args(configs, comment, group)
                exec_cmd = f"tmc -c '{exec_cmd}' {tmc_arg}"
            
            logger.info(f"Execution CMD is {exec_cmd}")  

            if args.run:
                process = subprocess.Popen(exec_cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in iter(process.stdout.readline, b''):
                    logger.info(str(line, encoding="utf-8").strip())
                    # Record title
                    pass
                    # Record value
                    pass
                    if subprocess.Popen.poll(process) is not None:
                        break
                process.communicate()
                try:
                    with open(log_path, "r") as f:
                        lines = f.readlines()
                        results_ = lines[1].strip().split(",")
                        Gops=results_[5]
                        time0=results_[8]
                        Gflops0=results_[9]
                        result = {"kernel": kernel, "precision": precision, "Gops": Gops, "time0": time0, "Gflops0": Gflops0}
                        logger.info(result)
                        results = results.append(result, ignore_index=True)
                except Exception as e:
                    logger.info(e)
        df.to_csv(os.path.join(configs["log_dir"], "benchdnn_kernel_results.csv"), index=False)
    return df

def set_env(precision):
    logger = logging.getLogger("set_env")
    env = os.environ
    if precision in ["amx_int8", "amx_bf16"]:
        env["ONEDNN_MAX_CPU_ISA"] = ""
    elif precision == "amx_fp16":
        env["ONEDNN_MAX_CPU_ISA"] = "AVX51E2_CORE_AMX_FP16"
    elif precision in ["avx_fp32", "avx_int8"]:
        env["ONEDNN_MAX_CPU_ISA"] = "avx512_core_vnni"
    else:
        logger.info("precision invalid! Exiting...")
        sys.exit()
    logger.info("ONEDNN_MAX_CPU_ISA={}".format(env["ONEDNN_MAX_CPU_ISA"]))
    
    env["KMP_AFFINITY"] = "granularity=fine,noduplicates,compact,1,0"
    logger.info("KMP_AFFINITY={}".format(env["KMP_AFFINITY"]))
	
    env["OMP_NUM_THREADS"] = omp_threads
    logger.info("OMP_NUM_THREADS={}".format(env["OMP_NUM_THREADS"]))
    return env
    
def set_cfg(precision):
    logger = logging.getLogger("set_cfg")
    if precision in ["amx_int8", "avx_int8"]:
        cfg="u8s8u8"
    elif precision == "amx_fp16":
        cfg="f16"
    elif precision == "amx_bf16":
        cfg="bf16bf16bf16"
    elif precision == "avx_fp32":
        cfg="f32"
    else:
        logger.info("precision invalid! Exiting...")
        sys.exit()
    return cfg

def set_cmd(cfg, kernel, log_path):
    logger = logging.getLogger("set_cmd")
    cores_per_socket = int(os.popen("lscpu | grep \"Core(s) per socket\"").read().strip().split(":")[1].strip()) - 1
    mb=32
    driver="conv"
    cmd = f"numactl --physcpubind {cores_per_socket} -m 0 /root/workspace/oneDNN/build/tests/benchdnn/benchdnn --{driver} --cfg={cfg} --mode=P  --mb={mb} {kernel} | tee {log_path}"
    logger.info(cmd)
    return cmd

def set_tmc_args(configs, comment, group):
    cmd = ""
    cmd += " -u"
    cmd += " -x {}".format(configs["user"])
    cmd += " -a {}".format(comment)
    cmd += " -i {}".format(comment)
    cmd += " -G {}".format(group)
    return cmd