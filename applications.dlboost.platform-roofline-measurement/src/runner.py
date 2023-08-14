import os 
import logging
import sys
import pandas as pd
import subprocess

class BenchdnnRunner():
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        # Kick off rn50 conv kernels with emon.
        df = pd.DataFrame()

        for precision in self.configs["precision"]:
            test_env = self.set_env(precision)
            
            for i, kernel in enumerate(self.configs["rn50_kernels"]):
                self.logger.info(f"precision is {precision}, kernel is {kernel}")
                kernel_log_folder = os.path.join(self.configs["log_dir"], "kernels")
                if not os.path.exists(kernel_log_folder):
                    os.mkdir(kernel_log_folder)
                log_path = os.path.join(kernel_log_folder, f"{precision}_{kernel}_summary_log.txt")
                
                test_cfg = self.set_cfg(precision)
                exec_cmd = self.set_cmd(test_cfg, kernel, log_path)
                
                if not self.args.no_tmc:
                    tmc_arg = self.set_tmc_args(precision, kernel)
                    exec_cmd = f"tmc -c '{exec_cmd}' {tmc_arg} 2>&1 | tee {log_path}"
                
                exec_cmd = f"source /opt/intel/oneapi/setvars.sh && {exec_cmd}"
                self.logger.info(f"Final Execution CMD is \n{exec_cmd}")  

                if self.args.run:
                    process = subprocess.Popen(exec_cmd, shell=True, env=test_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    for line in iter(process.stdout.readline, b''):
                        self.logger.info(str(line, encoding="utf-8").strip())
                        if subprocess.Popen.poll(process) is not None:
                            break
                    process.communicate()
                    if process.returncode != 0:
                        self.logger.error("Command failed with {}".format(process.returncode))
                    self.logger.info(f"kernel output saved at {log_path}")
        return

    def set_env(self, precision):
        env = os.environ
        if self.configs["system_info"]["platform"] == "sierraforest":
            env = self.print_set_env(env, "ONEDNN_MAX_CPU_ISA", "AVX2_VNNI_2")
            env = self.print_set_env(env, "DNNL_MAX_CPU_ISA", "AVX2_VNNI_2")
        else:
            if precision in ["amx_int8", "amx_bfloat16"]:
                env = self.print_set_env(env, "ONEDNN_MAX_CPU_ISA", "AVX512_CORE_AMX")
                env = self.print_set_env(env, "DNNL_MAX_CPU_ISA", "AVX512_CORE_AMX")
            elif precision == "amx_fp16":
                env = self.print_set_env(env, "ONEDNN_MAX_CPU_ISA", "AVX512_CORE_AMX_FP16")
                env = self.print_set_env(env, "DNNL_MAX_CPU_ISA", "AVX512_CORE_AMX_FP16")
            elif precision in ["avx_fp32", "avx_int8"]:
                env = self.print_set_env(env, "ONEDNN_MAX_CPU_ISA", "avx512_core_vnni")
                env = self.print_set_env(env, "DNNL_MAX_CPU_ISA", "avx512_core_vnni")
            else:
                self.logger.info("precision invalid! Exiting...")
                sys.exit()
            env = self.print_set_env(env, "KMP_AFFINITY", "granularity=fine,noduplicates,compact,1,0")
            env = self.print_set_env(env, "OMP_NUM_THREADS", self.configs["system_info"]["cores_per_numa"])
        return env

    def print_set_env(self, env, name, value):
        env[name] = str(value)
        self.logger.info("{}={}".format(name, value))
        return env
        
    def set_cfg(self, precision):
        if precision in ["amx_int8", "avx_int8"]:
            cfg="u8s8u8"
        elif precision == "amx_fp16":
            cfg="f16"
        elif precision == "amx_bfloat16":
            cfg="bf16bf16bf16"
        elif precision == "avx_fp32":
            cfg="f32"
        else:
            self.logger.info("precision invalid! Exiting...")
            sys.exit()
        return cfg

    def set_cmd(self, cfg, kernel, log_path):
        mb = self.configs["system_info"]["cores_per_numa"]
        driver="conv"
        max_test_time = self.configs["max_test_time"]
        cmd = []
        for i in range(self.configs["system_info"]["total_numas"]):
            start_cpu = int(i* self.configs["system_info"]["cores_per_numa"])
            end_cpu = int((i + 1)* self.configs["system_info"]["cores_per_numa"] - 1)
            instance_log_path = log_path.replace("_summary_log.txt", f"_instance_{i}_log.txt")
            cmd.append(f"numactl --physcpubind {start_cpu}-{end_cpu} -m {i} /root/workspace/oneDNN/build/tests/benchdnn/benchdnn --{driver} --cfg={cfg} --mode=P --max-ms-per-prb={max_test_time} --mb={mb} {kernel} 2>&1 | tee {instance_log_path}")
        test_cmd = " & ".join(cmd)
        self.logger.info(f"Test Command is \n{test_cmd}")
        return test_cmd

    def set_tmc_args(self, precision, kernel):
        cmd = ""
        cmd += " -u -n"
        if self.configs["system_info"]["platform"] == "sierraforest":
            cmd += " -Z metrics2"
        cmd += " -x {}".format(self.configs["user"])
        cmd += " -a {}_{}_{}_{}".format(self.configs["comment"], precision, kernel, self.configs["timestamp"])
        cmd += " -i {}_{}_{}_{}".format(self.configs["comment"], precision, kernel, self.configs["timestamp"])
        cmd += " -G {}".format(self.configs["group"])
        return cmd