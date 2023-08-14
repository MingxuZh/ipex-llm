import os
import sys
import subprocess
import pandas as pd
import numpy as np
import logging
import glob
import uuid
import pytz
from datetime import datetime
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import arg_parser, load_config
from src.system_info import SystemInfo
from src.pre_process import preprocess



class PostProcessor():
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)

    def download_edp(self):
        # Find dcsotracker path
        keyword = "rsync -ratlz dcso@"
        path = ""
        edp_file_list = {}
        for precision in self.configs["precision"]:
            edp_file_list[precision] = {}
            for kernel in self.configs["rn50_kernels"]:
                log_path = os.path.join(self.configs["log_dir"], "kernels", f"{precision}_{kernel}_summary_log.txt")
                if not os.path.isfile(log_path):
                    self.logger.error(f"Can not find log path of {precision} {kernel}. Skip this case.")
                    continue
                with open(log_path, "r") as f:
                    for line in f.readlines():
                        if keyword in line:
                            path = line.strip().split(keyword)[1].split(" ")[0]
                            self.logger.info(f"Find edp path {path}")
                            break
                if path == "":
                    self.logger.error(f"Can not find edp weblink! Skip case {precision} {kernel}.")
                    edp_file_list[precision][kernel] = None
                    continue
                folder_name = path.split("/")[-2]
                edp_remote_path = os.path.join(path, f"{folder_name}_edp_r1/{folder_name}_edp_r1.xlsx")
                
                # Setup local path
                edp_local_folder = os.path.join(self.configs["log_dir"], "edp_logs")
                if not os.path.exists(edp_local_folder):
                    os.mkdir(edp_local_folder)
                edp_local_path = os.path.join(edp_local_folder, f"{precision}_{kernel}_edp_r1.xlsx")

                # Generate download command.
                cmd = "sshpass -p {} rsync -ratlz dcso@{} {}".format(self.configs["dcsomc_download_pw"], edp_remote_path, edp_local_path)
                self.logger.info(f"Downloading EDP file command: {cmd}")
                
                if self.args.skip_download_edp and os.path.isfile(edp_local_path):
                    self.logger.info(f"Skip download EDP files, exsiting edp file list is {edp_local_path}")
                else:
                    # Download EDP files
                    process = subprocess.Popen(cmd, env=os.environ, shell=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
                    
                    for line in iter(process.stdout.readline, b''):
                        self.logger.info(str(line, encoding="utf-8"))
                        if subprocess.Popen.poll(process) is not None:
                            break
                    process.communicate()
                
                if os.path.isfile(edp_local_path):
                    self.logger.info("Successfully download edp file in {} .".format(edp_local_path))
                    
                    edp_file_list[precision][kernel] = edp_local_path
                else:
                    self.logger.error(f"Download edp Failed. Skip case {precision} {kernel}.")
                    edp_file_list[precision][kernel] = None
        return edp_file_list

    def get_kernel_output(self):
        ''' Traverse all precision and kernels, only record cases with results.
            Only Record the last output title and result.
        '''
        title = []
        value_list = []
        precision_list = []
        kernel_list = []
        instance_list = []
        for precision in self.configs["precision"]:
            for kernel in self.configs["rn50_kernels"]:
                log_keyword = os.path.join(self.configs["log_dir"], "kernels", f"{precision}_{kernel}_instance_*_log.txt")
                files = glob.glob(log_keyword)
                for file in files:
                    with open(file, "r") as f:
                        title_i = []
                        value_i = []
                        for i, line in enumerate(f.readlines()):
                            if self.configs["benchdnn_title_keyword"] in line:
                                self.logger.debug(f"Find Output Template in {file} line {i}.")
                                title_i = line.strip().split("Output template: ")[1].split(",")
                            elif self.configs["benchdnn_value_keyword"] in line:
                                self.logger.debug(f"Find Output result in {file} line {i}.")
                                value_i = line.strip().split(",")

                        
                        if len(title_i) <= 0:
                            self.logger.error(f"Cannot find output templete. Skip case {precision} {kernel}.")
                            return None
                        else:
                            title = title_i
                        
                        if len(value_i) <= 0:
                            self.logger.error(f"Cannot find output result. Skip case {precision} {kernel}.")
                        else: 
                            value_kernel = value_i
                        value_list.append(value_kernel)
                        precision_list.append(precision)
                        kernel_list.append(kernel)
                        instance_list.append(file)
        
        df = pd.DataFrame(value_list, columns=title)
        df["precision"] = precision_list
        df["kernel"] = kernel_list                    
        df["file"] = instance_list
        # Convert String to Float
        kpi_list = ["%Gops%", "%-time%", "%-Gflops%",  "%0time%", "%0Gflops%"]
        for kpi in kpi_list:
            df[kpi] = df[kpi].astype(float)               
        self.logger.info(f"Benchdnn kernel output summary: \n{df}")
        kernel_output_summary_path = os.path.join(self.configs["log_dir"], "kernel_output_summary.csv")
        df.to_csv(kernel_output_summary_path)
        self.logger.info(f"Benchdnn all kernel output summary saved at {kernel_output_summary_path}")
        return df

    def get_edp_freq(self, edp_file_list):
        ''' Traverse all precision and kernels, only record freq with results.
        '''
        core_freq_list = []
        uncore_freq_list = []
        precision_list = []
        kernel_list = []
        for precision in self.configs["precision"]:
            for kernel in self.configs["rn50_kernels"]:
                if precision not in edp_file_list or kernel not in edp_file_list[precision]:
                    self.logger.error(f"Cannot find edp file path for {precision} {kernel}. Skip this case.")
                    continue
                edp_path = edp_file_list[precision][kernel]
                if edp_path is not None:
                    try:
                        df = pd.read_excel(edp_path, sheet_name="system view")
                        core_freq_ind = df[df.iloc[:, 0] ==  self.configs["core_freq_keyword"]].index.tolist()[0]
                        uncore_freq_ind = df[df.iloc[:, 0] ==  self.configs["uncore_freq_keyword"]].index.tolist()[0]
                        core_freq = df.loc[core_freq_ind, self.configs["aggregated"]]
                        uncore_freq = df.loc[uncore_freq_ind, self.configs["aggregated"]]
                        core_freq_list.append(core_freq)
                        uncore_freq_list.append(uncore_freq)
                        precision_list.append(precision)
                        kernel_list.append(kernel)
                    except Exception as e:
                        self.logger.error(e)
                        self.logger.error(f"Fail to get edp freq. Skip case {precision} {kernel}.")
                else:
                    self.logger.error(f"{precision} {kernel} edp file path is None. Skip this case.")
                    continue
        array = np.asarray([precision_list, kernel_list, core_freq_list, uncore_freq_list]).T
        df_freq = pd.DataFrame(array, columns=["precision", "kernel", "core_freq", "uncore_freq"])
        df_freq["core_freq"] = df_freq["core_freq"].astype("float")
        df_freq["uncore_freq"] = df_freq["uncore_freq"].astype("float")
        self.logger.info(f"Kernel freq: \n{df_freq}")
        edp_freq_summary_path = os.path.join(self.configs["log_dir"], "edp_freq_summary.csv")
        df_freq.to_csv(edp_freq_summary_path)
        self.logger.info(f"Kernel EDP freq details saved at {edp_freq_summary_path}")
        return df_freq
    
    def compute_benchdnn_freq(self, df_kernel, df_edp):
        # Merge freq/gflops/weight in one dataframe.
        df_kernel_mean_gflops = df_kernel.groupby(["precision", "kernel"])["%0time%"].agg([("%0time%", "mean")]).reset_index()
        df_summary = pd.merge(df_kernel_mean_gflops, df_edp, on=["precision", "kernel"])
        df_summary["weight"] = df_summary["kernel"].apply(lambda x: self.configs["rn50_kernel_weight"][x])
        

        # Compute aggragated benchdnn freq.
        df_summary["weighted_time"] = df_summary["weight"] * df_summary["%0time%"]
        df_summary["overall_core_freq"] = df_summary["weighted_time"] * df_summary["core_freq"]
        df_summary["overall_uncore_freq"] = df_summary["weighted_time"] * df_summary["uncore_freq"]
        self.logger.info(f"Benchdnn freq and gflops details:\n{df_summary}")
        df_internal_computing = df_summary.groupby(["precision"])[["weighted_time", "overall_core_freq", "overall_uncore_freq"]].agg("sum")
        df_final_report = pd.DataFrame()
        df_final_report["benchdnn_core_freq"] = df_internal_computing["overall_core_freq"] / df_internal_computing["weighted_time"]
        df_final_report["benchdnn_uncore_freq"] = df_internal_computing["overall_uncore_freq"] / df_internal_computing["weighted_time"]
        self.logger.info(f"Benchdnn freq final report:\n{df_final_report}")
        
        # Save df to csv
        freq_merge_details_path = os.path.join(self.configs["log_dir"], "benchdnn_freq_computing_details.csv")
        final_report_path = os.path.join(self.configs["log_dir"], "benchdnn_freq_report.csv")
        df_summary.to_csv(freq_merge_details_path)
        df_final_report.to_csv(final_report_path)
        self.logger.info(f"Benchdnn freq and gflops details saved at {freq_merge_details_path}")
        self.logger.info(f"Benchdnn freq final report saved at {final_report_path}")
        return df_final_report
    
    def convert_benchdnn_format(self, benchdnn_table:pd.DataFrame):
        benchdnn_results = []
        for p in benchdnn_table.index:
            benchdnn_result = pd.DataFrame()
            benchdnn_result["@timestamp"] = [datetime.strptime(self.configs["timestamp"], "%Y%m%d%H%M%S").replace(microsecond=0).replace(tzinfo=None).isoformat()] * 2
            benchdnn_result["kubernetes.namespace_name"] = [f"AI_PNP_{self.args.label}"] * 2
            benchdnn_result["kubernetes.pod_id"] = [str(uuid.uuid4())] * 2
            benchdnn_result["kubernetes.host"] = [self.configs["system_info"]["sut_name"]] * 2
            benchdnn_result["cluster-name"] = ["No-cluster"] * 2
            benchdnn_result["WorkloadName"] = ["AI BenchDNN"] * 2
            benchdnn_result["WorkloadVersion"] = ["v1.0.0"] * 2
            if "amx" in p:
                isa = "AMX"
                
            elif self.configs["system_info"]["platform"] == "sierraforest":
                isa = "AVX256"
            else:
                isa = "AVX512"
            precision = p.split("_")[-1].upper()
            # TODO: extract batch_size
            bs = self.configs["system_info"]["cores_per_numa"]
            benchdnn_result["WorkloadPreset"] = ["BS={} | {}".format(bs, isa)] * 2
            benchdnn_result["IterationIndex"] = ["0"] * 2
            benchdnn_result["IterationID"] =  benchdnn_result["kubernetes.pod_id"] + "_" + benchdnn_result["IterationIndex"]
            benchdnn_result["TestName"] = ["AI BenchDNN ResNet50-v1-5 {} core_freq (GHz)".format(precision), "AI BenchDNN ResNet50-v1-5_{}_uncore_freq (GHz)".format(precision)]
            benchdnn_result["Value"] = [benchdnn_table.loc[p, "benchdnn_core_freq"], benchdnn_table.loc[p, "benchdnn_uncore_freq"]]
            
            status_core = "SUCCESS" if benchdnn_table.loc[p, "benchdnn_core_freq"] is not None else "FAIL"
            status_uncore = "SUCCESS" if benchdnn_table.loc[p, "benchdnn_uncore_freq"] is not None else "FAIL"
            benchdnn_result["Status"] = [status_core, status_uncore]
            benchdnn_results.append(benchdnn_result)
        benchdnn_results = pd.concat(benchdnn_results, axis=0)
        self.logger.info(f"Benchdnn results format: \n{benchdnn_results}")
        csv_path = os.path.join(self.configs["log_dir"], "TestResult.csv")
        benchdnn_results.to_csv(csv_path, index=False)
        self.logger.info(f"Save benchdnn results in {csv_path}")
        return benchdnn_results, csv_path
    
    def run(self):
        # Post Process benchdnn kernel time and Gflops
        df_time = self.get_kernel_output()

        edp_file_list = self.download_edp()
        df_freq = self.get_edp_freq(edp_file_list)

        # Generate summary table and final freq.
        df_kernel = df_time.copy()
        df_edp = df_freq.copy()
        benchdnn_table = self.compute_benchdnn_freq(df_kernel, df_edp)
        benchdnn_formated_results, benchdnn_results_csv_path  = self.convert_benchdnn_format(benchdnn_table)
        return

if __name__ == "__main__":
    DATE=datetime.now(pytz.utc).strftime("%Y%m%d%H%M%S")
    # Parse Arguments
    args = arg_parser()
    
    # Load Config
    configs = load_config()
    
    # Set logging Format
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)
    log_level = logging.INFO
    
    sh = logging.StreamHandler()
    logging.basicConfig(
                    level=log_level,
                    format=format_str,
                    handlers=[sh]
                    )
    
    logger = logging.getLogger()

    # Check system info
    SystemInfo(args, configs).run()

    WORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.log_folder_name == "":
        logger.error("Do not specify log_dir! Exit...")
        sys.exit()
    else:
        log_dir = os.path.join(WORK_ROOT, "results", args.log_folder_name)
        log_path = os.path.join(log_dir, "output.log")
        fh = logging.FileHandler(filename=log_path, mode="a+")
        configs["timestamp"] = DATE
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        configs["log_dir"] = log_dir
        preprocess(args, configs)
        post_processor = PostProcessor(args, configs)
        
        post_processor.run()