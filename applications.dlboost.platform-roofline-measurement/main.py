# -*- coding: utf-8 -*-
import os 
import subprocess
import sys
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils import arg_parser, load_config
from src.pre_process import preprocess
from src.runner import BenchdnnRunner
from src.post_process import PostProcessor
from src.system_info import SystemInfo
import pandas as pd
from datetime import datetime
from time import sleep
import pytz
WORK_ROOT = os.path.dirname(os.path.abspath(__file__))
DATE=datetime.now(pytz.utc).strftime("%Y%m%d%H%M%S")

def main():
    # Parse Arguments
    args = arg_parser()
    
    # Load Config
    configs = load_config()
    
    # Set logging Format
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)
    log_level = logging.INFO
    
    sh = logging.StreamHandler()
    
    if args.log_folder_name == "":
        log_dir = os.path.join(WORK_ROOT, "results", DATE)
    else:
        log_dir = os.path.join(WORK_ROOT, "results", args.log_folder_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    configs["log_dir"] = log_dir
    configs["timestamp"] = DATE

    log_path = os.path.join(log_dir, "output.log")
    fh = logging.FileHandler(filename=log_path, mode="a+")
    
    logging.basicConfig(
                    level=log_level,
                    format=format_str,
                    handlers=[sh, fh]
                    )
    
    logger = logging.getLogger()
    
    # Check system info
    SystemInfo(args, configs).run()

    # Preprocess to get test precision list.
    preprocess(args, configs)
    
    # Kick off rn50 conv kernels with emon.
    benchdnnrunner = BenchdnnRunner(args, configs)
    benchdnnrunner.run()

    if args.run:
        sleep_time = 60
        logger.info(f"Sleep {sleep_time}s to wait for DCSOMC finish EDP parsing.")
        sleep(sleep_time)
        logger.info("----------Start Post Processing----------")
        
        post_processor = PostProcessor(args, configs)
        post_processor.run()

    logger.info("Output log saved at {}".format(configs["log_dir"]))
if __name__ == "__main__":
    main()