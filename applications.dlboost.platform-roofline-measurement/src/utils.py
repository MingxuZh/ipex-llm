import os 
import logging
from argparse import ArgumentParser
import json

WORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def arg_parser():
    argument_parser = ArgumentParser()
    argument_parser.add_argument(
        "--run",
        help="Emulate the process steps & print them.. Don't run anything",
        dest="run", default=False, action='store_true')
    argument_parser.add_argument(
        "--precision",
        help="Specify precision you want to run.",
        dest="precision", default=None, type=str)
    argument_parser.add_argument(
        "--skip_download_edp",
        help="Find edp files are already downloaded and skip download edp again in post processing part.",
        dest="skip_download_edp", default=False, action="store_true")
    argument_parser.add_argument(
        "--no_tmc",
        help="Remove tmc collection",
        dest="no_tmc", default=False, action="store_true")
    argument_parser.add_argument(
        "--log_folder_name",
        help="Specify log folder name to save all output logs under $WORKROOT/results/.",
        dest="log_folder_name", default="", type=str)
    argument_parser.add_argument(
        "--label",
        help="Record label",
        dest="label", default="", type=str)
    argument_parser.add_argument(
        "--qdf",
        help="Record qdf",
        dest="qdf", default="", type=str)
    args = argument_parser.parse_args()

    return args

def load_config():
    """
    Load config.json from 
    """
    
    try:
        config_path = os.path.join(WORK_ROOT, "config.json")
        with open(config_path, 'r', encoding="UTF-8") as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise ValueError(e)