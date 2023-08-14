import os 
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def preprocess(args, configs):
    logger = logging.getLogger("PreProcess")
    default_precisions = configs["precision"]
    if args.precision is None:
        precisions = default_precisions
    else:
        if not set(args.precision.split(",")).issubset(set(default_precisions)):
            logger.error(f"Error! Precision {args.precision} is not correct! Please refer to config.json to set correct precision.")
            sys.exit()
        else:
            precisions = args.precision.split(",")
            configs["precision"] = precisions
            logger.info(f"Change Precision to customized {precisions}")
    
    return