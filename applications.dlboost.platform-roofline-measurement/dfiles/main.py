from argparse import ArgumentParser
from config import support_topologies,support_modes,support_precision,support_platform,support_function
import sys,os
import script

functions = {
            "spr": {
                  "ssd_mobilenet_v1": script.spr.run_ssd_mobilenet_v1.run_ssd_mobilenet_v1,
                  "resnet50_v15": script.spr.run_resnet50_v15.run_resnet50_v15,
                  "bert_large": script.spr.run_bert_large.run_bert_large,
                  "ssd_rn34": script.spr.run_resnet34.run_resnet34,
                  "mobilenet_v1": script.spr.run_mobilenet_v1.run_mobilenet_v1
                  },
            "icx": {
                  "ssd_mobilenet_v1": script.icx.run_ssd_mobilenet_v1.run_ssd_mobilenet_v1,
                  "resnet50_v15": script.icx.run_resnet50_v15.run_resnet50_v15,
                  "bert_large": script.icx.run_bert_large.run_bert_large,
                  "ssd_rn34": script.icx.run_resnet34.run_resnet34,
                  "mobilenet_v1": script.icx.run_mobilenet_v1.run_mobilenet_v1
                  }
        }

if __name__ == "__main__":

    cores_per_socket = int(os.popen("lscpu | grep 'Core(s) per socket'").read().strip().split(":")[1].strip())
    socket_num = int(os.popen("lscpu | grep 'Socket(s)'").read().strip().split(":")[1].strip())
    print("Platform info: socket_num:{} cores_per_socket:{}".format(socket_num,cores_per_socket))

    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--mode",
        help="the running modes: latency, throughput or accuracy",
        dest="mode", default="throughput")
    arg_parser.add_argument(
        "--topology",
        help="the running topology: ssd_mobilenet_v1, resnet50_v15, bert_large, ssd_rn34, mobilenet_v1",
        dest="topology", default="resnet50_v15")
    arg_parser.add_argument(
        "--precision",
        help="the running precision: fp32, vnni_int8, amx_int8",
        dest="precision", default="fp32")
    arg_parser.add_argument(
        "--function",
        help="run topology in inference or training mode: inference, training",
        dest="function", default="inference")
    arg_parser.add_argument(
        "--platform",
        help="the running platform: spr, icx",
        dest="platform", default="spr")
    args = arg_parser.parse_args()

    if args.topology.lower() not in support_topologies:
        sys.exit("Not supported topology: {}!\nrun 'python main.py -h' to check the running options".format(args.topology))
    if args.mode.lower() not in support_modes:
        sys.exit("Not supported modes: {}!\nrun 'python main.py -h' to check the running options".format(args.mode))
    if args.precision.lower() not in support_precision:
        sys.exit("Not supported precision: {}!\nrun 'python main.py -h' to check the running options".format(args.precision))
    if args.function.lower() not in support_function:
        sys.exit(
            "Not supported function: {}!\nrun 'python main.py -h' to check the running options".format(args.function))
    if args.platform.lower() not in support_platform:
        sys.exit("Not supported platform: {}!\nrun 'python main.py -h' to check the running options".format(args.platform))
    functions[args.platform.lower()][args.topology.lower()](args.mode.lower(), args.precision.lower(),
                                                            args.function.lower(), socket_num, cores_per_socket)
