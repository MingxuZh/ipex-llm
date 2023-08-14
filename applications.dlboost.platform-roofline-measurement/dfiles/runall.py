import sys,os
import time
import parse_file as pf
import csv
from argparse import ArgumentParser

support_topologies = ["ssd_mobilenet_v1", "resnet50_v15", "mobilenet_v1", "ssd_rn34"]
support_modes = ["throughput", "latency"]
support_precision = ["fp32","vnni_int8", "amx_int8"]
support_function = ["inference","training"]
support_platforms = ["spr","icx"]

def printhelp():
    for topology in support_topologies:
        for mode in support_modes:
            for prceision in support_precision:
                for function in support_function:
                    for platform in support_platforms:
                        print("{} --plaform {} --mode {} --topology {} --function {} --precision {}".format(sys.argv[0],platform, mode, topology, function, precision))
    sys.exit(0)

platform = "spr"

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--test",
        help="name of the test",
        dest="test", default="benchmarking")
    arg_parser.add_argument(
        "--mode",
        help="the running modes: latency, throughput or accuracy",
        dest="mode", default=None)
    arg_parser.add_argument(
        "--topology",
        help="the running topology: ssd_mobilenet_v1, resnet50_v15, bert_large, ssd_rn34, mobilenet_v1",
        dest="topology", default=None)
    arg_parser.add_argument(
        "--precision",
        help="the running precision: fp32, vnni_int8, amx_int8",
        dest="precision", default=None)
    arg_parser.add_argument(
        "--function",
        help="run topology in inference or training mode: inference, training",
        dest="function", default=None)
    arg_parser.add_argument(
        "--platform",
        help="the running platform: spr, icx",
        dest="platform", default="spr")
    arg_parser.add_argument('--example', dest='example', action='store_true')
    args = arg_parser.parse_args()
    if args.example: printhelp()
    if args.mode : support_modes = [args.mode]
    if args.topology : support_topology = [args.topology]
    if args.precision : support_precision = [args.precision]
    if args.function : support_function = [args.function]

    logfile_dir = os.path.join(sys.path[0], "log")
    if os.path.isdir(logfile_dir) is False:
        os.mkdir(logfile_dir)
    os.system("rm -rf {}/*".format(logfile_dir))
    pf.write_to_csv('TestName','Platform', 'Topology', 'Mode', 'Precision', 'Function', 'Inst Count', 'Throughput(img/sec)','Latency(ms/img)')
    for topology in support_topologies:
        for mode in support_modes:
            for precision in support_precision:
                for function in support_function:
                    logfile = os.path.join(logfile_dir, "test_{0}_{1}_{2}_{3}_{4}.log".format(platform,topology,mode,precision,function))
                    command = "python main.py --mode {0} --topology {1} --precision {2} --function {3} --platform {4} 2>&1 | tee {5}".format(mode,topology,precision,function,platform,logfile)
                    print("running command is {}".format(command))
                    os.system(command)
                    inst_count, throughput = pf.parse_file(logfile, topology)
                    latency = round((1000*inst_count)/throughput,2)
                    pf.write_to_csv(args.test, platform, topology, mode, precision, function, inst_count, throughput, latency)
                    time.sleep(10)


