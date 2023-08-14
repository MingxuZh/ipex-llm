#!/usr/bin/env python
############################################################################
# INTEL CONFIDENTIAL
# Copyright 2020 Intel Corporation All Rights Reserved.
#
# The source code contained or described herein and all documents related
# to the source code ("Material") are owned by Intel Corporation or its
# suppliers or licensors. Title to the Material remains with Intel Corp-
# oration or its suppliers and licensors. The Material may contain trade
# secrets and proprietary and confidential information of Intel Corpor-
# ation and its suppliers and licensors, and is protected by worldwide
# copyright and trade secret laws and treaty provisions. No part of the
# Material may be used, copied, reproduced, modified, published, uploaded,
# posted, transmitted, distributed, or disclosed in any way without
# Intel's prior express written permission.
#
# No license under any patent, copyright, trade secret or other intellect-
# ual property right is granted to or conferred upon you by disclosure or
# delivery of the Materials, either expressly, by implication, inducement,
# estoppel or otherwise. Any license under such intellectual property
# rights must be express and approved by Intel in writing.
############################################################################
"""
   INTEL CONFIDENTIAL - DO NOT RE-DISTRUBUTE
   Copyright 2021 Intel Corporation All Rights Reserved.

   Author   : Chinnaiyan, Rajendrakumar
   e-mail   : rajendrakumar.chinnaiyan@intel.com
   Date     : Feb 2021

   Package Dependancy:
    pip install xlsxwriter
   Description:  Simple Script to parse Tensorflow Inference results from Internal DLBoost BKC

"""

import re
import os
import sys
from collections import OrderedDict
import re
import copy
import numpy as np
from statistics import mean 
from math import fsum

mnv1_regex="Average Throughput:(.*)images/s on.*?iterations"
ssd_mobilenet_regex="Total samples/sec:(.*)samples/s"
rn50_regex="Throughput:(.*)images/sec"
ssd_rn34="Total samples/sec:(.*)samples/s"

try:
    import pandas as pd
except:
    print("ERROR: Required module 'pandas' missing. Please install and try again")
    quit()

try:
    import xlsxwriter
except:
    print("ERROR: Required module 'xlsxwriter' missing. Please install and try again")
    quit()

def getdirs(dirpath):
    a = [s for s in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, s)) and s.endswith(".log")]
    a.sort(key=lambda s: os.path.getmtime(os.path.join(dirpath, s)))
    return a

def decode_testcasename(test_case):
    t = test_case.split("_")
    config_details = {}
    config_details["platform"] = t[1]
    config_details["topology"] = "_".join(t[2:4])
    config_details["mode"] = t[-4]
    config_details["precision"] = t[-3]
    config_details["function"] = t[-2]
    config_details["trial"] = t[-1].replace(".log","")
    return config_details

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def parse_rn50(test_file):
    textfile = open(test_file, 'r')
    filetext = textfile.read()
    textfile.close()
    matches = re.findall(rn50_regex, filetext)
    if matches and len(matches):
        result={}
        for x in range(len(matches)):
            result[x+1] = float(matches[x].strip(' '))
        return result, len(matches)
    return 0,0

def parse_ssmn(test_file):
    textfile = open(test_file, 'r')
    filetext = textfile.read()
    textfile.close()
    matches = re.findall(ssd_mobilenet_regex, filetext)
    if matches and len(matches):
        result={}
        for x in range(len(matches)):
            result[x+1] = float(matches[x].strip(' '))
        return result, len(matches)
    return 0,0

def parse_mnv1(test_file):
    textfile = open(test_file, 'r')
    filetext = textfile.read()
    textfile.close()
    matches = re.findall(mnv1_regex, filetext)
    if matches and len(matches):
        result={}
        for x in range(len(matches)):
            result[x+1] = float(matches[x].strip(' '))
        return result, len(matches)
    return 0,0

def parse_ssdrn34(test_file):
    textfile = open(test_file, 'r')
    filetext = textfile.read()
    textfile.close()
    matches = re.findall(ssd_mobilenet_regex, filetext)
    if matches and len(matches):
        result={}
        for x in range(len(matches)):
            result[x+1] = float(matches[x].strip(' '))
        return result, len(matches)
    return 0,0


def parse_testcase(result_file, model):
    detailed_results = {}
    throughput=0
    instances=0
    if model == "mobilenet_v1":
        throughput, instances = parse_mnv1(result_file)
    elif model == "resnet50_v15":
        throughput, instances = parse_rn50(result_file)
    elif model == "ssd_mobilenet":
        throughput, instances = parse_ssmn(result_file)
    elif model == "ssd_rn34":
        throughput, instances = parse_ssdrn34(result_file)
    detailed_results[instances] = throughput
    return instances, throughput

def generate_detailed_sheet(detailed_data):
    summary_header = None
    summary_result = []
    for test, config in detailed_data.items():
        throughput = config.pop("throughput","")
        summary_header = ["testcase"] + list(config.keys()) + ["instance", "throughput"]
        config_values = list(config.values())
        if config['instances'] > 0:
            for instance, result in throughput.items():
                results = [test] + config_values + [instance, result]
                summary_result.append(results)
        else:
            results = [test] + config_values + [0, 0]
            summary_result.append(results)

    df_detailed_result = pd.DataFrame(summary_result, columns=summary_header)
    return df_detailed_result

def get_summary(summary_data):
    summary_header = None
    summary_result = []
    for testcase, summary in summary_data.items():
        summary_header = summary.keys()
        summary_result.append(summary.values())
    df_summary = pd.DataFrame(summary_result, columns=summary_header)
    return df_summary

def testsuite_generate_detailed_sheet(test_suitedata):
    summary_header = None
    summary_result = []
    for suite, detailed_data in test_suitedata.items():
        for test, config in detailed_data.items():
            throughput = config.pop("throughput","")
            summary_header = ["suite", "testcase"] + list(config.keys()) + ["instance", "throughput"]
            config_values = list(config.values())
            for instance, result in throughput.items():
                results = [suite, test] + config_values + [instance, result]
                summary_result.append(results)
    df_detailed_result = pd.DataFrame(summary_result, columns=summary_header)
    return df_detailed_result

def testsuite_get_summary(suite_summary_data):
    summary_header = None
    summary_result = []
    for suite, summary_data in suite_summary_data.items():
        for testcase, summary in summary_data.items():
            summary_header = ["suite", "testcase"] + list(summary.keys())
            summary_result.append([suite, testcase] + list(summary.values()))
    df_summary = pd.DataFrame(summary_result, columns=summary_header)
    return df_summary


def writedown_results(data, runname):
    try:
        summary_file = os.path.join(result_dir, "{}.xlsx".format(runname))
        writer = pd.ExcelWriter(summary_file, engine='xlsxwriter', options={'strings_to_numbers': True})
        for sheet_name, dataframe in data.items():
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False, float_format = "%0.2f")
            worksheet = writer.sheets[sheet_name]  # pull worksheet object
            for idx, col in enumerate(dataframe):  # loop through all columns
                series = dataframe[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                    )) + 1  # adding a little extra space
                worksheet.set_column(idx, idx, max_len)  # set column width            
        writer.save()
        print("Parsing done, summary and detailed results in : \n{}".format(os.path.abspath(summary_file)))
    except Exception as e:
        print("Failed to parse the results\n")


def parse_logs(result_dir):
    try:
        runname = os.path.basename(result_dir.strip('/'))
    except:
        runname = "result"
    print(runname)    
    test_cases = getdirs(result_dir)
    summary_data = {}
    detailed_data = {}
    all_dfs = {}
    for test_case in test_cases:
        config = decode_testcasename(test_case)
        test_case_results = os.path.join(result_dir, test_case)
        instances, throughput = parse_testcase(test_case_results, config['topology'])
        config["instances"] = instances
        summary = copy.deepcopy(config)
        summary["throughput"] = throughput
        if throughput:
            config["throughput"] = round(fsum(throughput.values()),2)
        summary_data[test_case] = config
        detailed_data[test_case] = summary

    detailed_df = generate_detailed_sheet(detailed_data)
    dfs= get_summary(summary_data)
    all_dfs["summary"] = dfs
    all_dfs["detailed"] = detailed_df
    writedown_results(all_dfs, runname)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: {} /path/to/root_dir".format(sys.argv[0]))
        quit()
    result_dir = sys.argv[1]
    parse_logs(result_dir)
