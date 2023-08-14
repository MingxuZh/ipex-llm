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
   Date     : January 2021 

   Package Dependancy:
    pip install xlsxwriter
   Description:  Simple Script to parse benchdnn results to check weight efficiency

"""
import os
import xlsxwriter
import glob
import sys, getopt
from xlsxwriter.utility import xl_rowcol_to_cell, xl_col_to_name
from typing import Optional
from xlsxwriter.worksheet import (
    Worksheet, cell_number_tuple, cell_string_tuple)

cores=32
fp32_freq=1.8
avx3_bf16_freq=1.8
amx_bf16_freq=1.8
vnni_int8_freq=1.8
amx_int8_freq=1.8

flops= { "cores": cores, 
         "measured_fp32_freq" : fp32_freq, 
         "measured_avx3_bf16_freq" : avx3_bf16_freq, 
         "measured_amx_bf16_freq" : amx_bf16_freq, 
         "measured_vnni_int8_freq" : vnni_int8_freq, 
         "measured_amx_int8_freq" : amx_int8_freq,
         "fp32":int(cores*fp32_freq*2*2*16), 
         "avx_bfloat16":int(cores*avx3_bf16_freq*32),
         "amx_bfloat16":int(cores*amx_bf16_freq*1024),
         "vnni_int8":int(cores*vnni_int8_freq*2*8*16), 
         "amx_int8":int(cores*amx_int8_freq*2048)}

def get_column_width(worksheet: Worksheet, column: int) -> Optional[int]:
    """Get the max column width in a `Worksheet` column."""
    strings = getattr(worksheet, '_ts_all_strings', None)
    if strings is None:
        strings = worksheet._ts_all_strings = sorted(
            worksheet.str_table.string_table,
            key=worksheet.str_table.string_table.__getitem__)
    lengths = set()
    for row_id, colums_dict in worksheet.table.items():  # type: int, dict
        data = colums_dict.get(column)
        if not data:
            continue
        if type(data) is cell_string_tuple:
            iter_length = len(strings[data.string])
            if not iter_length:
                continue
            lengths.add(iter_length)
            continue
        if type(data) is cell_number_tuple:
            iter_length = len(str(data.number))
            if not iter_length:
                continue
            lengths.add(iter_length)
    if not lengths:
        return None
    return max(lengths)


def set_column_autowidth(worksheet: Worksheet, column: int):
    """
    Set the width automatically on a column in the `Worksheet`.
    !!! Make sure you run this function AFTER having all cells filled in
    the worksheet!
    """
    maxwidth = get_column_width(worksheet=worksheet, column=column)
    if maxwidth is None:
        return
    worksheet.set_column(first_col=column, last_col=column, width=maxwidth)

def parse_isaused(text):
    if "avx512_core_amx_int8" in text:
        return "amx_int8"
    elif "avx512_core_vnni" in text:
        return "vnni_int8"
    elif "avx512_core_bf16" in text:
        return "avx_bfloat16"
    elif "avx512_core_amx_bf16" in text:
        return "amx_bfloat16"
    elif "avx512_common" in text or "avx512_core" in text:
        return "fp32"
    else:
        print(text)

def parse_config(text):
    config = {}
    t = text.split("--")
    for item in t:
        item = item.strip()
        if "=" in item:
            key, value = item.split("=")
            if key == "mb": key = "minibatch"
            value = value.strip()
            try:
                if isinstance(int(value), int): value = int(value)
            except Exception as e:
                pass
            config[key.strip()] = value
        elif item in ["conv","ip"]:
            config["type"] = item
    return config

def parse_config_int(text):
    config = {}
    text=text.replace("--","")
    t = text.strip().split(" ")
    for item in t:
        item = item.strip()
        if "=" in item:
            key, value = item.split("=")
            config[key.strip()] = value.strip()
    config["type"] = t[0].strip()
    return config

def parse_perf(text):
    result = {}
    perfs = text.split(",")
    multiples = 1
    try:
        multi = perfs[3].replace('"','').split('*')
        result['name'] = multi[0]
        if len(multi) > 1:
            multiples = multi[1]
    except Exception as e:
        print(e)
    config = parse_config_int(perfs[3])
    config['max_isa'] = parse_isaused(perfs[2])
    result['multiples'] = int(multiples)
    result['avg_flops'] = float(perfs[-1])
    result['avg_time'] = float(perfs[-2])
    result['max_flops'] = float(perfs[-3])
    result['min_time'] = float(perfs[-4])
    return result, config

def summarize(config, result):
    cfg = config.get('max_isa')
    gflops = flops.get(cfg)
    flops_efficiency = result['max_flops'] / gflops
    avg_flops_efficiency = result['avg_flops'] / gflops
    execution_time  = result['min_time'] * result['multiples']
    weighted_efficiency = flops_efficiency * execution_time
    result['hw_peak_flops'] = gflops
    result['avg_flops_efficiency'] = avg_flops_efficiency
    result['flops_efficiency'] = flops_efficiency
    result['execution_time'] = execution_time
    result['weighted_efficiency'] = weighted_efficiency
    return {**config, **result} 

def parse_details(contents):
    baseconfig = perf_data = None
    header = None
    result = []
    for line in contents.split("\n"):
        if "benchdnn" in line:
            baseconfig = parse_config(line)
        elif line.startswith('perf'):
            perf_data, config = parse_perf(line)
            config.update(baseconfig)
            summary = summarize(config, perf_data)
            if not header:
                header = summary.keys()
            result.append(summary.values())
    return header, result

def write_results(workbook, socket, header, result):
    worksheet = workbook.add_worksheet(socket)
    cell_format = workbook.add_format()
    cell_format.set_bold()
    cell_format_percentage = workbook.add_format()
    cell_format_percentage.set_bold()
    cell_format_percentage.set_num_format(10)
    cell_format_header = workbook.add_format()
    cell_format_header.set_bold()
    cell_format_header.set_bg_color('#8db4e2')
    rownum = 0
    col = 0
    for cfg, flop in flops.items():
        worksheet.write(rownum, 0, cfg, cell_format_header)
        worksheet.write(rownum, 1, flop)
        rownum += 1
    col = 0
    
    col_mapping={}

    for value in header:
        worksheet.write(rownum, col, value, cell_format_header)
        col_mapping[value] = col
        col += 1
    col_key_list = list(col_mapping.keys())
    rownum += 1
    maxcol = 0
    for res in result:
        start_row = rownum+1
        for row in res:
            col = 0 
            for value in row:
                if col_key_list[list(col_mapping.values()).index(col)] == "avg_flops_efficiency":
                    formula = "={}{}/{}{}".format(xl_col_to_name(col_mapping['avg_flops']),rownum+1,xl_col_to_name(col_mapping['hw_peak_flops']),rownum+1)
                    worksheet.write(rownum, col, formula)
                elif col_key_list[list(col_mapping.values()).index(col)] == "flops_efficiency":
                    formula = "={}{}/{}{}".format(xl_col_to_name(col_mapping['max_flops']),rownum+1,xl_col_to_name(col_mapping['hw_peak_flops']),rownum+1)
                    worksheet.write(rownum, col, formula)
                elif col_key_list[list(col_mapping.values()).index(col)] == "execution_time":
                    formula = "={}{}*{}{}".format(xl_col_to_name(col_mapping['min_time']),rownum+1,xl_col_to_name(col_mapping['multiples']),rownum+1)
                    worksheet.write(rownum, col, formula)
                else:
                    worksheet.write(rownum, col, value)
                col += 1
            rownum +=1  


        weighted_efficiency_col = col_mapping['weighted_efficiency']
        weighted_efficiency_col_name = xl_col_to_name(weighted_efficiency_col)

        excution_time_col = col_mapping['execution_time']
        excution_time_col_name = xl_col_to_name(excution_time_col)

        max_flops_efficiency_col = col_mapping['flops_efficiency']
        max_flops_efficiency_col_name = xl_col_to_name(max_flops_efficiency_col)

        avg_flops_efficiency_col = col_mapping['avg_flops_efficiency']
        avg_flops_efficiency_col_name = xl_col_to_name(avg_flops_efficiency_col)
        
        end_row = rownum

        avg_row=end_row+1

        max_flops_efficiency_formula = "=(AVERAGE({}{}:{}{}))".format(max_flops_efficiency_col_name, start_row, max_flops_efficiency_col_name, end_row)
        worksheet.write(avg_row, max_flops_efficiency_col, max_flops_efficiency_formula, cell_format)

        avg_flops_efficiency_formula = "=(AVERAGE({}{}:{}{}))".format(avg_flops_efficiency_col_name, start_row, avg_flops_efficiency_col_name, end_row)
        worksheet.write(avg_row, avg_flops_efficiency_col, avg_flops_efficiency_formula, cell_format)
        
        sum_exec_time_sum_formula = "=(SUM({}{}:{}{}))".format(excution_time_col_name, start_row, excution_time_col_name, end_row)
        worksheet.write(avg_row, excution_time_col, sum_exec_time_sum_formula, cell_format)

        sum_weighted_efficiency_formula = "=(SUM({}{}:{}{}))".format(weighted_efficiency_col_name, start_row, weighted_efficiency_col_name, end_row)
        worksheet.write(avg_row, weighted_efficiency_col, sum_weighted_efficiency_formula, cell_format)

        summary_formula = "={}{}/{}{}".format(weighted_efficiency_col_name,avg_row+1,excution_time_col_name,avg_row+1)
        
        #Write Summary at top
        
        worksheet.write(2, 4, "Average Weights Efficiency", cell_format_header)
        worksheet.write(2, 5, summary_formula, cell_format)
        
        worksheet.write(3, 4, "Max Flops Efficiency", cell_format_header)
        worksheet.write(3, 5, max_flops_efficiency_formula, cell_format_percentage)

        worksheet.write(4, 4, "Average Flops Efficiency", cell_format_header)
        worksheet.write(4, 5, avg_flops_efficiency_formula, cell_format_percentage)
        

        maxcol=col+2

    for ccol in range(0, maxcol):
        set_column_autowidth(worksheet, ccol)

def search_logs(log_path):
    workbook = xlsxwriter.Workbook('{}/summary.xlsx'.format(log_path))    
    all_logs = glob.glob("{}/*.txt".format(log_path))
    results = {}
    header = None
    for log in all_logs:
        base=os.path.basename(log)
        filename = os.path.splitext(base)[0]
        socket = "-".join(filename.split("_")[2:8])
        socket = filename.split("_")[-1].replace(".txt","")
        with open(log) as f:
            contents = f.read()
            header, result = parse_details(contents)
            if socket in results:
                results[socket].append(result)
            else:
                results[socket] = [result]
    for socket, result in results.items():
        write_results(workbook, socket, header, result)
    workbook.close()

def search_logfile(log_file):
    workbook = xlsxwriter.Workbook('summary.xlsx')    
    results = {}
    header = None
    base=os.path.basename(log_file)
    filename = os.path.splitext(base)[0]
    with open(log_file) as f:
        contents = f.read()
        header, result = parse_details(contents)
        results[filename] = [result]
    for socket, result in results.items():
        write_results(workbook, socket, header, result)
    workbook.close()


def main(argv):
   log_path = ''
   log_file = ""
   try:
      opts, args = getopt.getopt(argv,"hl:f:",["logpath=","logfile="])
   except getopt.GetoptError:
      print('weight_eff.py -l <log path> | -f <log file>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('weight_eff.py -l <log path>')
         sys.exit()
      elif opt in ("-l", "--logpath"):
         log_path = arg
      elif opt in ("-f", "--logfile"):
         log_file = arg

   if log_path:
       search_logs(log_path)
   elif log_file:
       search_logfile(log_file)

if __name__ == "__main__":
   main(sys.argv[1:])
