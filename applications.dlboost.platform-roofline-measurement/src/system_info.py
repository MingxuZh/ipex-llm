# -*- coding:utf-8 -*-

import sys
import os
import logging
import json
WORKDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

class SystemInfo():
    ''' Check system info, including
            software version,
            sut name, 
            system platform name, 
            system CPU QDF, 
            system BIOS version,
            ...
        Record system info in configs.
    '''
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.function = {
            "platform": PlatformInfo,
            "sut_name": SutNameInfo,
            "kernel": KernelInfo,
            "qdf": QdfInfo,
            "bios_version": BiosInfo,
            "detail": DetailInfo
        }
        # if self.args.debug:
        #     self.function.update({"svr_info": SVRInfo})

    def run(self):
        self.configs["system_info"] = dict()
        for f in self.function:
            self.function[f](self.args, self.configs).run()
        return


class PlatformInfo():
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)
        

    def run(self):
        self.configs["system_info"]["platform"] = self.read_platform()
        return

    def read_platform(self, default_platform=""):
        # Set default platform as spr.
        emon_platform = self.emon_read_platform()
        fmscpu_platform = self.fms_read_platform()
        platform_tuple = [
            ("emon", emon_platform),
            ("FMS", fmscpu_platform),
            ("default", default_platform)
            ]
        plat_ind = next((i for i, (name, value) in enumerate(platform_tuple) if value), None)
        if plat_ind is not None:
            platform_source = platform_tuple[plat_ind][0]
            platform_name = platform_tuple[plat_ind][1]
            self.logger.info(f"{platform_source} detect platform {platform_name}.")
        else:
            platform_name = ""
            self.logger.warning(f"Can not Detect any valid platform, use {platform_name}.")
        return platform_name

    def fms_read_platform(self):
        platform_list_path = os.path.join(WORKDIR, 'platform_list.json')
        with open(platform_list_path, 'r', encoding='utf-8') as config:
            platform_list = json.load(config) 
        try:   
            # Read lscpu to get Model/Family/Stepping
            cpu_family = int(os.popen('lscpu |grep -v BIOS | grep "CPU family:"').read().split(':')[1].strip())
            cpu_family = '{:02x}'.format(cpu_family)[-1]
            model = int(os.popen('lscpu |grep "Model:"').read().split(':')[1].strip())
            model = '{:02x}'.format(model)[-2:]
            stepping = int(os.popen('lscpu |grep "Stepping:"').read().split(':')[1].strip())
            stepping = '{:02x}'.format(stepping)[-1]
            
            # Combine FMS to get a string.
            current_fms = '{}0{}{}{}'.format(model[0], cpu_family, model[1], stepping)
            
            if current_fms in platform_list["fms"]:
                detected_platform = platform_list["fms"][current_fms]["platform"]
                detected_stepping = platform_list["fms"][current_fms]["stepping"]
                self.logger.info(f"FMS Detected platform is {current_fms}, which maps to platform {detected_platform} {detected_stepping}.")
            else:
                self.logger.warning(f"FMS Detected platform {current_fms} is unregistered.")
                detected_platform = ""
        except Exception as e:
            self.logger.error(f"fail to read platform through fms with error {e}.")
            detected_platform = ""

        return detected_platform

    def emon_read_platform(self):
        try:
            emon_platform = os.popen("source /opt/intel/sep/sep_vars.sh > /dev/null && emon -v | grep cpu_family| grep named | cut -d ' ' -f 8").read().strip().lower()
            self.logger.info(f"EMON Detected platform is {emon_platform}.")
        except Exception as e:
            self.logger.error(f"fail to read platform through emon with error {e}.")
            emon_platform = ""
        return emon_platform
            


class SutNameInfo():
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self):
        sut_name = os.popen("uname -n").read().splitlines()[0]
        self.configs["system_info"]["sut_name"] = sut_name
        self.logger.info(f"sut name is {sut_name}")
        return


class QdfInfo():
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self):
        self.configs["system_info"]["qdf"] = self.args.qdf
        return


class KernelInfo():
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self):
        self.configs["system_info"]["kernel"] = os.popen("uname -r").read().strip()
        return


class BiosInfo():
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self):
        self.configs["system_info"]["bios_version"] = self.get_version()
        return
    
    def get_version(self):
        bios_version = os.popen("dmidecode -t bios | grep Version | cut -d ':' -f 2").read().strip()
        self.logger.info(f"bios version of this system is {bios_version}")
        return bios_version


class SVRInfo():
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self):
        cmd = "cd /home/svr-info && mkdir -p /home/svr-info/results && ./svr-info -benchmark cpu -format txt -output /home/svr-info/results"
        os.popen(cmd)
        self.configs["system_info"]["svr_info"] = None
        return

class DetailInfo():
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        self.configs["system_info"]["cores_per_socket"] = int(os.popen("lscpu | grep \"Core(s) per socket\"").read().strip().split(":")[1].strip()) - 1
        self.configs["system_info"]["total_cpus"] = int(os.popen("lscpu | sed -n 's/^CPU(s):[ \t]*//p'").read().strip())
        self.configs["system_info"]["total_sockets"] = int(os.popen("lscpu | sed -n 's/^Socket(s):[ \t]*//p'").read().strip())
        self.configs["system_info"]["total_numas"] = int(os.popen("lscpu | sed -n 's/^NUMA node(s):[ \t]*//p'").read().strip())
        self.configs["system_info"]["threads_per_cores"] = int(os.popen("lscpu | sed -n 's/^Thread(s) per core:[ \t]*//p'").read().strip())
        self.configs["system_info"]["total_cores"] = int(self.configs["system_info"]["total_cpus"] / self.configs["system_info"]["threads_per_cores"])
        self.configs["system_info"]["cores_per_numa"] = int(self.configs["system_info"]["total_cores"] / self.configs["system_info"]["total_numas"])
        self.configs["system_info"]["family"] = os.popen("lscpu  | grep 'CPU family:' | awk '{ print $3}'").read().strip()
        self.configs["system_info"]["model"] = os.popen("lscpu  | grep 'Model:' | awk '{ print $2}'").read().strip()
        self.configs["system_info"]["stepping"] = os.popen("lscpu  | grep 'Stepping:' | awk '{ print $2}'").read().strip()
        self.configs["system_info"]["l3cache"] = os.popen("lscpu  | grep 'L3 cache:' | awk '{ print $3}'").read().strip()
        return