#!/bin/bash

#set -eux
set -o pipefail

#export https_proxy=http://child-prc.intel.com:913
#export http_proxy=http://child-prc.intel.com:913
export http_proxy=http://proxy-dmz.intel.com:912
export https_proxy=http://proxy-dmz.intel.com:912

install_deps () {
        rh_family=(rhel centos fedora)
        debian_family=(ubuntu)
        os=$(awk -F "=" '/^ID=/{print $NF}' /etc/*release | tr -d '"')
        if [[ "${rh_family[@]}" =~ "$os" ]]; then
                dnf updateinfo
                dnf install gcc-c++.x86_64 cmake gcc git wget make numactl -y
        elif [[ "$os" =~ "${debian_family[@]}" ]]; then
        	apt update && apt install cmake gcc g++ git wget make numactl -y
        else
                echo "Error: Unknown Operating system $os"
                exit 1
        fi
}

if [[ ! "$(systemctl status docker)" =~ "active (running)" ]]; then
   echo "Install deps"
   install_deps
fi

FILE=/opt/intel/oneapi/setvars.sh
if test -f "$FILE"; then
    echo "HPC Kit already installed."
else
    pushd /tmp
    no_proxy="" wget -q https://registrationcenter-download.intel.com/akdlm/irc_nas/17229/l_HPCKit_b_2021.1.10.2477_offline.sh
    chmod +x ./l_HPCKit_b_2021.1.10.2477_offline.sh
    ./l_HPCKit_b_2021.1.10.2477_offline.sh -s -a --silent --eula accept
    rm -rf ./l_HPCKit_b_2021.1.10.2477_offline.sh
    popd
fi
python3 -m pip install -r requirements.txt

oneDNNdir=~/workspace/oneDNN

mkdir -p ~/workspace/

FILE=/etc/docker
if [ -d "$oneDNNdir" ]; then
    echo "$oneDNNdir already exists"
    cd $oneDNNdir
    git pull
    rm -rf $oneDNNdir/build
else
    git clone https://github.com/intel-innersource/libraries.performance.math.onednn.git $oneDNNdir
fi

mkdir -p $oneDNNdir/build
cd ~/workspace/oneDNN/build
source /opt/intel/oneapi/setvars.sh
echo "Setting up benchdnn"
CXX=icpx CC=icx cmake ..
make -j

cd $HOME/workspace/oneDNN/build
lib_path=$(find /opt/intel/ -name libiomp5.so | grep -m1 libiomp5.so)
cp $lib_path $HOME/workspace/oneDNN/build/src/
echo "$(pwd)/src" > /etc/ld.so.conf.d/lib_omp_dnnl.conf
ldconfig

# Install emon
if test -f /opt/intel/sep/sep_vars.sh; then
    echo "EMON already installed!"
else
    cd $HOME
    wget https://dcsorepo.jf.intel.com:8080/utils/emon/sep_private_5_40_beta_linux_03021406.tar.bz2
    tar xvf sep_private_5_40_beta_linux_03021406.tar.bz2
    cd sep_private_5_40_beta_linux_03021406
    echo "Installing emon..."
    ./sep-installer.sh -u -i --accept-license -ni --no-udev --install-boot-script
    echo "Emon installation Done!"
    source /opt/intel/sep/sep_vars.sh
fi

# Install dcsomc
if which tmc; then
    echo "TMC already installed!"
else
    cd $HOME
    rm -rf dcsomc
    git clone git clone https://github.com/intel-sandbox/tools.dcso.telemetry.client.git $HOME/dcsomc
    cd dcsomc
    bash install.sh
    echo "TMC installation Done!"
fi



# create known hosts
if [ ! -d "/root/.ssh/known_hosts" ]; then
    mkdir -p /root/.ssh
    touch /root/.ssh/known_hosts
fi

# clear

# echo -e "
# ***************************************************************************
# Running the workload/benchmark
# ***************************************************************************
# WORKDIR=\"\$HOME/workspace/oneDNN/build/tests/benchdnn\"
# cd \$WORKDIR
# cores_per_socket=\$(lscpu | awk '/Core\(s\) per socket:/{print $NF}')
# source /opt/intel/oneapi/setvars.sh
# export OMP_NUM_THREADS=\$cores_per_socket
# export KMP_AFFINITY=compact,1,0,granularity=fine
 
# ***************************************************************************
# Functionality and Precision=AMX (INT8)
# ***************************************************************************
# export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX 
# export DNNL_VERBOSE=1
# ./benchdnn --mode=c --conv --dir=fwd_b --cfg=u8s8u8 --skip-impl='ref' --mb=\$cores_per_socket --batch=inputs/conv/shapes_resnet_50_v1_5

# *************************************************************************** 
# Functionality and Precision=Non-AMX/Legacy-VNNI (INT8)
# ***************************************************************************
# export DNNL_MAX_CPU_ISA=""
# export DNNL_VERBOSE=1
# ./benchdnn --mode=c --conv --dir=fwd_b --cfg=u8s8u8 --skip-impl='ref' --mb=\$cores_per_socket --batch=inputs/conv/shapes_resnet_50_v1_5
 
# ***************************************************************************
# Functionality and Precision=fp32 (FP32)
# ***************************************************************************
# export DNNL_MAX_CPU_ISA=""
# export DNNL_VERBOSE=1
# ./benchdnn --mode=c --conv --dir=fwd_b --cfg=f32 --skip-impl='ref' --mb=\$cores_per_socket --batch=inputs/conv/shapes_resnet_50_v1_5
 
# ***************************************************************************
# Performance and Precision=AMX
# ***************************************************************************
# export DNNL_VERBOSE=0
# export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX 
# ./benchdnn --mode=p --conv --dir=fwd_b --cfg=u8s8u8 --skip-impl='ref' --mb=\$cores_per_socket --batch=inputs/conv/shapes_resnet_50_v1_5

# *************************************************************************** 
# Performance and Precision=Non-AMX/Legacy-VNNI
# ***************************************************************************
# export DNNL_VERBOSE=0
# export DNNL_MAX_CPU_ISA=""
# ./benchdnn --mode=p --conv --dir=fwd_b --cfg=u8s8u8 --skip-impl='ref' --mb=\$cores_per_socket --batch=inputs/conv/shapes_resnet_50_v1_5

# *************************************************************************** 
# Performance and Precision=fp32
# ***************************************************************************
# export DNNL_VERBOSE=0
# export DNNL_MAX_CPU_ISA=""
# ./benchdnn --mode=p --conv --dir=fwd_b --cfg=f32 --skip-impl='ref' --mb=$
# "

echo -e "\n*** Benchdnn setup completed and ready for tests ***\n"


