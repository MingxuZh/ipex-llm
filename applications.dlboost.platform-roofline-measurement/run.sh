#!/usr/bin/bash

log_path='./results'
# echo "Backup previous results folder......"
date_now=$(date "+%F%H%M%S")
# if [ -d $log_path ];then
#     mv $log_path results_$date_now
# fi

date_now=$(date "+%F%H%M%S")
if [ $1 == "run" ]; then
    echo "============= BenchDNN Starting ============================="
    sleep 10

    python3 main.py --run --log_folder_name $date_now
    # python3 src/post_process.py --log_folder_name <timestamp you want to parse>
    sleep 10
    echo "============= Test Finished ================================="
else
    echo "=============== BenchDNN DryRun ============================="
    
    python3 main.py --log_folder_name $date_now
    
    echo "=============== DryRun Finished ============================="
fi
if [ ! -d /root/nfs_ipu/ ]; then
    mkdir /root/nfs_ipu/
    mount -t nfs spiv-pnp-test1.sh.intel.com:/data_bck/nfs/IPU/in-progress/ /root/nfs_ipu/
    if [ $? -eq 0 ]; then
        echo "nfs mount passed"
    else
        echo "nfs mount failed"
    fi
else
    mountpoint -q /root/nfs_ipu/
    if [ $? -eq 0 ]; then
        echo "nfs have been mount"
    else
        mount -t nfs spiv-pnp-test1.sh.intel.com:/data_bck/nfs/IPU/in-progress/ /root/nfs_ipu/
    fi
fi

# get uuid from result.csv
uuid=$(uuidgen)
tar -zcvf $uuid.tar.gz $log_path/$date_now
md5=($(md5sum $uuid.tar.gz))
mv $uuid.tar.gz $uuid.${md5[0]}.tar.gz
cp $uuid.${md5[0]}.tar.gz /root/nfs_ipu/
echo $uuid.${md5[0]}