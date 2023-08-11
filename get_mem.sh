#!/bin/bash
set -xe

while true
do
    sleep 3
    if [ $(ps -ef |grep "run_.*.py.*device cpu" |wc -l) -lt 2 ];then
        break
    fi
    echo "$(date +%s), $(numastat -p $(ps -ef |grep "run_.*.py.*device cpu" |grep -v grep |awk '{printf("%s  ", $2)}'))"
done