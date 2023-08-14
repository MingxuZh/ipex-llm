#!/bin/bash
FILES=$1/*
for f in $FILES
do
    result=$(cat $f | grep "Throughput" | awk ' { sum += $2 } END { if (NR > 0) printf "= %s",sum } {printf "%s|",$2;} END{print "\n"}')
    filename="${f##*/}"
    echo "$filename = $result"
done