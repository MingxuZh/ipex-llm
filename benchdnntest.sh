#!/bin/bash

export logsdir=/dev/shm/frequency/log/

./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc) mb1_ic256oc256_ih14oh14kh3sh1dh0ph1_iw14ow14kw3sw1dw0pw1 2>&1 | tee "$logsdir/benchdnn_ins1.log" &
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc) mb1_ic64oc256_ih56oh56kh1sh1dh0ph0_iw56ow56kw1sw1dw0pw0 2>&1 | tee "$logsdir/benchdnn_ins2.log" &
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc) mb1_ic512oc512_ih7oh7kh3sh1dh0ph1_iw7ow7kw3sw1dw0pw1 2>&1 | tee "$logsdir/benchdnn_ins3.log" &
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc) mb1_ic256oc1024_ih14oh14kh1sh1dh0ph0_iw14ow14kw1sw1dw0pw0 2>&1 | tee "$logsdir/benchdnn_ins4.log" &
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc) mb1_ic128oc128_ih28oh28kh3sh1dh0ph1_iw28ow28kw3sw1dw0pw1 2>&1 | tee "$logsdir/benchdnn_ins5.log" &
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc) mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1 2>&1 | tee "$logsdir/benchdnn_ins6.log" &
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc) mb1_ic128oc512_ih28oh28kh1sh1dh0ph0_iw28ow28kw1sw1dw0pw0 2>&1 | tee "$logsdir/benchdnn_ins7.log" &
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc) mb1_ic512oc2048_ih7oh7kh1sh1dh0ph0_iw7ow7kw1sw1dw0pw0 2>&1 | tee "$logsdir/benchdnn_ins8.log" &
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc) mb1_ic3oc64_ih224oh112kh7sh2dh0ph3_iw224ow112kw7sw2dw0pw3 2>&1 | tee "$logsdir/benchdnn_ins9.log" &
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc) mb1_ic1024oc256_ih14oh14kh1sh1dh0ph0_iw14ow14kw1sw1dw0pw0 2>&1 | tee "$logsdir/benchdnn_ins10.log" &

./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc)
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc)
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc)
./benchdnn --conv --mode=P --max-ms-per-prb=10e7 --mb=$(nproc)