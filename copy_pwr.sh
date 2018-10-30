#!/bin/bash

for sclk in `seq 0 7`;
do
for mclk in `seq 0 3`;
do
    echo "Shader Clock Mode:" $sclk "Memory Clock Mode:" $mclk
    /opt/rocm/bin/rocm-smi --setsclk $sclk &> /dev/null
    /opt/rocm/bin/rocm-smi --setmclk $mclk &> /dev/null
    ./enable_l2.out -b
done
done


for sclk in `seq 0 7`;
do
for mclk in `seq 0 3`;
do
    echo "Shader Clock Mode:" $sclk "Memory Clock Mode:" $mclk
    /opt/rocm/bin/rocm-smi --setsclk $sclk &> /dev/null
    /opt/rocm/bin/rocm-smi --setmclk $mclk &> /dev/null
    ./disable_l2.out -b
done
done
