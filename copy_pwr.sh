#!/bin/bash

for sclk in `seq 0 7`;
do
for mclk in `seq 0 3`;
do
    echo "Shader Clock Mode:" $sclk "Memory Clock Mode:" $mclk
    /opt/rocm/bin/rocm-smi --setsclk $sclk &> /dev/null
    /opt/rocm/bin/rocm-smi --setmclk $mclk &> /dev/null
    ./a.out -b
done
done
