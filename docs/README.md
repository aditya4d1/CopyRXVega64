## Benchmark buffer copy on RXVega64

This repo contains asm and hip source code to benchmark performance of buffer copy using single workgroup, different configurations of loop unrolling and different number of workitems per workgroup with different shader and memory clock speeds.

You can view the [source](https://github.com/adityaatluri/CopyRXVega64) and run on your AMD ROCm hardware.

### Introduction

#### Workgroup configurations
In this experiment, we launch a single workgroup of `256`, `512` and `1024` workitems doing a buffer copy of **1MB**; with each workitem loading a `dwordx4 (128bit)` at the same time. With these number of workitems per workgroup, there are multiple ways to write a kernel, one way to use a loop whose body does single copy from source buffer to destination buffer; another possible way is to do double copy from source buffer to destination buffer thereby decreasing the number of loops by half. Therefore, we use different unroll factors to do loop unrolling for each workgroup dimensions; the ones tested are `2`, `4`, `8` and `16` (for `1024` workitems workgroup, we use only until unroll factor of `8` as the number of registers available per workitem can't fit unroll factor of `16`).

```
foo_<total loops>_<unroll factor>_<number of loops>_<number of workitems>
```
***
We write the kernels in both **hip** and **asm** to find which one performs better than the other.

#### Naming
There is a naming scheme used for naming kernel names (and assembly files too).  The kernels have total loops, unroll factors, number of loops, number of workitems in a workgroup and implementation of the kernel in their names. For example, `foo_128_8_16_256_asm` means the number of workitems are `256`, as each workitem loads `4 32-bit` values or `16 bytes` the total number of loops each workgroups should do to transfer 1MB of data is `128`. We pick loop unroll factor as `8` therefore, the number of loops each workitem does after unrolling is `16` (`128/8`).
***
Output buffer is validated after running the kernel checking of mistakes in implementation.

#### Power
Power states of _shader clock_ and _memory clock_ are changed to find the best performance of the combination.

Below, are power states of the GPU used **RX Vega 64**. There are **8** _shader clock_ (_alu clock_) power states and **4** _memory clock_ power states
```
====================    ROCm System Management Interface    ====================
================================================================================
GPU[0] 		: Supported GPU clock frequencies on GPU0
GPU[0] 		: 0: 852Mhz 
GPU[0] 		: 1: 991Mhz 
GPU[0] 		: 2: 1084Mhz 
GPU[0] 		: 3: 1138Mhz 
GPU[0] 		: 4: 1200Mhz 
GPU[0] 		: 5: 1401Mhz 
GPU[0] 		: 6: 1536Mhz 
GPU[0] 		: 7: 1630Mhz
GPU[0] 		: 
GPU[0] 		: Supported GPU Memory clock frequencies on GPU0
GPU[0] 		: 0: 167Mhz 
GPU[0] 		: 1: 500Mhz 
GPU[0] 		: 2: 800Mhz 
GPU[0] 		: 3: 945Mhz
GPU[0] 		: 
================================================================================
====================           End of ROCm SMI Log          ====================
```

### Performance

The following chart shows performance of different kernels 
![alt text](https://raw.githubusercontent.com/adityaatluri/CopyRXVega64/master/docs/results.png)

- The assembly kernel performance is shown on the left side of the chart
- The hip kernel performance is shown on the right side of the chart
- The bandwidth at different power states are shows on Y-axis
- Different kernels used for benchmarking is shown on X-axis