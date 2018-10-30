LLVM-MC=~/llvm80/bin/llvm-mc
LLVM-LD=~/llvm80/bin/ld.lld
HIPCC=/opt/rocm/bin/hipcc

.SUFFIXES: .s .co

all: asm disable_l2 enable_l2

HIP_SOURCES=copy_1wg_bm.cpp

ASM_SOURCES=copy_64_2_32_1024.s copy_64_4_16_1024.s copy_64_8_8_1024.s copy_128_2_64_512.s copy_128_4_32_512.s copy_128_8_16_512.s copy_128_16_8_512.s copy_256_2_128_256.s copy_256_4_64_256.s copy_256_8_32_256.s copy_256_16_16_256.s
ASM_COBJECTS=$(ASM_SOURCES:.s=.co)

disable_l2:
	$(HIPCC) -DENABLE_L2=0 $(HIP_SOURCES) -o $@.out

enable_l2:
	$(HIPCC) -DENABLE_L2=1 $(HIP_SOURCES) -o $@.out

asm: $(ASM_SOURCES) $(ASM_COBJECTS)

.s.co:
	$(LLVM-MC) -arch=amdgcn -mcpu=gfx900 -filetype=obj $< -o $<.o
	$(LLVM-LD) -shared $<.o -o $@

clean:
	rm -rf *.o *.co *.out
