SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .cu .cu.o .c .o

CUDA_PATH = /usr/local/cuda
SDK_PATH = /Developer/GPU\ Computing/

NVCC = $(CUDA_PATH)/bin/nvcc
CC = llvm-gcc-4.2
LINK = $(CC) -fPIC -Xlinker -rpath $(CUDA_PATH)/lib

INCLUDES = -I. -I$(CUDA_PATH)/include/ -I$(SDK_PATH)/C/common/inc/
LIBS = -L$(CUDA_PATH)/lib -lcuda -lcudart -L$(SDK_PATH)/C/lib/ -lcutil_x86_64

OBJ_CPU = main_cpu.o
OBJ_GPU = main_gpu.cu.o

NVCCFLAGS   = -m64 -O0 -g -G -Xcompiler "-O0 -g3 -fno-strict-aliasing -m64"
NVCCFLAGS  += --compiler-bindir=llvm-gcc-4.2

FLAGS = -O0 -g3 -fbounds-check -Wunused-variable -Wunused-parameter \
	      -pedantic-errors -Wall -pedantic -ftree-vrp -std=c99 -m64

%.cu.o : %.cu
	$(NVCC) -c -o $@ $< $(NVCCFLAGS) $(INCLUDES)

%.o : %.c
	$(CC) -c -o $@ $< $(FLAGS) $(INCLUDES)

all : test_cpu test_gpu

test_cpu : $(OBJ_CPU)
	$(LINK) -o test_cpu $(OBJ_CPU) $(LIBS) $(FLAGS)

test_gpu : $(OBJ_GPU)
	$(LINK) -o test_gpu $(OBJ_GPU) $(LIBS) $(FLAGS)

.PHONY : clean
clean :
	rm -f $(OBJ_CPU) $(OBJ_GPU) test_cpu test_gpu