SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .cu .cu.o .c .o

CUDA_PATH = /usr/local/cuda
SDK_PATH = /Developer/GPU\ Computing

NVCC = $(CUDA_PATH)/bin/nvcc
CC = gcc
LINK = llvm-gcc-4.2 -fPIC -Xlinker -rpath $(CUDA_PATH)/lib

#FLAGS, L=0 for testing, L=4 for optimization
ifndef L
  L = 4
endif

INCLUDES = -I. -I$(CUDA_PATH)/include/ -I$(SDK_PATH)/C/common/inc/
LIBS = -L$(CUDA_PATH)/lib -lcuda -L$(SDK_PATH)/C/lib/ -lcutil_x86_64

OBJ_CPU = main_cpu.o
OBJ_GPU = main_gpu.cu.o

# flags
ifeq ("$(L)", "0")
	NVCCFLAGS   = -m64 -O0 -g -G
	#NVCCFLAGS  += -Xcompiler "-O0 -g -fno-strict-aliasing -m64"
	
	FLAGS = -g -O0 -fbounds-check -Wunused-variable -Wunused-parameter \
	      -pedantic-errors -Wall -pedantic -ftree-vrp -std=c99 -m64
	FLAGS += -da -Q
else ifeq ("$(L)", "4")
	NVCCFLAGS   = -O3 -use_fast_math -arch=sm_20 -m64
	#NVCCFLAGS  += -Xcompiler "-O3 -ffast-math"
	FLAGS = -O3 -ffast-math -std=c99 -m64
endif

NVCCFLAGS  += --compiler-bindir=llvm-gcc-4.2

%.cu.o : %.cu
	$(NVCC) -c -o $@ $< $(NVCCFLAGS) $(INCLUDES)

%.o : %.c
	$(CC) -c -o $@ $< $(FLAGS) $(INCLUDES)

all : test_cpu test_gpu

test_cpu : $(OBJ_CPU)
	$(CC) -o test_cpu $(OBJ_CPU) $(LIBS) $(FLAGS)

test_gpu : $(OBJ_GPU)
	$(LINK) -o test_gpu $(OBJ_GPU) $(LIBS) $(FLAGS)

.PHONY : clean
clean :
	rm -f $(OBJ_CPU) $(OBJ_GPU) test_cpu test_gpu