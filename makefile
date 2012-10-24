SHELL = /bin/sh

.SUFFIXES:
.SUFFIXES: .cu .cu.o .c .o

CUDA_PATH = /usr/local/cuda
SDK_PATH = /usr/local/cuda/samples/common/inc/

NVCC = $(CUDA_PATH)/bin/nvcc
CC   = /usr/local/gcc-4.6.2/bin/gcc
PGCC = pgcc

LINK = $(CC) -fPIC -Xlinker -rpath $(CUDA_PATH)/lib64

#FLAGS, L=0 for testing, L=4 for optimization
ifndef L
  L = 4
endif

# paths
INCLUDES = -I. -I$(CUDA_PATH)/include/ -I$(SDK_PATH)
LIBS = -L$(CUDA_PATH)/lib64 -lcuda -lcudart

# flags
CCFLAGS   = -O3 -ffast-math -std=c99 -m64
PGCCFLAGS = -fast
ACCFLAGS  = -acc -ta=nvidia,cuda4.2,cc20 -Minfo=accel -lpgacc #-Minline=levels:2
OMPFLAGS  = -fast -mp -Minfo
NVCCFLAGS = -O3 -arch=sm_20 -m64

%.cu.o : %.cu
	$(NVCC) -c -o $@ $< $(NVCCFLAGS) $(INCLUDES)

all : laplace_cpu laplace_gpu laplace_omp laplace_acc

laplace_cpu : laplace_cpu.c
	$(CC) -o $@ $< $(CCFLAGS)

laplace_omp : laplace_cpu.c
	$(PGCC) -o $@ $< $(PGCCFLAGS) $(OMPFLAGS)
# alternatively, gcc can be used
#	$(CC) -o $@ $< -O3 -fopenmp -lm

laplace_acc : laplace_cpu.c
	$(PGCC) -o $@ $< $(PGCCFLAGS) $(ACCFLAGS)

laplace_gpu : laplace_gpu.cu.o
	$(LINK) -o $@ $< $(LIBS) $(CCFLAGS)

.PHONY : clean
clean :
	rm -f *.cu.o laplace_cpu laplace_gpu laplace_omp laplace_acc
