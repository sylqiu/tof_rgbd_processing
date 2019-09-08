# Makefile

TF_INC = `python -c "import tensorflow; print(tensorflow.sysconfig.get_include())"`

ifndef CUDA_HOME
    CUDA_HOME := /usr/local/cuda
endif
CC        = gcc -O2 -pthread
CXX       = g++
GPUCC     = nvcc
CFLAGS    = -std=c++11 -I$(TF_INC) -I"$(CUDA_HOME)/include" -DGOOGLE_CUDA=1
GPUCFLAGS = -c
LFLAGS    = -pthread -shared -fPIC
GPULFLAGS = -x cu -Xcompiler -fPIC
CGPUFLAGS = -L$(CUDA_HOME)/lib -L$(CUDA_HOME)/lib64 -lcudart

OUT_DIR   =  ../build

WARP_SRC = "warp_by_flow_kernel.cc" "warp_by_flow_op.cc"
GPU_SRC_WARP = warp_by_flow_kernel.cu.cc
GPU_PROD_WARP = $(OUT_DIR)/warp_by_dlow_kernel_gpu.o
WARP_PROD = $(OUT_DIR)/warp_by_flow.so

CFLAGS += -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__ -D_GLIBCXX_USE_CXX11_ABI=0

all: warp_by_flow

warp_by_flow:
	$(GPUCC) -g $(CFLAGS) $(GPUCFLAGS) $(GPU_SRC_WARP) $(GPULFLAGS) $(GPUDEF) -o $(GPU_PROD_WARP)
	$(CXX) -g $(CFLAGS) $(WARP_SRC) $(GPU_PROD_WARP) $(LFLAGS) $(CGPUFLAGS) -o $(WARP_PROD)

clean:
	rm -f $(GPU_PROD_WARP)
