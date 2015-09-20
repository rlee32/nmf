CUDA_PATH = /usr/local/cuda-6.5
MYARCH = 20
CC = c++
NVCC = $(CUDA_PATH)/bin/nvcc
CFLAGS = -L$(CUDA_PATH)/lib64 -Wl,-rpath -Wl,$(CUDA_PATH)/lib64 -lcudart # -O3 -std=c++11
NVCCFLAGS= -arch=compute_$(MYARCH) -code=sm_$(MYARCH) -I$(CUDA_SDK_PATH)/C/common/inc --ptxas-options=-v -fmad=true -use_fast_math

CUSRCS = math.cu
CSRCS = main.cpp math.cpp
EXECNAME = gputest

CUOBJS = $(CUSRCS:.cu=.o__cu)
COBJS = $(CSRCS:.c=.o__c)

$(EXECNAME): $(CUOBJS) $(COBJS)
	$(CC) $(CFLAGS) $^ -o $@ 

%.o__c: %.c
	$(CC) -o $@ -c $<

%.o__cu: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

clean:
	rm -f core *.o__cu *.o__c *~ $(EXECNAME)
