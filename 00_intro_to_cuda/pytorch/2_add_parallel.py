######################################################################
# Author: Jose A. Iglesias-Guitian                                   #
# PyTorch code							     #
# Introduction to CUDA                                               #
######################################################################

# Element wise add operation

from __future__ import absolute_import
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit

import torch
import numpy

from pycuda.curandom import rand as curand

# Vector size
N = 10000

a_gpu = curand((N,))
b_gpu = 1 - a_gpu

c_cpu = torch.cuda.FloatTensor(N)

from pycuda.elementwise import ElementwiseKernel
func_kernel = ElementwiseKernel(
        "float *a, float *b, float *c",
        "c[i] = a[i] + b[i]",
        "add")

c_gpu = gpuarray.empty_like(a_gpu)

func_kernel(a_gpu, b_gpu, c_gpu)

# Copy result to host
#cuda.memcpy_dtoh(c_cpu, c_gpu)

c_cpu = c_gpu.get()

# Display results
print("Should be %d" % pycuda.gpuarray.sum(c_gpu).get())
print("Results: %d" % numpy.sum(c_cpu))
