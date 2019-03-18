#!/usr/bin/env python

import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
import libcudnn, ctypes
import numpy as np

# Create a cuDNN context
cudnn_context = libcudnn.cudnnCreate()

# Set some options and tensor dimensions
tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CROSS_CORRELATION']
#convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CONVOLUTION']
#convolution_fwd_pref = libcudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_PREFER_FASTEST']
convolution_fwd_pref = libcudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT']

start, end = (drv.Event(), drv.Event())

def start_bench():
    start.record()

def end_bench(op):
    end.record()
    end.synchronize()
    msecs  = end.time_since(start)
    print("%7.3f msecs" % (msecs))

n_input = 64
filters_in = 128
filters_out = 128
height_in = 112
width_in = 112
height_filter = 7
width_filter = 7
pad_h = 3
pad_w = 3
vertical_stride = 1
horizontal_stride = 1
upscalex = 1
upscaley = 1
alpha = 1.0
beta = 1.0

# Input tensor
X = gpuarray.to_gpu(np.random.rand(n_input, filters_in, height_in, width_in)
    .astype(np.float32))

# Filter tensor
filters = gpuarray.to_gpu(np.random.rand(filters_out,
    filters_in, height_filter, width_filter).astype(np.float32))

# Descriptor for input
X_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensor4dDescriptor(X_desc, tensor_format, data_type,
    n_input, filters_in, height_in, width_in)

# Filter descriptor
filters_desc = libcudnn.cudnnCreateFilterDescriptor()
libcudnn.cudnnSetFilter4dDescriptor(filters_desc, data_type, tensor_format, filters_out,
    filters_in, height_filter, width_filter)

# Convolution descriptor
conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
libcudnn.cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w,
    vertical_stride, horizontal_stride, upscalex, upscaley,
    convolution_mode, data_type)

# Get output dimensions (first two values are n_input and filters_out)
_, _, height_output, width_output = libcudnn.cudnnGetConvolution2dForwardOutputDim(
    conv_desc, X_desc, filters_desc)

# Output tensor
Y = gpuarray.empty((n_input, filters_out, height_output, width_output), np.float32)
Y_desc = libcudnn.cudnnCreateTensorDescriptor()
libcudnn.cudnnSetTensor4dDescriptor(Y_desc, tensor_format, data_type, n_input,
    filters_out, height_output, width_output)

# Get pointers to GPU memory
X_data = ctypes.c_void_p(int(X.gpudata))
filters_data = ctypes.c_void_p(int(filters.gpudata))
Y_data = ctypes.c_void_p(int(Y.gpudata))

for x in range(0, 3000):
        # Perform convolution
        # 18446744073709551615 maximum allowed workspace size (64 bits machines theoretical limit)
        algo = libcudnn.cudnnGetConvolutionForwardAlgorithm(cudnn_context, X_desc,
            filters_desc, conv_desc, Y_desc, convolution_fwd_pref, (2**x))

        print("Iteration = %d" % x + " bytes limit: %d" % (134217728 + 2**x))
        print("Cudnn algorithm = %d" % algo.value)

        ws_size = libcudnn.cudnnGetConvolutionForwardWorkspaceSize(cudnn_context, X_desc, filters_desc, conv_desc, Y_desc, algo)
        ws_ptr  = drv.mem_alloc(ws_size.value) if ws_size.value > 0 else 0
        ws_data = ctypes.c_void_p(int(ws_ptr))

        start_bench()

        libcudnn.cudnnConvolutionForward(cudnn_context, alpha, X_desc, X_data,
            filters_desc, filters_data, conv_desc, algo, ws_data, ws_size.value, beta,
            Y_desc, Y_data)

        end_bench("fprop")

        ws_ptr = None


# Clean up
libcudnn.cudnnDestroyTensorDescriptor(X_desc)
libcudnn.cudnnDestroyTensorDescriptor(Y_desc)
libcudnn.cudnnDestroyFilterDescriptor(filters_desc)
libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
libcudnn.cudnnDestroy(cudnn_context)
