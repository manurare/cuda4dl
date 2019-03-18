######################################################################
# Author: Jose A. Iglesias-Guitian                                   #
# PyTorch code							     #
# Introduction to CUDA                                               #
######################################################################

# Add with a single thread on the GPU

import torch

a_gpu = torch.cuda.FloatTensor(1)
b_gpu = torch.cuda.FloatTensor(1)

c_gpu = torch.cuda.FloatTensor(1)

h_cpu = torch.FloatTensor(1)

a_gpu.fill_(2)
b_gpu.fill_(7)

# operation run in GPU
c_gpu = a_gpu + b_gpu

# data copied back to CPU
h_cpu = c_gpu.cpu()

print(c_gpu) # data in GPU tensor
print(h_cpu) # data in CPU tensor

