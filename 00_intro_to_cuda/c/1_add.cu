/**********************************************************************\
 * Author: Jose A. Iglesias-Guitian                                   *
 * C/C++ code							      *
 * Introduction to CUDA						      *
/**********************************************************************/

// Instructions: How to compile this program.
// nvcc 1_add.cu -L /usr/local/cuda/lib -lcudart -o 1_add

#include<stdio.h>

__global__ void add(int *a, int *b, int *c)
{
    printf("(GPU) Hello from thread id (%d,%d)\n", threadIdx.x, threadIdx.y);
    *c = *a + *b;
}

int main(void) {
    int a, b, c;	            // host copies of a, b, c
    int *d_a, *d_b, *d_c;	     // device copies of a, b, c
    int size = sizeof(int);
    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    // Setup input values
    a = 2;
    b = 7;
    // Copy inputs to device
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU
    add<<<1,1>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    printf("(CPU) Add result is %d\n", c);
    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
