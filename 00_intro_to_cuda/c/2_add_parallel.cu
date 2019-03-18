/**********************************************************************\
 * Author: Jose A. Iglesias-Guitian                                   *
 * C/C++ code							      *
 * Introduction to CUDA						      *
/**********************************************************************/

// Instructions: How to compile this program.
// nvcc 2_add_parallel.cu -L /usr/local/cuda/lib -lcudart -o 2_add_parallel

// Multiple blocks, one thread each

#include<stdio.h>

__global__ void add(int *a, int *b, int *c)  {
    int id = blockIdx.x;

    c[id] = a[id] + b[id];
}

int main(void) {
     // Vector size
    int N = 10000;

    // Host vectors
    int *a, *b;
    int *c; // output vector

    // Device vectors
    int *d_a, *d_b;
    int *d_c;  // device copies

    // Size in bytes of each vector
    size_t size = N*sizeof(int);

    // Allocate host memory
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    // Allocate device memory
    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    // Initialize host vectors
    for( int i = 0; i < N; i++) {
      a[i] = i;
      b[i] = -(i-1);
    }

    // Copy host input vectors to device
    cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

    // Number of thread per block
    int blockCount = N;

    // Launch add() on GPU
    add<<<blockCount,1>>>(d_a, d_b, d_c);

    // Copy result to host
    cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost);

    // Results should sum up to N
    int sum = 0;
    for (int i = 0; i < N; i++) {
      if (i < 5) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
      }
      sum += c[i];
    }
    printf("...\n");

    printf("Should be %d\nResults: %d\n", N,sum);

    // Cleanup host
    free(a);
    free(b);
    free(c);

    // Cleanup device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
