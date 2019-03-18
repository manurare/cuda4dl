/**********************************************************************\
 * Author: Jose A. Iglesias-Guitian                                   *
 * C/C++ code							      *
 * Introduction to CUDA						      *
/**********************************************************************/

// Instructions: How to compile this program.
// nvcc 0_hello_world.cu -L /usr/local/cuda/lib -lcudart -o 0_hello-world

#include<stdio.h>

int main(void) {
    printf("Hello World! \n");
    return 0;
}
