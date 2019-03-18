#include<stdio.h>

int main(void) {
  // int* a = (int *)malloc(sizeof(int));
  // *a = 4;
  // printf("%p\n",a);
  // printf("%d\n",*a);
  int a = 5;
  int* d_a = NULL;
  d_a = &a;
  printf("variable a = %d\n", a);
  printf("contenido variable d_a = %d\n", *d_a);
  return 0;
}
