#include <stdio.h>

__global__ void firstParallel() {
  printf("This is running in parallel.\n");
}

int main() {
  firstParallel<<<5, 5>>>(); // 在GPU中为核函数分配5个具有5个线程的线程块，将运行25次；
  cudaDeviceSynchronize(); // 同步
}

// nvcc basic_parallel.cu -o basic_parallel