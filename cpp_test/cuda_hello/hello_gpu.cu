#include <stdio.h>

void helloCPU() {
  printf("Hello from the CPU.\n");
}

// __global__ 表明这是一个全局GPU核函数.
__global__ void helloGPU() {
  printf("Hello from the GPU.\n");
}

int main() {
  helloCPU(); // 调用CPU函数

   /* 使用 <<<...>>> 配置核函数的GPU参数，
   * 第一个1表示1个线程块，第二个1表示每个线程块1个线程。*/
  helloGPU<<<1, 1>>>(); // 调用GPU函数
  cudaDeviceSynchronize(); // `cudaDeviceSynchronize` 同步CPU和GPU
}

// nvcc hello_gpu.cu -o hello_gpu