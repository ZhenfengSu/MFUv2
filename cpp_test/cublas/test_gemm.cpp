/*
https://godweiyang.com/2021/08/24/gemm/
 */
#include <sys/time.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include<iostream>
#include<fstream>
using namespace std;
int8_t float2int8(float f, float scale) {
    int8_t i = int8_t(f * scale);
    if (i < -127) i = -127;
    if (i > 127) i = 127;
    return i;
}

template <typename T, typename S>
void allocate_memory(int m, int n, int k, T **A, T **B, S **C) {
    cudaMallocManaged(A, m * k * sizeof(T));
    cudaMallocManaged(B, k * n * sizeof(T));
    cudaMallocManaged(C, m * n * sizeof(S));
}

template <typename T, typename S>
void free_memory(T *A, T *B, S *C) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

template <typename T, typename S>
int cublas_gemm_ex(cublasHandle_t handle, cublasOperation_t transA, cublasOperation_t transB,
                   int m, int n, int k, T *A, T *B, S *C, int lda, int ldb, int ldc,
                   S *alpha, S *beta, int algo) {
    cudaDataType_t AType, BType, CType, ComputeType;
    if (std::is_same<T, float>::value) {
        AType = BType = CType = ComputeType = CUDA_R_32F;
    } else if (std::is_same<T, __half>::value) {
        AType = BType = CType = ComputeType = CUDA_R_16F;
    } else if (std::is_same<T, int8_t>::value) {
        AType = BType = CUDA_R_8I;
        CType = ComputeType = CUDA_R_32I;
    } else {
        printf("Not supported data type.");
        return -1;
    }
    cublasStatus_t status;
    status = cublasGemmEx(handle,
                          transA,
                          transB,
                          m,
                          n,
                          k,
                          alpha,
                          A,
                          AType,
                          lda,
                          B,
                          BType,
                          ldb,
                          beta,
                          C,
                          CType,
                          ldc,
                          ComputeType,
                          static_cast<cublasGemmAlgo_t>(algo));

    if (status == CUBLAS_STATUS_SUCCESS)
        return 1;
    else
        return -1;
}

template <typename T, typename S>
float test_gemm(cublasHandle_t handle, int m, int n, int k, T *A, T *B, S *C,
               S *alpha, S *beta, int algo, int iteration) {
    float total_time = 0;
    // 先运行一次，预热
    int warm_up = cublas_gemm_ex(handle,
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_N,
                                 n,
                                 m,
                                 k,
                                 B,
                                 A,
                                 C,
                                 n,
                                 k,
                                 n,
                                 alpha,
                                 beta,
                                 static_cast<cublasGemmAlgo_t>(algo));
    // 正式测试
    for (int i = 0; i < iteration; ++i) {
        struct timeval start, end;
        cudaDeviceSynchronize();
        cudaProfilerStart();
        gettimeofday(&start, NULL);
        int success = cublas_gemm_ex(handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     n,
                                     m,
                                     k,
                                     B,
                                     A,
                                     C,
                                     n,
                                     k,
                                     n,
                                     alpha,
                                     beta,
                                     static_cast<cublasGemmAlgo_t>(algo));
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        cudaProfilerStop();
        if (success > 0 && i > 0)
            total_time += (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
    }
    if (total_time > 0)
        printf("algo %d: %.3f ms\n", algo, total_time / (iteration - 1));

    // 返回平均时间
    float avg_time;
    avg_time = total_time / (iteration - 1);
    return avg_time;    
}

int main() {
    int m = 4096, n = 8192, k = 1024;
    float m_fp = 4096.0, n_fp = 8192.0, k_fp = 1024.0;
    printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);
    // 不使用tensor core的算法
    int start_algo = CUBLAS_GEMM_DEFAULT; // 不同的gemm算法
    int end_algo = CUBLAS_GEMM_ALGO23;
    // 使用tensor core的算法
    int start_algo_t_op = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    int end_algo_t_op = CUBLAS_GEMM_ALGO15_TENSOR_OP;
    int iteration = 1000;

    // 初始化输入数据和输出数据的内存空间
    float *fA, *fB, *fC;
    __half *hA, *hB, *hC;
    int8_t *iA, *iB; int32_t *iC;
    float f_alpha = 1, f_beta = 0;
    __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
    int32_t i_alpha = 1, i_beta = 0;
    allocate_memory(m, n, k, &fA, &fB, &fC);
    allocate_memory(m, n, k, &hA, &hB, &hC);
    allocate_memory(m, n, k, &iA, &iB, &iC);
    for (int i = 0; i < m * k; ++i) {
        fA[i] = float(i % 255 - 127) / 127;
        hA[i] = __float2half_rn(fA[i]);
        iA[i] = float2int8(fA[i], 127);
    } 
    for (int i = 0; i < k * n; ++i) {
        fB[i] = float(i % 255 - 127) / 127;
        hB[i] = __float2half_rn(fB[i]);
        iB[i] = float2int8(fB[i], 127);
    }
    // 创建cublas句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    // 创建时间记录的列表
    int total_algo = end_algo - start_algo + end_algo_t_op - start_algo_t_op + 2;
    printf("total algo number: %d\n", total_algo);
    float time_list_fp32[total_algo]={0};
    float time_list_fp16[total_algo]={0};
    float time_list_int8[total_algo]={0};
    printf(">>>>>>>>>>>>>>>>> test fp32 >>>>>>>>>>>>>>>>>\n");
    int i = 0;
    float time_fp32;
    // 不使用tensor core
    for (int algo = start_algo; algo <= end_algo; ++algo)
        {
            time_fp32 = test_gemm(handle, m, n, k, fA, fB, fC, &f_alpha, &f_beta, algo, iteration);
            time_list_fp32[i] = time_fp32;
            i++;
        }
    // 使用tensor core
    for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo)
        {
            time_fp32 = test_gemm(handle, m, n, k, fA, fB, fC, &f_alpha, &f_beta, algo, iteration);
            time_list_fp32[i] = time_fp32;
            i++;
        }

    printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
    i = 0;
    float time_fp16;
    // 不使用tensor core
    for (int algo = start_algo; algo <= end_algo; ++algo)
        {
            time_fp16 = test_gemm(handle, m, n, k, hA, hB, hC, &h_alpha, &h_beta, algo, iteration);
            time_list_fp16[i] = time_fp16;
            i++;
        }
        
    // 使用tensor core
    for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo)
        {
            time_fp16 = test_gemm(handle, m, n, k, hA, hB, hC, &h_alpha, &h_beta, algo, iteration);
            time_list_fp16[i] = time_fp16;
            i++;
        }

    printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
    i = 0;
    float time_int8;
    // 不使用tensor core
    for (int algo = start_algo; algo <= end_algo; ++algo)
    {
        time_int8 = test_gemm(handle, m, n, k, iA, iB, iC, &i_alpha, &i_beta, algo, iteration);
        time_list_int8[i] = time_int8;
        i++;
    }
    // 使用tensor core
    for (int algo = start_algo_t_op; algo <= end_algo_t_op; ++algo)
    {
        time_int8 = test_gemm(handle, m, n, k, iA, iB, iC, &i_alpha, &i_beta, algo, iteration);
        time_list_int8[i] = time_int8;
        i++;
    }
    // 打印时间
    printf(">>>>>>>>>>>>>>>>> compare time >>>>>>>>>>>>>>>>>\n");
    float min_time_fp32 = 1000000;
    float min_time_fp16 = 1000000;
    float min_time_int8 = 1000000;
    for (int i = 0; i < total_algo; ++i)
    {
        printf("algo %d: fp32 %.3f ms, fp16 %.3f ms, int8 %.3f ms\n", i, time_list_fp32[i], time_list_fp16[i], time_list_int8[i]);
        if (time_list_fp32[i] < min_time_fp32)
        {
            min_time_fp32 = time_list_fp32[i];
        }

        if (time_list_fp16[i] < min_time_fp16)
        {
            min_time_fp16 = time_list_fp16[i];
        }

        if (time_list_int8[i] < min_time_int8)
        {
            min_time_int8 = time_list_int8[i];
        }
        
    }
    // 打印最快时间
    printf(">>>>>>>>>>>>>>>>> compare min time >>>>>>>>>>>>>>>>>\n");
    printf("fp32: %.3f ms\n", min_time_fp32);
    printf("fp16: %.3f ms\n", min_time_fp16);
    printf("int8: %.3f ms\n", min_time_int8);
    float mfu_fp32;
    float mfu_fp16;
    float mfu_int8;
    // 转化为浮点数
    float FLOPS_matrix = 2 * m_fp * n_fp * k_fp / 1000000000000; //TFLOPS
    float FLOPS_fp32_4090 = 82.6;
    float FLOPS_fp16_4090 = 330.3;
    float FLOPS_int8_4090 = 660.6;
    printf("FLOPS_matrix: %.3f TFLOPS\n", FLOPS_matrix);
    mfu_fp32 = FLOPS_matrix / (min_time_fp32/1000) / FLOPS_fp32_4090*100;
    mfu_fp16 = FLOPS_matrix / (min_time_fp16/1000) / FLOPS_fp16_4090*100;
    mfu_int8 = FLOPS_matrix / (min_time_int8/1000) / FLOPS_int8_4090*100;
    printf(">>>>>>>>>>>>>>>>> compare MFU >>>>>>>>>>>>>>>>>\n");
    printf("fp32: %.3f \n", mfu_fp32);
    printf("fp16: %.3f \n", mfu_fp16);
    printf("int8: %.3f \n", mfu_int8);
    // 计算每个类型的最快时间
    // printf(">>>>>>>>>>>>>>>>> compare result >>>>>>>>>>>>>>>>>\n");
    // printf("fp32: ");
    // for (int i = 0; i < 10; ++i)
    //     printf("%.5f%c", fC[i], " \n"[i==9]);
    // printf("fp16: ");
    // for (int i = 0; i < 10; ++i)
    //     printf("%.5f%c", float(hC[i]), " \n"[i==9]);
    // printf("int8: ");
    // for (int i = 0; i < 10; ++i)
    //     printf("%.5f%c", float(iC[i])/127/127, " \n"[i==9]);

    free_memory(iA, iB, iC);
    free_memory(fA, fB, fC);
    free_memory(hA, hB, hC);
    return 0;
}
// nvcc test_gemm.cpp -o test_gemm -L/usr/local/cuda/lib64 -lcudart -lcuda -lcublas