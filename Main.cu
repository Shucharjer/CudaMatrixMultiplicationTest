#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <exception>
#include <random>
#include <iostream>

/*因为使用了模板，就有奇奇怪怪的不能分头文件的bug，再加上nvcc对多文件编译不是很友好，就直接写在了一个文件里*/ 
template <class T> T GetRandomValue();
template <class T> void GetRandomMatrix(T* dst, int rows, int cols);
template <class T> void PrintMatrix(T* dst, unsigned int rows, unsigned int cols);
template <class T> void PrintMatrixLine(T* dst, unsigned int rows, unsigned int cols, unsigned int rowToPrint);
template <class T> __device__ T Min(T a, T b);
template <class T> __global__ void kMatrixMultiplication(T* dDst, T* dSrc1, T* dSrc2, unsigned int rowsOfSrc1, unsigned int colsOfSrc1, unsigned int colsOfSrc2, unsigned int totalThreads, unsigned int calCount);
template <class T> cudaError_t cMatrixMultiplication(T* dst, T* src1, T* src2, unsigned int rowsOfSrc1, unsigned int colsOfSrc1, unsigned int rowsOfSrc2, unsigned int colsOfSrc2);

int main()
{
    const unsigned int arraySize = 1000;
    // malloc
    float* src1 = new float[arraySize * arraySize];
    float* src2 = new float[arraySize * arraySize];
    float* dst = new float[arraySize * arraySize];


    GetRandomMatrix((float*)src1, arraySize, arraySize);
    GetRandomMatrix((float*)src2, arraySize, arraySize);
    printf_s("start!\n");
    cMatrixMultiplication((float*)dst, (float*)src1, (float*)src2, arraySize, arraySize, arraySize, arraySize);
    printf_s("end!\n");
    
    // free
    delete[]src1;
    delete[]src2;
    delete[]dst;
    return 0;
}

template <class T>
T GetRandomValue()
{
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<T> distr(-100, 100);
    return distr(eng);
}

template <class T>
void GetRandomMatrix(T* dst, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            *(dst + i * cols + j) = GetRandomValue<T>();
    return;
}

template <class T>
void PrintMatrix(T* dst, unsigned int rows, unsigned int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            T val = *(dst + i * cols + j);
            printf_s("%f ", val);
        }
        printf_s("\n");
    }
}

template <class T>
void PrintMatrixLine(T* dst, unsigned int rows, unsigned int cols, unsigned int rowToPrint)
{
    if (rowToPrint < rows)
    {
        for (int i = 0; i < cols; i++)
        {
            printf_s("%f ", *(dst + rowToPrint * cols + i));
        }
    }
    else
    {
        printf_s("rowToPrint too big!");
        return;
    }
    printf_s("\n");
}

template <class T>
__device__ T Min(T a, T b)
{
    return a <= b ? a : b;
}

template <class T>
__global__ void kMatrixMultiplication(T* dDst, T* dSrc1, T* dSrc2, unsigned int rowsOfSrc1, unsigned int colsOfSrc1, unsigned int colsOfSrc2, unsigned int totalThreads, unsigned int calCount)
{
    // dst[i][j] = \Sigma src1[i][k] * src2[k][j];

    unsigned int blockId = /*blockIdx.z * (gridDim.x * gridDim.y) + */blockIdx.y * gridDim.x + blockIdx.x;
    unsigned int threadId = /*threadIdx.z * (blockDim.x * blockDim.y) + */threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int gThreadId = blockId * (blockDim.x * blockDim.y) + threadId;

    unsigned int offset = gThreadId * calCount;

    unsigned int end = offset + calCount;
    if (gThreadId == totalThreads - 1) end = rowsOfSrc1 * colsOfSrc2;

    const unsigned int blockSize = 16;

    for (unsigned int k = 0; k < colsOfSrc1; k += blockSize)
    {
        for (unsigned int index = offset; index < end; index += blockSize)
        {
            for (unsigned int K = k; K < Min(colsOfSrc1, k + blockSize); K++)
            {
#pragma unroll
                for (unsigned int Index = index; Index < Min(end, index + blockSize); Index++)
                {
                    *(dDst + Index) += *(dSrc1 + (Index / colsOfSrc1) * colsOfSrc1 + K) * *(dSrc2 + K * colsOfSrc2 + (Index % colsOfSrc2));
                }
            }
        }
    }
}

template <class T>
cudaError_t cMatrixMultiplication(T* dst, T* src1, T* src2, unsigned int rowsOfSrc1, unsigned int colsOfSrc1, unsigned int rowsOfSrc2, unsigned int colsOfSrc2)
{
    if (colsOfSrc1 != rowsOfSrc2)
    {
        printf("colsOfSrc1 not equals rowsOfSrc2!\n");
        return cudaErrorInvalidSource;
    }

    cudaError_t cudaStatus{ };

    T* dDst = nullptr;
    T* dSrc1 = nullptr;
    T* dSrc2 = nullptr;

    // threads与blocks的选择，也就是总线程数的选择是根据自己的gpu核心数来的，一般总线程数大于核心数才能发挥最大性能
    dim3 threads(5, 5, 1);
    dim3 blocks(11, 11, 1);

    unsigned int totalThreads = blocks.x * blocks.y * blocks.z * threads.x * threads.y * threads.z;
    unsigned int calCount = rowsOfSrc1 * colsOfSrc2 / totalThreads;

    // Get device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        printf("failed at cudaSetDevice!\n");
        goto Error;
    }

    // Malloc
    cudaStatus = cudaMalloc((void**)&dDst, sizeof(T) * rowsOfSrc1 * colsOfSrc2);
    if (cudaStatus != cudaSuccess) {
        printf("failed at cudaMalloc!\n");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dSrc1, sizeof(T) * rowsOfSrc1 * colsOfSrc1);
    if (cudaStatus != cudaSuccess) {
        printf("failed at cudaMalloc!\n");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dSrc2, sizeof(T) * rowsOfSrc2 * colsOfSrc2);
    if (cudaStatus != cudaSuccess) {
        printf("failed at cudaMalloc!\n");
        goto Error;
    }

    // Memcpy
    cudaStatus = cudaMemcpy(dSrc1, src1, sizeof(T) * rowsOfSrc1 * colsOfSrc1, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("failed at cudaMemcpy!\n");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dSrc2, src2, sizeof(T) * rowsOfSrc2 * colsOfSrc2, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("failed at cudaMemcpy!\n");
        goto Error;
    }

    // Calculate
    cudaMemset(dDst, 0, sizeof(T) * rowsOfSrc1 * colsOfSrc2);
    kMatrixMultiplication<<<blocks, threads>>>(dDst, dSrc1, dSrc2, rowsOfSrc1, colsOfSrc1, colsOfSrc2, totalThreads, calCount);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("failed at running!\n");
        goto Error;
    }

    // Sync
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        printf("failed at cudaDeviceSynchronize!\n");
        goto Error;
    }

    // Copy result
    cudaStatus = cudaMemcpy(dst, dDst, sizeof(T) * rowsOfSrc1 * colsOfSrc2, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("failed at cudaMemcpy!\n");
        goto Error;
    }

    // free
Error:
    cudaFree(dDst);
    cudaFree(dSrc1);
    cudaFree(dSrc2);
    cudaDeviceReset();
    return cudaStatus;
}