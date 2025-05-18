#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define NUM_ELEMENTS (512 * 50000)
#define BLOCK_DIM 256
#define ITERATIONS 100
#define CHECK_CUDA_ERROR(err) if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error [%s:%d]: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); }

__global__ void processVectors(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const float product = A[idx] * B[idx];
        float sum = 0.0f;
        
        // Развернутый цикл для уменьшения количества итераций
        #pragma unroll(10)
        for (int j = 0; j < ITERATIONS; j++) {
            sum += __sinf(j + product);  // Используем быструю версию sinf
        }
        
        C[idx] = sum;
    }
}

void initializeData(float* A, float* B, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = sinf(i);
        B[i] = cosf(2 * i - 5);
    }
}

int main() {
    const size_t data_size = NUM_ELEMENTS * sizeof(float);
    const int grid_size = (NUM_ELEMENTS + BLOCK_DIM - 1) / BLOCK_DIM;
    
    // Выделение pinned памяти на хосте
    float *h_A, *h_B, *h_C;
    cudaError_t err = cudaHostAlloc(&h_A, data_size, cudaHostAllocDefault);
    err |= cudaHostAlloc(&h_B, data_size, cudaHostAllocDefault);
    err |= cudaHostAlloc(&h_C, data_size, cudaHostAllocDefault);
    CHECK_CUDA_ERROR(err);
    printf("[SYSTEM] Allocated pinned host memory for %d elements\n", NUM_ELEMENTS);

    // Инициализация данных
    printf("[SYSTEM] Initializing data...\n");
    initializeData(h_A, h_B, NUM_ELEMENTS);

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    err = cudaMalloc(&d_A, data_size);
    err |= cudaMalloc(&d_B, data_size);
    err |= cudaMalloc(&d_C, data_size);
    CHECK_CUDA_ERROR(err);
    printf("[SYSTEM] Allocated device memory\n");

    // Создание CUDA stream
    cudaStream_t stream;
    err = cudaStreamCreate(&stream);
    CHECK_CUDA_ERROR(err);

    // Асинхронное копирование данных
    printf("[SYSTEM] Starting asynchronous data transfer...\n");
    err = cudaMemcpyAsync(d_A, h_A, data_size, cudaMemcpyHostToDevice, stream);
    err |= cudaMemcpyAsync(d_B, h_B, data_size, cudaMemcpyHostToDevice, stream);
    CHECK_CUDA_ERROR(err);

    // Таймер
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Запуск ядра
    printf("[SYSTEM] Launching kernel (%d blocks, %d threads each)...\n", 
           grid_size, BLOCK_DIM);
    processVectors<<<grid_size, BLOCK_DIM, 0, stream>>>(d_A, d_B, d_C, NUM_ELEMENTS);
    
    // Проверка ошибок ядра
    err = cudaGetLastError();
    CHECK_CUDA_ERROR(err);

    // Асинхронное копирование результатов
    printf("[SYSTEM] Transferring results to host...\n");
    err = cudaMemcpyAsync(h_C, d_C, data_size, cudaMemcpyDeviceToHost, stream);
    CHECK_CUDA_ERROR(err);

    // Синхронизация и замер времени
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("[SYSTEM] Computation completed in %.2f ms\n", elapsed_time);
    printf("[SYSTEM] Throughput: %.2f GB/s\n", 
           (3 * data_size) / (elapsed_time * 1e6));  // 3 массива: 2 чтения, 1 запись

    // Освобождение ресурсов
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("[SYSTEM] Resources released\n");
    return EXIT_SUCCESS;
}