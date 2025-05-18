#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define ELEMENTS (512 * 50000)
#define BLOCK_SIZE 512
#define GRID_SIZE ((ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE)

__global__ void vectorAddition(const float* __restrict__ A, 
                              const float* __restrict__ B,
                              float* __restrict__ C, 
                              int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

void initializeData(float* A, float* B, int size) {
    for (int i = 0; i < size; ++i) {
        const float x = i + 1.0f;
        A[i] = 1.0f / (x * x);
        B[i] = expf(1.0f / x);
    }
}

int main() {
    const size_t data_size = ELEMENTS * sizeof(float);
    
    // Выделение pinned-памяти с проверкой ошибок
    float *hA, *hB, *hC;
    cudaError_t err = cudaHostAlloc(&hA, data_size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned memory for hA: %s\n", 
                cudaGetErrorString(err));
        return 1;
    }
    err = cudaHostAlloc(&hB, data_size, cudaHostAllocDefault);
    err |= cudaHostAlloc(&hC, data_size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned memory: %s\n", 
                cudaGetErrorString(err));
        return 1;
    }

    // Инициализация данных
    printf("[CUDA] Initializing host data...\n");
    initializeData(hA, hB, ELEMENTS);

    // Выделение памяти на устройстве
    float *dA, *dB, *dC;
    err = cudaMalloc(&dA, data_size);
    err |= cudaMalloc(&dB, data_size);
    err |= cudaMalloc(&dC, data_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    // Копирование данных на устройство
    printf("[CUDA] Copying data to device...\n");
    err = cudaMemcpy(dA, hA, data_size, cudaMemcpyHostToDevice);
    err |= cudaMemcpy(dB, hB, data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to device: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    // Запуск ядра
    printf("[CUDA] Launching kernel (%d blocks, %d threads each)...\n",
           GRID_SIZE, BLOCK_SIZE);
    vectorAddition<<<GRID_SIZE, BLOCK_SIZE>>>(dA, dB, dC, ELEMENTS);
    
    // Проверка ошибок ядра
    if ((err = cudaGetLastError()) != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    // Копирование результатов обратно
    printf("[CUDA] Copying results to host...\n");
    if ((err = cudaMemcpy(hC, dC, data_size, cudaMemcpyDeviceToHost)) {
        fprintf(stderr, "Failed to copy results: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    // Освобождение ресурсов
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    printf("[CUDA] Computation completed successfully\n");
    return 0;
}