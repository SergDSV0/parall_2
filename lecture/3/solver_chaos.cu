#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MATRIX_SIZE 2048
#define GLOBAL_TOLERANCE 1e-6
#define LOCAL_TOLERANCE 1e-15
#define BLOCK_DIM 16

__global__ void computeMatrixA(double* matrix, const double* vector, int size) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < size && col < size) {
        const double x_col = vector[col];
        const double x_row = vector[row];
        const double value = pow(sin(x_col) * cos(x_row), 2.0);
        matrix[row + col * size] = value + (row == col ? (double)size : 0.0);
    }
}

void checkCudaError(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", context, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Выделение pinned-памяти с проверкой ошибок
    double *host_vector, *host_function, *host_delta;
    cudaError_t err = cudaHostAlloc(&host_vector, MATRIX_SIZE * sizeof(double), 
                                   cudaHostAllocDefault);
    checkCudaError(err, "host_vector allocation");
    
    err = cudaHostAlloc(&host_function, MATRIX_SIZE * sizeof(double), 
                                      cudaHostAllocDefault);
    checkCudaError(err, "host_function allocation");
    
    err = cudaHostAlloc(&host_delta, MATRIX_SIZE * sizeof(double), 
                                   cudaHostAllocDefault);
    checkCudaError(err, "host_delta allocation");
    
    printf("[SYSTEM] Pinned host memory allocated for %d elements\n", MATRIX_SIZE);

    // Инициализация данных
    printf("[SYSTEM] Initializing data...\n");
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        host_vector[i] = 1.0;
        host_function[i] = 0.5;
    }

    // Выделение памяти на устройстве
    double *device_vector, *device_function, *device_matrix, *device_delta;
    err = cudaMalloc(&device_vector, MATRIX_SIZE * sizeof(double));
    checkCudaError(err, "device_vector allocation");
    
    err = cudaMalloc(&device_function, MATRIX_SIZE * sizeof(double));
    checkCudaError(err, "device_function allocation");
    
    err = cudaMalloc(&device_matrix, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
    checkCudaError(err, "device_matrix allocation");
    
    err = cudaMalloc(&device_delta, MATRIX_SIZE * sizeof(double));
    checkCudaError(err, "device_delta allocation");
    
    printf("[SYSTEM] Device memory allocated\n");

    // Создание CUDA stream
    cudaStream_t computation_stream;
    err = cudaStreamCreate(&computation_stream);
    checkCudaError(err, "stream creation");
    printf("[SYSTEM] CUDA stream created\n");

    // Асинхронное копирование данных
    printf("[SYSTEM] Starting asynchronous data transfer...\n");
    err = cudaMemcpyAsync(device_vector, host_vector, 
                         MATRIX_SIZE * sizeof(double),
                         cudaMemcpyHostToDevice, computation_stream);
    checkCudaError(err, "vector copy to device");
    
    err = cudaMemcpyAsync(device_function, host_function,
                         MATRIX_SIZE * sizeof(double),
                         cudaMemcpyHostToDevice, computation_stream);
    checkCudaError(err, "function copy to device");

    // Настройка параметров запуска ядра
    const dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    const dim3 grid_dim((MATRIX_SIZE + BLOCK_DIM - 1) / BLOCK_DIM,
                       (MATRIX_SIZE + BLOCK_DIM - 1) / BLOCK_DIM);

    // Запуск ядра для вычисления матрицы
    printf("[SYSTEM] Launching matrix computation kernel (%dx%d blocks, %dx%d threads)...\n",
           grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    computeMatrixA<<<grid_dim, block_dim, 0, computation_stream>>>
        (device_matrix, device_vector, MATRIX_SIZE);
    
    // Проверка ошибок ядра
    err = cudaGetLastError();
    checkCudaError(err, "kernel launch");

    // Синхронизация
    printf("[SYSTEM] Synchronizing stream...\n");
    err = cudaStreamSynchronize(computation_stream);
    checkCudaError(err, "stream synchronization");
    
    printf("[SYSTEM] All operations completed successfully\n");

    // Освобождение ресурсов
    cudaFreeHost(host_vector);
    cudaFreeHost(host_function);
    cudaFreeHost(host_delta);
    cudaFree(device_vector);
    cudaFree(device_function);
    cudaFree(device_matrix);
    cudaFree(device_delta);
    cudaStreamDestroy(computation_stream);
    
    printf("[SYSTEM] Resources released\n");
    return EXIT_SUCCESS;
}