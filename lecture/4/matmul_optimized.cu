#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>

#define BLOCK_DIM 32
#define MATRIX_SIZE 2048
#define PADDING 1  // Для устранения банковских конфликтов

// Ядро с оптимизацией через разделяемую память и padding
__global__ void matrixMultiplyShared(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int width) {
    // Блокировка разделяемой памяти с padding
    __shared__ float As[BLOCK_DIM][BLOCK_DIM + PADDING];
    __shared__ float Bs[BLOCK_DIM][BLOCK_DIM + PADDING];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // Вычисление глобальных координат элемента
    int row = by * BLOCK_DIM + ty;
    int col = bx * BLOCK_DIM + tx;
    
    float sum = 0.0f;

    // Итерация по блокам матрицы
    for (int k = 0; k < width; k += BLOCK_DIM) {
        // Загрузка данных в разделяемую память
        if (row < width && (k + tx) < width) {
            As[ty][tx] = A[row * width + (k + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((k + ty) < width && col < width) {
            Bs[ty][tx] = B[(k + ty) * width + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        // Вычисление произведения для текущего блока
        for (int i = 0; i < BLOCK_DIM; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }

    // Запись результата
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = (float)(rand() % 100) / 10.0f;
    }
}

void verifyResult(const float* A, const float* B, const float* C, int size) {
    float eps = 1e-4f;
    int errors = 0;
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            
            if (fabs(C[i * size + j] - sum) > eps) {
                if (errors++ < 5) {
                    printf("Mismatch at (%d,%d): CPU=%f, GPU=%f\n",
                           i, j, sum, C[i * size + j]);
                }
            }
        }
    }
    
    if (errors == 0) {
        printf("Matrix multiplication verified successfully!\n");
    } else {
        printf("Found %d errors in matrix multiplication\n", errors);
    }
}

int main() {
    const int size = MATRIX_SIZE;
    const int bytes = size * size * sizeof(float);
    
    // Выделение хостовой памяти
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);
    assert(h_A && h_B && h_C);
    printf("[SYSTEM] Allocated host memory for %dx%d matrices\n", size, size);

    // Инициализация матриц
    printf("[SYSTEM] Initializing matrices with random values...\n");
    initializeMatrix(h_A, size);
    initializeMatrix(h_B, size);

    // Выделение device памяти
    float *d_A, *d_B, *d_C;
    cudaError_t err = cudaMalloc(&d_A, bytes);
    err |= cudaMalloc(&d_B, bytes);
    err |= cudaMalloc(&d_C, bytes);
    assert(err == cudaSuccess);
    printf("[SYSTEM] Allocated device memory\n");

    // Копирование данных на устройство
    printf("[SYSTEM] Copying data to device...\n");
    err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    err |= cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);

    // Настройка параметров запуска
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((size + BLOCK_DIM - 1) / BLOCK_DIM,
                 (size + BLOCK_DIM - 1) / BLOCK_DIM);

    // Создание событий для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Запуск ядра
    printf("[SYSTEM] Launching kernel (%dx%d blocks, %dx%d threads)...\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    cudaEventRecord(start);
    matrixMultiplyShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, size);
    cudaEventRecord(stop);

    // Проверка ошибок ядра
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Синхронизация и замер времени
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("[SYSTEM] Kernel execution time: %.2f ms\n", elapsed_time);

    // Копирование результатов обратно
    printf("[SYSTEM] Copying results to host...\n");
    err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);

    // Проверка результатов (опционально)
    printf("[SYSTEM] Verifying results...\n");
    verifyResult(h_A, h_B, h_C, size);

    // Освобождение ресурсов
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("[SYSTEM] Resources released\n");
    return EXIT_SUCCESS;
}