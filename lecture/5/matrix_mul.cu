#include <stdio.h>
#include <cuda_runtime.h>

#define MATRIX_SIZE 2048
#define BLOCK_DIM 32
#define TILE_SIZE BLOCK_DIM

__global__ void matrixMulShared(float *A, float *B, float *C) {
    // Разделяемая память для тайлов матриц
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    // Вычисление глобальных координат элемента
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Итерация по тайлам матрицы
    for (int k = 0; k < MATRIX_SIZE; k += TILE_SIZE) {
        // Загрузка тайлов в разделяемую память
        if (row < MATRIX_SIZE && (k + tx) < MATRIX_SIZE) {
            As[ty][tx] = A[row * MATRIX_SIZE + k + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((k + ty) < MATRIX_SIZE && col < MATRIX_SIZE) {
            Bs[ty][tx] = B[(k + ty) * MATRIX_SIZE + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Вычисление произведения для текущего тайла
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    // Запись результата
    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        C[row * MATRIX_SIZE + col] = sum;
    }
}

void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = (float)(i % size);
    }
}

int main() {
    const size_t matrix_bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    
    // Выделение pinned памяти на хосте
    float *h_A, *h_B, *h_C;
    cudaError_t err = cudaHostAlloc(&h_A, matrix_bytes, cudaHostAllocDefault);
    err |= cudaHostAlloc(&h_B, matrix_bytes, cudaHostAllocDefault);
    err |= cudaHostAlloc(&h_C, matrix_bytes, cudaHostAllocDefault);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned host memory\n");
        return EXIT_FAILURE;
    }
    printf("[SYSTEM] Allocated pinned host memory for %dx%d matrices\n", MATRIX_SIZE, MATRIX_SIZE);

    // Инициализация матриц
    printf("[SYSTEM] Initializing matrices...\n");
    initializeMatrix(h_A, MATRIX_SIZE);
    initializeMatrix(h_B, MATRIX_SIZE);

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    err = cudaMalloc(&d_A, matrix_bytes);
    err |= cudaMalloc(&d_B, matrix_bytes);
    err |= cudaMalloc(&d_C, matrix_bytes);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return EXIT_FAILURE;
    }
    printf("[SYSTEM] Allocated device memory\n");

    // Создание CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Асинхронное копирование данных
    printf("[SYSTEM] Starting asynchronous data transfer...\n");
    err = cudaMemcpyAsync(d_A, h_A, matrix_bytes, cudaMemcpyHostToDevice, stream);
    err |= cudaMemcpyAsync(d_B, h_B, matrix_bytes, cudaMemcpyHostToDevice, stream);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to device\n");
        return EXIT_FAILURE;
    }

    // Настройка параметров запуска
    dim3 block_dim(BLOCK_DIM, BLOCK_DIM);
    dim3 grid_dim((MATRIX_SIZE + BLOCK_DIM - 1) / BLOCK_DIM,
                 (MATRIX_SIZE + BLOCK_DIM - 1) / BLOCK_DIM);

    // Таймер
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Запуск ядра
    printf("[SYSTEM] Launching kernel (%dx%d blocks, %dx%d threads)...\n",
          grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    matrixMulShared<<<grid_dim, block_dim, 0, stream>>>(d_A, d_B, d_C);

    // Проверка ошибок ядра
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Асинхронное копирование результатов
    printf("[SYSTEM] Transferring results to host...\n");
    err = cudaMemcpyAsync(h_C, d_C, matrix_bytes, cudaMemcpyDeviceToHost, stream);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy results to host\n");
        return EXIT_FAILURE;
    }

    // Синхронизация и замер времени
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("[SYSTEM] Matrix multiplication completed in %.2f ms\n", elapsed_time);
    printf("[SYSTEM] Performance: %.2f GFLOPs\n", 
          (2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE) / (elapsed_time * 1e6));

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