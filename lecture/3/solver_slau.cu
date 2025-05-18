#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TOLERANCE 1e-15
#define SYSTEM_SIZE 10240
#define BLOCK_SIZE 256
#define MAX_ITERATIONS 1000

__global__ void jacobiIteration(const double* __restrict__ A,
                               const double* __restrict__ F,
                               const double* __restrict__ X_old,
                               double* __restrict__ X_new,
                               int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double sum = 0.0;
        double a_ii = A[i * size + i];  // Diagonal element
        
        for (int j = 0; j < size; j++) {
            if (j != i) {
                sum += A[i * size + j] * X_old[j];
            }
        }
        
        X_new[i] = (F[i] - sum) / a_ii;
    }
}

__global__ void computeResidual(const double* __restrict__ X_old,
                               const double* __restrict__ X_new,
                               double* __restrict__ residual,
                               int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        residual[i] = fabs(X_new[i] - X_old[i]);
    }
}

void checkCudaError(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in %s: %s\n", context, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void initializeSystem(double* A, double* F, double* X, int size) {
    for (int i = 0; i < size; i++) {
        F[i] = 1.0;
        X[i] = 0.0;
        for (int j = 0; j < size; j++) {
            A[i * size + j] = (i == j) ? 2.0 : 0.1;  // Diagonal dominance
        }
    }
}

int main() {
    const int matrix_bytes = SYSTEM_SIZE * SYSTEM_SIZE * sizeof(double);
    const int vector_bytes = SYSTEM_SIZE * sizeof(double);
    const int grid_size = (SYSTEM_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Host memory allocation
    double *h_A, *h_F, *h_X, *h_X_new, *h_residual;
    h_A = (double*)malloc(matrix_bytes);
    h_F = (double*)malloc(vector_bytes);
    h_X = (double*)malloc(vector_bytes);
    h_X_new = (double*)malloc(vector_bytes);
    h_residual = (double*)malloc(vector_bytes);
    
    if (!h_A || !h_F || !h_X || !h_X_new || !h_residual) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    printf("[SYSTEM] Host memory allocated for %d equations\n", SYSTEM_SIZE);

    // Initialize system
    printf("[SYSTEM] Initializing system...\n");
    initializeSystem(h_A, h_F, h_X, SYSTEM_SIZE);

    // Device memory allocation
    double *d_A, *d_F, *d_X, *d_X_new, *d_residual;
    cudaError_t err = cudaMalloc(&d_A, matrix_bytes);
    err |= cudaMalloc(&d_F, vector_bytes);
    err |= cudaMalloc(&d_X, vector_bytes);
    err |= cudaMalloc(&d_X_new, vector_bytes);
    err |= cudaMalloc(&d_residual, vector_bytes);
    checkCudaError(err, "device memory allocation");

    // Copy data to device
    printf("[SYSTEM] Transferring data to device...\n");
    err = cudaMemcpy(d_A, h_A, matrix_bytes, cudaMemcpyHostToDevice);
    err |= cudaMemcpy(d_F, h_F, vector_bytes, cudaMemcpyHostToDevice);
    err |= cudaMemcpy(d_X, h_X, vector_bytes, cudaMemcpyHostToDevice);
    checkCudaError(err, "data transfer to device");

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Iterative solver
    printf("[SYSTEM] Starting Jacobi iterations...\n");
    double residual = 1.0;
    int iteration = 0;
    
    while (residual > TOLERANCE && iteration < MAX_ITERATIONS) {
        // Perform Jacobi iteration
        jacobiIteration<<<grid_size, BLOCK_SIZE>>>(d_A, d_F, d_X, d_X_new, SYSTEM_SIZE);
        
        // Compute residual
        computeResidual<<<grid_size, BLOCK_SIZE>>>(d_X, d_X_new, d_residual, SYSTEM_SIZE);
        
        // Copy residual back to host
        err = cudaMemcpy(h_residual, d_residual, vector_bytes, cudaMemcpyDeviceToHost);
        checkCudaError(err, "residual copy");
        
        // Calculate average residual
        residual = 0.0;
        for (int i = 0; i < SYSTEM_SIZE; i++) {
            residual += h_residual[i];
        }
        residual /= SYSTEM_SIZE;
        
        // Swap pointers for next iteration
        double* temp = d_X;
        d_X = d_X_new;
        d_X_new = temp;
        
        iteration++;
        if (iteration % 10 == 0) {
            printf("Iteration %4d: residual = %e\n", iteration, residual);
        }
    }

    // Record execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("[SYSTEM] Solver converged in %d iterations\n", iteration);
    printf("[SYSTEM] Final residual: %e\n", residual);
    printf("[SYSTEM] GPU execution time: %.2f ms\n", elapsed_time);

    // Free resources
    free(h_A);
    free(h_F);
    free(h_X);
    free(h_X_new);
    free(h_residual);
    cudaFree(d_A);
    cudaFree(d_F);
    cudaFree(d_X);
    cudaFree(d_X_new);
    cudaFree(d_residual);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("[SYSTEM] Resources released\n");
    return EXIT_SUCCESS;
}