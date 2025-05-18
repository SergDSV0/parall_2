#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define ELEMENT_COUNT (512 * 50000)
#define THREADS_PER_BLOCK 512
#define ITERATIONS 100

__global__ void computeTrigonometricProduct(float *input1, float *input2, float *output, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements) {
        float product = input1[idx] * input2[idx];
        float accumulatedValue = 0.0f;
        
        for (int cycle = 0; cycle < ITERATIONS; cycle++) {
            accumulatedValue += sinf(product + cycle);
        }
        
        output[idx] = accumulatedValue;
    }
}

void initializeData(float *data1, float *data2, int count) {
    for (int i = 0; i < count; i++) {
        data1[i] = sinf(i);
        data2[i] = cosf(2 * i - 5);
    }
}

int main() {
    size_t dataSize = ELEMENT_COUNT * sizeof(float);
    
    // Allocate pinned host memory
    float *hostDataA, *hostDataB, *hostResult;
    cudaMallocHost(&hostDataA, dataSize);
    cudaMallocHost(&hostDataB, dataSize);
    cudaMallocHost(&hostResult, dataSize);
    
    printf("[SYSTEM] Preparing input data...\n");
    initializeData(hostDataA, hostDataB, ELEMENT_COUNT);
    
    // Allocate device memory
    float *deviceDataA, *deviceDataB, *deviceResult;
    cudaMalloc(&deviceDataA, dataSize);
    cudaMalloc(&deviceDataB, dataSize);
    cudaMalloc(&deviceResult, dataSize);
    
    // Transfer data to device
    cudaMemcpy(deviceDataA, hostDataA, dataSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDataB, hostDataB, dataSize, cudaMemcpyHostToDevice);
    
    // Calculate grid dimensions
    int blockCount = (ELEMENT_COUNT + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    printf("[SYSTEM] Executing computation kernel (%d blocks, %d threads each)...\n", 
           blockCount, THREADS_PER_BLOCK);
    computeTrigonometricProduct<<<blockCount, THREADS_PER_BLOCK>>>(
        deviceDataA, deviceDataB, deviceResult, ELEMENT_COUNT);
    
    // Retrieve results
    printf("[SYSTEM] Transferring results to host...\n");
    cudaMemcpy(hostResult, deviceResult, dataSize, cudaMemcpyDeviceToHost);
    
    // Cleanup resources
    cudaFreeHost(hostDataA);
    cudaFreeHost(hostDataB);
    cudaFreeHost(hostResult);
    cudaFree(deviceDataA);
    cudaFree(deviceDataB);
    cudaFree(deviceResult);
    
    printf("[SYSTEM] Computation completed and resources released.\n");
    
    return 0;
}