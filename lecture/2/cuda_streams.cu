#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define TOTAL_ELEMENTS (512 * 50000)
#define STREAM_COUNT 4
#define BLOCK_SIZE 512

__global__ void computeVectorSum(const float* __restrict__ input1,
                                const float* __restrict__ input2,
                                float* __restrict__ output,
                                int elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elements) {
        output[idx] = input1[idx] + input2[idx];
    }
}

void initializeHostData(float* data1, float* data2, int count) {
    for (int i = 0; i < count; i++) {
        data1[i] = 1.0f / ((i + 1.0f) * (i + 1.0f));
        data2[i] = expf(1.0f / (i + 1.0f));
    }
}

int main() {
    const size_t data_size = TOTAL_ELEMENTS * sizeof(float);
    const int chunk_size = TOTAL_ELEMENTS / STREAM_COUNT;
    const int grid_size = (chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate pinned host memory
    float *host_A, *host_B, *host_C;
    cudaHostAlloc(&host_A, data_size, cudaHostAllocDefault);
    cudaHostAlloc(&host_B, data_size, cudaHostAllocDefault);
    cudaHostAlloc(&host_C, data_size, cudaHostAllocDefault);
    printf("[SYSTEM] Pinned host memory allocated for %d elements\n", TOTAL_ELEMENTS);

    // Initialize host data
    printf("[SYSTEM] Initializing host data...\n");
    initializeHostData(host_A, host_B, TOTAL_ELEMENTS);

    // Allocate device memory
    float *device_A, *device_B, *device_C;
    cudaMalloc(&device_A, data_size);
    cudaMalloc(&device_B, data_size);
    cudaMalloc(&device_C, data_size);
    printf("[SYSTEM] Device memory allocated\n");

    // Create CUDA streams
    cudaStream_t compute_streams[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamCreate(&compute_streams[i]);
    }
    printf("[SYSTEM] Created %d CUDA streams\n", STREAM_COUNT);

    // Process data in streams
    for (int i = 0; i < STREAM_COUNT; i++) {
        const int offset = i * chunk_size;
        
        // Async memory transfers
        cudaMemcpyAsync(device_A + offset, host_A + offset,
                       chunk_size * sizeof(float),
                       cudaMemcpyHostToDevice, compute_streams[i]);
        
        cudaMemcpyAsync(device_B + offset, host_B + offset,
                       chunk_size * sizeof(float),
                       cudaMemcpyHostToDevice, compute_streams[i]);

        // Kernel launch
        computeVectorSum<<<grid_size, BLOCK_SIZE, 0, compute_streams[i]>>>(
            device_A + offset, device_B + offset, device_C + offset, chunk_size);

        // Async result copy
        cudaMemcpyAsync(host_C + offset, device_C + offset,
                       chunk_size * sizeof(float),
                       cudaMemcpyDeviceToHost, compute_streams[i]);
    }

    // Synchronize all streams
    cudaDeviceSynchronize();
    printf("[SYSTEM] All stream operations completed\n");

    // Cleanup resources
    cudaFreeHost(host_A);
    cudaFreeHost(host_B);
    cudaFreeHost(host_C);
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamDestroy(compute_streams[i]);
    }
    printf("[SYSTEM] Resources released\n");

    return 0;
}