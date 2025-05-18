#include <stdlib.h>
#include <cuda.h>

#define ELEMENTS 1024
#define THREADS_PER_BLOCK 256
#define BLOCK_COUNT (ELEMENTS / THREADS_PER_BLOCK)

__global__ void computeSum(int *input1, int *input2, int *output) {
    int element_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_id < ELEMENTS) {
        output[element_id] = input1[element_id] + input2[element_id];
    }
}

void initializeArrays(int *arr1, int *arr2, int size) {
    for (int i = 0; i < size; i++) {
        arr1[i] = i;
        arr2[i] = i * 2;
    }
}

void displaySample(int *data, const char *name, int count) {
    printf("%s samples: ", name);
    for (int i = 0; i < count; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");
}

int verifyResults(int *result, int *src1, int *src2, int size) {
    int error_count = 0;
    for (int i = 0; i < size; i++) {
        if (result[i] != src1[i] + src2[i]) {
            error_count++;
            if (error_count <= 3) {
                printf("Mismatch at position %d: expected %d, got %d\n", 
                      i, src1[i] + src2[i], result[i]);
            }
        }
    }
    return error_count;
}

int main() {
    int *host_input1, *host_input2, *host_output;
    int *device_input1, *device_input2, *device_output;

    // Host memory allocation
    printf("Allocating host memory for %d elements...\n", ELEMENTS);
    host_input1 = (int*)malloc(ELEMENTS * sizeof(int));
    host_input2 = (int*)malloc(ELEMENTS * sizeof(int));
    host_output = (int*)malloc(ELEMENTS * sizeof(int));

    // Data initialization
    printf("Preparing initial data...\n");
    initializeArrays(host_input1, host_input2, ELEMENTS);

    // Display sample data
    printf("\nSample input values:\n");
    displaySample(host_input1, "Array A", 5);
    displaySample(host_input2, "Array B", 5);

    // Device memory allocation
    printf("\nAllocating device memory...\n");
    cudaMalloc(&device_input1, ELEMENTS * sizeof(int));
    cudaMalloc(&device_input2, ELEMENTS * sizeof(int));
    cudaMalloc(&device_output, ELEMENTS * sizeof(int));

    // Data transfer to device
    printf("Transferring data to device...\n");
    cudaMemcpy(device_input1, host_input1, ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_input2, host_input2, ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);

    // Kernel execution
    printf("Executing kernel with %d blocks of %d threads...\n", BLOCK_COUNT, THREADS_PER_BLOCK);
    computeSum<<<BLOCK_COUNT, THREADS_PER_BLOCK>>>(device_input1, device_input2, device_output);

    // Results transfer back
    printf("Retrieving results from device...\n");
    cudaMemcpy(host_output, device_output, ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

    // Error checking
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cuda_status));
    }

    // Display results
    printf("\nComputation results:\n");
    displaySample(host_output, "Output", 5);

    // Verification
    int total_errors = verifyResults(host_output, host_input1, host_input2, ELEMENTS);
    if (total_errors == 0) {
        printf("\nAll calculations are correct!\n");
    } else {
        printf("\nFound %d discrepancies in results\n", total_errors);
    }

    // Resource cleanup
    printf("\nReleasing allocated resources...\n");
    cudaFree(device_input1);
    cudaFree(device_input2);
    cudaFree(device_output);
    free(host_input1);
    free(host_input2);
    free(host_output);

    return EXIT_SUCCESS;
}