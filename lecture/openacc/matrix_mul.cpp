#include <iostream>
#include <openacc.h>
#include <chrono>

const int N = 2048;
const int BLOCK_SIZE = 32;

int main() {
    float *A = new float[N*N];
    float *B = new float[N*N];
    float *C = new float[N*N];

    // Инициализация матриц
    #pragma acc parallel loop copyout(A[0:N*N], B[0:N*N])
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i*N + j] = 2*j + i;
            B[i*N + j] = j - i;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Перемножение матриц с оптимизацией
    #pragma acc data copyin(A[0:N*N], B[0:N*N]) copyout(C[0:N*N])
    {
        #pragma acc parallel loop tile(BLOCK_SIZE, BLOCK_SIZE) \
            present(A, B, C) async
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                #pragma acc loop seq
                for (int k = 0; k < N; ++k) {
                    sum += A[i*N + k] * B[k*N + j];
                }
                C[i*N + j] = sum;
            }
        }
    }
    #pragma acc wait

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "GPU Time: " << duration << " ms" << std::endl;

    delete[] A; 
    delete[] B; 
    delete[] C;
    return 0;
}
