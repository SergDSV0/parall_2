#include <iostream>
#include <cmath>
#include <openacc.h>
#include <chrono>
#include <cassert>

const int N = 512 * 50000; // Размер массивов

int main() {
    float *a = nullptr, *b = nullptr, *c = nullptr;

    // Выделение pinned-памяти на устройстве
    a = (float*)acc_malloc(N * sizeof(float));
    b = (float*)acc_malloc(N * sizeof(float));
    c = (float*)acc_malloc(N * sizeof(float));

    // Проверка успешности выделения памяти
    assert(a != nullptr && b != nullptr && c != nullptr);

    // Инициализация массивов на GPU (асинхронная)
    #pragma acc parallel loop async deviceptr(a, b)
    for(int i = 0; i < N; ++i) {
        a[i] = 1.0f / ((i + 1.0f) * (i + 1.0f));
        b[i] = expf(1.0f / (i + 1.0f));
    }

    // Параллельное сложение на GPU
    auto start = std::chrono::high_resolution_clock::now();
    #pragma acc parallel loop async deviceptr(a, b, c)
    for(int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
    #pragma acc wait // Критическая синхронизация

    auto end = std::chrono::high_resolution_clock::now();
    auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Проверка первых 5 элементов через явное копирование
    float* host_c = new float[5];
    #pragma acc update self(c[0:5]) // Явное копирование данных
    for(int i = 0; i < 5; ++i) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }
    delete[] host_c;

    // Освобождение памяти
    acc_free(a);
    acc_free(b);
    acc_free(c);

    std::cout << "\nGPU time: " << gpu_time << " ms" << std::endl;
    return 0;
}
