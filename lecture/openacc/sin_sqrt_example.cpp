#include <iostream>
#include <cmath>
#include <openacc.h>

const int N = 1024 * 1024; // Размер массива

int main() {
    float *x = new float[N]; // Инициализация массива x на CPU
    float *y = new float[N]; // Инициализация массива y на CPU

    // Инициализация массива x: x_i = (2π/N) * i
    #pragma acc parallel loop copyout(x[0:N]) // Копирование x на GPU после инициализации
    for (int i = 0; i < N; ++i) {
        x[i] = 2.0f * M_PI * static_cast<float>(i) / static_cast<float>(N);
    }

    // Вычисление y[i] = sin(sqrt(x[i])) на GPU
    #pragma acc parallel loop copyin(x[0:N]) copyout(y[0:N]) // Копирование x на GPU и y обратно на CPU
    for (int i = 0; i < N; ++i) {
        y[i] = sinf(sqrtf(x[i]));
    }

    // Вывод первых 5 элементов для проверки (на английском)
    std::cout << "First 5 elements of y[i] = sin(sqrt(x[i])):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "y[" << i << "] = " << y[i] << std::endl;
    }

    delete[] x;
    delete[] y;
    return 0;
}
