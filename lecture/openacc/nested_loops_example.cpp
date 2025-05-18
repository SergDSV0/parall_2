#include <iostream>
#include <cmath>
#include <openacc.h>

const int N = 1024 * 1024;

int main() {
    float *x = new float[N];
    float *y = new float[N];

    // Инициализация массива x на GPU
    #pragma acc parallel loop copyout(x[0:N])
    for(int i = 0; i < N; ++i) {
        x[i] = i * 0.001f;
    }

    // Вычисление y[i] = sin(x[i]) * cos(x[i]) с оптимизацией доступа
    #pragma acc parallel loop copyin(x[0:N]) copyout(y[0:N])
    for(int i = 0; i < N; ++i) {
        float val = x[i];
        y[i] = sinf(val) * cosf(val);
    }

    // Вывод результатов
    std::cout << "First 5 elements of y[i] = sin(x[i]) * cos(x[i]):" << std::endl;
    for(int i = 0; i < 5; ++i) {
        std::cout << "y[" << i << "] = " << y[i] << std::endl;
    }

    delete[] x;
    delete[] y;
    return 0;
}
