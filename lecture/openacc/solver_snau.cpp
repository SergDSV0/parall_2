#include <iostream>
#include <cmath>
#include <openacc.h>
#include <chrono>

const int N = 2048; // Размер системы
const double EPS_G = 1e-6; // Точность для СНАУ
const double TAU = 0.1; // Шаг метода

int main() {
    double *X0 = new double[N]; // Текущее приближение
    double *X1 = new double[N]; // Следующее приближение
    double *V = new double[N];  // Поправка
    double *Phi = new double[N]; // Вектор невязки

    // Инициализация начального приближения
    #pragma acc parallel loop copyout(X0[0:N])
    for (int i = 0; i < N; ++i) X0[i] = 1.0;

    auto start = std::chrono::high_resolution_clock::now();
    double eps_g = 1.0;
    int iter = 0;

    #pragma acc data copy(X0[0:N]) create(X1[0:N], V[0:N], Phi[0:N])
    while (eps_g > EPS_G && iter < 100) {
        // Вычисление невязки Phi(X0)
        #pragma acc parallel loop present(X0, Phi)
        for (int i = 0; i < N; ++i) {
            Phi[i] = sin(X0[i]) * cos(X0[i]) * X0[i] - 0.5;
        }

        // Вычисление поправки V (упрощенный вариант)
        #pragma acc parallel loop present(X0, V)
        for (int i = 0; i < N; ++i) {
            V[i] = -Phi[i] / (2.0 * cos(2.0 * X0[i]));
        }

        // Обновление решения
        #pragma acc parallel loop present(X0, X1, V)
        for (int i = 0; i < N; ++i) {
            X1[i] = X0[i] + TAU * V[i];
        }

        // Вычисление погрешности
        #pragma acc parallel loop present(X0, X1, Phi)
        for (int i = 0; i < N; ++i) {
            Phi[i] = fabs(X1[i] - X0[i]);
            X0[i] = X1[i];
        }

        #pragma acc update self(Phi[0:N])
        eps_g = 0.0;
        for (int i = 0; i < N; ++i) eps_g += Phi[i];
        eps_g /= N;

        iter++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Iterations: " << iter << std::endl;
    std::cout << "Final error: " << eps_g << std::endl;
    std::cout << "GPU time: " << duration << " ms" << std::endl;

    delete[] X0;
    delete[] X1;
    delete[] V;
    delete[] Phi;
    return 0;
}
