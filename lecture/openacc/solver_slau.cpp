#include <iostream>
#include <cmath>
#include <openacc.h>
#include <chrono>

const int N = 10240; // Размер системы
const double EPS = 1e-15; // Точность
const int MAX_ITER = 1000; // Максимум итераций

int main() {
    double *A = new double[N*N]; // Матрица коэффициентов
    double *F = new double[N];   // Правая часть
    double *X0 = new double[N];  // Текущее приближение
    double *X1 = new double[N];  // Следующее приближение
    double *delta = new double[N]; // Разница между итерациями

    // Инициализация матрицы A (пример: диагональное преобладание)
    #pragma acc parallel loop collapse(2) copyout(A[0:N*N])
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i*N + j] = (i == j) ? 2.0 * N : 1.0;
        }
    }

    // Инициализация вектора F (отдельный цикл)
    #pragma acc parallel loop copyout(F[0:N])
    for (int i = 0; i < N; ++i) {
        F[i] = 1.0; // Произвольная правая часть
    }

    // Начальное приближение
    #pragma acc parallel loop copyout(X0[0:N])
    for (int i = 0; i < N; ++i) X0[i] = 0.0;

    // Итерационный процесс
    double eps = 1.0;
    int iter = 0;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma acc data copyin(A[0:N*N], F[0:N]) create(X1[0:N], delta[0:N]) copy(X0[0:N])
    while (eps > EPS && iter < MAX_ITER) {
        // Обновление решения
        #pragma acc parallel loop present(A, F, X0, X1)
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j) {
                if (j != i) sum += A[i*N + j] * X0[j];
            }
            X1[i] = (F[i] - sum) / A[i*N + i];
        }

        // Вычисление погрешности
        #pragma acc parallel loop present(X0, X1, delta)
        for (int i = 0; i < N; ++i) {
            delta[i] = fabs(X1[i] - X0[i]);
            X0[i] = X1[i];
        }

        // Суммирование погрешности на CPU
        #pragma acc update self(delta[0:N])
        eps = 0.0;
        for (int i = 0; i < N; ++i) eps += delta[i];
        eps /= N;

        iter++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Вывод результатов
    std::cout << "Iterations: " << iter << std::endl;
    std::cout << "Final error: " << eps << std::endl;
    std::cout << "GPU time: " << duration << " ms" << std::endl;

    delete[] A;
    delete[] F;
    delete[] X0;
    delete[] X1;
    delete[] delta;
    return 0;
}
