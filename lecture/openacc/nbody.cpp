#include <iostream>
#include <cmath>
#include <cstdlib>
#include <openacc.h>
#include <chrono>

const int N = 10240;      // Количество частиц
const int NT = 10;        // Число шагов по времени
const float tau = 0.001f; // Шаг по времени

int main() {
    float *X = new float[NT * N];
    float *Y = new float[NT * N];
    float *VX = new float[N];
    float *VY = new float[N];
    float *AX = new float[N];
    float *AY = new float[N];

    // Инициализация начальных условий на CPU
    for (int i = 0; i < N; ++i) {
        float phi = rand() % 1000 * 0.001f;
        X[i] = cos(phi) * 1e-4f;
        Y[i] = sin(phi) * 1e-4f;
        VX[i] = -Y[i] * 10.0f;
        VY[i] = X[i] * 10.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Копирование данных на GPU
    #pragma acc data copy(X[0:NT*N], Y[0:NT*N], VX[0:N], VY[0:N]) create(AX, AY)
    {
        for (int step = 1; step < NT; ++step) {
            // Вычисление ускорений на GPU
            #pragma acc parallel loop present(X, Y, AX, AY) async
            for (int i = 0; i < N; ++i) {
                float ax = 0.0f, ay = 0.0f;
                for (int j = 0; j < N; ++j) {
                    if (i == j) continue;
                    float dx = X[j] - X[i];
                    float dy = Y[j] - Y[i];
                    float r = sqrtf(dx*dx + dy*dy);
                    if (r > 0.01f) {
                        float inv_r3 = 10.0f / (r * r * r);
                        ax += dx * inv_r3;
                        ay += dy * inv_r3;
                    }
                }
                AX[i] = ax;
                AY[i] = ay;
            }

            // Обновление координат на GPU
            #pragma acc parallel loop present(X, Y, VX, VY, AX, AY) async
            for (int i = 0; i < N; ++i) {
                int idx = step * N + i;
                X[idx] = X[(step-1)*N + i] + VX[i] * tau + AX[i] * tau * tau * 0.5f;
                Y[idx] = Y[(step-1)*N + i] + VY[i] * tau + AY[i] * tau * tau * 0.5f;
                VX[i] += AX[i] * tau;
                VY[i] += AY[i] * tau;
            }
        }
        #pragma acc wait
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "GPU Time: " << duration << " ms" << std::endl;

    delete[] X; 
    delete[] Y; 
    delete[] VX; 
    delete[] VY; 
    delete[] AX; 
    delete[] AY;
    return 0;
}
