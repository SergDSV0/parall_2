#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define SHARED_MEM_SIZE (2 * BLOCK_SIZE * sizeof(float))
#define MIN_DISTANCE 0.01f
#define GRAVITY_CONST 10.0f
#define INITIAL_RADIUS 1e-4f

__global__ void computeAccelerations(float* positions_x,
                                    float* positions_y,
                                    float* accelerations_x,
                                    float* accelerations_y,
                                    int current_step,
                                    int num_particles) {
    extern __shared__ float shared_positions[];

    float* shared_x = shared_positions;
    float* shared_y = &shared_positions[blockDim.x];

    int particle_id = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = (current_step - 1) * num_particles;

    // Загрузка данных в разделяемую память
    shared_x[threadIdx.x] = positions_x[particle_id + offset];
    shared_y[threadIdx.x] = positions_y[particle_id + offset];
    __syncthreads();

    float acc_x = 0.0f, acc_y = 0.0f;
    float current_x = shared_x[threadIdx.x];
    float current_y = shared_y[threadIdx.x];

    for (int j = 0; j < blockDim.x; j++) {
        if (j != threadIdx.x) {
            float dx = shared_x[j] - current_x;
            float dy = shared_y[j] - current_y;
            float distance_sq = dx * dx + dy * dy;
            
            if (distance_sq > MIN_DISTANCE * MIN_DISTANCE) {
                float inv_dist_cube = GRAVITY_CONST / (distance_sq * sqrtf(distance_sq));
                acc_x += dx * inv_dist_cube;
                acc_y += dy * inv_dist_cube;
            }
        }
    }

    if (particle_id < num_particles) {
        accelerations_x[particle_id] = acc_x;
        accelerations_y[particle_id] = acc_y;
    }
}

__global__ void updatePositions(float* positions_x,
                               float* positions_y,
                               float* velocities_x,
                               float* velocities_y,
                               float* accelerations_x,
                               float* accelerations_y,
                               float time_step,
                               int current_step,
                               int num_particles) {
    int particle_id = threadIdx.x + blockIdx.x * blockDim.x;
    int prev_offset = (current_step - 1) * num_particles;
    int curr_offset = current_step * num_particles;

    if (particle_id < num_particles) {
        // Обновление позиций (метод Верле)
        positions_x[particle_id + curr_offset] = positions_x[particle_id + prev_offset] + 
                                               velocities_x[particle_id] * time_step + 
                                               0.5f * accelerations_x[particle_id] * time_step * time_step;
        
        positions_y[particle_id + curr_offset] = positions_y[particle_id + prev_offset] + 
                                               velocities_y[particle_id] * time_step + 
                                               0.5f * accelerations_y[particle_id] * time_step * time_step;

        // Обновление скоростей
        velocities_x[particle_id] += accelerations_x[particle_id] * time_step;
        velocities_y[particle_id] += accelerations_y[particle_id] * time_step;
    }
}

void initializeParticles(float* x, float* y, float* vx, float* vy, int num_particles) {
    for (int i = 0; i < num_particles; i++) {
        float angle = (float)rand() / RAND_MAX * 2.0f * M_PI;
        x[i] = cosf(angle) * INITIAL_RADIUS;
        y[i] = sinf(angle) * INITIAL_RADIUS;
        
        float velocity = (x[i] * x[i] + y[i] * y[i]) * 10.0f;
        vx[i] = -velocity * sinf(angle);
        vy[i] = velocity * cosf(angle);
    }
}

int main() {
    const int num_particles = 10240;
    const int num_steps = 10;
    const float time_step = 0.001f;

    size_t single_array_size = num_particles * sizeof(float);
    size_t full_array_size = num_steps * num_particles * sizeof(float);

    // Выделение хостовой памяти
    float *h_pos_x, *h_pos_y, *h_vel_x, *h_vel_y, *h_acc_x, *h_acc_y;
    h_pos_x = (float*)malloc(full_array_size);
    h_pos_y = (float*)malloc(full_array_size);
    h_vel_x = (float*)malloc(single_array_size);
    h_vel_y = (float*)malloc(single_array_size);
    h_acc_x = (float*)malloc(single_array_size);
    h_acc_y = (float*)malloc(single_array_size);

    if (!h_pos_x || !h_pos_y || !h_vel_x || !h_vel_y || !h_acc_x || !h_acc_y) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Инициализация частиц
    printf("Initializing particle system with %d particles...\n", num_particles);
    initializeParticles(h_pos_x, h_pos_y, h_vel_x, h_vel_y, num_particles);

    // Выделение device памяти
    float *d_pos_x, *d_pos_y, *d_vel_x, *d_vel_y, *d_acc_x, *d_acc_y;
    cudaMalloc(&d_pos_x, full_array_size);
    cudaMalloc(&d_pos_y, full_array_size);
    cudaMalloc(&d_vel_x, single_array_size);
    cudaMalloc(&d_vel_y, single_array_size);
    cudaMalloc(&d_acc_x, single_array_size);
    cudaMalloc(&d_acc_y, single_array_size);

    // Копирование данных на устройство
    cudaMemcpy(d_pos_x, h_pos_x, full_array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, h_pos_y, full_array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_x, h_vel_x, single_array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_y, h_vel_y, single_array_size, cudaMemcpyHostToDevice);

    // Настройка параметров запуска
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim((num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Таймер
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    // Основной цикл моделирования
    printf("Starting simulation for %d time steps...\n", num_steps);
    cudaEventRecord(start);
    
    for (int step = 1; step < num_steps; step++) {
        computeAccelerations<<<grid_dim, block_dim, SHARED_MEM_SIZE>>>(
            d_pos_x, d_pos_y, d_acc_x, d_acc_y, step, num_particles);
        
        updatePositions<<<grid_dim, block_dim>>>(
            d_pos_x, d_pos_y, d_vel_x, d_vel_y, d_acc_x, d_acc_y, 
            time_step, step, num_particles);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Simulation completed in %.2f ms\n", elapsed_time);

    // Освобождение ресурсов
    free(h_pos_x); free(h_pos_y); free(h_vel_x); free(h_vel_y); free(h_acc_x); free(h_acc_y);
    cudaFree(d_pos_x); cudaFree(d_pos_y); cudaFree(d_vel_x); cudaFree(d_vel_y); cudaFree(d_acc_x); cudaFree(d_acc_y);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}