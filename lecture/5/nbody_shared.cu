#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define NUM_PARTICLES 20480
#define TIME_STEPS 10
#define BLOCK_SIZE 256
#define MIN_DISTANCE 0.01f
#define GRAVITY_CONST 10.0f
#define INITIAL_RADIUS 1e-4f
#define SHARED_MEM_SIZE (2 * BLOCK_SIZE * sizeof(float))

__global__ void nbodySimulation(float* positions_x,
                                float* positions_y,
                                float* velocities_x,
                                float* velocities_y,
                                float time_step,
                                int current_step) {
    extern __shared__ float shared_positions[];

    float* shared_x = shared_positions;
    float* shared_y = &shared_positions[blockDim.x];

    int particle_id = threadIdx.x + blockIdx.x * blockDim.x;
    int prev_offset = (current_step - 1) * NUM_PARTICLES;
    int curr_offset = current_step * NUM_PARTICLES;

    // Загрузка данных в разделяемую память
    shared_x[threadIdx.x] = positions_x[particle_id + prev_offset];
    shared_y[threadIdx.x] = positions_y[particle_id + prev_offset];
    __syncthreads();

    float acc_x = 0.0f, acc_y = 0.0f;
    float current_x = shared_x[threadIdx.x];
    float current_y = shared_y[threadIdx.x];

    // Вычисление ускорений
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

    // Обновление позиций и скоростей (метод Верле)
    if (particle_id < NUM_PARTICLES) {
        float vel_x = velocities_x[particle_id];
        float vel_y = velocities_y[particle_id];

        positions_x[particle_id + curr_offset] = current_x + vel_x * time_step + 
                                               0.5f * acc_x * time_step * time_step;
        positions_y[particle_id + curr_offset] = current_y + vel_y * time_step + 
                                               0.5f * acc_y * time_step * time_step;

        velocities_x[particle_id] = vel_x + acc_x * time_step;
        velocities_y[particle_id] = vel_y + acc_y * time_step;
    }
}

void initializeParticles(float* x, float* y, float* vx, float* vy) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        float angle = (float)rand() / RAND_MAX * 2.0f * M_PI;
        x[i] = cosf(angle) * INITIAL_RADIUS;
        y[i] = sinf(angle) * INITIAL_RADIUS;
        
        float velocity = (x[i] * x[i] + y[i] * y[i]) * 10.0f;
        vx[i] = -velocity * sinf(angle);
        vy[i] = velocity * cosf(angle);
    }
}

int main() {
    const float time_step = 0.001f;
    const size_t single_array_size = NUM_PARTICLES * sizeof(float);
    const size_t full_array_size = TIME_STEPS * NUM_PARTICLES * sizeof(float);

    // Выделение pinned памяти на хосте
    float *h_pos_x, *h_pos_y, *h_vel_x, *h_vel_y;
    cudaError_t err = cudaHostAlloc(&h_pos_x, full_array_size, cudaHostAllocDefault);
    err |= cudaHostAlloc(&h_pos_y, full_array_size, cudaHostAllocDefault);
    err |= cudaHostAlloc(&h_vel_x, single_array_size, cudaHostAllocDefault);
    err |= cudaHostAlloc(&h_vel_y, single_array_size, cudaHostAllocDefault);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned host memory\n");
        return EXIT_FAILURE;
    }
    printf("[SYSTEM] Allocated memory for %d particles over %d time steps\n", 
           NUM_PARTICLES, TIME_STEPS);

    // Инициализация частиц
    printf("[SYSTEM] Initializing particle system...\n");
    initializeParticles(h_pos_x, h_pos_y, h_vel_x, h_vel_y);

    // Выделение памяти на устройстве
    float *d_pos_x, *d_pos_y, *d_vel_x, *d_vel_y;
    err = cudaMalloc(&d_pos_x, full_array_size);
    err |= cudaMalloc(&d_pos_y, full_array_size);
    err |= cudaMalloc(&d_vel_x, single_array_size);
    err |= cudaMalloc(&d_vel_y, single_array_size);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return EXIT_FAILURE;
    }

    // Копирование данных на устройство
    printf("[SYSTEM] Transferring data to device...\n");
    err = cudaMemcpy(d_pos_x, h_pos_x, full_array_size, cudaMemcpyHostToDevice);
    err |= cudaMemcpy(d_pos_y, h_pos_y, full_array_size, cudaMemcpyHostToDevice);
    err |= cudaMemcpy(d_vel_x, h_vel_x, single_array_size, cudaMemcpyHostToDevice);
    err |= cudaMemcpy(d_vel_y, h_vel_y, single_array_size, cudaMemcpyHostToDevice);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy data to device\n");
        return EXIT_FAILURE;
    }

    // Настройка параметров запуска
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim((NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Создание CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Таймер
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);

    // Основной цикл моделирования
    printf("[SYSTEM] Starting simulation...\n");
    for (int step = 1; step < TIME_STEPS; step++) {
        nbodySimulation<<<grid_dim, block_dim, SHARED_MEM_SIZE, stream>>>(
            d_pos_x, d_pos_y, d_vel_x, d_vel_y, time_step, step);
    }

    // Синхронизация и замер времени
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("[SYSTEM] Simulation completed in %.2f ms\n", elapsed_time);
    printf("[SYSTEM] Performance: %.2f million interactions/second\n",
          (NUM_PARTICLES * NUM_PARTICLES * TIME_STEPS) / (elapsed_time * 1e3));

    // Освобождение ресурсов
    cudaFreeHost(h_pos_x);
    cudaFreeHost(h_pos_y);
    cudaFreeHost(h_vel_x);
    cudaFreeHost(h_vel_y);
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_vel_x);
    cudaFree(d_vel_y);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("[SYSTEM] Resources released\n");
    return EXIT_SUCCESS;
}