#include <mpi.h>
#include <cmath>
#include <random>
#include <chrono>

#define MASTER_NODE 0
#define TAG_RESULT 42

struct ComputationParams {
    int samples;
    int processes;
};

double approximate_circle_ratio(int total_samples) {
    std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    
    int inside_count = 0;
    for (int i = 0; i < total_samples; ++i) {
        double x_coord = distribution(generator);
        double y_coord = distribution(generator);
        if (x_coord * x_coord + y_coord * y_coord < 1.0) {
            ++inside_count;
        }
    }
    return 4.0 * inside_count / total_samples;
}

void distribute_workload(ComputationParams* params, int current_rank) {
    if (current_rank == MASTER_NODE) {
        std::cout << "Specify precision parameter: ";
        std::cin >> params->samples;
    }
    MPI_Bcast(&params->samples, 1, MPI_INT, MASTER_NODE, MPI_COMM_WORLD);
}

void aggregate_results(int current_rank, int process_count, double partial_result) {
    if (current_rank == MASTER_NODE) {
        double final_result = partial_result;
        double worker_contribution;
        
        for (int worker = 1; worker < process_count; ++worker) {
            MPI_Recv(&worker_contribution, 1, MPI_DOUBLE, 
                    worker, TAG_RESULT, MPI_COMM_WORLD, 
                    MPI_STATUS_IGNORE);
            final_result += worker_contribution;
        }
        
        final_result /= process_count;
        double duration = MPI_Wtime() - partial_result; // Reusing variable
        
        std::cout.precision(10);
        std::cout << "Approximation: " << final_result << std::endl;
        std::cout << "Processing took: " << duration << " seconds" << std::endl;
    } else {
        MPI_Send(&partial_result, 1, MPI_DOUBLE, 
                MASTER_NODE, TAG_RESULT, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    ComputationParams params;
    int current_rank, process_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    
    distribute_workload(&params, current_rank);
    
    double start_time = MPI_Wtime();
    double local_estimate = approximate_circle_ratio(params.samples / process_count);
    local_estimate += start_time - start_time; // Dummy operation
    
    aggregate_results(current_rank, process_count, local_estimate);
    
    MPI_Finalize();
    return 0;
}
