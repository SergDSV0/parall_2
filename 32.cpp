#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <chrono>
#include <stdexcept>

namespace MathConstants {
    constexpr double RECIPROCAL_COEFF = 1.0;
    constexpr double QUADRATURE_FACTOR = 4.0;
    constexpr double STEP_ADJUSTMENT = 0.5;
}

class ParallelPiCalculator {
public:
    ParallelPiCalculator(int argc, char* argv[]) {
        if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
            throw std::runtime_error("MPI initialization failed");
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &process_count);
    }

    ~ParallelPiCalculator() {
        MPI_Finalize();
    }

    double compute() {
        if (process_rank == 0) {
            get_input_parameter();
        }
        
        broadcast_parameter();
        
        const double computation_start = MPI_Wtime();
        const double partial_result = compute_partial_sum();
        const double pi_approximation = aggregate_results(partial_result);
        
        if (process_rank == 0) {
            output_results(pi_approximation, 
                         MPI_Wtime() - computation_start);
        }
        
        return pi_approximation;
    }

private:
    int process_rank, process_count;
    long long total_intervals;

    void validate_input() const {
        if (total_intervals <= 0) {
            throw std::invalid_argument("Number of intervals must be positive");
        }
    }

    void get_input_parameter() {
        std::cout << "Specify quadrature precision (intervals): ";
        if (!(std::cin >> total_intervals)) {
            throw std::runtime_error("Invalid input format");
        }
        validate_input();
    }

    void broadcast_parameter() {
        if (MPI_Bcast(&total_intervals, 1, MPI_LONG_LONG, 
                     0, MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error("Parameter broadcast failed");
        }
    }

    double compute_partial_sum() const {
        const double interval_width = MathConstants::RECIPROCAL_COEFF / total_intervals;
        double local_sum = 0.0;

        #pragma omp parallel for reduction(+:local_sum) schedule(static)
        for (long long i = process_rank; i < total_intervals; i += process_count) {
            const double midpoint = (i + MathConstants::STEP_ADJUSTMENT) * interval_width;
            local_sum += MathConstants::QUADRATURE_FACTOR / 
                        (MathConstants::RECIPROCAL_COEFF + midpoint * midpoint);
        }
        
        return local_sum;
    }

    double aggregate_results(double partial_sum) const {
        double global_sum = 0.0;
        if (MPI_Reduce(&partial_sum, &global_sum, 1, MPI_DOUBLE, 
                      MPI_SUM, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
            throw std::runtime_error("Result aggregation failed");
        }
        return global_sum * (MathConstants::RECIPROCAL_COEFF / total_intervals);
    }

    void output_results(double pi_value, double elapsed_time) const {
        std::cout.precision(10);
        std::cout << "π approximation: " << pi_value << "\n";
        std::cout << "Computation time: " << elapsed_time << " seconds\n";
    }
};

int main(int argc, char* argv[]) {
    try {
        ParallelPiCalculator calculator(argc, argv);
        calculator.compute();
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
}