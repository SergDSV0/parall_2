#include <mpi.h>
#include <iostream>
#include <random>
#include <chrono>
#include <stdexcept>

class MonteCarloPiCalculator {
    int rank_;
    int size_;
    long total_points_;
    double calculation_time_;

    void validate_input(long N) const {
        if (N <= 0) {
            throw std::runtime_error("Number of points must be positive");
        }
        if (N < size_) {
            throw std::runtime_error("Number of points should be >= number of processes");
        }
    }

    void broadcast_input() {
        if (rank_ == 0) {
            std::cout << "Enter number of points (calculation accuracy): ";
            std::cin >> total_points_;
            validate_input(total_points_);
        }
        MPI_Bcast(&total_points_, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    }

    long count_points_in_circle() const {
        std::mt19937 generator(rank_ + 1);  // Seed with rank for reproducibility
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        const long local_points = total_points_ / size_;
        long local_in_circle = 0;

        for (long i = 0; i < local_points; ++i) {
            double x = distribution(generator);
            double y = distribution(generator);
            if (x * x + y * y <= 1.0) {
                ++local_in_circle;
            }
        }
        return local_in_circle;
    }

    double calculate_pi(long total_in_circle) const {
        return 4.0 * total_in_circle / total_points_;
    }

    void print_results(double pi) const {
        if (rank_ != 0) return;

        std::cout.precision(8);
        std::cout << "Calculated Pi: " << pi << "\n";
        std::cout.precision(6);
        std::cout << "Execution Time: " << calculation_time_ << " seconds\n";
    }

public:
    MonteCarloPiCalculator(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    ~MonteCarloPiCalculator() {
        MPI_Finalize();
    }

    void run() {
        try {
            const auto start_time = MPI_Wtime();
            
            broadcast_input();
            long local_in_circle = count_points_in_circle();
            
            // Reduce all local counts to rank 0
            long total_in_circle = 0;
            MPI_Reduce(&local_in_circle, &total_in_circle, 1, 
                      MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            calculation_time_ = MPI_Wtime() - start_time;

            if (rank_ == 0) {
                double pi = calculate_pi(total_in_circle);
                print_results(pi);
            }

        } catch (const std::exception& e) {
            if (rank_ == 0) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
    }
};

int main(int argc, char* argv[]) {
    MonteCarloPiCalculator calculator(argc, argv);
    calculator.run();
    return 0;
}