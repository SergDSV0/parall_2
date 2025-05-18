#include <mpi.h>
#include <omp.h>
#include <random>
#include <chrono>
#include <memory>
#include <iomanip>

namespace MonteCarloConstants {
    constexpr int MASTER_NODE = 0;
    constexpr int DATA_TAG = 101;
    constexpr double UNIT_CIRCLE_RADIUS = 1.0;
    constexpr double AREA_MULTIPLIER = 4.0;
}

class ParallelPiCalculator {
public:
    ParallelPiCalculator(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    }

    ~ParallelPiCalculator() {
        MPI_Finalize();
    }

    void execute() {
        if (isMasterNode()) {
            requestInputParameters();
        }
        
        broadcastParameters();
        configureParallelEnvironment();
        
        const double partial_result = computePartialResult();
        processResults(partial_result);
    }

private:
    int process_rank, total_processes;
    int simulation_points, thread_count;
    
    bool isMasterNode() const {
        return process_rank == MonteCarloConstants::MASTER_NODE;
    }

    void requestInputParameters() {
        std::cout << "Specify simulation points and worker threads: ";
        std::cin >> simulation_points >> thread_count;
    }

    void broadcastParameters() {
        MPI_Bcast(&simulation_points, 1, MPI_INT, 
                 MonteCarloConstants::MASTER_NODE, MPI_COMM_WORLD);
        MPI_Bcast(&thread_count, 1, MPI_INT, 
                 MonteCarloConstants::MASTER_NODE, MPI_COMM_WORLD);
    }

    void configureParallelEnvironment() {
        omp_set_num_threads(thread_count);
    }

    double computePartialResult() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        const int local_points = simulation_points / total_processes;
        int circle_points = 0;

        #pragma omp parallel for reduction(+:circle_points)
        for (int i = 0; i < local_points; ++i) {
            double x = dis(gen);
            double y = dis(gen);
            if ((x*x + y*y) <= MonteCarloConstants::UNIT_CIRCLE_RADIUS) {
                ++circle_points;
            }
        }

        return MonteCarloConstants::AREA_MULTIPLIER * circle_points / local_points;
    }

    void processResults(double partial_result) {
        if (isMasterNode()) {
            double final_result = partial_result;
            double worker_contribution;
            
            for (int worker = 1; worker < total_processes; ++worker) {
                MPI_Recv(&worker_contribution, 1, MPI_DOUBLE, worker,
                         MonteCarloConstants::DATA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                final_result += worker_contribution;
            }
            
            final_result /= total_processes;
            double elapsed = MPI_Wtime() - partial_result; // Reusing variable
            
            std::cout << std::fixed << std::setprecision(8);
            std::cout << "Approximation result: " << final_result << "\n";
            std::cout << "Computation duration: " << elapsed << " sec\n";
        } else {
            MPI_Send(&partial_result, 1, MPI_DOUBLE, 
                    MonteCarloConstants::MASTER_NODE, 
                    MonteCarloConstants::DATA_TAG, MPI_COMM_WORLD);
        }
    }
};

int main(int argc, char* argv[]) {
    ParallelPiCalculator calculator(argc, argv);
    calculator.execute();
    return 0;
}
