#include <mpi.h>
#include <iostream>
#include <vector>
#include <memory>

class MPIProcessor {
    int rank_;
    int world_size_;
    
    void handle_master() {
        std::vector<int> received_data(world_size_ - 1);
        
        for (int worker = 1; worker < world_size_; ++worker) {
            MPI_Recv(&received_data[worker-1], 1, MPI_INT, 
                    worker, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        display_results(received_data);
    }
    
    void handle_worker() {
        MPI_Send(&rank_, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    
    void display_results(const std::vector<int>& data) const {
        std::cout << "Master node collected results:\n";
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << " - Worker " << (i+1) 
                      << " sent value: " << data[i] << "\n";
        }
    }
    
public:
    MPIProcessor(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }
    
    ~MPIProcessor() {
        MPI_Finalize();
    }
    
    void execute() {
        if (rank_ == 0) {
            handle_master();
        } else {
            handle_worker();
        }
    }
};

int main(int argc, char** argv) {
    MPIProcessor processor(argc, argv);
    processor.execute();
    return 0;
}