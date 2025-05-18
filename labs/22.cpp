#include <mpi.h>
#include <iostream>
#include <vector>
#include <stdexcept>

class AllToAllCommunicator {
    int rank_;
    int size_;
    std::vector<int> received_messages_;

    void validate_environment() const {
        if (size_ < 2) {
            throw std::runtime_error("Minimum 2 processes required");
        }
    }

    void exchange_messages() {
        // Non-blocking send all ranks except self
        std::vector<MPI_Request> send_requests(size_ - 1);
        int request_index = 0;
        
        for (int dest = 0; dest < size_; ++dest) {
            if (dest != rank_) {
                MPI_Isend(&rank_, 1, MPI_INT, dest, 0, 
                         MPI_COMM_WORLD, &send_requests[request_index++]);
            }
        }

        // Receive messages from all other ranks
        received_messages_.resize(size_ - 1);
        for (int i = 0; i < size_ - 1; ++i) {
            MPI_Status status;
            MPI_Recv(&received_messages_[i], 1, MPI_INT, 
                    MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            
            log_received_message(status.MPI_SOURCE, received_messages_[i]);
        }

        // Wait for all sends to complete
        MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
    }

    void log_received_message(int source, int message) const {
        std::cout << "Process " << rank_ 
                 << " received value " << message
                 << " from process " << source << "\n";
    }

public:
    AllToAllCommunicator(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    ~AllToAllCommunicator() {
        MPI_Finalize();
    }

    void run() {
        try {
            validate_environment();
            exchange_messages();
        } catch (const std::exception& e) {
            if (rank_ == 0) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
    }
};

int main(int argc, char* argv[]) {
    AllToAllCommunicator communicator(argc, argv);
    communicator.run();
    return 0;
}
