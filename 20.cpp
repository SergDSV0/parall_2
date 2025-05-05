#include <mpi.h>
#include <iostream>
#include <string>
#include <stdexcept>

class AsyncMPICommunicator {
    int rank_;
    int size_;
    std::string message_;
    
    void validate_environment() const {
        if (size_ < 2) {
            throw std::runtime_error("At least 2 processes are required");
        }
    }

    void handle_sender() {
        MPI_Request request;
        MPI_Isend(message_.data(), message_.size() + 1, 
                 MPI_CHAR, 1, 0, MPI_COMM_WORLD, &request);
        
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    void handle_receiver() {
        MPI_Request request;
        MPI_Status status;
        char buffer[100];

        MPI_Irecv(buffer, sizeof(buffer), 
                 MPI_CHAR, 0, 0, MPI_COMM_WORLD, &request);

        MPI_Wait(&request, &status);
        display_received_message(buffer);
    }

    void display_received_message(const char* msg) const {
        std::cout << "Process " << rank_ << " received message: '"
                 << msg << "'\n";
    }

public:
    AsyncMPICommunicator(int argc, char* argv[]) 
        : message_("45") {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    ~AsyncMPICommunicator() {
        MPI_Finalize();
    }

    void run() {
        try {
            validate_environment();
            
            if (rank_ == 0) {
                handle_sender();
            } else if (rank_ == 1) {
                handle_receiver();
            }
        } catch (const std::exception& e) {
            if (rank_ == 0) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
    }
};

int main(int argc, char* argv[]) {
    AsyncMPICommunicator communicator(argc, argv);
    communicator.run();
    return 0;
}