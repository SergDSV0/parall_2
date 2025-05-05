#include <mpi.h>
#include <iostream>
#include <memory>

class RingCommunicator {
    int world_size;
    int world_rank;
    
    void handle_root() {
        int message = 0;
        
        // Start the ring
        MPI_Send(&message, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        
        // Receive completion
        MPI_Recv(&message, 1, MPI_INT, world_size - 1, 0, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        log_message(message);
    }
    
    void handle_node() {
        int message;
        
        // Receive from predecessor
        MPI_Recv(&message, 1, MPI_INT, world_rank - 1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        log_message(message);
        
        // Process and forward
        message++;
        int target = (world_rank == world_size - 1) ? 0 : world_rank + 1;
        MPI_Send(&message, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
    }
    
    void log_message(int msg) const {
        std::cout << "Processor " << world_rank 
                 << " processed value: " << msg << "\n";
    }
    
public:
    RingCommunicator(int argc, char** argv) {
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    }
    
    ~RingCommunicator() {
        MPI_Finalize();
    }
    
    void run() {
        if (world_rank == 0) {
            handle_root();
        } else {
            handle_node();
        }
    }
};

int main(int argc, char** argv) {
    RingCommunicator comm(argc, argv);
    comm.run();
    return 0;
}