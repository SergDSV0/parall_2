#include <iostream>
#include <string>
#include <mpi.h>

class MessagePasser {
    int current_rank;
    
public:
    MessagePasser(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
    }
    
    ~MessagePasser() {
        MPI_Finalize();
    }
    
    void run() {
        if (is_sender()) {
            transmit_data();
        } else if (is_receiver()) {
            receive_and_display();
        }
    }

private:
    bool is_sender() const { return current_rank == 0; }
    bool is_receiver() const { return current_rank == 1; }

    void transmit_data() {
        const std::string data = "45";
        MPI_Send(data.c_str(), data.size() + 1, 
                MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    }

    void receive_and_display() {
        char buffer[100];
        MPI_Recv(buffer, sizeof(buffer), 
                MPI_CHAR, 0, 0, MPI_COMM_WORLD, 
                MPI_STATUS_IGNORE);
        std::cout << "Received transmission: '" 
                 << buffer 
                 << "'" 
                 << std::endl;
    }
};

int main(int argc, char* argv[]) {
    MessagePasser messenger(argc, argv);
    messenger.run();
    return 0;
}