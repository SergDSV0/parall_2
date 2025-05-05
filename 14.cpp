#include <iostream>
#include <mpi.h>

void initializeMPI(int* argc, char*** argv, int* processId, int* totalProcesses) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, processId);
    MPI_Comm_size(MPI_COMM_WORLD, totalProcesses);
}

void displayMessage(int processId, int totalProcesses) {
    std::cout << "Process #" << processId << " out of " << totalProcesses << " is active\n";
}

void finalizeMPI() {
    MPI_Finalize();
}

int main(int argc, char** argv) {
    int myRank, processCount;
    
    initializeMPI(&argc, &argv, &myRank, &processCount);
    displayMessage(myRank, processCount);
    finalizeMPI();
    
    return 0;
}