#include <mpi.h>
#include <vector>
#include <string>
#include <iostream>

class MPIHandler {
    int rank_;
    int size_;

public:
    MPIHandler(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    ~MPIHandler() { MPI_Finalize(); }

    int rank() const { return rank_; }
    int size() const { return size_; }
};

std::string getProcessMessage(int rank, int size) {
    if (rank == 0) {
        return "Total processes: " + std::to_string(size);
    } else if (rank % 2 == 0) {
        return "Process " + std::to_string(rank) + " (EVEN) reporting!";
    } else {
        return "Process " + std::to_string(rank) + " (ODD) active!";
    }
}

int main(int argc, char* argv[]) {
    MPIHandler mpi(argc, argv);

    auto message = getProcessMessage(mpi.rank(), mpi.size());
    std::cout << message << std::endl;

    return 0;
}