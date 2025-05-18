#include <mpi.h>
#include <sstream>
#include <string>

struct MPIEnvironment {
    int current_rank;
    int total_processes;

    MPIEnvironment(int argc, char* argv[]) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    }

    ~MPIEnvironment() {
        MPI_Finalize();
    }
};

std::string generateProcessMessage(int rank, int total) {
    std::ostringstream stream;
    stream << "[Rank " << rank << "/" << total << "] Reporting for duty!";
    return stream.str();
}

int main(int argc, char* argv[]) {
    MPIEnvironment env(argc, argv);

    auto message = generateProcessMessage(env.current_rank, env.total_processes);
    std::cout << message << std::endl;

    return 0;
}
