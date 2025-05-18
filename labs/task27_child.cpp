#include <mpi.h>
#include <cstdio>
#include <unistd.h>

#define ERROR_MSG "Parent process connection failed\n"
#define SUCCESS_MSG "Process %d of %d initialized (Parent ID: %d)\n"

int initialize_process_hierarchy(MPI_Comm* main_comm) {
    MPI_Comm progenitor;
    MPI_Comm_get_parent(&progenitor);
    
    if (progenitor == MPI_COMM_NULL) {
        fprintf(stderr, ERROR_MSG);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return -1;
    }
    
    *main_comm = progenitor;
    return 0;
}

void display_process_info(int current_id, int total_processes, int ancestor_id) {
    printf(SUCCESS_MSG, current_id, total_processes, ancestor_id);
}

int main(int argc, char** argv) {
    MPI_Comm root_communicator;
    int process_id, total_processes;
    const int root_process = 0;
    
    MPI_Init(&argc, &argv);
    
    if (initialize_process_hierarchy(&root_communicator) != 0) {
        return EXIT_FAILURE;
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
    
    display_process_info(process_id, total_processes, root_process);
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
