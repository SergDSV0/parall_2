#include <mpi.h>
#include <cstdio>
#include <vector>

#define DEFAULT_ROOT 0

void spawn_child_processes(int proc_count, MPI_Comm* external_comm) {
    const char* child_binary = "./bin/child_process.bin";
    MPI_Comm_spawn(child_binary, MPI_ARGV_NULL, proc_count, 
                  MPI_INFO_NULL, DEFAULT_ROOT, MPI_COMM_WORLD,
                  external_comm, MPI_ERRCODES_IGNORE);
}

void display_process_hierarchy(int current_rank, int total_procs) {
    printf("Process hierarchy:\nCurrent rank: %d\nTotal processes: %d\nParent: root\n",
           current_rank, total_procs);
}

int main(int argc, char** argv) {
    MPI_Comm child_comm;
    int current_rank, total_procs, child_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);

    if (current_rank == DEFAULT_ROOT) {
        fprintf(stdout, "Specify child process count: ");
        fscanf(stdin, "%d", &child_count);
        
        spawn_child_processes(child_count, &child_comm);
        display_process_hierarchy(current_rank, total_procs + child_count);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
