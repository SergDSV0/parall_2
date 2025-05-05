#include <mpi.h>
#include <omp.h>
#include <unistd.h>

#define MASTER_NODE 0
#define OUTPUT_PREFIX "Process Data"

typedef struct {
    int mpi_id;
    int mpi_total;
    int omp_id;
    int omp_total;
} ParallelInfo;

void display_parallel_info(ParallelInfo* info) {
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    printf("%s | Host: %s | MPI: %d/%d | OMP: %d/%d\n",
           OUTPUT_PREFIX, hostname,
           info->mpi_id, info->mpi_total,
           info->omp_id, info->omp_total);
}

void execute_parallel_region(int mpi_rank, int mpi_size) {
    #pragma omp parallel default(none) shared(mpi_rank, mpi_size)
    {
        ParallelInfo thread_info = {
            .mpi_id = mpi_rank,
            .mpi_total = mpi_size,
            .omp_id = omp_get_thread_num(),
            .omp_total = omp_get_num_threads()
        };
        
        display_parallel_info(&thread_info);
    }
}

void initialize_mpi_environment(int* argc, char*** argv, int* rank, int* size) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
}

int main(int argc, char* argv[]) {
    int process_rank, process_count;
    
    initialize_mpi_environment(&argc, &argv, &process_rank, &process_count);
    
    if (process_rank == MASTER_NODE) {
        printf("Initializing hybrid MPI+OpenMP environment...\n");
    }
    
    execute_parallel_region(process_rank, process_count);
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}