#include <mpi.h>
#include <omp.h>
#include <cstdio>

#define MAIN_NODE 0
#define COMMUNICATION_TAG 100

void initializeParallelEnvironment(int* processId, int* processCount) {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, processId);
    MPI_Comm_size(MPI_COMM_WORLD, processCount);
}

void getThreadCount(int* threadCount, int processId) {
    if (processId == MAIN_NODE) {
        std::fprintf(stdout, "Specify parallel worker count: ");
        std::fscanf(stdin, "%d", threadCount);
    }
    MPI_Bcast(threadCount, 1, MPI_INT, MAIN_NODE, MPI_COMM_WORLD);
}

void executeParallelWork(int processId, int processCount, int threadCount) {
    const int totalWorkers = threadCount * processCount;
    
    #pragma omp parallel num_threads(threadCount)
    {
        int localWorkerId = omp_get_thread_num();
        std::printf("Worker %d in process %d (Total workers: %d)\n",
                   localWorkerId, processId, totalWorkers);
    }
}

int main(int argc, char** argv) {
    int currentProcessId, totalProcesses, workerCount;
    
    initializeParallelEnvironment(&currentProcessId, &totalProcesses);
    getThreadCount(&workerCount, currentProcessId);
    executeParallelWork(currentProcessId, totalProcesses, workerCount);
    
    MPI_Finalize();
    return 0;
}