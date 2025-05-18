#include <mpi.h>
#include <cstdio>
#include <string>

#define DATA_SIZE 11

int execute_parallel_program(int proc_id, int total_procs, const char* input_msg) {
    char received_data[DATA_SIZE];
    MPI_Comm partitioned_comm;
    
    // Разделение коммуникатора на чётные и нечётные процессы
    int split_key = (proc_id % 2 == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, split_key, proc_id, &partitioned_comm);
    
    // Только чётные процессы участвуют в широковещательной рассылке
    if (proc_id % 2 == 0) {
        MPI_Bcast((void*)received_data, DATA_SIZE, MPI_CHAR, 0, partitioned_comm);
    }
    
    // Получение информации о новом коммуникаторе
    int new_proc_id = -1, new_total_procs = -1;
    if (proc_id % 2 == 0) {
        MPI_Comm_rank(partitioned_comm, &new_proc_id);
        MPI_Comm_size(partitioned_comm, &new_total_procs);
    }
    
    // Вывод информации о процессе
    printf("Global ID: %d/%d | Local group: %s/%s | Data: %s\n",
           proc_id, total_procs,
           (new_proc_id == -1) ? "N/A" : std::to_string(new_proc_id).c_str(),
           (new_total_procs == -1) ? "N/A" : std::to_string(new_total_procs).c_str(),
           (proc_id == 0) ? input_msg : received_data);
    
    // Освобождение ресурсов
    if (proc_id % 2 == 0) {
        MPI_Comm_free(&partitioned_comm);
    }
    
    return 0;
}

int main(int argc, char** argv) {
    int proc_id, total_procs;
    char user_input[DATA_SIZE] = {0};
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &total_procs);
    
    if (proc_id == 0) {
        printf("Input message (max 10 chars): ");
        scanf("%10s", user_input);
    }
    
    execute_parallel_program(proc_id, total_procs, user_input);
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
