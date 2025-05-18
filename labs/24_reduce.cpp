#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <iostream>

int main(int argc, char *argv[]) {
    int    rank, size, N;
    long   local_in_circle = 0, total_in_circle;
    double pi,start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter a number (calculation accuracy): ");
        std::cin >> N;
        start_time = MPI_Wtime(); 
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    srand(rank + 1);
    for (int i = 0; i < N / size; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x * x + y * y <= 1) {
            local_in_circle++;
        }
    }

    MPI_Reduce(&local_in_circle, &total_in_circle, 
               1, MPI_LONG, MPI_SUM, 
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi = 4.0 * total_in_circle / N;
        end_time = MPI_Wtime();

        printf("Calculated Pi: %.8f\n", pi);
        printf("Execution Time: %.6f seconds\n", 
               end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
