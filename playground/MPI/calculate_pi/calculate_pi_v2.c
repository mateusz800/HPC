/**
 * Calculate PI value 
 * PI/4 = 1/1 - 1/3 + 1/5 - 1/7 + ...
 * -----------------------------------------------------------------
 * compile program by command mpicc calculate_pi.c -o calculate_pi
 * run it by mpirun -np <process count> ./calculate_pi
 * -----------------------------------------------------------------
 *  
*/

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    double precision = 1000000000;
    int currentProcessRank;
    int processCount;
    double pi, pi_final = 0;
    double startTime, endTime;
    int sign;
    int denominator;

    MPI_Init(&argc, &argv);

    startTime = MPI_Wtime();

    // find out my rank
    MPI_Comm_rank(MPI_COMM_WORLD, &currentProcessRank);
    /* MPI_COMM_WORLD is default communicator. It groups all processes when the program started.
        It is our context. */

    // find out the number of all processes
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    printf("process count : %d \n", processCount);

    // calculate computation on one process
    pi = 0;
    denominator = currentProcessRank * 2 + 1;
    sign = ((denominator - 1) / 2) % 2 ? -1 : 1;
    while (denominator < precision)
    {
        pi += sign / (double)denominator;
        denominator += 2 * processCount;
        sign = ((denominator - 1) / 2) % 2 ? -1 : 1;
    }
    // merge the values
    MPI_Reduce(&pi, &pi_final, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (currentProcessRank == 0) // if smain process
    {
        pi_final *= 4;
        endTime = MPI_Wtime();
        printf("pi = %f \n", pi_final);
        printf("The process took %f seconds", endTime - startTime);
    }
    MPI_Finalize();
    return 0;
}