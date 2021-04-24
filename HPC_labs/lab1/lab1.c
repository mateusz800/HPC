#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>

// naive method
bool isPrime(const int p)
{
    for(int i=2;i<p;i++){
        if(p % i == 0){
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    int range_start = 30000000;
    int range_end = 40000000;

    int myrank, proccount;
    unsigned int count, count_final;
    int mine_start, interval_size;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // find out my rank
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // find out the number of processes in MPI_COMM_WORLD
    MPI_Comm_size(MPI_COMM_WORLD, &proccount);


    // each process performs computations on its part
    count = 0;
    interval_size = (range_end - range_start) / proccount;
    mine_start = myrank * interval_size + range_start;
    for (int i = mine_start; i < mine_start + interval_size; i++)
    {
       count += isPrime(i) ? 1:0;
    }
    

    // now merge the numbers to rank 0
    MPI_Reduce(&count, &count_final, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (!myrank)
    {
        printf("count=%d", count_final);
    }

    // Shut down MPI
    MPI_Finalize();

    return 0;
}