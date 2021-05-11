#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

#define RANGESTART 40000000
#define RANGEEND 50000000
#define RANGESIZE 500000
#define DATA 0
#define RESULT 1
#define FINISH 2

int is_prime(const int p)
{
    for (int i = 3; i <= sqrt(p); i++)
    {
        if (p % i == 0)
        {
            return 0;
        }
    }
    return 1;
}

int goldbach(int number, int rank)
{
    int i = 2;

    for (int j = number - i; j > 2; j--, i++)
    {
        if (is_prime(i) == 1 && is_prime(j) == 1)
        {
            printf("[Rank %d] The first sum is %d + %d = %d \n", rank, i, j, number);
            return 1;
        }
    }
    return 0;
}

int call_goldbach_range(int start, int end, int rank)
{
    int result;
    for (int i = start; i <= end; i++)
    {
        if (goldbach(i, rank) == 0)
        {
            printf("i = %d", i);
            return 0;
        }
    }
    return 1;
}

void printtime(struct timeval *start, struct timeval *stop)
{
    long time = 1000000 * (stop->tv_sec - start->tv_sec) + stop->tv_usec - start->tv_usec;
    printf("\nMPI execution time=%ld microseconds\n", time);
}

int main(int argc, char **argv)
{

    struct timeval start, stop;
    int myrank, nproc;
    MPI_Status status;
    MPI_Request request;


    int range[2]; // each process range
    int result;   // 0 if Goldbach's conjecture failed

    MPI_Init(&argc, &argv);
    // obtain my rank
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // and the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (!myrank)
        gettimeofday(&start, NULL);

    // run your computations here (including MPI communication)
    if (nproc < 2)
    {
        printf("Run with at least 2 processes");
        MPI_Finalize();
        return -1;
    }
    if (((RANGEEND - RANGESTART) / RANGESIZE) < 2 * (nproc - 1))
    {
        printf("More subranges needed");
        MPI_Finalize();
        return -1;
    }

    // now the master will distribute the data and slave processes will perform computations
    if (myrank == 0)
    {
        range[0] = RANGESTART;
        // first distribute some ranges to all slaves
        for (int i = 1; i < nproc; i++)
        {
            range[1] = range[0] + RANGESIZE;
            // send it to process i
            MPI_Isend(range, 2, MPI_INT, i, DATA, MPI_COMM_WORLD, &request);
            range[0] = range[1];
        }
        do
        {
            // distribute remaining subranges to the processes which have completed their parts
            MPI_Irecv(&result, 1, MPI_INT, MPI_ANY_SOURCE, RESULT,
                      MPI_COMM_WORLD, &request);
            // in meantime calculate new range
            range[1] = range[0] + RANGESIZE;
            if (range[1] > RANGEEND)
                range[1] = RANGEEND;
            MPI_Wait(&request, &status);
            if (result == 0) // Golbach's conjecture doesn't work
            {
                printf("Golbach's conjecture failed");
                MPI_Finalize();
                return -1;
            }
            MPI_Isend(range, 2, MPI_INT, status.MPI_SOURCE, DATA,
                      MPI_COMM_WORLD, &request);
            range[0] = range[1];
        } while (range[1] < range_end);

        // now receive results from the processes
        for (int i = 0; i < (nproc - 1); i++)
        {
            MPI_Irecv(&result, 1, MPI_INT, MPI_ANY_SOURCE, RESULT,
                      MPI_COMM_WORLD, &request);
            if (result == 0)
            {
                printf("Golbach's conjecture failed");
                MPI_Finalize();
                return -1;
            }
        }
        // shut down the slaves
        for (int i = 1; i < nproc; i++)
        {
            MPI_Isend(NULL, 0, MPI_INT, i, FINISH, MPI_COMM_WORLD, &request);
        }
    }
    else // slave
    {
        do
        {
            // blocking function
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == DATA)
            {
                // the same as using blocking function
                MPI_Irecv(range, 2, MPI_INT, 0, DATA, MPI_COMM_WORLD, &request);
                // wait for message
                MPI_Wait(&request, &status);

                // compute my part
                result = call_goldbach_range(range[0], range[1], myrank);
                // send the result back
                MPI_Isend(&result, 1, MPI_INT, 0, RESULT,
                          MPI_COMM_WORLD, &request);
            }
        } while (status.MPI_TAG != FINISH);
    }

    // synchronize/finalize your computations

    if (myrank == 0)
    {
        MPI_Wait(&request, &status);
        gettimeofday(&stop, NULL);
        printtime(&start, &stop);
    }

    MPI_Finalize();
}
