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

int goldbach(int number, int rank, int first)
{
    int i = 2;
    for (int j = number - i; j > 2; j--, i++)
    {
        if (is_prime(i) == 1 && is_prime(j) == 1)
        {
            if (first)
            {
                printf("[Rank %d] The first sum is %d + %d = %d \n", rank, i, j, number);
            }
            return 1;
        }
    }
    return 0;
}

int goldbach_range(int start, int end, int rank)
{
    for (int i = start; i <= end; i++)
    {
        if (goldbach(i, rank, i == start) == 0)
        {
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
    MPI_Request *requests;
    int requestcount = 0;
    int requestcompleted;
    int myrank, proccount;
    int *ranges;
    int range[2];
    int *resulttemp;
    int sentcount = 0;
    int recvcount = 0;
    int i;
    MPI_Status status;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // find out my rank
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // find out the number of processes in MPI_COMM_WORLD
    MPI_Comm_size(MPI_COMM_WORLD, &proccount);

    if (proccount < 2)
    {
        printf("Run with at least 2 processes");
        MPI_Finalize();
        return -1;
    }

    if (((RANGEEND - RANGESTART) / RANGESIZE) < 2 * (proccount - 1))
    {
        printf("More subranges needed");
        MPI_Finalize();
        return -1;
    }

    resulttemp = (int *)malloc((proccount - 1) * sizeof(int));

    // now the master will distribute the data and slave processes will perform computations
    if (myrank == 0)
    {
        // requests to receive
        // requests to send
        // requests send to end
        requests = (MPI_Request *)malloc(3 * (proccount - 1) * sizeof(MPI_Request));

        if (!requests)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }
        // master send range start and end. After that it send again next range
        ranges = (int *)malloc(4 * (proccount - 1) * sizeof(int));
        if (!ranges)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }

        range[0] = RANGESTART;

        // first distribute some ranges to all slaves
        for (i = 1; i < proccount; i++)
        {
            range[1] = range[0] + RANGESIZE;
            // send it to process i
            // Nonblocking becouse it is beginning of processing - slaves do nothing except waiting for message
            MPI_Send(range, 2, MPI_INT, i, DATA, MPI_COMM_WORLD);
            sentcount++;
            range[0] = range[1];
        }

        // the first proccount requests will be for receiving, the latter ones for sending
        for (i = 0; i < 2 * (proccount - 1); i++)
            requests[i] = MPI_REQUEST_NULL; // none active at this point

        // start receiving for results from the slaves
        for (i = 1; i < proccount; i++)
            MPI_Irecv(&(resulttemp[i - 1]), 1, MPI_INT, i, RESULT,
                      MPI_COMM_WORLD, &(requests[i - 1]));

        // start sending new data parts to the slaves
        for (i = 1; i < proccount; i++)
        {
            range[1] = range[0] + RANGESIZE;
            ranges[2 * i - 2] = range[0];
            ranges[2 * i - 1] = range[1];

            // send it to process i
            MPI_Isend(&(ranges[2 * i - 2]), 2, MPI_INT, i, DATA,
                      MPI_COMM_WORLD, &(requests[proccount - 2 + i]));

            sentcount++;
            range[0] = range[1];
        }
        while (range[1] < RANGEEND)
        {
            // wait for completion of any of the requests
            MPI_Waitany(2 * proccount - 2, requests, &requestcompleted,
                        MPI_STATUS_IGNORE);

            // if it is a result then send new data to the process
            // and add the result
            if (requestcompleted < (proccount - 1))
            {
                if (!resulttemp[requestcompleted])
                {
                    printf("Goldbach's conjecture doesn't work\n");
                    MPI_Finalize();
                    return -1;
                }
                recvcount++;

                // first check if the send has terminated
                MPI_Wait(&(requests[proccount - 1 + requestcompleted]),
                         MPI_STATUS_IGNORE);

                // now send some new data portion to this process
                range[1] = range[0] + RANGESIZE;

                if (range[1] > RANGEEND)
                    range[1] = RANGEEND;
                ranges[2 * requestcompleted] = range[0];
                ranges[2 * requestcompleted + 1] = range[1];
                MPI_Isend(&(ranges[2 * requestcompleted]), 2, MPI_INT,
                          requestcompleted + 1, DATA, MPI_COMM_WORLD,
                          &(requests[proccount - 1 + requestcompleted]));
                sentcount++;
                range[0] = range[1];

                // now issue a corresponding recv
                MPI_Irecv(&(resulttemp[requestcompleted]), 1,
                          MPI_INT, requestcompleted + 1, RESULT,
                          MPI_COMM_WORLD,
                          &(requests[requestcompleted]));
            }
        }
        // now send the FINISHING ranges to the slaves
        // shut down the slaves
        range[0] = range[1];
        for (i = 1; i < proccount; i++)
        {

            ranges[2 * i - 4 + 2 * proccount] = range[0];
            ranges[2 * i - 3 + 2 * proccount] = range[1];
            MPI_Isend(range, 2, MPI_INT, i, DATA, MPI_COMM_WORLD,
                      &(requests[2 * proccount - 3 + i]));
        }

        // now receive results from the processes - that is finalize the pending requests
        MPI_Waitall(3 * proccount - 3, requests, MPI_STATUSES_IGNORE);

        for (i = 0; i < (proccount - 1); i++)
        {
            if (!resulttemp[i])
            {
                printf("Goldbach's conjecture doesn't work");
                MPI_Finalize();
                return -1;
            }
        }
        // now receive results for the initial sends
        for (i = 0; i < (proccount - 1); i++)
        {
            MPI_Recv(&(resulttemp[i]), 1, MPI_INT, i + 1, RESULT,
                     MPI_COMM_WORLD, &status);
            if (!resulttemp[i])
            {
                printf("Goldbach's conjecture doesn't work");
                MPI_Finalize();
                return -1;
            }
            recvcount++;
        }
        // now display the result
        printf("\n The Goldbach;s hypothesis is true in given range\n");
    }
    else //slave
    {
        requests = (MPI_Request *)malloc(2 * sizeof(MPI_Request));

        if (!requests)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }

        requests[0] = requests[1] = MPI_REQUEST_NULL;
        ranges = (int *)malloc(2 * sizeof(int));

        if (!ranges)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }
        // result from two ranges
        resulttemp = (int *)malloc(2 * sizeof(int));

        if (!resulttemp)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }

        // first receive the initial data
        MPI_Recv(range, 2, MPI_INT, 0, DATA, MPI_COMM_WORLD, &status);
        while (range[0] < range[1])
        {
            // if there is some data to process
            // before computing the next part start receiving a new data part
            MPI_Irecv(ranges, 2, MPI_INT, 0, DATA, MPI_COMM_WORLD,
                      &(requests[0]));

            // compute my part
            resulttemp[1] = goldbach_range(range[0], range[1], myrank);
            //printf(" result %d, waiting for all\n", resulttemp[0]);
            // now finish receiving the new part
            // and finish sending the previous results back to the master
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
            //printf("After wait\n");
            range[0] = ranges[0];
            range[1] = ranges[1];
            resulttemp[0] = resulttemp[1];

            // and start sending the results back
            MPI_Isend(&resulttemp[0], 1, MPI_INT, 0, RESULT,
                      MPI_COMM_WORLD, &(requests[1]));

            // now finish sending the last results to the master
            MPI_Wait(&(requests[1]), MPI_STATUS_IGNORE);
        }
    }
    // Shut down MPI
    MPI_Finalize();
    return 0;
}