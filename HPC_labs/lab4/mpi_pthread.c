#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <pthread.h>
#include <math.h>

#define THREADNUM 8
#define RANGESTART 40000000
#define RANGEEND 50000000
#define RANGESIZE 500000
#define RESULT 0

typedef struct thread_arguments
{
  int range_start;
  int range_end;
  int rank;
} thargs_t;

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

void *goldbach_range(void *args)
{
  thargs_t arguments = *((thargs_t *)args);
  //int start, int end, int rank;
  int result = 1;
  for (int i = arguments.range_start; i <= arguments.range_end; i++)
  {
    if (goldbach(i, arguments.rank, i == arguments.range_start) == 0)
    {
      result = 0;
    }
  }
  MPI_Send(&result, 1, MPI_INT, arguments.rank, RESULT, MPI_COMM_WORLD);
  //return 1;
}

void printtime(struct timeval *start, struct timeval *stop)
{
  long time = 1000000 * (stop->tv_sec - start->tv_sec) + stop->tv_usec - start->tv_usec;

  printf("\nMPI+pthread execution time=%ld microseconds\n", time);
}

int main(int argc, char **argv)
{

  pthread_t thread[THREADNUM];
  pthread_attr_t attr;
  int startValue[THREADNUM];
  int endValue[THREADNUM];
  double precision = 1000000000;
  int step, threadstep;
  int range[2];
  double pilocal = 0;
  int myrank, proccount;

  double pi_final = 0;
  int i, j;
  int threadsupport;
  void *threadstatus;
  MPI_Status status;
  thargs_t thread_arguments[THREADNUM];

  // Initialize MPI
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadsupport);

  if (threadsupport != MPI_THREAD_MULTIPLE)
  {
    printf("\nThe implementation does not support MPI_THREAD_MULTIPLE, it supports level %d\n", threadsupport);
    MPI_Finalize();
    exit(-1);
  }

  // find out my rank
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  // find out the number of processes in MPI_COMM_WORLD
  MPI_Comm_size(MPI_COMM_WORLD, &proccount);

  // now distribute the required precision
  if (precision < proccount)
  {
    printf("Precision smaller than the number of processes - try again.");
    MPI_Finalize();
    return -1;
  }

  // now start the threads in each process
  // define the thread as joinable
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // initialize the step value
  step = (RANGEEND - RANGESTART) / proccount;
  threadstep = step / THREADNUM;
  range[0] = RANGESTART + myrank * step;
  range[1] = RANGESTART + myrank * step + step;
  for (i = 0; i < THREADNUM; i++)
  {
    // initialize the start value
    thread_arguments[i].range_start = range[0] + i * threadstep;
    thread_arguments[i].range_end = range[0] + (i + 1) * threadstep;
    thread_arguments[i].rank = myrank;

    // launch a thread for calculations
    pthread_create(&thread[i], &attr, goldbach_range,
                   (void *)(&(thread_arguments[i])));
  }

  // receive results from the threads

  double resulttemp;
  for (j = 0; j < THREADNUM; j++)
  {
    MPI_Recv(&resulttemp, 1, MPI_INT, myrank, RESULT, MPI_COMM_WORLD, &status);
    if (!resulttemp)
    {
      printf("Goldbach's conjecture doesn't work");
      MPI_Finalize();
      return -1;
    }
  }

  // now synchronize the threads
  for (i = 0; i < THREADNUM; i++)
    pthread_join(thread[i], &threadstatus);
  /*
       // now merge the numbers to rank 0                                                                                                     
       MPI_Reduce(&pilocal,&pi_final,1, 
       MPI_DOUBLE,MPI_SUM,0, 
       MPI_COMM_WORLD); 
    */

  if (!myrank)
  {
    printf("\n The Goldbach;s hypothesis is true in given range\n");
  }

  // Shut down MPI
  MPI_Finalize();
  return 0;
}
