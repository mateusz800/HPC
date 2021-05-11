#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>


void printtime(struct timeval *start,struct timeval *stop) {
  long time=1000000*(stop->tv_sec-start->tv_sec)+stop->tv_usec-start->tv_usec;

  printf("\nMPI+OpenMP execution time=%ld microseconds\n",time);

}

int main(int argc,char **argv) {

  struct timeval start,stop;
  int threadsupport;
  int myrank,nproc;

  // Initialize MPI with desired support for multithreading -- state your desired support level

  MPI_Init_thread(&argc, &argv,MPI_THREAD_FUNNELED,&threadsupport); 

  if (threadsupport<MPI_THREAD_FUNNELED) {
    printf("\nThe implementation does not support MPI_THREAD_FUNNELED, it supports level %d\n",threadsupport);
    MPI_Finalize();
    return -1;
  }
  
  // obtain my rank
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  // and the number of processes
  MPI_Comm_size(MPI_COMM_WORLD,&nproc);

  if (!myrank)
    gettimeofday(&start,NULL);
  
  // run your computations here (including MPI communication and OpenMP stuff)



  // synchronize/finalize your computations

  if (!myrank) {
    gettimeofday(&stop,NULL); 
    printtime(&start,&stop);
  }
    
  MPI_Finalize();
  
}
