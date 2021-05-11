#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>


void printtime(struct timeval *start,struct timeval *stop) {
  long time=1000000*(stop->tv_sec-start->tv_sec)+stop->tv_usec-start->tv_usec;

  printf("\nOpenMP execution time=%ld microseconds\n",time);

}

int main(int argc,char **argv) {

  struct timeval start,stop;


  gettimeofday(&start,NULL);
  
  // run your computations here (including OpenMP stuff)



  
  // synchronize/finalize your computations

  gettimeofday(&stop,NULL);

  printtime(&start,&stop);

}
