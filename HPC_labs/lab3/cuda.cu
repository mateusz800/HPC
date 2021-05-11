#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>



__host__
void printtime(struct timeval *start,struct timeval *stop) {
  long time=1000000*(stop->tv_sec-start->tv_sec)+stop->tv_usec-start->tv_usec;

  printf("\nCUDA execution time=%ld microseconds\n",time);

}

int main(int argc,char **argv) {

    struct timeval start,stop;

    gettimeofday(&start,NULL);

    // run your CUDA kernel(s) here



    // synchronize/finalize your CUDA computations

    gettimeofday(&stop,NULL);

    printtime(&start,&stop);


}
