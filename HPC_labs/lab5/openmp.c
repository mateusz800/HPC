
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

#define PACKET_COUNT 10000
#define RESOLUTION 10000000


typedef struct {
  void *data;
} t_data;


typedef struct {
  void *result;
} t_result;


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


double dtime()
{
  double tseconds = 0.0;
  struct timeval mytime;
  gettimeofday(&mytime,(struct timezone*)0);
  tseconds = (double)(mytime.tv_sec +mytime.tv_usec*1.0e-6);
  return( tseconds );
}


t_data *partition(t_data inputdata,int partition_count) {}


double f(double x) {
  return sin(x)*sin(x)/x;
}


double one_shot_integration(double start,double end) {
  return ((end-start)*(f(start)+f(end))/2);
}


double adaptive_integration(double start,double end,int k) {
  // first check if we need to go any deeper
  double middle=(start+end)/2;
  double a1,a2,a;
  double r;
  int threadid;
  a=one_shot_integration(start,end);
  a1=one_shot_integration(start,middle);
  a2=one_shot_integration(middle,end);
  if (k<3) {
    if (fabs(a-a1-a2)>0.000000001) { // go deeper
      r=0;
#pragma omp parallel sections firstprivate(start,middle,end,k) reduction(+:r) num_threads(2)
      {
#pragma omp section
	{
	  r=adaptive_integration(start,middle,k+1);
	}
#pragma omp section
	{
	  r=adaptive_integration(middle,end,k+1);
	}
      }
      return r;
    }
  } else {
    if (fabs(a-a1-a2)>0.000000001) { // go deeper
      a1=adaptive_integration(start,middle,k+1);
      a2=adaptive_integration(middle,end,k+1);
      a=a1+a2;
    }
  }
  return a;
}


void process(t_data data,t_result result) {
  // use the adaptive quadrature approach i.e. check whether we need
  // to dive deeper or whether the current resolution if fine
  // this can also be parallelized using OpenMP - later put it into a framework
  // compute an integrate of function f()
  // over range [data.data[0], data.data[1]]
  // do it adaptively
  double pivot;
  double r_a=((double *)data.data)[0];
  double r_b=((double *)data.data)[1];
  double integrate=0;
  double r_length=(r_b-r_a)/RESOLUTION;
  double x=r_a;
  int i;
  double t_start,t_stop;
  t_start=dtime();
  integrate=adaptive_integration(r_a,r_b,0);
  *((double *)result.result)=integrate; // store the result
}


t_result *allocate_results(int *packet_count) {
  int i;
  // prepare space for results
  t_result *r_results=(t_result *)malloc(sizeof(t_result)*PACKET_COUNT);
  if (r_results==NULL) {
    perror("Not enough memory");
    exit(-1);
  }
  for(i=0;i<PACKET_COUNT;i++) {
    r_results[i].result=malloc(sizeof(double));
    if (r_results[i].result==NULL) {
      perror("Not enough memory");
      exit(-1);
    }
  }
  *packet_count=PACKET_COUNT;
  return r_results;
}


t_data *generate_data(int *packet_count) {
  // prepare the input data
  // i.e. the given range is to be divided into packets
  int i;
  double a=40000000,b=5000000;
  double packet_size=(b-a)/PACKET_COUNT;
  t_data *p_packets;
  double *p_data;
  double r_a,r_b;
  // prepare PACKET_COUNT number of packets
  t_data *packets=(t_data *)malloc(sizeof(t_data)*PACKET_COUNT);
  if (packets==NULL) {
    perror("Not enough memory");
    exit(-1);
  }
  r_a=a;
  r_b=a+packet_size;
  p_packets=packets; // pointer to the beginning of the packets
  for(i=0;i<PACKET_COUNT;i++) {
    packets[i].data=malloc(2*sizeof(double));
    if (packets[i].data==NULL) {
      perror("Not enough memory");
      exit(-1);
    }
    // populate the packet with the data
    p_data=(double *)packets[i].data;
    *p_data=r_a;
    *(p_data+1)=r_b;
    r_a+=packet_size;
    r_b+=packet_size;
  }
  *packet_count=PACKET_COUNT;
  return packets;
}


t_data *data;
t_result *results;

main(int argc,char **argv) {
  // prepare the input data
  // i.e. the given range is to be divided into packets
  // now the main processing loop using OpenMP
  int counter;
  int my_data;
  double result;
  int i;
  int packet_count;
  int results_count;
  int threadid;
  int threadclass; // for using various critical sections
  double t_start,t_stop,t_total,t_current,t_min;
  int t_counter=0;
  t_min=100000000;
  do {
    data=generate_data(&packet_count);
    results=allocate_results(&results_count);
    counter=0;
    t_total=0;
    t_start=dtime();
    omp_set_nested(1);
    omp_set_dynamic(0);
#pragma omp parallel private(my_data) firstprivate(threadid,threadclass) shared(counter) num_threads(60)
    {
      threadid=omp_get_thread_num();
      do {
	// each thread will try to get its data from the available list
#pragma omp critical
	{
	  my_data=counter;
	  counter++; // also write result to the counter
	}
	// process and store result -- this can be done without synchronization
	//results[my_data]=
	if (my_data<PACKET_COUNT)
	  process(data[my_data],results[my_data]); // note that processing
	// may take various times for various data packets
      } while (my_data<PACKET_COUNT); // otherwise simply exit because
      // there are no more data packets to process
    }
    t_stop=dtime();
#pragma omp barrier
    // now just add results
    result=0;
    for(i=0;i<PACKET_COUNT;i++) {
      // printf("\nVal[%d]=%.8f",i,*((double *)results[i].result));
      // fflush(stdout);
      result+=*((double *)results[i].result);
    }
    t_counter++;
    t_current=t_stop-t_start;
    if (t_current<t_min)
      t_min=t_current;
  } while ((t_counter<4) && (t_current<100));
  printf("\nFinished");
  printf("\nThe total value of the integrate is %.5f\n",result);
  printf("\nTotal time elapsed=%.8f\n",t_min);
}
