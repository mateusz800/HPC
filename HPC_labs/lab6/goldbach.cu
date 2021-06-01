
 #include <stdio.h>
 #include <cuda_runtime.h>

 #define RANGESTART 40000000
 #define RANGEEND 50000000
 
__device__
int is_prime(const int p)
{
  for (int i = 3; i <= sqrtf(p); i++)
  {
    if (p % i == 0)
    {
      return 0;
    }
  }
  return 1;
}

__global__
void goldbach(int* result)
{
  int id = blockIdx.x*blockDim.x+threadIdx.x;
  if( (id+RANGESTART)%2 == 1){
      return;
  }
  int i = 2;
  for (int j = id  + RANGESTART - i; j > 2; j--, i++)
  {
    if (is_prime(i) == 1 && is_prime(j) == 1)
    {
      if (id<20)
      {
        printf("[Thread %d] The first sum is %d + %d = %d \n", id, i, j, id + RANGESTART);
      }
      result[id] = 0;
      return;
    }
  }
  result[id] = 1;
}

 int main(void)
 {
     // Error code to check return values for CUDA calls
     cudaError_t err = cudaSuccess;

     int numElements = RANGEEND - RANGESTART;
     size_t size = numElements * sizeof(int);
     size_t primes_size = RANGEEND*sizeof(int);

     int *h_result = (int *)malloc(size);
 
     // Verify that allocations succeeded
     if (h_result == NULL)
     {
         fprintf(stderr, "Failed to allocate host vectors!\n");
         exit(EXIT_FAILURE);
     }
 
     
     // Allocate the device result
     int *d_result = NULL;
     cudaMalloc((void **)&d_result, size);
     cudaMemset(d_result, 0, size);

     err = cudaGetLastError();
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to allocate data (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
   

     // Launch the Kernel
     int threadsPerBlock = 256;
     int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
     printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

     goldbach<<<blocksPerGrid, threadsPerBlock>>>(d_result);
     err = cudaGetLastError();
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     // Copy the device result vector in device memory to the host result vector
     // in host memory.
     printf("Copy output data from the CUDA device to the host memory\n");
     err = cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     // Verify that the result vector is correct
     for (int i = 0; i < numElements; ++i)
     {
         if (h_result[i] == 1)
         {
             fprintf(stderr, "Test failed for number %d!\n", RANGESTART + i);
             exit(EXIT_FAILURE);
         }
     }
     printf("Test PASSED: Goldbach was right\n");
 
     // Free device global memory
     cudaFree(d_result);
     err = cudaGetLastError();
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to free device memory (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
   
     // Free host memory
     free(h_result);
 
     // Reset the device and exit
     err = cudaDeviceReset();
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
     return 0;
 }
 