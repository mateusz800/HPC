#include <stdio.h>
#include <cuda_runtime.h>

/*
Application adds two vectors declared in the code
*/

__global__ void vecAdd(int* a, int* b , int* c, int size){
    // calculate thread id
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id < size){
        c[id] = a[id] + b[id];
    }
}

void printVector(int* vec, int size){
    printf("[%d", vec[0]);
    for(int i=1;i<size; i++){
        printf(", %d", vec[i]);
    }
    printf("]\n");
}

int main(int argc, char**argv){
    int size = 5;
    size_t vectorSize = size * sizeof(int);

    //initialize host variables
    int* h_vecA = (int*) malloc(vectorSize);
    int* h_vecB = (int*) malloc(vectorSize);
    int* h_vecResult = (int*) malloc(vectorSize);

    for(int i = 0; i < size; i++){
		h_vecA[i] = i;
		h_vecB[i] = i*i;
	}

    // initialize device variables
    int * d_vecA, *d_vecB, *d_vecResult;
    cudaMalloc(&d_vecA, vectorSize);
    cudaMalloc(&d_vecB, vectorSize);
    cudaMalloc(&d_vecResult, vectorSize);

    cudaMemcpy(d_vecA, h_vecA, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vecB, h_vecB, vectorSize, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(1, 1, 1);
	dim3 threadsPerBlock(size, 1, 1);

    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_vecA, d_vecB, d_vecResult, size);


    // copy the result to the device
    cudaMemcpy(h_vecResult, d_vecResult, vectorSize, cudaMemcpyDeviceToHost);
     

    printf("The result: \n");
    printVector(h_vecResult, size);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess)
    {
        fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
        exit(-1);    
    }
    return 0;
}