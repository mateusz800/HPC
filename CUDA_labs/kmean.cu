#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

__device__ 
float calculateDistance(float* data, float* center, int size){
    // euclidean distance
    float distance = 0;
    for(int i=0;i<size;i++){
        distance += pow(data[i] - center[i], 2);
    }
    return sqrt(distance);
}

__global__ 
void kmean(float* data, int* centers, int rows, int cols, int k, int* result){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id > rows){
        return;
    }
    float distance;
    float min_distance;
    float min_distance_cluster;
    for(int i=0;i<k;i++){
        if(i == 0 ){
            min_distance = calculateDistance(&data[id*cols], &data[centers[i]*cols], cols );
            min_distance_cluster = i;
        } else {
            distance = calculateDistance(&data[id*cols], &data[centers[i]*cols], cols );
            if(distance < min_distance){
                min_distance = distance;
                min_distance_cluster = i;
            }
        }
    }
    //printf("cluster: %d\n", min_distance_cluster);
    result[id] = min_distance_cluster;

}


void loadData(char* file_url, int col_start, int cols, int rows, float* data_ptr){
    FILE* stream = fopen(file_url, "r");
    char line[1024];
    char* clipboard;

    // skip first row
    fgets(line, 1024, stream);

    for(int i=0;i<rows;i++){
        fgets(line, 1024, stream);
        clipboard = strtok( line, "," );
        int j = 0;
        while( clipboard != NULL )
        {
            if(j >= col_start){
                data_ptr[i * cols + j-col_start] = atof(clipboard);
            }
            clipboard = strtok( NULL, "," );
            j++;
            if(j >= col_start + cols){
                break;
            }
        }
    }
}

void saveResultAsCsv(int * data, int rows){
    FILE * file = fopen("result.csv", "w+");
    fprintf(file,"Id, Cluster\n");
    for(int i=0;i<rows;i++){
        fprintf(file, "%d, %d\n", i, data[i]);
    }
    fclose(file);
}


int main(int argc, char** argv){
    int k = 10;
    int rows = 1024*1024;
    int cols = 4;

    size_t data_size = cols * rows * sizeof(float);
    size_t centers_size = k * sizeof(int);
    size_t calc_classes_size = rows * sizeof(int);

    float * h_data = (float*) malloc(data_size); // data[row * cols + col]
    int * h_center_indexes = (int *) malloc(centers_size); // center[k*cols + col]
    int * h_calc_classes = (int*) malloc(calc_classes_size);
    srand(time(NULL)); 
    loadData("minute_weather_shuffled.csv", 2, cols, rows, h_data);
    // initialize clusters centers
    for(int i=0; i < k ; i++){
        h_center_indexes[i] = rand() % rows;
    }

    float * d_data;
    int * d_center_indexes;
    int * d_calc_classes;
    cudaError_t error;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_center_indexes, centers_size);
    cudaMalloc(&d_calc_classes, calc_classes_size);
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_center_indexes, h_center_indexes, centers_size, cudaMemcpyHostToDevice);

    // call device function
    dim3 blocksPerGrid(1024, 1, 1);
	dim3 threadsPerBlock(1024, 1, 1);
    kmean<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_center_indexes, rows, cols, k,  d_calc_classes);

    // send results to host 
    cudaMemcpy(h_calc_classes, d_calc_classes, calc_classes_size, cudaMemcpyDeviceToHost);

    error = cudaGetLastError();
    if(error != cudaSuccess){
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }


    // recalculate clusters

   

    saveResultAsCsv(h_calc_classes, rows);

    cudaFree(d_data);
    free(h_data);

    return 0;
}