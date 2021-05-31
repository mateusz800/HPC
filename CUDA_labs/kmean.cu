#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

__device__ 
double calculateDistance(double* data, double* center, int size){
    // euclidean distance
    double distance = 0;
    for(int i=0;i<size;i++){
        distance += pow(data[i] - center[i], 2);
    }
    return sqrt(distance);
}

__global__ 
void kmean(int offset, double* data, double* centers, int rows, int cols, int k, int* result, int* class_count){
    int id = blockIdx.x*blockDim.x+threadIdx.x +offset;
    if(id >= rows){
        return;
    }
    double distance;
    double min_distance;
    int min_distance_cluster;
    for(int i=0;i<k;i++){
        if(i == 0 ){
            min_distance = calculateDistance(&data[id*cols], &centers[i*cols], cols );
            min_distance_cluster = i;
        } else {
            distance = calculateDistance(&data[id*cols], &centers[i*cols], cols );
            if(distance < min_distance){
                min_distance = distance;
                min_distance_cluster = i;
            }
        }
    }
    //printf("cluster: %d\n", min_distance_cluster);
    result[id] = min_distance_cluster;
    atomicAdd(&class_count[min_distance_cluster], 1) ;

}


void loadData(char* file_url, int col_start, int cols, int rows, double* data_ptr){
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
    fprintf(file,"Id,Cluster\n");
    for(int i=0;i<rows;i++){
        fprintf(file, "%d, %d\n", i, data[i]);
    }
    fclose(file);
}
void saveClusterCentersAsCsv(double * data, int k, int cols){
    FILE * file = fopen("centers.csv", "w+");
    //fprintf(file,"Cluster id, Cluster\n");
    for(int i=0;i<k;i++){
        for(int j=0;j<cols;j++){
            if(j==cols-1){
                fprintf(file, "%f\n", data[i*cols+j]);
            } else {
                fprintf(file, "%f,", data[i*cols+j]);
            }
        }
    }
    fclose(file);
}


int main(int argc, char** argv){
    int k = 12;
     // weather - cols:6
    // iris - cols:2, rows:150, k: 3
    int rows = 1586822;
    int cols = 6;
    int steps = 30;
    int offset = 0;
    char* file_url = "wind_data_prepared.csv";

    size_t data_size = cols * rows * sizeof(double);
    size_t centers_size = k * cols * sizeof(double);
    size_t calc_classes_size = rows * sizeof(int);
    size_t class_count_size = k * sizeof(int);

    double * h_data = (double*) malloc(data_size); // data[row * cols + col]
    double * h_centers = (double *) malloc(centers_size); // center[k*cols + col]
    int * h_calc_classes = (int*) malloc(calc_classes_size);
    int * h_class_count = (int*) malloc(class_count_size);

   
    // load arguments
    if(argc == 5){
        file_url = argv[1];
        rows = atoi(argv[2]);
        cols = atoi(argv[3]);
        k = atoi(argv[4]);
    } else {
        printf("Continue with default parameters \n");
    }
        

    srand(time(NULL)); 
    loadData(file_url, 1, cols, rows, h_data);
    // initialize clusters centers
    for(int i=0; i < k ; i++){
        int data_index = (rand() % rows) *cols;
        for(int j=0;j<cols;j++){
            h_centers[i*cols +j] = h_data[data_index + j];
        }
       
    }

    double * d_data;
    double * d_centers;
    int * d_calc_classes;
    int * d_class_count;
    cudaError_t error;
    dim3 blocksPerGrid(1024, 1, 1);
	dim3 threadsPerBlock(1024, 1, 1);
    int max_rows = blocksPerGrid.x * threadsPerBlock.x;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_centers, centers_size);
    cudaMalloc(&d_calc_classes, calc_classes_size);
    cudaMalloc(&d_class_count, class_count_size);
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

    for(int step=0;step<steps;step++){
        cudaMemcpy(d_centers, h_centers, centers_size, cudaMemcpyHostToDevice);
        cudaMemset(d_class_count, 0, class_count_size);

        for(offset=0;offset<rows;offset+=max_rows){
            // call device function
            kmean<<<blocksPerGrid, threadsPerBlock>>>(offset, d_data, d_centers, rows, cols, k,  d_calc_classes, d_class_count);

            if(offset+max_rows < rows){
                calc_classes_size = max_rows * sizeof(int);
            } else{
                calc_classes_size = (rows - offset) * sizeof(int);
            }
         
            // send results to host 
            cudaMemcpy(h_calc_classes + offset , d_calc_classes+offset, calc_classes_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_class_count, d_class_count, class_count_size, cudaMemcpyDeviceToHost);
            
            // check if error occured
            error = cudaGetLastError();
            if(error != cudaSuccess){
                fprintf(stderr, "Error: %s\n", cudaGetErrorString(error));
                exit(-1);
            }
        }
       
        // recalculate clusters (mean value)
        if(step != steps-1){
            // reset cluster centers
            for(int i=0;i<k;i++){
                for(int j=0;j<cols;j++){
                    h_centers[i*cols+j] = 0;
                }
            }
            for(int i=0;i<rows;i++){
                // sum all values
                for(int j=0;j<cols;j++){
                    h_centers[h_calc_classes[i] * cols + j] += h_data[i*cols + j];
                }
     
               
            }
            for(int i=0;i<k;i++){
                for(int j=0;j<cols;j++){
                    if(h_class_count[i] != 0){
                        h_centers[i*cols + j] /= h_class_count[i];
                    }
                }
            }
        }
    }

    saveResultAsCsv(h_calc_classes, rows);
    saveClusterCentersAsCsv(h_centers, k, cols);

    cudaFree(d_data);
    free(h_data);

    return 0;
}