#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>


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

int main(int argc, char** argv){
    int k = 10;
    int rows = 10;
    int cols = 4;

    size_t data_size = cols * rows * sizeof(float);
    size_t centers_size = k * cols * sizeof(float);

    float * h_data = (float*) malloc(data_size); // data[row * cols + col]
    float * h_center = (float *) malloc(centers_size); // center[k*cols + col]
    loadData("minute_weather.csv", 2, cols, rows, h_data);

    float * d_data;
    float * d_center;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_center, centers_size);
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_center, h_center, centers_size, cudaMemcpyHostToDevice);

    // call device function

    // send results to host (row - class)

    cudaFree(d_data);
    free(h_data);

    return 0;
}