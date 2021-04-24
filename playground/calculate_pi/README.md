# Calculating Pi number

Pi value can be calculated from the given formula:

pi/4 = 1/1 - 1/3 + 1/5 - 1/7 + 1/9 - ...

## First version
*calculate_pi_v1.c*

Each process calculate sum of adjacent fractions e.g  
  process 1 calculate : 1/1 -1/3 + 1/5   
  process 2 calculate : -1/7 + 1/9 - 1/11  
  and so on

### The results
| Number of processes | time [s] |
|:-------------------:|----------|
|1                    |3.580916  |
|2                    |3.655705  |
|3                    |3.644040  |
|4                    |3.687655  |


## Second version
*calculate_pi_v2.c*  
Each process sum up every process_count-th fraction e.g.
for two processes:  
process 1: 1/1 + 1/5 + 1/9 + 1/13 + ...  
process 2: -1/3 - 1/7 - 1/11 - ...

### The results
| Number of processes | time [s] |
|:-------------------:|----------|
|1                    |3.605191  |
|2                    |1.811258  |
|3                    |1.248927  |
|4                    |0.940016  |



## Running the program
Compile it using command:
```bash
mpicc calculate_pi_v<version number>.c -o calculate_pi_v<version number>
```
And run it by:
```bash
mpirun -np <process count> ./calculate_pi_v<version number>
```