## Problem
 How many prime numbers are there in the range 30.000.000 - 40.000.000

### Answer
There are 575795 prime numbers in that range.


 ### Version 1
 Program uses naive method to determine if the number is prime.

 #### The results
| Number of processes | time [s] |
|:-------------------:|----------|
|4                    |to long   |


 ### Version 2
Determining if the number is prime by dividing it by each next values less or equal square root of the given number.

 #### The results
| Number of processes | time [s] |
|:-------------------:|----------|
|1                    |25.390    |
|2                    |13.698    |
|3                    |9.383     |
|4                    |7.106     |



