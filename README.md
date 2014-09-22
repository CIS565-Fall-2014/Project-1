Project 1
=========
#Jianqiao Li
# Project 1 : Introduction to CUDA

## PERFORMANCE ANALYSIS
* How does changing the tile and block sizes change performance? Why?
* How does changing the number of planets change performance? Why?
The following table shows the fps with different number of particles and block size.
--------------------------------------------------------
| particle #|     Block Size=128    |  Block Size=256  | Block Size=512|
|:---------:|:-----------------:|:-----------------:|:-----------:|
|    1000     |         60       |       60       |  60|
|    2000     |         40        |       40       | 40|
|   5000     |         20        |       20        | 20|
------------------------------------------------------------


* Without running experiments, how would you expect the serial and GPU verions
  of matrix_math to compare?  Why?



