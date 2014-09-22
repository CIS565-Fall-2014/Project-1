Project 1
=========
#Jianqiao Li
# Project 1 : Introduction to CUDA

## PERFORMANCE ANALYSIS
* How does changing the tile and block sizes change performance? Why?
* How does changing the number of planets change performance? Why?
The following table shows the fps with different number of particles and block size.
| particle #|     Block Size    |  approximate fps  |
|:---------:|:-----------------:|:-----------------:|
|    50     |         no        |       12.77       |
|    64     |         no        |       10.23       |
|   128     |         no        |       5.67        |
|   256     |         no        |       1.28        |
|    50     |        yes        |       10.43       |
|    64     |        yes        |       9.56        |
|   128     |        yes        |       5.72        |
|   256     |        yes        |       1.34        |



* Without running experiments, how would you expect the serial and GPU verions
  of matrix_math to compare?  Why?



