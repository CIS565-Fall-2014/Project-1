Project 1
=========
#Jianqiao Li
# Project 1 : Introduction to CUDA

## PERFORMANCE ANALYSIS
* How does changing the tile and block sizes change performance? Why?
* How does changing the number of planets change performance? Why?
The following table shows the fps with different number of particles and block size.
--------------------------------------------------------
| particle #|     Block Size    |  approximate fps  |
|:---------:|:-----------------:|:-----------------:|
|    100     |         128        |       12.77       |
|    500     |         128        |       10.23       |
|   1000     |         128        |       5.67        |
|   5000     |         128       |       1.28        |
|    50     |        yes        |       10.43       |
|    64     |        yes        |       9.56        |
|   128     |        yes        |       5.72        |
|   256     |        yes        |       1.34        |
--------------------------------------------------------


* Without running experiments, how would you expect the serial and GPU verions
  of matrix_math to compare?  Why?



