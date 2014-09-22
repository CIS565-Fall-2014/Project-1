Project 1
=========

# Project 1 : Introduction to CUDA

Eric Lee

## Performance Analysis

* How does changing the tile and block sizes change performance? Why?
Increasing tile and block size should increase performance up to a point. It did not
affect this simulation too much.

* How does changing the number of planets change performance? Why?
5 planets - 60 FPS
200 planets - 60 FPS
2500 planets - 60 FPS
4000 planets - 40 FPS
5000 planets - 30 FPS
10000 planets - 10 FPS
20000 planets - 3 FPS

The more planets there are, the bigger performance hit the simulation takes. Inverse
relation between number of planets and performance. There are parts of the code (accelerate) that are not taking full advantage of the GPU.  

* Without running experiments, how would you expect the serial and GPU verions
  of matrix_math to compare?  Why?
Because we are dealing with such small matrices in this project, the performance
difference is extremely negligible. However, as the matrices scale larger and larger,
we should should a significant increase in performance for the GPU version of 
matrix_math because each element in the resulting matrix can be calculated in
parallel.
