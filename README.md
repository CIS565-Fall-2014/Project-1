## PART 4 : PERFORMANCE ANALYSIS

* How does changing the tile and block sizes change performance? Why?

Changing the tile and block sizes may increase the speed if we can utilize the shared memory efficiently. Smaller the block size, smaller the shared memory threads can have. If threads frequently access global memory instead of shared memory, then the performance go down. 
For Nbody problem, we actually calculated forces between two paired body twice. If we can store them into shared memory, then the performance can be doubled. 

* How does changing the number of planets change performance? Why?

As we can see from the result, there is only 1.5 FPS for 5000 bodies, and 60FPS for 100 bodies. Performance decreses in O(N2) if the number of planets increases. The main reason is the accelerate() function has a for loop which integrates all forces from other bodies serially.To increase the performance, we can try to make the additions also in parallell.

* Without running experiments, how would you expect the serial and GPU verions
  of matrix_math to compare?  Why?

GPU version of matrix_math runs faster than serial computation. Since the computation for each element in the matrix is independent, all computations can be done parallely, which means those computations can be done simultaneously rather than one-by-one in CPU. 
