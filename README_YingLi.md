1. How does changing the tile and block sizes change performance? Why?
Increase the number of blocks and the threads per block will improve the performance; because more threads computes at the same time.
For N-Body simulation, if the block size is set to 10, fps is near 37. If it is set to 50 or larger, fps is near 60 and the each frame runs more smoothly.

2. How does changing the number of planets change performance? Why?
Larger number of planets, longer each frame runs; because we have to compute the force between every pair of planets.

3. Without running experiments, how would you expect the serial and GPU verions of matrix_math to compare?  Why?
The computing cost of matrix addition and subtration on CPU is O(width *width). The GPU cost is O(width) if number of threads is no less than the matrix size.