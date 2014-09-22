Project 1
=========

# Performance Analysis

## Changing the tile and block sizes
Changing the block sizes in Part 1 did not affect the FPS rate at all; for the range of nbody counts that I tested, they were identical whether the size of blocks and tiles were increased or decreased. This is likely due to the lack of conditional statements so that all but one of the blocks is being fully utilized, regardless of their size.

## Changing the number of planets
Increasing the number of planets up to about 6000 nbodies did not decrease the FPS, but after that the FPS dropped off.
* 0 - 6000 nbodies -> 60 FPS
* 7000 -> 25 FPS
* 8000 -> 15 FPS
* 9000 -> 14 FPS
* 10000 -> 12 FPS
<br>This is because the number of cycles needed to process all the threads was greater than the latency that the number of warps could hide.

## Serial vs GPU verions of matrix_math
Because the size of the matrices was so small, operations on the GPU would be slower than on the CPU because of the latency associated with copying the data into device memory. The gain from parallel processing would be lost to that, so serial would actually be faster. (Timers in the matrix_math file did confirm this; CPU times were about 2 to 2.5 times faster than GPU times.)
