For part 2, I change the acc in sendToPBO() as vec3(0,0,0), as the adapter will crash if I don't make this change.

## PART 4 : PERFORMANCE ANALYSIS
1.How does changing the tile and block sizes change performance? Why?
Test when setting particle number as 5000
block sizes  estimated average FPS
  128              5.01
  256              4.88
  512              4.92
  1024             4.60

It seems that block sizes don't have much influence on the performance. 

Why: I think the global memory has long latency and thus most of the computing time for each frame is spent on writing/reading global memory. Therefore,
the block sizes don't have much influence on the performance. 
  

2.How does changing the number of planets change performance? Why?
Test when set block sizes as 128
particles  estimated average FPS
  100              60.00
  500              60.00
  1000             60.00
  5000              5.02
  10000             1.31

The performance is good when particle number is low and become slower when I set larger particle number. 

Why: When the particle number is low, the memory latency is not shown as we have enough threads to hide it. But when we have large particle number,
(when the latency can't be hidden even we use all threads) the latency will increase with the increase of global memory.

3.Without running experiments, how would you expect the serial and GPU verions of matrix_math to compare? Why?
I think the serial and GPU verions of matrix_math will both perform well when the size of matrix is small. But when the size of matrix is large, the GPU version 
wroks much faster than the serial version.

Why:The GPU version is multi-threads, while the serial version is single thread. And when matrix size is large, GPU version is faster as it has fewer loops.

