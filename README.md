Project 1
=========

Performance Analysis

* How does changing the tile and block sizes change performance? Why?

numPlanets = 5000
tile/blockSize: 128   256   512   1024
	   FPS:	1.43  1.41  1.39  1.39

Chaning the tile/block size doesn't seem to have noticeable effects on performance because no matter the block size, the threads were running in parallel,
and since no shared memory or register was utilized, there was no difference in reading from memory either.



* How does changing the number of planets change performance? Why?

numPlanets: 5000   2500   1250   600   300   150
       FPS: 1.43   2.84   5.64   11.6  22.6  43.01

From the data it's clear that the run time is in linear relationship with numPlanets.
The reason is that the outerloop that goes through each planet and accelerates it was 
implemented in parallel and the inner loop that calculates contribution from all other planets wasn't 
in parallel leaving us a O(N) on numPlanets.



* Without running experiments, how would you expect the serial and GPU verions
  of matrix_math to compare?  Why?

I expect the GPU version has O(N) on dimension of the input matrix but O(N^3) for the CPU version because the two 
outer loops in the GPU version was done in parallel.

## SUBMISSION
Please commit your changes to your forked version of the repository and open a
pull request.  Please write your performance analysis in your README.md.
Remember to email Harmony (harmoli+CIS565@seas.upenn.edu) your grade and why.
