Project 1
=========

# Project 1 : Introduction to CUDA


## PART 4 : PERFORMANCE ANALYSIS
Since this is the first project, we will guide you with some example
questions.  In future projects, please answer at least these questions, as
they go through basic performance analysis.  Please go above and beyond the
questions we suggest and explore how different aspects of your code impact
performance as a whole. 

We have provided a frame counter as a metric, but feel free to add cudaTimers,
etc. to do more fine-grained benchmarking of various aspects. 

NOTE : Performance should be measured in comparison to a baseline.  Be sure to
describe the changes you make between experiments and how you are benchmarking.

* How does changing the tile and block sizes change performance? Why?
The number of tile and block sizes depend on the program and graphic cards. We cannot set block size too big also not too small. But the number of threads per block must be a multiple of 32 which is the warp size. Also each SM unit on the GPU must have enough active warps to sufficiently hide all latency.


* How does changing the number of planets change performance? Why?
The more planets are there, the slower the program will run. Because to get the position, velocity and acceleration of each planet we will need to calculate the previous value and the delta value. All of them will based on the other planets. So the more planets we have in the program, the more complicated calculation it will be and it slows down the program.



* Without running experiments, how would you expect the serial and GPU verions
  of matrix_math to compare?  Why?
  
  I believe the serial version will be much slower than the matrix math on GPU. Since the matrix math requires simple calculation many times which can be operated on the threads from GPU. CPU has limited threads compared to GPU so the serial version will take much longer time than GPU matrix math.
  
  


## SUBMISSION
Please commit your changes to your forked version of the repository and open a
pull request.  Please write your performance analysis in your README.md.
Remember to email Harmony (harmoli+CIS565@seas.upenn.edu) your grade and why.
