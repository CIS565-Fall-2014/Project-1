

* How does changing the tile and block sizes change performance? Why?
  -This is a hard question to answer.  These values are dependent on both
  -the specific architecture on the card, as well as how the problem is being
  -distributed.  One rule that you can use is that block size should generally be
  -a multiple of 32, as the card will round up to a multiple of that anyways.
  -Other than that, often times experimentation is required to find the best
  -configuration.  You do needs to have enough warps to hide latency, though.
  -So to a point, increasing them will increase performance.
* How does changing the number of planets change performance? Why?
  -The lower the number of planets, the faster the code will run.  We're not exactly
  -optimizing the simulation (lots of global memory access for example), so each 
  -additional body adds compute time.
* Without running experiments, how would you expect the serial and GPU verions
  of matrix_math to compare?  Why?
  -The serial versions should be much slower than the GPU versions.  The difference
  -should be much more prevalent the higher the matrix dimensions grow, as well.