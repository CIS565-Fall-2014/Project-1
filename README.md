
Project 1 PERFORMANCE ANALYSIS
=========
----------------------------------------------------------

The following table shows the fps with different number of planets and block size.

--------------------------------------------------------
| planets #|     Block Size=128    |  Block Size=256  | Block Size=512|
|:---------:|:-----------------:|:-----------------:|:-----------:|
|    1000     |         60       |       60       |  60|
|    2000     |         40        |       40       | 40|
|   5000     |         20        |       20        | 20|
------------------------------------------------------------
* Changing the block size doesn't change the performance. I think this is because we are not using shared memory. Thus changing the number of blocks doesn't affect the performance.
* Changing the number of planets affects the performance a lot. This makes sense because our for loop in the device function will run the N times, where N is the number of planets. 
* Without running experiments, how would you expect the serial and GPU verions
  of matrix_math to compare?  Why?



