
Project 1 PERFORMANCE ANALYSIS
=========
----------------------------------------------------------

The following table shows the fps with different number of planets and block size.


| planets #|     Block Size=128    |  Block Size=256  | Block Size=512|
|:---------:|:-----------------:|:-----------------:|:-----------:|
|    1000     |         60       |       60       |  60|
|    2000     |         40        |       40       | 40|
|   5000     |         20        |       20        | 20|


* Changing the block size doesn't change the performance. I think this is because we are not using shared memory. Thus changing the number of blocks doesn't affect the performance.
* Changing the number of planets affects the performance a lot. This makes sense because our for loop in the device function will run the N times, where N is the number of planets. 

----------------------------------------------------------------
* Without running experiments, I would expect mat_add funciton and mat_sub function running on GPU will be much faster than CPU version. Because the add and subtracte operations are totally parellel. Although matrix multiply operation is also parellel, it has to spend a lot of time to read memory. Thus the running time could be similar for GPU and CPU version of mat_mult. 



