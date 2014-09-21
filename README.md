Performance Analysis:
I used: N = 1000, blockSize = 128, without share memory, fps: 9.3 to be my benchmark. 
I found that the blockSize under 128 will decrease the performance dramatically.
With the same N, when I set the blockSize to 64, the fps will be 6.7. 
When blockSize is set to 32, the fps will be 3.3. On the other side, increase the blockSize to 256 or 512 doesn't change the frame rate.
In addition, if the blockSize can't be divided by 128, it will also decrease the performance. I ever set the blockSize to 129, and the fps will be 7.5.
I thought that is because my ALU could handle 128 threads simultaneously. 
Therefore, the blockSize which can't be divided by 128 would waste some calculation power and increase unnecessary computation loading.  
Generally, I think that under the restriction of register number and the memory limitation, the higher the blockSize is, the better performace of our application is.

On the other hand, add the share memory seems doesn't increase the fps. 
I am not so sure about the reason, but I will guess that the cache of the global memory is big enough to store our data and the share memory doesn't provide any benefit.

The most influential part is N, the number of planets. When N is 5000, my fps is only 1.9. 
The reason is obvious, the computation loading decrease from 5000x5000 to 1000x1000 and that is 25 times less.


When comparing to serial and papallel matrix_math problems, I believe that the performance of parallel computation will perform much better when matrix size become larger.
Because every element inside the result matrix could invoke a kernel to compute simultaneously instead of one by one computing.
In serial computation, every element has to wait until the previous element finishes computing.
However, I don't think our 5x5 matrix will have performance difference between serial and parallel computation.
The matrix is too small, the parallel computation can't provide much benefit here. 
Besides, we have to copy the memory from CPU to GPU to compute, and this process will take some time. 
That's why I don't think the GPU computation here is faster than CPU computation.



