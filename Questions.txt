1, The detailed description about the data changes according to the different parameters can be find in the enclosed file named "performance.xls".
a) With Block size equals 128 and I changed the number of particles to see the average fps of the program. The fps don't change much when particles less than 4000. That should be a lock of the system where the highest fps is 60. 
b) When the particles larger than 4000, from 4000 to 4200, the fps drops dramatically from 42 to 28. And from 4200 to 8000, the fps drops linearly with about 200/frame. After 9000, fps drops about 1000/frame.
c) A first guess of the reason is that the particles exceeded the overall assignable threads numbers. But I have not yet verified my guess. One SM can hold 1024 thread and there are 2 on my GT750M card. The overall performance turns when threads exceeded the maximum threads larger than 2048, 4096, 6144, etc.
2, The detailed description about the data changes according to the different parameters can be find in the enclosed file named "performance.xls". 
when the blocksize is larger than 128, fps does not change much. When each blocks contains less threads, there will be more blocks to execute.

3, Without experiments, I think the GPU should be much faster than the CPU version since the GPU version is O(n) in time where the CPU is O(n^3). 