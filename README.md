Performance Analysis:
-----------------------------------------------------------

N-Body

- Impact of tile/block sizes:
 For constant N-Bodies = 1000
  Block Size  updateF (ms)  updateS (ms)
      32       0.55          0.0055
      64       0.55          0.0055
     128       0.55          0.0055
     256       0.55          0.0058
     512       0.67          0.0068
    1024       0.68          0.0089
Changing the block size makes very little impact, though it looks like too many threads per block starts to run into issues with running out of local memory, leading to performance falling off a cliff at 512 threads per block. There would be more of an impact on the lower side if there were use of shared memory in the code that were impacted with fewer warps that could share.

- Number of Planets:
 For constant Block Size = 128
   N-Bodies   updateF (ms)  updateS (ms)
     500       0.27          0.0055
    1000       0.55          0.0055
    2000       1.10          0.0059
    3000       1.68          0.0065
    4000       2.33          0.0076    
    5000       2.88          0.0076      
The updateF step clearly is O(n), which is good since N-body is usually O(n^2). The parallel implementation got rid of the outer loop, but not the inner loop. The updateS step should be O(1) since all bodies are integrated in parallel, but there is a performance cliff that I am unsure about... perhaps local memory?

Matrix Math

- My implementation lacks optimization using shared memory that was discussed and shown in class.

- For small matrices, I would expect the GPU performance to lag behind the CPU implementation since CPU optimizations like Out of Order execution will help to speed it up and loading memory from the CPU to GPU will take time. It likely will take at least 100+ elements in the matrix before it becomes worth parallelizing.

- 5x5 GPU/CPU Comparison
CPU Performance
 Mat Add  -> 0.0014 ms
 Mat Sub  -> 0.0015 ms
 Mat Mult -> 0.0013 ms
GPU Performance
 Mat Add  -> 0.037 ms
 Mat Sub  -> 0.038 ms
 Mat Mult -> 0.053 ms

- 12x12 GPU/CPU Comparison
CPU Performance
 Mat Add  -> 0.0016 ms
 Mat Sub  -> 0.0014 ms
 Mat Mult -> 0.0014 ms
GPU Performance
 Mat Add  -> 0.082 ms
 Mat Sub  -> 0.035 ms
 Mat Mult -> 0.055 ms