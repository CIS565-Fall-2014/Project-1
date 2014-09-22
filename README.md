# MICHAEL B LI Project-1
## Notes and performance analysis

### Notes on issues with running the n-body simulator

 * In Moore 100C the code only works on machines with NVS 310s.
 * N_FOR_VIS needs to be reduced drastically from the given value
   of 5000 to get something that runs (with like more than 1 fps).
   I have it set to something low.
 * \_\_device\_\_ needs to be declared on const planetMass at the
   top of kernel.cu.
 * Perhaps there is something wrong with my equation setup for F,
   but the numbers being returned are way too large. This is why
   I added a "/factor" in the code.
 * There are some leftover comments (especially in *accelerate*)
   I decided to keep so that I can understand my thought process
   if I look back at this code later. Obviously this is bad style,
   so if you'd like to have me get rid of those comments on future
   assignments, I can.
 * Perhaps my spotty understanding of the hardware implementation of 
   float operations has resulted in an inefficient implementation for
   calculating stuff in *accelerate*. I will probably want to review
   240/371 material at some point. Until then, though, this is my excuse
   for why my program runs slowly.

### Performance Analysis

#### Effects of block size

Note: it looks like some sort of VSync is turned on, as fps outputs **tend** to be factors of 60.
Perhaps more accurate measurements can be taken with VSync off. However not everything is a factor
of 60 so I'm not really sure what's actually going on.

This can be modified in kernel.h. On an NVS 310, with N_FOR_VIS always 500, with Release configuration:
 * The block size of 128 (baseline) gives an fps of about 15.
 * 1     ->     30    fps
 * 2     ->     30    fps
 * 3     ->     30    fps
 * 4     ->    ~36    fps
 * 8     ->     60    fps
 * 16    ->    ~00.63 fps

Note that there is no \_\_shared\_\_ memory in the project, so the "tile width" consideration on slide
93 of the CUDA introduction 2/2 deck is irrelevant. (Basically, the explanation for how block size
affects fps does not involve "shared memory limitations".)

text, p2
