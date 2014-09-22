# MICHAEL B LI Project-1
## Notes and performance analysis

### Notes on issues with running the n-body simulator

 * In Moore 100C the code only works on machines with NVS 310s.
 * N_FOR_VIS needs to be reduced drastically from the given value of 5000 to get something that runs (with like more than 1 fps). I have it set to 100.
 * \_\_device\_\_ needs to be declared on const planetMass at the top of kernel.cu.
 * Perhaps there is something wrong with my equation setup for F, but the numbers being returned are way too large. This is why I added a "/factor" in the code.
 * There are some leftover comments (especially in *accelerate*) I decided to keep so that I can understand my thought process if I look back at this code later. Obviously this is bad style, so if you'd like to have me get rid of those comments on future assignments, I can.

### Performance Analysis

text for performance analysis, p1

text, p2
