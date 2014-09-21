Project 1
=========

* Kai Ninomiya <kainino> (Arch Linux, Intel i5-4xxx, GTX 750)


Part 4
------

### N-body performance analysis

* How does changing the tile and block sizes change performance? Why?

* How does changing the number of planets change performance? Why?

![N-Body performance by block size (using minimum necessary block count)](perftests.png)

TODO

### Performance predictions: matrix_math

* Without running experiments, how would you expect the serial and GPU verions
of matrix_math to compare? Why?

For matrix addition and subtraction, I don't expect much
of a performance increase, if any, using the GPU. Since these operations are
linear-time in the size of the matrix _n_, the linear-time memory copy
still dominates.

For matrix multiplication, it reduces the runtime from _O(n^1.5)_ to _O(n)_.
For smaller matrices, it should be possible to do this within one block, and
use shared memory instead of global memory, which may improve performance.
