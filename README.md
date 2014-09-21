Project 1
=========

* Kai Ninomiya <kainino> (Arch Linux, Intel i5-4xxx, GTX 750)


Part 4
------



*Expected matrix_math GPU/CPU comparison*

For matrix addition and subtraction, I don't expect much
of a performance increase, if any, using the GPU. Since these operations are
linear-time in the size of the matrix _n_, the linear-time memory copy
still dominates.

For matrix multiplication, it reduces the runtime from _O(n^1.5)_ to _O(n)_.
For smaller matrices, it should be possible to do this within one block, and
use shared memory instead of global memory, which may improve performance.
