Project 1 Writeup
=========

Performance Analysis

1. Changing the tile/block sizes
	Comparing the differences between block sizes of powers of two indicates that there is an
	optimal size (128) and going up or down decreases the performance.  Increasing the block
	size should allow more threads to share memory and decrease the downtime from loading the
	data, but reduces the number of memory fetches that can be performed at once?  I'm not sure.

2. Changing the number of planets
	Increasing the number of planets decreases performance, while decreasing the number of
	planets increases performance.  There are a few causes for this.  First of all, increasing
	the number of planets increases the number of calculations per thread (by a linear factor).
	Second, the number of threads increases (also by a linear factor) because more planets need
	to be processed (I don't know if I have the terms right here).

3. Comparing serial/GPU versions of matrix math
	The GPU version should take far, far less time than the serial version for large matrices.
	With the 5x5 array that we have, the difference is small.  However, the GPU version's add
	and subtract should scale linearly with the width, while a serial calculation scales with
	width^2.  The multiply should scale both by another factor of the width.

4. Improving the performance of nbody by not using glm
	I noticed while writing the nbody simulation code that glm::normalize() was significantly
	reducing the performance.  I had 3 calls to the method, and it took my down to 2fps.
	With a manual version (vector/width), I got 30 fps instead.  In addition, replacing my
	calls to glm::length() with sqrt(x^2+y^2+z^2) improved it to 60 fps.  My guess is that
	it is simply faster to do things locally if possible, since it doesn't need to use
	external code.