Project 1
=========

# Project 1 : Introduction to CUDA

## NOTE :
This project (and all other projects in this course) requires a NVIDIA graphics
card with CUDA capabilityi!  Any card with compute capability 2.0 and up will
work.  This means any card from the GeForce 400 and 500 series and afterwards
will work.  If you do not have a machine with these specs, feel free to use
computers in the SIG Lab.  All computers in SIG lab and Moore 100 C have CUDA 
capable cards and should already have the CUDA SDK installed. 

## PART 1 : INSTALL NSIGHT
To help with benchmarking and performance analysis, we will be using NVIDIA's
profiling and debugging tool named NSight. Download and install it from the
following link for whichever IDE you will be using:
http://www.nvidia.com/object/nsight.html. 

NOTE : If you are using Linux / Mac, most of the screenshots and class usage of
NSight will be in Visual Studio.  You are free to use to the Eclipse version
NSight during these in class labs, but we will not be able to help you as much.

## PART 2 : NBODY SIMULATION
To get you used to using CUDA kernels, we will be writing a simple 2D nbody 
simulator.  The following source files are included in the project:

* main.cpp : sets up graphics stuff for visualization
* kernel.cu : this contains the CUDA kernel calls

All the code that you will need to modify is in kernel.cu and is marked clearly
by TODOs.

## PART 3 : MATRIX MATH
In this portion we will walk you through setting up a project that writes some
simple matrix math functions. Please put this portion in a folder marked Part2
in your repository. 

### Step 1 : Create your project.
Using the instructions on the Google forum, please set up a new Visual Studio project that
compiles using CUDA. For uniformity, please write your main function and all
your code in a file named matrix_math.cu.

### Step 2 : Setting up CUDA memory.
As we discussed in class, there is host memory and device memory.  Host memory
is the memory that exists on the CPU, whereas device memory is memory on the
GPU.  

In order to create/reserve memory on the GPU, we need to explicitly do so
using cudaMalloc.  By calling cudaMalloc, we are calling malloc on the GPU to
reserve a portion of its memory.  Like malloc, cudaMalloc simply mallocs a
portion of memory and returns a pointer. This memory is only accessible on the
device unless we explicitly copy memory from the GPU to the CPU.  The reverse is
also true.  

We can copy memory to and from the GPU using the function cudaMemcpy. Like the
POSIX C memcpy, you will need to specify the size of memory you are copying.
The last argument is used to specify the direction of the copy (from GPU to CPU
or the other way around).

Please initialize 2 5 x 5 matrices represented as an array of floats on the CPU
and the GPU where each of the entry is equal to its position (i.e. A_00 = 0,
A_01 = 1, A_44 = 24). 

### Step 3 : Creating CUDA kernels. 
In the previous part, we explicitly divided the CUDA kernels from the rest of
the file for stylistic purposes.  Since there will be far less code in this
project, we will write the global and device functions in the same file as the
main function.

Given a matrix A and matrix B (both represented as arrays of floats), please
write the following functions :
* mat_add : A + B
* mat_sub : A - B
* mat_mult : A * B

You may assume for all matrices that the dimensions of A and B are the same and
that they are square.

Use the 2 5 x 5 matrices to test your code either by printing directly to the
console or writing an assert.

THINGS TO REMEMBER :
* global and device functions only have access to memory that is explicitly on
  the device, meaning you MUST copy memory from the CPU to the GPU if you would
  like to use it there
* The triple triangle braces "<<<" begin and end the global function call.  This
  provides parameters with which CUDA uses to set up tile size, block size and
  threads for each warp.
* Do not fret if Intellisense does not understand CUDA keywords (if they have
  red squiggly lines underneath CUDA keywords).  There is a way to integrate
  CUDA syntax highlighting into Visual Studio, but it is not the default.

### Step 4 : Write a serial version.
For comparison, write a single-threaded CPU version of mat_add, mat_sub and
mat_mult. We will not introduce timing elements in this project, but please
keep them in mind as the upcoming lab will introduce more on this topic. 

## PART 4 : PERFORMANCE ANALYSIS
Since this is the first project, we will guide you with some example
questions.  In future projects, please answer at least these questions, as
they go through basic performance analysis.  Please go above and beyond the
questions we suggest and explore how different aspects of your code impact
performance as a whole. 

We have provided a frame counter as a metric, but feel free to add cudaTimers,
etc. to do more fine-grained benchmarking of various aspects. 

NOTE : Performance should be measured in comparison to a baseline.  Be sure to
describe the changes you make between experiments and how you are benchmarking.

* How does changing the tile and block sizes change performance? Why?
* How does changing the number of planets change performance? Why?
* Without running experiments, how would you expect the serial and GPU verions
  of matrix_math to compare?  Why?

## SUBMISSION
Please commit your changes to your forked version of the repository and open a
pull request.  Please write your performance analysis in your README.md.
Remember to email Harmony (harmoli+CIS565@seas.upenn.edu) your grade and why.
