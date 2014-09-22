/* 
 * MICHAEL B LI
 * CIS 565 Project-1
 * 
 * "Hello world" made possible with support from http://www.brainlings.com/2011/11/hello-world-in-cuda/
 * 
 * 
 */

#include <iostream>

#define MAT_WIDTH 5

// forward declare CPU helper method to display results
void printResults(float*, char*);
// and the serial versions of stuff
void cpu_mat_add (float*, float*, float*);
void cpu_mat_sub (float*, float*, float*);
void cpu_mat_mult (float*, float*, float*);

// kernels to run on GPU
__global__ void mat_add(float* Md, float* Nd, float* Pd) {
	int tx = threadIdx.x; // COLUMN counter: incr 1 -> add 1
	int ty = threadIdx.y; // ROW counter:    incr 1 -> add 5 (or whatever MAT_WIDTH is)

	int index = ty * MAT_WIDTH + tx;

	Pd[index] = Md[index] + Nd[index];
}

__global__ void mat_sub(float* Md, float* Nd, float* Pd) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int index = ty * MAT_WIDTH + tx;

	Pd[index] = Md[index] - Nd[index];
}

__global__ void mat_mult(float* Md, float* Nd, float* Pd) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float runningSum = 0.f; // to be used for calculation of the dot product

	// M: move along a row, increment COLUMN counter
	// N: move along a column, increment ROW counter
	for (int k = 0; k < MAT_WIDTH; k++) {
		runningSum += Md[ty*MAT_WIDTH + k] * Nd[tx + k*MAT_WIDTH];
	}

	Pd[ty * MAT_WIDTH + tx] = runningSum;
}


int main(int argc, char** argv) {

	//std::cout << "Hello there!" << std::endl;

	//std::cin.ignore();

	//return 0;

	//#################################################################################


	// CPU float array
	// I will just be using a 1D array to hold the floats.
	// MAT_WIDTH is defined to be 5 above; change it to test with something else. All matrices assumed to be SQUARE.

	int size = MAT_WIDTH * MAT_WIDTH * sizeof(float); // how much memory needed

	// initialize 2 matrices on the CPU
	float* M = new float[MAT_WIDTH*MAT_WIDTH]();
	float* N = new float[MAT_WIDTH*MAT_WIDTH]();
	for (int i = 0; i < MAT_WIDTH*MAT_WIDTH; i++) {
		M[i] = float(i);
		N[i] = float(i);
	}

	// allocate space for output on CPU
	float* P = (float*) malloc(size);


	// use cudaMalloc to allocate on the GPU

	// pointers for device memory
	float* Md;
	float* Nd;
	float* Pd;
	

	cudaMalloc((void**)&Md, size);
	cudaMalloc((void**)&Nd, size);
	cudaMalloc((void**)&Pd, size);

	// transfer M and N onto the GPU's Md and Nd

	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

	// invoke the kernel HERE
	// NOTE: using global memory only (cudaMalloc?), since using __shared__ was not specified in the directions
	// and this project is already difficult enough such that I'd rather not give myself extra work.
	// Also the matrices are really small, so the optimization would add extra overhead.
	// I could indeed look through the "CUDA Part 2" slides and copy over code optimized with tiles & shared memory
	// but I'd rather not for this exercise. I will not be giving myself any bonus points for effort.

	dim3 dimBlock(MAT_WIDTH, MAT_WIDTH);
	dim3 dimGrid(1,1);


	// print results. use the same P for each kernel - print before running the next operation
	for (int i = 0; i < 3; i++) {
		char* whichOp;
		switch (i) {
		case 0:
			whichOp = "ADD";
			mat_add<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
			break;
		case 1:
			whichOp = "SUBTRACT";
			mat_sub<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
			break;
		case 2:
			whichOp = "MULTIPLY";
			mat_mult<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
			break;
		default:
			//do nothing, this won't happen, just look at the for loop
			break;
		}


		// copy back onto CPU from GPU to let me print results to console
		cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
		printResults(P, whichOp);
	}

	// free device matrices
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);

	// junk I'll keep in case I want it later

	//for (int i = 0; i < 26; i++) {
	//	std::cout << M[i] << std::endl;
	//}

	// print out serial results
	cpu_mat_add(P, M, N);
	printResults(P, "SERIAL ADD");
	cpu_mat_sub(P, M, N);
	printResults(P, "SERIAL SUBTRACT");
	cpu_mat_mult(P, M, N);
	printResults(P, "SERIAL MULTIPLY");


	std::cin.ignore();
	return 0;
}

// just prints results
void printResults (float* R, char* message) {
	std::cout << message << std::endl;

	for (int i = 0; i < MAT_WIDTH; i++) {
		for (int j = 0; j < MAT_WIDTH; j++) {
			int index = i * MAT_WIDTH + j;
			std::cout << R[index] << '\t';
		}
		std::cout << std::endl;
	}
}

//serial versions. code is basically the same as the GPU versions, with extra nested for loops
void cpu_mat_add (float* R, float* M, float* N) {
	for (int i = 0; i < MAT_WIDTH; i++) {
		for (int j = 0; j < MAT_WIDTH; j++) {
			int index = i*MAT_WIDTH + j;
			R[index] = M[index] + N[index];
		}
	}
}

void cpu_mat_sub (float* R, float* M, float* N) {
	for (int i = 0; i < MAT_WIDTH; i++) {
		for (int j = 0; j < MAT_WIDTH; j++) {
			int index = i*MAT_WIDTH + j;
			R[index] = M[index] - N[index];
		}
	}
}

void cpu_mat_mult (float* R, float* M, float* N) {
	for (int i = 0; i < MAT_WIDTH; i++) {
		for (int j = 0; j < MAT_WIDTH; j++) {
			int index = i*MAT_WIDTH + j;
			
			float runningSum = 0.f;
			
			for (int k = 0; k < MAT_WIDTH; k++) {
				runningSum += M[i*MAT_WIDTH + k] * N[k*MAT_WIDTH + j];
			}
			
			R[index] = runningSum;
		}
	}
}
