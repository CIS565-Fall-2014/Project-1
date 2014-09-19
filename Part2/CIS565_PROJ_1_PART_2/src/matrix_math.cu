#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h> 
#include <assert.h>
#include <iostream>
using std::cout;

const int width = 5; 
int size = width * width * sizeof(float); 

dim3 threadsPerBlock(16, 16);
dim3 numBlocks(1); 

float * Ad; 
float * Bd;

float * Cdadd; 
float * Cdsub; 
float * Cdmult; 

float * A; 
float * B; 

float * Cadd; 
float * Csub; 
float * Cmult; 


// initialize matrices A and B
void initMat() 
{
	int index = 0; 
	for (int r = 0; r < width; r++) 
	{
		for (int c = 0; c < width; c++) 
		{
			A[r*width+c] = index; 
			B[r*width+c] = index; 
			index++; 
		}
	}
}

// matrix addition kernel
__global__ void mat_add_kernel(float* M, float* N, float* P, int width) 
{
	// 2D thread ID 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 

	P[ty*width+tx] = M[ty*width+tx] + N[ty*width+tx]; 
}

// matrix subtraction kernel
__global__ void mat_sub_kernel(float* M, float* N, float* P, int width) 
{
	// 2D thread ID 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 

	P[ty*width+tx] = M[ty*width+tx] - N[ty*width+tx]; 
}

// matrix multiplication kernel
__global__ void mat_mult_kernel(float* M, float* N, float* P, int width) 
{
	// 2D thread ID 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 


	float sum = 0.0f; 
	for (int i = 0; i < width; i++) 
	{
		sum += (M[ty*width+i]*N[i*width+tx]); 
	}
	P[ty*width+tx] = sum; 
}


// serial versions of mat_add, mat_sub, and mat_mult  

void mat_add_serial(float* M, float* N, float* P) 
{
	for (int r = 0; r < width; r++) 
	{
		for (int c = 0; c < width; c++) 
		{
			P[r*width+c] = M[r*width+c] + N[r*width+c]; 
		}
	}
}

void mat_sub_serial(float* M, float* N, float* P) 
{
	for (int r = 0; r < width; r++) 
	{
		for (int c = 0; c < width; c++) 
		{
			P[r*width+c] = M[r*width+c] - N[r*width+c]; 
		}
	}
}

void mat_mult_serial(float* M, float* N, float* P) 
{
	for (int r = 0; r < width; r++) 
	{
		for (int c = 0; c < width; c++) 
		{
			float sum = 0.0f; 
			for (int i = 0; i < width; i++) 
			{
				sum += (M[r*width+i]*N[i*width+c]); 
			}
			P[r*width+c] = sum; 
		}
	}
}

int main() 
{
	// initialize matrices
	A = new float[width*width];  
	B = new float[width*width];  
	Cadd = new float[width*width];  
	Csub = new float[width*width];  
	Cmult = new float[width*width];  

	// populate matrices A and B
	initMat(); 

	// set up memory and data transfer 
	cudaMalloc((void**)&Ad, size); 
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice); 

	cudaMalloc((void**)&Bd, size); 
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice); 

	cudaMalloc((void**)&Cdadd, size); 
	cudaMalloc((void**)&Cdsub, size); 
	cudaMalloc((void**)&Cdmult, size); 

	// add, subtract, and multiply matrices
	mat_add_kernel<<<numBlocks, threadsPerBlock>>>(Ad, Bd, Cdadd, width);
	cudaMemcpy(Cadd, Cdadd, size, cudaMemcpyDeviceToHost); 
	std::cout<<Cadd[0];std::cout<<" "; std::cout<<Cadd[1];std::cout<<" "; std::cout<<Cadd[2];std::cout<<" "; std::cout<<Cadd[3];std::cout<<" "; std::cout<<Cadd[4];std::cout<<"\n"; 
	std::cout<<Cadd[5];std::cout<<" "; std::cout<<Cadd[6];std::cout<<" "; std::cout<<Cadd[7];std::cout<<" "; std::cout<<Cadd[8];std::cout<<" "; std::cout<<Cadd[9];std::cout<<"\n"; 
	std::cout<<Cadd[10];std::cout<<" "; std::cout<<Cadd[11];std::cout<<" "; std::cout<<Cadd[12];std::cout<<" "; std::cout<<Cadd[13];std::cout<<" "; std::cout<<Cadd[14];std::cout<<"\n"; 
	std::cout<<Cadd[15];std::cout<<" "; std::cout<<Cadd[16];std::cout<<" "; std::cout<<Cadd[17];std::cout<<" "; std::cout<<Cadd[18];std::cout<<" "; std::cout<<Cadd[19];std::cout<<"\n"; 
	std::cout<<Cadd[20];std::cout<<" "; std::cout<<Cadd[21];std::cout<<" "; std::cout<<Cadd[22];std::cout<<" "; std::cout<<Cadd[23];std::cout<<" "; std::cout<<Cadd[24];std::cout<<"\n"; 

	mat_sub_kernel<<<numBlocks, threadsPerBlock>>>(Ad, Bd, Cdsub, width);
	cudaMemcpy(Csub, Cdsub, size, cudaMemcpyDeviceToHost); 
	std::cout<<Csub[0];std::cout<<" "; std::cout<<Csub[1];std::cout<<" "; std::cout<<Csub[2];std::cout<<" "; std::cout<<Csub[3];std::cout<<" "; std::cout<<Csub[4];std::cout<<"\n"; 
	std::cout<<Csub[5];std::cout<<" "; std::cout<<Csub[6];std::cout<<" "; std::cout<<Csub[7];std::cout<<" "; std::cout<<Csub[8];std::cout<<" "; std::cout<<Csub[9];std::cout<<"\n"; 
	std::cout<<Csub[10];std::cout<<" "; std::cout<<Csub[11];std::cout<<" "; std::cout<<Csub[12];std::cout<<" "; std::cout<<Csub[13];std::cout<<" "; std::cout<<Csub[14];std::cout<<"\n"; 
	std::cout<<Csub[15];std::cout<<" "; std::cout<<Csub[16];std::cout<<" "; std::cout<<Csub[17];std::cout<<" "; std::cout<<Csub[18];std::cout<<" "; std::cout<<Csub[19];std::cout<<"\n"; 
	std::cout<<Csub[20];std::cout<<" "; std::cout<<Csub[21];std::cout<<" "; std::cout<<Csub[22];std::cout<<" "; std::cout<<Csub[23];std::cout<<" "; std::cout<<Csub[24];std::cout<<"\n"; 

	mat_mult_kernel<<<numBlocks, threadsPerBlock>>>(Ad, Bd, Cdmult, width);
	cudaMemcpy(Cmult, Cdmult, size, cudaMemcpyDeviceToHost); 
	std::cout<<Cmult[0];std::cout<<" "; std::cout<<Cmult[1];std::cout<<" "; std::cout<<Cmult[2];std::cout<<" "; std::cout<<Cmult[3];std::cout<<" "; std::cout<<Cmult[4];std::cout<<"\n"; 
	std::cout<<Cmult[5];std::cout<<" "; std::cout<<Cmult[6];std::cout<<" "; std::cout<<Cmult[7];std::cout<<" "; std::cout<<Cmult[8];std::cout<<" "; std::cout<<Cmult[9];std::cout<<"\n"; 
	std::cout<<Cmult[10];std::cout<<" "; std::cout<<Cmult[11];std::cout<<" "; std::cout<<Cmult[12];std::cout<<" "; std::cout<<Cmult[13];std::cout<<" "; std::cout<<Cmult[14];std::cout<<"\n"; 
	std::cout<<Cmult[15];std::cout<<" "; std::cout<<Cmult[16];std::cout<<" "; std::cout<<Cmult[17];std::cout<<" "; std::cout<<Cmult[18];std::cout<<" "; std::cout<<Cmult[19];std::cout<<"\n"; 
	std::cout<<Cmult[20];std::cout<<" "; std::cout<<Cmult[21];std::cout<<" "; std::cout<<Cmult[22];std::cout<<" "; std::cout<<Cmult[23];std::cout<<" "; std::cout<<Cmult[24];std::cout<<"\n"; 

	// free suda memory 
	cudaFree(Ad); 
	cudaFree(Bd); 
	cudaFree(Cdadd); 
	cudaFree(Cdsub); 
	cudaFree(Cdmult); 

	// serial add, subtract, multiply 
	mat_add_serial(A, B, Cadd); 
	mat_sub_serial(A, B, Csub); 
	mat_mult_serial(A, B, Cmult); 

	std::cout<<Cadd[0];std::cout<<" "; std::cout<<Cadd[1];std::cout<<" "; std::cout<<Cadd[2];std::cout<<" "; std::cout<<Cadd[3];std::cout<<" "; std::cout<<Cadd[4];std::cout<<"\n"; 
	std::cout<<Cadd[5];std::cout<<" "; std::cout<<Cadd[6];std::cout<<" "; std::cout<<Cadd[7];std::cout<<" "; std::cout<<Cadd[8];std::cout<<" "; std::cout<<Cadd[9];std::cout<<"\n"; 
	std::cout<<Cadd[10];std::cout<<" "; std::cout<<Cadd[11];std::cout<<" "; std::cout<<Cadd[12];std::cout<<" "; std::cout<<Cadd[13];std::cout<<" "; std::cout<<Cadd[14];std::cout<<"\n"; 
	std::cout<<Cadd[15];std::cout<<" "; std::cout<<Cadd[16];std::cout<<" "; std::cout<<Cadd[17];std::cout<<" "; std::cout<<Cadd[18];std::cout<<" "; std::cout<<Cadd[19];std::cout<<"\n"; 
	std::cout<<Cadd[20];std::cout<<" "; std::cout<<Cadd[21];std::cout<<" "; std::cout<<Cadd[22];std::cout<<" "; std::cout<<Cadd[23];std::cout<<" "; std::cout<<Cadd[24];std::cout<<"\n"; 

	std::cout<<Csub[0];std::cout<<" "; std::cout<<Csub[1];std::cout<<" "; std::cout<<Csub[2];std::cout<<" "; std::cout<<Csub[3];std::cout<<" "; std::cout<<Csub[4];std::cout<<"\n"; 
	std::cout<<Csub[5];std::cout<<" "; std::cout<<Csub[6];std::cout<<" "; std::cout<<Csub[7];std::cout<<" "; std::cout<<Csub[8];std::cout<<" "; std::cout<<Csub[9];std::cout<<"\n"; 
	std::cout<<Csub[10];std::cout<<" "; std::cout<<Csub[11];std::cout<<" "; std::cout<<Csub[12];std::cout<<" "; std::cout<<Csub[13];std::cout<<" "; std::cout<<Csub[14];std::cout<<"\n"; 
	std::cout<<Csub[15];std::cout<<" "; std::cout<<Csub[16];std::cout<<" "; std::cout<<Csub[17];std::cout<<" "; std::cout<<Csub[18];std::cout<<" "; std::cout<<Csub[19];std::cout<<"\n"; 
	std::cout<<Csub[20];std::cout<<" "; std::cout<<Csub[21];std::cout<<" "; std::cout<<Csub[22];std::cout<<" "; std::cout<<Csub[23];std::cout<<" "; std::cout<<Csub[24];std::cout<<"\n"; 

	std::cout<<Cmult[0];std::cout<<" "; std::cout<<Cmult[1];std::cout<<" "; std::cout<<Cmult[2];std::cout<<" "; std::cout<<Cmult[3];std::cout<<" "; std::cout<<Cmult[4];std::cout<<"\n"; 
	std::cout<<Cmult[5];std::cout<<" "; std::cout<<Cmult[6];std::cout<<" "; std::cout<<Cmult[7];std::cout<<" "; std::cout<<Cmult[8];std::cout<<" "; std::cout<<Cmult[9];std::cout<<"\n"; 
	std::cout<<Cmult[10];std::cout<<" "; std::cout<<Cmult[11];std::cout<<" "; std::cout<<Cmult[12];std::cout<<" "; std::cout<<Cmult[13];std::cout<<" "; std::cout<<Cmult[14];std::cout<<"\n"; 
	std::cout<<Cmult[15];std::cout<<" "; std::cout<<Cmult[16];std::cout<<" "; std::cout<<Cmult[17];std::cout<<" "; std::cout<<Cmult[18];std::cout<<" "; std::cout<<Cmult[19];std::cout<<"\n"; 
	std::cout<<Cmult[20];std::cout<<" "; std::cout<<Cmult[21];std::cout<<" "; std::cout<<Cmult[22];std::cout<<" "; std::cout<<Cmult[23];std::cout<<" "; std::cout<<Cmult[24];std::cout<<"\n"; 


	return 0;
}