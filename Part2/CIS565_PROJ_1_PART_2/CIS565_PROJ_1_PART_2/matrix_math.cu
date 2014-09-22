#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

// Eric Lee
// CIS 565 Project 1

using namespace std;

// Matrices
const int dim = 5;
int memSize = dim * dim * sizeof(float); 
float* matrixA;
float* matrixB;

float* matrixAGPU;
float* matrixBGPU;

float* resultMatrix;
float* resultGPU;

// CUDA
dim3 dimGrid(1, 1);
dim3 dimBlock(dim, dim);

void initMatrices() 
{
	// CPU
	matrixA = new float[dim * dim];
	matrixB = new float[dim * dim];

	for (int i = 0; i < 25; i++) {
		matrixA[i] = i;
		matrixB[i] = i;
	}

	resultMatrix = new float[dim * dim];

	// GPU
	// explicitly reserve memory
	cudaMalloc((void**)&matrixAGPU, memSize); 
	cudaMemcpy(matrixAGPU, matrixA, memSize, cudaMemcpyHostToDevice); 

	cudaMalloc((void**)&matrixBGPU, memSize); 
	cudaMemcpy(matrixBGPU, matrixB, memSize, cudaMemcpyHostToDevice); 

	cudaMalloc((void**)&resultGPU, memSize); 
	cudaMemcpy(resultGPU, resultMatrix, memSize, cudaMemcpyHostToDevice); 
}

void freeMatrices()
{
	delete[] matrixA;
	delete[] matrixB;
	delete[] resultMatrix;

	cudaFree(matrixAGPU);
	cudaFree(matrixBGPU);
	cudaFree(resultGPU);
}

// GPU kernel versions
__global__ void mat_add(float* M, float* N, float* result)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float newValue = M[ty * dim + tx] + N[ty * dim + tx];

	result[ty * dim + tx] = newValue;
}

__global__ void mat_sub(float* M, float* N, float* result)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float newValue = M[ty * dim + tx] - N[ty * dim + tx];

	result[ty * dim + tx] = newValue;
}

__global__ void mat_mult(float* M, float* N, float* result)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float newValue = 0;

	for (int i = 0; i < dim; i++) {
		float Melement = M[ty * dim + i];
		float Nelement = N[i * dim + tx];
		newValue += Melement * Nelement;
	}

	result[ty * dim + tx] = newValue;
}

// CPU serial versions
void mat_add_serial(float* M, float* N, float* result) 
{
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			result[i * dim + j] = M[i * dim + j] + N[i * dim + j]; 
		}
	}
}

void mat_sub_serial(float* M, float* N, float* result) 
{
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			result[i * dim + j] = M[i * dim + j] - N[i * dim + j]; 
		}
	}
}

void mat_mult_serial(float* M, float* N, float* result) 
{
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			float sum = 0;

			for (int k = 0; k < dim; k++) {
				float a = M[i * dim + k];
				float b = N[k * dim + j];
				sum += a * b;
			}
			result[i * dim + j] = sum; 
		}
	}
}

void matrixToString(float* matrix, char* string)
{
	int index = 0;
	
	cout << string << endl;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim; j++) {
			cout << matrix[index] << " ";
			index++;
		}
		cout << endl;
	}
	cout << "----------------" << endl;
	cout << endl;
}

int main()
{
	// initialize 2 5x5 matrices on GPU
	initMatrices();

	matrixToString(matrixA, "Matrix A Initial");
	matrixToString(matrixB, "Matrix B Initial");

	// TESTING
	// Kernel
	mat_add<<<dimGrid, dimBlock>>>(matrixAGPU, matrixBGPU, resultGPU);
	cudaMemcpy(resultMatrix, resultGPU, memSize, cudaMemcpyDeviceToHost); 
	matrixToString(resultMatrix, "Kernel Add");

	mat_sub<<<dimGrid, dimBlock>>>(matrixAGPU, matrixBGPU, resultGPU);
	cudaMemcpy(resultMatrix, resultGPU, memSize, cudaMemcpyDeviceToHost); 
	matrixToString(resultMatrix, "Kernel Sub");

	mat_mult<<<dimGrid, dimBlock>>>(matrixAGPU, matrixBGPU, resultGPU);
	cudaMemcpy(resultMatrix, resultGPU, memSize, cudaMemcpyDeviceToHost); 
	matrixToString(resultMatrix, "Kernel Mult");

	// Serial
	mat_add_serial(matrixA, matrixB, resultMatrix);
	matrixToString(resultMatrix, "Serial Add");

	mat_sub_serial(matrixA, matrixB, resultMatrix);
	matrixToString(resultMatrix, "Serial Sub");

	mat_mult_serial(matrixA, matrixB, resultMatrix);
	matrixToString(resultMatrix, "Serial Mult");

	// prevent window from auto closing
	getchar();

	freeMatrices();

	return 0;
}