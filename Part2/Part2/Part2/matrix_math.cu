#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <conio.h>
#include <ctime>
#include <Windows.h>


using namespace std;

// Prints a square matrix with side length of size.
void print(float* mat, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			cout << mat[i*size + j] << " ";
		cout << endl;
	}
	cout << endl;
}

bool equals(float* A, float* B, int size)
{
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			if (A[i * 5 + j] - B[i * 5 + j])
				return false;
		}
	}
	return true;
}

__global__ void mat_add(float* A, float* B, float* C, int len)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  C[ty * len + tx] = A[ty * len + tx] + B[ty * len + tx];
}

__global__ void mat_sub(float* A, float* B, float* D, int len)
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  D[ty * len + tx] = A[ty * len + tx] - B[ty * len + tx];
}

__global__ void mat_mult(float* A, float* B, float* M, int len)
{
  __shared__ float Ads[5][5];
  __shared__ float Bds[5][5];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int r = by * len + ty; int c = bx * len + tx;

  float val = 0;
  Ads[ty][tx] = A[r * len + tx];
  Bds[ty][tx] = B[c + len * ty];
  __syncthreads();
  for (int i = 0; i < len; i++)
    val += Ads[ty][i] * Bds[i][tx];
  __syncthreads();

  M[r * len + c] = val;
}

int main(int argc, char** argv)
{
	float* A = new float[25];
	float* B = new float[25];
	dim3 dimBlock(5, 5);
	dim3 dimGrid(1, 1);

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			A[i * 5 + j] = i * 5 + j;
			B[i * 5 + j] = i * 5 + j;
		}
	}

	LARGE_INTEGER begin, end;
	LARGE_INTEGER frequency;
	QueryPerformanceFrequency(&frequency);

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Parallel Matrix Add
	float* parC = new float[25];
	float *Ad, *Bd, *Cd;
	int size = 5 * 5 * sizeof(float);
	cudaMalloc(&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	cudaMalloc(&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
	cudaMalloc(&Cd, size);

	cudaEventRecord(start, 0);
	mat_add<<<dimGrid, dimBlock>>>(Ad, Bd, Cd, 5);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(parC, Cd, size, cudaMemcpyDeviceToHost);
	cudaFree(Cd);
	cudaEventElapsedTime(&time, start, stop);
	printf("Kernel: %f ms\t", time);

	// Serial Matrix Add
	QueryPerformanceCounter(&begin);
	float* serC = new float[25];
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			serC[i * 5 + j] = A[i * 5 + j] + B[i * 5 + j];
		}
	}
	QueryPerformanceCounter(&end);
	printf("CPU: %f ms\n", (end.QuadPart - begin.QuadPart) * 1000.0 / frequency.QuadPart);

	if (equals(parC, serC, 5)) {
		cout << "Correct matrix add" << endl;
		print(serC, 5);
	} else {
		cout << "Error in matrix add" << endl;
		print(parC, 5);
		print(serC, 5);
	}
	_getch();

	// Parallel Matrix Sub
	float* parD = new float[25];
	float* Dd;
	cudaMalloc(&Dd, size);

	cudaEventRecord(start, 0);
	mat_sub<<<dimGrid, dimBlock>>>(Ad, Bd, Dd, 5);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(parD, Dd, size, cudaMemcpyDeviceToHost);
	cudaFree(Dd);
	cudaEventElapsedTime(&time, start, stop);
	printf("Kernel: %f ms\t", time);

	// Serial Matrix Sub
	QueryPerformanceCounter(&begin);;
	float* serD = new float[25];
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			serD[i * 5 + j] = A[i * 5 + j] - B[i * 5 + j];
		}
	}
	QueryPerformanceCounter(&end);
	printf("CPU: %f ms\n", (end.QuadPart - begin.QuadPart) * 1000.0 / frequency.QuadPart);

	if (equals(parD, serD, 5)) {
		cout << "Correct matrix sub" << endl;
		print(serD, 5);
	} else {
		cout << "Error in matrix sub" << endl;
		print(parD, 5);
		print(serD, 5);
	}
	_getch();

	// Parallel Matrix Mult
	float* parE = new float[25];
	float* Ed;
	cudaMalloc(&Ed, size);

	cudaEventRecord(start, 0);
	mat_mult<<<dimGrid, dimBlock>>>(Ad, Bd, Ed, 5);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	cudaMemcpy(parE, Ed, size, cudaMemcpyDeviceToHost);
	cudaFree(Ed);
	cudaEventElapsedTime(&time, start, stop);
	printf("Kernel: %f ms\t", time);

	cudaMemcpy(parE, Ed, size, cudaMemcpyDeviceToHost);
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Ed);
	
	// Serial Matrix Mult
	QueryPerformanceCounter(&begin);
	float* serE = new float[25];
	int sum = 0;
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			for (int k = 0; k < 5; k++)
				sum += A[i * 5 + k] * B[k * 5 + j];
			serE[i * 5 + j] = sum;
			sum = 0;
		}
	}
	QueryPerformanceCounter(&end);
	printf("CPU: %f ms\n", (end.QuadPart - begin.QuadPart) * 1000.0 / frequency.QuadPart);

	if (equals(parE, serE, 5)) {
		cout << "Correct matrix mult" << endl;
		print(serE, 5);
	} else {
		cout << "Error in matrix mult" << endl;
		print(parE, 5);
		print(serE, 5);
	}
	_getch();

	delete A;
	delete B;
	delete serC;
	delete serD;
	delete serE;
	delete parC;
	delete parD;
	delete parE;

	cout << "\nDONE" << endl;

	return 0;
}