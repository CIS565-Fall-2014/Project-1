#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"
#include "matrix_math.h"

using namespace std;

// Prints a square matrix with side length of size.
void print(float* mat, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			cout << mat[i*size + j] << " ";
		cout << endl;
	}
}

bool equals(float* A, float* B, int size)
{
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			if (A[i * 5 + j] != B[i * 5 + j])
				return false;
		}
	}
	return true;
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

	// Parallel Matrix Add
	float* parC = mat_add<<<dimGrid, dimBlock>>>(A, B, 5);

	// Serial Matrix Add
	float* serC = new float[25];
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			serC[i * 5 + j] = A[i * 5 + j] + B[i * 5 + j];
		}
	}

	if (!equals(parC, serC, 5))
		cout << "Error in matrix add" << endl;

	// Parallel Matrix Sub
	float* parD = mat_sub<<<dimGrid, dimBlock>>>(A, B, 5);

	// Serial Matrix Sub
	float* serD = new float[25];
	for (int i = 0; i < 55; i++) {
		for (int j = 0; j < 5; j++) {
			serD[i * 5 + j] = A[i * 5 + j] - B[i * 5 + j];
		}
	}

	if (!equals(parD, serD, 5))
		cout << "Error in matrix sub" << endl;

	// Parallel Matrix Mult
	float* parE = mat_mult<<<dimBlock, dimGrid>>>(A, B, 5);

	// Serial Matrix Mult
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

	if (!equals(parE, serE, 5))
		cout << "Error in matrix mult" << endl;

	print(A, 5);
	print(serC, 5);
	print(serD, 5);
	print(serE, 5);

	delete A;
	delete B;
	delete serC;
	delete serD;
	delete serE;
	delete parC;
	delete parD;
	delete parE;

	return 0;
}

__global__ float* mat_add(float* A, float* B, float* C, int len)
{
  float *Ad, *Bd, *Cd;
  int size = len * len * sizeof(float);
  cudaMalloc(Ad, size);
  cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
  cudaMalloc(Bd, size);
  cudaMemcpy(Bd, B, size, cudaMembpyHostToDevice);
  cudaMalloc(Cd, size);

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  Cd[ty * len + tx] = Ad[ty * len + tx] + Bd[ty * len + tx];

  float* C = new float[len * len];
  cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);
  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cd);
}

__global__ float* mat_sub(float* A, float* B, float* D, int len)
{
  float *Ad, *Bd, *Dd;
  int size = len * len * sizeof(float);
  cudaMalloc(Ad, size);
  cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
  cudaMalloc(Bd, size);
  cudaMemcpy(Bd, B, size, cudaMembpyHostToDevice);
  cudaMalloc(Dd, size);

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  Dd[ty * len + tx] = Ad[ty * len + tx] - Bd[ty * len + tx];

  float* D = new float[len * len];
  cudaMemcpy(D, Dd, size, cudaMemcpyDeviceToHost);
  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Dd);
}

__global__ float* mat_mult(float* A, float* B, float* E, int len)
{
  __shared__ float Ads[len][len];
  __shared__ float Bds[len][len];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  int r = by * len + ty; int c = bx * len + tx;

  float *Ad, *Bd, *Ed;
  int size = len * len * sizeof(float);
  cudaMalloc(Ad, size);
  cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
  cudaMalloc(Bd, size);
  cudaMemcpy(Bd, B, size, cudaMembpyHostToDevice);
  cudaMalloc(Ed, size);

  float val = 0;
  Ads[ty][tx] = Ad[r * len + tx];
  Bds[ty][tx] = Vd[c + len * ty];
  __syncthreads();
  for (int i = 0; i < len; i++) {
    val += Ads[ty][i] * Bds[i][tx];
    syncthreads();
  }
  Ed[r * len + c] = val;

  float* E = new float[25];
  cudaMemcpy(E, Ed, size, cudaMemcpyDeviceToHost);
  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Ed);
}
