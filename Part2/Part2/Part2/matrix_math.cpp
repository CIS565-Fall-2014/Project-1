#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
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
	float* parC = new float[25];
	//mat_add<<<dimGrid, dimBlock>>>(A, B, parC, 5);

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
	float* parD = new float[25];
	//mat_sub<<<dimGrid, dimBlock>>>(A, B, parD, 5);

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
	float* parE = new float[25];
	//mat_mult<<<dimBlock, dimGrid>>>(A, B, parE, 5);

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
