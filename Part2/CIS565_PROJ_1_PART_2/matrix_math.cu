#pragma once

#include <cuda_runtime.h>
#include <iostream>

const int MATRIX_WIDTH = 5;
const int MATRIX_SIZE = MATRIX_WIDTH * MATRIX_WIDTH;

// Function prototypes.
__global__ void matAdd( float *mat1, float *mat2, float *result );
__global__ void matSub( float *mat1, float *mat2, float *result );
__global__ void matMult( float *mat1, float *mat2, float *result );
void matAdd( int N, float *mat1, float *mat2, float *result );
void matSub( int N, float *mat1, float *mat2, float *result );
void matMult( int N, float *mat1, float *mat2, float *result );
void debugPrintMat( int N, float *mat );

int main(int argc, char** argv)
{
	float *h_matrix1, *h_matrix2, *h_result;
	float *d_matrix1, *d_matrix2, *d_result;
	int size = sizeof( float ) * MATRIX_SIZE;

	// Allocate memory for host data.
	h_matrix1 = ( float* )malloc( size );
	h_matrix2 = ( float* )malloc( size );
	h_result = ( float* )malloc( size );

	// Allocate memory for device data.
	cudaMalloc( (void**)&d_matrix1, size );
	cudaMalloc( (void**)&d_matrix2, size );
	cudaMalloc( (void**)&d_result, size );

	// Populate host matrices.
	for ( int i = 0; i < MATRIX_SIZE; ++i ) {
		h_matrix1[i] = i;
		h_matrix2[i] = i;
	}
	
	// Copy data in host matrices to device matrices.
	cudaMemcpy( d_matrix1, h_matrix1, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_matrix2, h_matrix2, size, cudaMemcpyHostToDevice );

	// Output operand matrices.
	std::cout << "Matrix 1:" << std::endl;
	debugPrintMat( MATRIX_SIZE, h_matrix1 );
	std::cout << "Matrix 2:" << std::endl;
	debugPrintMat( MATRIX_SIZE, h_matrix2 );

	// Call matAdd kernel, copy result to host, and output.
	matAdd<<< 1, MATRIX_SIZE >>>( d_matrix1, d_matrix2, d_result );
	cudaMemcpy( h_result, d_result, size, cudaMemcpyDeviceToHost );
	std::cout << "Parallel add:" << std::endl;
	debugPrintMat( MATRIX_SIZE, h_result );

	// Call matSub kernel, copy result to host, and output.
	matSub<<< 1, MATRIX_SIZE >>>( d_matrix1, d_matrix2, d_result );
	cudaMemcpy( h_result, d_result, size, cudaMemcpyDeviceToHost );
	std::cout << "Parallel subtract:" << std::endl;
	debugPrintMat( MATRIX_SIZE, h_result );

	// Call matMult kernel, copy result to host, and output.
	matMult<<< 1, MATRIX_SIZE >>>( d_matrix1, d_matrix2, d_result );
	cudaMemcpy( h_result, d_result, size, cudaMemcpyDeviceToHost );
	std::cout << "Parallel multiply:" << std::endl;
	debugPrintMat( MATRIX_SIZE, h_result );

	// Output results of seriel matrix operations to compare results with parallel implementations.
	matAdd( MATRIX_SIZE, h_matrix1, h_matrix2, h_result );
	std::cout << "Seriel add:" << std::endl;
	debugPrintMat( MATRIX_SIZE, h_result );
	matSub( MATRIX_SIZE, h_matrix1, h_matrix2, h_result );
	std::cout << "Seriel subtract:" << std::endl;
	debugPrintMat( MATRIX_SIZE, h_result );
	matMult( MATRIX_SIZE, h_matrix1, h_matrix2, h_result );
	std::cout << "Seriel multiply:" << std::endl;
	debugPrintMat( MATRIX_SIZE, h_result );

	// Wait for user input to stall console from exiting, so we can see the printed results.
	std::cin.ignore();

	// Release allocated memory.
	free( h_matrix1 );
	free( h_matrix2 );
	free( h_result );
	cudaFree( d_matrix1 );
	cudaFree( d_matrix2 );
	cudaFree( d_result );

	return 0;
}


////////////////////////////////////////////////////
// Parallel algorithms.
////////////////////////////////////////////////////

__global__
void matAdd( float *mat1, float *mat2, float *result )
{
	int index = threadIdx.x;
	result[index] = mat1[index] + mat2[index];
}

__global__
void matSub( float *mat1, float *mat2, float *result )
{
	int index = threadIdx.x;
	result[index] = mat1[index] - mat2[index];
}

__global__
void matMult( float *mat1, float *mat2, float *result )
{
	int index = threadIdx.x;
	result[index] = mat1[index] * mat2[index];
}


////////////////////////////////////////////////////
// Seriel algorithms.
////////////////////////////////////////////////////

void matAdd( int N, float *mat1, float *mat2, float *result )
{
	for ( int i = 0; i < N; ++i ) {
		result[i] = mat1[i] + mat2[i];
	}
}


void matSub( int N, float *mat1, float *mat2, float *result )
{
	for ( int i = 0; i < N; ++i ) {
		result[i] = mat1[i] - mat2[i];
	}
}


void matMult( int N, float *mat1, float *mat2, float *result )
{
	for ( int i = 0; i < N; ++i ) {
		result[i] = mat1[i] * mat2[i];
	}
}


////////////////////////////////////////////////////
// Debug print.
////////////////////////////////////////////////////

void debugPrintMat( int N, float *mat )
{
	for ( int i = 0; i < N; ++i ) {
		std::cout << mat[i] << " ";
	}
	std::cout << std::endl;
}