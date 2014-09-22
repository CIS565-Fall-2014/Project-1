#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <iostream>

float cpu_array [25];
float cpu_output_array [25];
float *gpu_array_A;
float *gpu_array_B;
float *gpu_output_array;
const int mat_width = 5;

dim3 dimBlock(mat_width, mat_width);
dim3 dimGrid(1, 1);

void initCuda(int width) {
	cudaMalloc((void**)&gpu_array_A, width*width*sizeof(float));
	cudaMemcpy(gpu_array_A, cpu_array, width*width*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&gpu_array_B, width*width*sizeof(float));
	cudaMemcpy(gpu_array_B, cpu_array, width*width*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&gpu_output_array, width*width*sizeof(float));
}

__global__ void mat_add (float* Ad, float* Bd, float* Pd, int width) {
	int index = threadIdx.y * width + threadIdx.x;

	Pd[index] = Ad[index] + Bd[index];
}

__global__ void mat_sub (float* Ad, float* Bd, float* Pd, int width) {
	int index = threadIdx.y * width + threadIdx.x;

	Pd[index] = Ad[index] - Bd[index];
}

__global__ void mat_mult (float* Ad, float* Bd, float* Pd, int width) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float Pvalue = 0;

	for (int k = 0; k < width; k++) {
		Pvalue += Ad[ty * width + k] * Bd[k * width + tx];
	}

	Pd[ty * width + tx] = Pvalue;
}

void cpu_mat_add (float* A, float* B, float* P, int width) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			P[j * width + i] = A[j * width + i] + B[j * width + i];
		}
	}
}

void cpu_mat_sub (float* A, float* B, float* P, int width) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			P[j * width + i] = A[j * width + i] - B[j * width + i];
		}
	}
}

/***
* Simple helper function for printing a matrix.
***/
void cpu_mat_mult (float* A, float* B, float* P, int width) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			float Psum = 0;
			for (int k = 0; k < width; k++) {
				Psum += A[j * width + k] * B[k * width + i];
			}
			P[j * width + i] = Psum;
		}
	}
}

/***
* Simple helper function for printing a matrix.
***/
void printMatrix (float* M, int width) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			std::cout << cpu_output_array[i * width + j] << " ";
		}
		std::cout << std::endl;
	}
}

int main(int argc, char** argv) {

	for (int i = 0; i < 25; i++) {
		cpu_array[i] = i;
	}
	
	initCuda(mat_width);
	
	mat_add<<<dimGrid, dimBlock>>>(gpu_array_A, gpu_array_B, gpu_output_array, mat_width);
	cudaMemcpy(cpu_output_array, gpu_output_array, mat_width*mat_width*sizeof(float), cudaMemcpyDeviceToHost);
	printMatrix(cpu_output_array, mat_width);

	cpu_mat_add(cpu_array, cpu_array, cpu_output_array, mat_width);
	printMatrix(cpu_output_array, mat_width);

	mat_sub<<<dimGrid, dimBlock>>>(gpu_array_A, gpu_array_B, gpu_output_array, mat_width);
	cudaMemcpy(cpu_output_array, gpu_output_array, mat_width*mat_width*sizeof(float), cudaMemcpyDeviceToHost);
	printMatrix(cpu_output_array, mat_width);

	cpu_mat_sub(cpu_array, cpu_array, cpu_output_array, mat_width);
	printMatrix(cpu_output_array, mat_width);

	mat_mult<<<dimGrid, dimBlock>>>(gpu_array_A, gpu_array_B, gpu_output_array, mat_width);
	cudaMemcpy(cpu_output_array, gpu_output_array, mat_width*mat_width*sizeof(float), cudaMemcpyDeviceToHost);
	printMatrix(cpu_output_array, mat_width);

	cpu_mat_mult(cpu_array, cpu_array, cpu_output_array, mat_width);
	printMatrix(cpu_output_array, mat_width);

	int a;
	std::cin>>a;
}