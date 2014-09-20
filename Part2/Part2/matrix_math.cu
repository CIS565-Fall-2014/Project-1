#include <stdio.h>
#include <cuda.h>
#include <cmath>

float * mat_1d;
float * mat_2d;
float * mat_3d;

float * mat_1;
float * mat_2;
float * mat_3;

const int N = 5;

// initialize function
void init() {
	int size = N*N;

	// Allocate CPU Memory
	mat_1 = (float*) malloc(size*sizeof(float));
	mat_2 = (float*) malloc(size*sizeof(float));
	mat_3 = (float*) malloc(size*sizeof(float));

	// Allocate GPU Memory
    cudaMalloc((void**)&mat_1d, size*sizeof(float));
	cudaMalloc((void**)&mat_2d, size*sizeof(float));
	cudaMalloc((void**)&mat_3d, size*sizeof(float));

	// Initialize CPU Memory
	for (size_t i = 0; i < size; i++) {
		mat_1[i] = 3.2*(i/5) - 1.2*(i%5) + 7.5;
		mat_2[i] = 1.6*(i/5) + 5.5*(i%5) - 2.2;
		mat_3[i] = 0;
	}

	// Initialize GPU Memory
	cudaMemcpy(mat_1d, mat_1, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(mat_2d, mat_2, size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(mat_3d, mat_3, size*sizeof(float), cudaMemcpyHostToDevice);
}

void cleanUp() {
	free(mat_1);
	free(mat_2);
	free(mat_3);

	cudaFree(mat_1d);
	cudaFree(mat_2d);
	cudaFree(mat_3d);
}

void printResults(float* m) {
	for (size_t i = 0; i < N; i++) {
		fprintf(stdout, "[%f, %f, %f, %f, %f] \n", m[i*N], m[i*N+1], m[i*N+2], m[i*N+3], m[i*N+4]);
	}
	fprintf(stdout, "\n");
}

void printResultsGPU(float* md) {
	float* temp = (float*) malloc(N*N*sizeof(float));

	cudaMemcpy(temp, md, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	printResults(temp);

	free(temp);
}

__global__ void mat_add(float* m1, float* m2, float* m3) {
    int index = (threadIdx.x * blockDim.x) + threadIdx.y;

	m3[index] = m1[index] + m2[index];
}

__global__ void mat_sub(float* m1, float* m2, float* m3) {
    int index = (threadIdx.x * blockDim.x) + threadIdx.y;

	m3[index] = m1[index] - m2[index];
}

__global__ void mat_mult(float* m1, float* m2, float* m3) {
    int index = (threadIdx.x * blockDim.x) + threadIdx.y;

	// Initialize the result value
	float value = 0.0f;

	// Determine the row and column number of the current element
	int row = threadIdx.x;
	int col = threadIdx.y;

	// Loop through and compute the dot product needed for this element
	for (size_t i = 0; i < N; i++) {
		value += m1[row*N + i] * m2[i*N + col];
	}

	m3[index] = value;
}

void mat_add_cpu(float* m1, float* m2, float* m3) {
	for (size_t i = 0; i < N; i++) {
		for (size_t j = 0; j < N; j++) {
			m3[N*i+j] = m1[N*i+j] + m2[N*i+j];
		}
	}
}

void mat_sub_cpu(float* m1, float* m2, float* m3) {
	for (size_t i = 0; i < N; i++) {
		for (size_t j = 0; j < N; j++) {
			m3[N*i+j] = m1[N*i+j] - m2[N*i+j];
		}
	}
}

void mat_mult_cpu(float* m1, float* m2, float* m3) {
	for (size_t i = 0; i < N; i++) {
		for (size_t j = 0; j < N; j++) {
			float sum = 0;
			for (size_t k = 0; k < N; k++) {
				float a = m1[N*i + k];
				float b = m2[k*N + j];
				sum += a * b;
			}
			m3[N*i+j] = sum;
		}
	}
}

int main(int argc, char** argv) {
	init();

	dim3 dimBlock(N,N);

	fprintf(stdout, "GPU: \n");

	// Do matrix addition on the GPU and see the result
	mat_add<<<1,dimBlock>>>(mat_1d, mat_2d, mat_3d);
	cudaThreadSynchronize();
	printResultsGPU(mat_3d);

	// Do matrix subtraction on the GPU and see the result
	mat_sub<<<1,dimBlock>>>(mat_1d, mat_2d, mat_3d);
	cudaThreadSynchronize();
	printResultsGPU(mat_3d);

	// Do matrix multiplication on the GPU and see the result
	mat_mult<<<1,dimBlock>>>(mat_1d, mat_2d, mat_3d);
	cudaThreadSynchronize();
	printResultsGPU(mat_3d);

	fprintf(stdout, "CPU: \n");

	// Do matrix addition on the CPU and see the result
	mat_add_cpu(mat_1, mat_2, mat_3);
	printResults(mat_3);

	// Do matrix subtraction on the CPU and see the result
	mat_sub_cpu(mat_1, mat_2, mat_3);
	printResults(mat_3);

	// Do matrix multiplication on the CPU and see the result
	mat_mult_cpu(mat_1, mat_2, mat_3);
	printResults(mat_3);

	cleanUp();

	while (true) {

	}

	return 0;
}
