
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void mat_add(const float *m1, const float *m2, float *m3)
{
    int i = threadIdx.x;
    m3[i] = m2[i] + m1[i];
}

__global__ void mat_sub(const float *m1, const float *m2, float *m3)
{
    int i = threadIdx.x;
    m3[i] = m1[i] - m2[i];
}

__global__ void mat_mult(const float *m1, const float *m2, float *m3, int matrix_size)
{
	int i = threadIdx.x;
	int row = i / matrix_size;
	int column = i - (i / matrix_size) * matrix_size;

	m3[i] = 0;
	for(int j=0; j<matrix_size; j++)
	{
		m3[i] += m1[row * matrix_size + j] * m2[j * matrix_size + column];
	}
    
}

void mat_add_serial(const float *m1, const float *m2, float *m3, int matrix_size)
{
    for(int i=0; i< matrix_size * matrix_size; i++)
	 m3[i] = m2[i] + m1[i];
}

void mat_sub_serial(const float *m1, const float *m2, float *m3, int matrix_size)
{
    for(int i=0; i< matrix_size * matrix_size; i++)
		m3[i] = m1[i] - m2[i];
}

void mat_mult_serial(const float *m1, const float *m2, float *m3, int matrix_size)
{
	for(int i=0; i< matrix_size * matrix_size; i++){
		int row = i / matrix_size;
		int column = i - (i / matrix_size) * matrix_size;

		m3[i] = 0;
		for(int j=0; j<matrix_size; j++)
		{
			m3[i] += m1[row * matrix_size + j] * m2[j * matrix_size + column];
		}
	}
}

int main()
{
	const int matrix_width = 5;
	const int matrix_height = 5;
	const float m1[matrix_width * matrix_height] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
	const float m2[matrix_width * matrix_height] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
	float m3[matrix_width * matrix_height] = {0};

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

	float *dev_m1 = 0;
    float *dev_m2 = 0;
    float *dev_m3 = 0;
	    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_m1, matrix_width * matrix_height * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_m2, matrix_width * matrix_height * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc((void**)&dev_m3, matrix_width * matrix_height * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

	 // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_m1, m1, matrix_width * matrix_height * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

    cudaStatus = cudaMemcpy(dev_m2, m2, matrix_width * matrix_height * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }

	// add
	mat_add<<<1, matrix_width * matrix_height>>>(dev_m1, dev_m2, dev_m3);
	cudaDeviceSynchronize();
    cudaMemcpy(m3, dev_m3, matrix_width * matrix_height * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0;i<matrix_height;i++)
		printf("{%.0f,%.0f,%.0f,%.0f,%.0f}\n",m3[i*matrix_width], m3[i*matrix_width+1], m3[i*matrix_width+2], m3[i*matrix_width+3], m3[i*matrix_width+4]);
	printf("\n");

	// sub
	mat_sub<<<1, matrix_width * matrix_height>>>(dev_m1, dev_m2, dev_m3);
    cudaDeviceSynchronize();
    cudaMemcpy(m3, dev_m3, matrix_width * matrix_height * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0;i<matrix_height;i++)
		printf("{%.0f,%.0f,%.0f,%.0f,%.0f}\n",m3[i*matrix_width], m3[i*matrix_width+1], m3[i*matrix_width+2], m3[i*matrix_width+3], m3[i*matrix_width+4]);
	printf("\n");

	//mult
	mat_mult<<<1, matrix_width * matrix_height>>>(dev_m1, dev_m2, dev_m3,matrix_width);
    cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(m3, dev_m3, matrix_width * matrix_height * sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0;i<matrix_height;i++)
		printf("{%.0f,%.0f,%.0f,%.0f,%.0f}\n",m3[i*matrix_width], m3[i*matrix_width+1], m3[i*matrix_width+2], m3[i*matrix_width+3], m3[i*matrix_width+4]);
	printf("\n");

	// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }



    cudaFree(dev_m1);
    cudaFree(dev_m2);
    cudaFree(dev_m3);

    // Add vectors in parallel.
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}