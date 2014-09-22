
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

#define BLOCK_SIZE 8

cudaError_t mat_add(float *a, float *b, float *c, unsigned int size);
cudaError_t mat_sub(float *a, float *b, float *c, unsigned int size);
cudaError_t mat_mult(float *a, float *b, float *c, unsigned int size);
__global__ void mat_add_kernel(const float *a, const float *b, float *c, unsigned int size)
{
   int row =  threadIdx.y;
   int col =  threadIdx.x;
   if(row < size && col < size)
   {
		c[row * size + col] = a[row * size +col] + b[ row * size + col];
   }
}
__global__ void mat_mult_kernel(const float *a, const float *b, float *c, unsigned int size)
{
   int row =  threadIdx.y;
   int col =  threadIdx.x;
   float cValue = 0;
   if(row < size && col < size)
   {
	   for(int i=0; i< size; ++i)
	   {
		   cValue += a[row * size + i] * b[i * size + col];
	   }
	   c[row * size + col] =cValue;
   }
}
__global__ void mat_sub_kernel(const float *a, const float *b, float *c, unsigned int size)
{
   int row =  threadIdx.y;
   int col =  threadIdx.x;
   if(row < size && col < size)
   {
		c[row * size + col] = a[row * size +col] - b[ row * size + col];
   }
}

void printMat(float *a, int size)
{
	for(int i=0; i<size; ++i)
	{
		for(int j=0; j < size; ++j)
		{
			std::cout<<a[i*size +j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
}

int main()
{
	unsigned int size = 5;
	unsigned int len = size * size;
	float *mat1 = (float*) malloc(sizeof(float)* len);
	float *mat2 =  (float*) malloc(sizeof(float)* len);
	float count = 0.0f;
	for(int i=0; i< size; ++i){
		for(int j=0; j < size;++j){
			mat1[i * size +j] = count;
			mat2[i * size +j] = count;
			count += 1.0f;
		}
	}
	std::cout<<"This is GPU version"<<std::endl;
	printMat(mat1,size);
	printMat(mat2,size);
	float *mat3 = (float*) malloc(sizeof(float) * len);
	mat_add(mat1,mat2,mat3,size);
	printMat(mat3,size);
	mat_sub(mat1,mat2,mat3,size);
	printMat(mat3,size);
	mat_mult(mat1,mat2,mat3,size);
	printMat(mat3,size);
	free(mat1);
	free(mat2);
	free(mat3);
    return 0;
}

cudaError_t mat_add(float *a, float *b, float *c, unsigned int size)
{
	float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
	unsigned int len = size * size;
    cudaError_t cudaStatus;
	 // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
	 // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc(&dev_c, len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	 cudaStatus = cudaMalloc(&dev_a, len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc(&dev_b, len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	 // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	 cudaStatus = cudaMemcpy(dev_c, c, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	// Launch a kernel on the GPU with one thread for each element.
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
	//dim3 dimGrid(size / dimBlock.x +1, size / dimBlock.y+1);
    mat_add_kernel<<<1, dimBlock>>>(dev_a, dev_b, dev_c, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
/*
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }*/

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, len * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
	return cudaStatus;
}
cudaError_t mat_sub(float *a, float *b, float *c, unsigned int size)
{
	float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
	unsigned int len = size * size;
    cudaError_t cudaStatus;
	 // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
	 // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc(&dev_c, len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	 cudaStatus = cudaMalloc(&dev_a, len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc(&dev_b, len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	 // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	 cudaStatus = cudaMemcpy(dev_c, c, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	// Launch a kernel on the GPU with one thread for each element.
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
	//dim3 dimGrid(size / dimBlock.x +1, size / dimBlock.y+1);
    mat_sub_kernel<<<1, dimBlock>>>(dev_a, dev_b, dev_c, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
/*
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }*/

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, len * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
	return cudaStatus;
}
cudaError_t mat_mult(float *a, float *b, float *c, unsigned int size)
{
	float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
	unsigned int len = size * size;
    cudaError_t cudaStatus;
	 // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
	 // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc(&dev_c, len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	 cudaStatus = cudaMalloc(&dev_a, len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc(&dev_b, len * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	 // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	 cudaStatus = cudaMemcpy(dev_c, c, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	// Launch a kernel on the GPU with one thread for each element.
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
	//dim3 dimGrid(size / dimBlock.x +1, size / dimBlock.y+1);
    mat_mult_kernel<<<1, dimBlock>>>(dev_a, dev_b, dev_c, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
/*
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }*/

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, len * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
	return cudaStatus;
}