#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void dev_matrix_add(int dim, float * A, float * B, float * result)
{
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(row < dim && col < dim) result[row * dim + col] = A[row * dim + col] + B[row * dim + col];
}

__global__ void dev_matrix_sub(int dim, float * A, float * B, float * result)
{
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(row < dim && col < dim) result[row * dim + col] = A[row * dim + col] - B[row * dim + col];
}

__global__ void dev_matrix_mult(int dim, float * A, float * B, float * result)
{
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(row >= dim || col >= dim) return;
	float sum = 0.0f;
	for (int i = 0; i < dim; i++)
	{
		sum += A[row * dim + i] * B[col * dim + i];
	}

	result[row * dim + col] = sum;
}


int main(int argc, char** argv)
{
	float * A, *B, * result, * dev_A, * dev_B, * dev_result;
	int M(9);
	int tileWidth = 2;
	int N = M * M * sizeof(float);

	A = (float*) malloc( N);
	B = (float*) malloc(N);
	result = (float*) malloc(N);

	cudaMalloc((void**) & dev_A, N);
	cudaMalloc((void**) & dev_B, N);
	cudaMalloc((void**) & dev_result,N);

	for(int i = 0;i<M*M;i++)
	{
		A[i] = 1.0f;
		B[i] = 1.0f;
	}

	cudaMemcpy(dev_A,A,N,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B,B,N,cudaMemcpyHostToDevice);

	dim3 gridDim((int)ceil((float)M/(float)tileWidth),(int)ceil((float)M/(float)tileWidth));
	dim3 blockDim(tileWidth,tileWidth);

	dev_matrix_mult<<<gridDim,blockDim>>>(M,dev_A,dev_B,dev_result);


	cudaMemcpy(result, dev_result,N,cudaMemcpyDeviceToHost);
	for(int i = 0;i<M*M;i++)
	{
		std::cout<<result[i]<<std::endl;
	}
    std::cout<<"test";
	std::cin.get();
    return 0;
}

