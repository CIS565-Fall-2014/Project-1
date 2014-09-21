#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
const int N = 3;

//data on device
float * dev_matA;
float * dev_matB;
float * dev_res;

//data on host
float * matA = (float *)malloc(N*N*sizeof(float));
float * matB = (float *)malloc(N*N*sizeof(float));
float * res = (float *)malloc(N*N*sizeof(float));

void initMat(float * mat)
{
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
			mat[i + j*N] = i + j * N;
}

void printRes()
{
	for(int j =0; j <N; j++)
	{
		cout << endl;
		for(int i=0;i<N;i++)
			cout << res[j*N + i] << " ";
	}
}

__global__ void mat_add(float * A, float * B, float * res)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	res[x + y*N] = A[x + y*N] + B[x + y*N];
}

__global__ void mat_sub(float * A, float * B, float * res)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	res[x + y*N] = A[x + y*N] - B[x + y*N];
}

__global__ void mat_mult(float * A, float * B, float * res)
{
	int x = threadIdx.x;
	int y = threadIdx.y;

	float result = 0;
	for(int i = 0; i < N; i++)
	{
		result += A[y * N + i] * B[x + i * N];
	}
	res[x + y*N] = result;
}

void mat_add_serial(float * A, float * B, float * res)
{
	for(int i = 0; i < N * N; i++)
	{
		res[i] = A[i] + B[i];
	}
}
void mat_sub_serial(float * A, float * B, float * res)
{
	for(int i = 0; i < N * N; i++)
	{
		res[i] = A[i] - B[i];
	}
}
void mat_mult_serial(float * A, float * B, float * res)
{
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
		{
			float result = 0;
			for(int k = 0; k < N; k++)
			{
				result += A[i*N + k] * B[k*N + j];
			}
			res[i*N + j] = result;
		}
}

int main(int argc, char** argv)
{
	dim3 threadsPerBlock(N,N);
	cudaMalloc((void**)&dev_matA, N*N*sizeof(float));
	cudaMalloc((void**)&dev_matB, N*N*sizeof(float));
	cudaMalloc((void**)&dev_res, N*N*sizeof(float));

	initMat(matA);
	initMat(matB);
	cudaMemcpy(dev_matA, matA, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matB, matB, N*N*sizeof(float), cudaMemcpyHostToDevice);

	mat_add<<<1, threadsPerBlock>>>(dev_matA, dev_matB, dev_res);
	cudaMemcpy(res, dev_res, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	cout<< endl << "###########parallel mat_add test######## ";
	printRes();

	mat_sub<<<1, threadsPerBlock>>>(dev_matA, dev_matB, dev_res);
	cudaMemcpy(res, dev_res, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	cout<< endl << "###########parallel mat_add test######## ";
	printRes();

	mat_mult<<<1, threadsPerBlock>>>(dev_matA, dev_matB, dev_res);
	cudaMemcpy(res, dev_res, N*N*sizeof(float), cudaMemcpyDeviceToHost);
	cout<< endl << "###########parallel mat_mul test######## ";
	printRes();

	mat_add_serial(matA, matB, res);
	cout<< endl << "###########serial mat_add test######## ";
	printRes();

	mat_sub_serial(matA, matB, res);
	cout<< endl << "###########serial mat_add test######## ";
	printRes();

	mat_mult_serial(matA, matB, res);
	cout<< endl << "###########serial mat_mul test######## ";
	printRes();

}