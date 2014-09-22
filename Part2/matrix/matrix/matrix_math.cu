#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <assert.h>
//#include <ctime>

using namespace std;

#define WIDTH 5
#define MSIZE 25
#define numBlocks 1
dim3 threadsPerBlock(WIDTH, WIDTH);

__global__ void matAdd(float* Ad, float *Bd, float *Pd)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float a = Ad[ty * WIDTH + tx];
	float b = Bd[ty * WIDTH + tx];
	
	Pd[ty * WIDTH + tx] = a + b;
}

__global__ void matSub(float* Ad, float *Bd, float *Pd)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float a = Ad[ty * WIDTH + tx];
	float b = Bd[ty * WIDTH + tx];
	
	Pd[ty * WIDTH + tx] = a - b;
}

__global__ void matMul(float* Ad, float *Bd, float *Pd)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float pValue = 0.0f;
	for (int k = 0; k < WIDTH; k++)
	{
		float a = Ad[ty * WIDTH + k];
		float b = Bd[k * WIDTH + tx];
		pValue += a * b;
	}
	
	
	Pd[ty * WIDTH + tx] = pValue;
}


void matSerialAdd(float *A, float *B, float *P)
{
	for (int r = 0; r < WIDTH; r++)
	{
		for (int c = 0; c < WIDTH; c++)
		{
			P[r * WIDTH + c] = A[r * WIDTH + c] + B[r * WIDTH + c];
		}
	}
}

void matSerialSub(float *A, float *B, float *P)
{
	for (int r = 0; r < WIDTH; r++)
	{
		for (int c = 0; c < WIDTH; c++)
		{
			P[r * WIDTH + c] = A[r * WIDTH + c] - B[r * WIDTH + c];
		}
	}
}

void matSerialMul(float *A, float *B, float *P)
{
	for (int r = 0; r < WIDTH; r++)
	{
		for (int c = 0; c < WIDTH; c++)
		{
			float pValue = 0.0f;
			for (int k = 0; k < WIDTH; k++)
			{
				pValue += A[r * WIDTH + k] * B[k * WIDTH + c];
			}
			P[r * WIDTH + c] = pValue;
		}
	}
}


int main()
{
	float *A = new float[MSIZE];
	float *B = new float[MSIZE];
	float *P = new float[MSIZE];
	float *serialP = new float[MSIZE];
	for (int i = 0; i < MSIZE; i++)
	{
		A[i] = i;
		B[i] = i;
	}

	//clock_t start;
	//double durationGPU, durationCPU;

	//load A, B to device memory
	int size = MSIZE * sizeof(float);
	float *Ad, *Bd, *Pd;

	cudaMalloc((void**)&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Pd, size);

	//add
	//start = clock();
	matAdd<<< numBlocks, threadsPerBlock >>>(Ad, Bd, Pd);
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	//durationGPU = (clock() - start) / (double)CLOCKS_PER_SEC;

	//start = clock();
	matSerialAdd(A, B, serialP);
	//durationCPU = (clock() - start) / (double)CLOCKS_PER_SEC;

	for (int i = 0; i < MSIZE; i++)
		assert(P[i] == serialP[i]);
	cout << "Matrix Addition Success!" << endl;
	//cout << "CPU Timing: " << durationCPU << endl;
	//cout << "GPU Timing: " << durationGPU << endl<<endl;

	//sub
	//start = clock();
	matSub<<< numBlocks, threadsPerBlock >>>(Ad, Bd, Pd);
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	//durationGPU = (clock() - start) / (double)CLOCKS_PER_SEC;

	//start = clock();
	matSerialSub(A, B, serialP);
	//durationCPU = (clock() - start) / (double)CLOCKS_PER_SEC;

	for (int i = 0; i < MSIZE; i++)
		assert(P[i] == serialP[i]);
	std::cout << "Matrix Subtraction Success!" << std::endl;
	//cout << "CPU Timing: " << durationCPU << endl;
	//cout << "GPU Timing: " << durationGPU << endl<<endl;

	//dot mul
	//start = clock();
	matMul<<< numBlocks, threadsPerBlock >>>(Ad, Bd, Pd);
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	//durationGPU = (clock() - start) / (double)CLOCKS_PER_SEC;

	//start = clock();
	matSerialMul(A, B, serialP);
	//durationCPU = (clock() - start) / (double)CLOCKS_PER_SEC;

	for (int i = 0; i < MSIZE; i++)
		assert(P[i] == serialP[i]);
	std::cout << "Matrix Dot Multiplication Success!" << std::endl;
	//cout << "CPU Timing: " << durationCPU << endl;
	//cout << "GPU Timing: " << durationGPU << endl<<endl;

	//free device memory
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Pd);

	return 0;
}