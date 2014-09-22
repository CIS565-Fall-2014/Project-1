#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WIDTH 5


__global__ void MatAddKernel (float* Ad, float* Bd, float* Rd, int Width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float RValue = 0;

	Rd[ty*Width+tx] = Ad[ty * Width + tx] + Bd[ty * Width + tx];
}

__global__ void MatMinKernel (float* Ad, float* Bd, float* Rd, int Width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float RValue = 0;

	Rd[ty*Width+tx] = Ad[ty * Width + tx] - Bd[ty * Width + tx];
}

__global__ void MatMulKernel(float* Ad, float* Bd, float* Rd, int Width)
{
	//2D Thread ID
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float RValue = 0;

	for (int k = 0; k < Width; ++k)
	{
		float Ad_ele = Ad[ty * Width + k];
		float Bd_ele = Bd[k * Width + tx];
		RValue += Ad_ele * Bd_ele;
	}

	Rd[ty*Width+tx] = RValue;
}


void MatrixMulOnDevice(float* A, float*B, float* R, int Width)
{
	int size = Width * Width * sizeof(float);
	float* Ad, *Bd, *Rd;
	//load A and B to device momory
	cudaMalloc((void**)&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(Width, Width);
	dim3 dimGrid(1,1);
	cudaMalloc((void**)&Rd, size);

	MatMulKernel<<<dimGrid, dimBlock>>>(Ad, Bd, Rd, Width);

	cudaMemcpy(R, Rd, size, cudaMemcpyDeviceToHost);
	cudaFree(Ad); cudaFree(Bd); cudaFree(Rd);
}

void MatrixAddOnDevice(float* A, float*B, float* R, int Width)
{
	int size = Width * Width * sizeof(float);
	float* Ad, *Bd, *Rd;
	//load A and B to device momory
	cudaMalloc((void**)&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(Width, Width);
	dim3 dimGrid(1,1);
	cudaMalloc((void**)&Rd, size);

	MatAddKernel<<<dimGrid, dimBlock>>>(Ad, Bd, Rd, Width);

	cudaMemcpy(R, Rd, size, cudaMemcpyDeviceToHost);
	cudaFree(Ad); cudaFree(Bd); cudaFree(Rd);
}

void MatrixMinOnDevice(float* A, float*B, float* R, int Width)
{
	int size = Width * Width * sizeof(float);
	float* Ad, *Bd, *Rd;
	//load A and B to device momory
	cudaMalloc((void**)&Ad, size);
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Bd, size);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(Width, Width);
	dim3 dimGrid(1,1);
	cudaMalloc((void**)&Rd, size);

	MatMinKernel<<<dimGrid, dimBlock>>>(Ad, Bd, Rd, Width);

	cudaMemcpy(R, Rd, size, cudaMemcpyDeviceToHost);
	cudaFree(Ad); cudaFree(Bd); cudaFree(Rd);
}

void MatrixMulOnHost(float* A, float* B, float* R, int Width)
{
	for (int i = 0; i<Width; i++)
		for (int j = 0; j<Width; j++)
	{
		float sum = 0;
		for (int k = 0; k<Width; k++)
		{
			float a = A[i*Width + k];
			float b = B[k*Width + j];
			sum += a * b;
		}
		R[i*Width + j] = sum;
	}
}
void MatrixAddOnHost(float* A, float* B, float* R, int Width)
{
	for (int i = 0; i<Width; i++)
		for (int j = 0; j<Width; j++)
	{
		R[i*Width + j] = A[i*Width + j] + B[i*Width + j];
	}
}

void MatrixMinOnHost(float* A, float* B, float* R, int Width)
{
	for (int i = 0; i<Width; i++)
		for (int j = 0; j<Width; j++)
	{
		R[i*Width + j] = A[i*Width + j] - B[i*Width + j];
	}
}


void main(){
//	__device__ float* M1_d = new float[WIDTH * WIDTH];
//	__device__ float* M2_d = new float[WIDTH * WIDTH];
//	__device__ float* R_d = new float[WIDTH * WIDTH];

	float* M1_h = new float[WIDTH * WIDTH]; 
	float* M2_h = new float[WIDTH * WIDTH];
	
	float* R_h_mul = new float[WIDTH * WIDTH]; 
	float* R_h_add = new float[WIDTH * WIDTH];
	float* R_h_min = new float[WIDTH * WIDTH]; 

	float* R_h_mulc = new float[WIDTH * WIDTH]; 
	float* R_h_addc = new float[WIDTH * WIDTH];
	float* R_h_minc = new float[WIDTH * WIDTH]; 

	for(int i = 0; i<WIDTH*WIDTH; i++)
	{
		M1_h[i] = i; M2_h[i] = i;
	}
	
	MatrixMulOnDevice(M1_h, M2_h, R_h_mul, WIDTH);
	MatrixAddOnDevice(M1_h, M2_h, R_h_add, WIDTH);
	MatrixMinOnDevice(M1_h, M2_h, R_h_min, WIDTH);

	MatrixMulOnHost(M1_h, M2_h, R_h_mulc, WIDTH);
	MatrixAddOnHost(M1_h, M2_h, R_h_addc, WIDTH);
	MatrixMinOnHost(M1_h, M2_h, R_h_minc, WIDTH);

	//results
	printf("M1 = \n");
	for(int j = 0; j<WIDTH*WIDTH; j++)
	{
		printf("%.1f ", M1_h[j]);
		if(j%5 == 4) printf("\n");
	}

	printf("\n M2 = \n");
	for(int j = 0; j<WIDTH*WIDTH; j++)
	{
		printf("%.1f ", M2_h[j]);
		if(j%5 == 4) printf("\n");
	}

	printf("\n ============= GPU =============\n");
	printf("\n Matrix Multiply: M1 * M2 = \n");
	for(int j = 0; j<WIDTH*WIDTH; j++)
	{
		printf("%.1f ", R_h_mul[j]);
		if(j%5 == 4) printf("\n");
	}

	printf("\n Matrix Addition: M1 + M2 = \n");
	for(int j = 0; j<WIDTH*WIDTH; j++)
	{
		printf("%.1f ", R_h_add[j]);
		if(j%5 == 4) printf("\n");
	}
	
	printf("\n Matrix Subtraction: M1 - M2 = \n");
	for(int j = 0; j<WIDTH*WIDTH; j++)
	{
		printf("%.1f ", R_h_min[j]);
		if(j%5 == 4) printf("\n");
	}

	printf("\n ============= CPU =============\n");
	printf("\n Matrix Multiply: M1 * M2 = \n");
	for(int j = 0; j<WIDTH*WIDTH; j++)
	{
		printf("%.1f ", R_h_mulc[j]);
		if(j%5 == 4) printf("\n");
	}
	printf("\n Matrix Addition: M1 + M2 = \n");
	for(int j = 0; j<WIDTH*WIDTH; j++)
	{
		printf("%.1f ", R_h_addc[j]);
		if(j%5 == 4) printf("\n");
	}
	printf("\n Matrix Subtraction: M1 - M2 = \n");
	for(int j = 0; j<WIDTH*WIDTH; j++)
	{
		printf("%.1f ", R_h_minc[j]);
		if(j%5 == 4) printf("\n");
	}
}