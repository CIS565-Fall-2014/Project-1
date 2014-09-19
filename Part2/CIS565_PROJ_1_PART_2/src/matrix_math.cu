#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h> 
#include <assert.h>

const int width = 5; 
int size = width * width * sizeof(float); 

dim3 threadsPerBlock(width, width);
dim3 numBlocks(1); 

//float ** Ad; 
//float ** Bd;
//
//float ** Cdadd; 
//float ** Cdsub; 
//float ** Cdmult; 
//
//float ** A; 
//float ** B; 
//
//float ** Cadd; 
//float ** Csub; 
//float ** Cmult; 


//// initialize matrices A and B
//void initMat() 
//{
//	int index = 0; 
//	for (int r = 0; r < width; r++) 
//	{
//		for (int c = 0; c < width; c++) 
//		{
//			A[r][c] = index; 
//			B[r][c] = index; 
//			index++; 
//		}
//	}
//}
//
//__global__ void mat_add_kernel(float** M[width][width], float** N[width][width], float** P[width][width]) 
//{
//	// 2D thread ID 
//	int tx = threadIdx.x; 
//	int ty = threadIdx.y; 
//
//	P[ty][tx] = M[ty][tx] + N[ty][tx]; 
//}
//
//void mat_add(float** M, float** N, float** P) 
//{
//	mat_add_kernel<<<numBlocks, threadsPerBlock>>>(M, N, P);
//}
//
//__global__ void mat_sub_kernel(float** M[width][width], float** N[width][width], float** P[width][width]) 
//{
//	// 2D thread ID 
//	int tx = threadIdx.x; 
//	int ty = threadIdx.y; 
//
//	P[ty][tx] = M[ty][tx] - N[ty][tx]; 
//}
//
//void mat_sub(float** M, float** N, float** P) 
//{
//	mat_sub_kernel<<<numBlocks, threadsPerBlock>>>(M, N, P);
//}
//
//__global__ void mat_mult_kernel(float** M, float** N, float** P) 
//{
//	// 2D thread ID 
//	int tx = threadIdx.x; 
//	int ty = threadIdx.y; 
//
//
//	float sum = 0.0f; 
//	for (int i = 0; i < width; i++) 
//	{
//		sum += (M[ty][i]*N[i][tx]); 
//	}
//	P[ty][tx] = sum; 
//}
//
//void mat_mult(float** M, float** N, float** P) 
//{
//	mat_mult_kernel<<<numBlocks, threadsPerBlock>>>(M, N, P);
//}
//
//
//
///* serial versions of:  
//	mat_add 
//	mat_sub 
//	mat_mult */ 
//
//void mat_add_serial(float** M, float** N, float** P) 
//{
//	for (int r = 0; r < width; r++) 
//	{
//		for (int c = 0; c < width; c++) 
//		{
//			P[r][c] = M[r][c] + N[r][c]; 
//		}
//	}
//}
//
//void mat_sub_serial(float** M, float** N, float** P) 
//{
//	for (int r = 0; r < width; r++) 
//	{
//		for (int c = 0; c < width; c++) 
//		{
//			P[r][c] = M[r][c] - N[r][c]; 
//		}
//	}
//}
//
//void mat_mult_serial(float** M, float** N, float** P) 
//{
//	for (int r = 0; r < width; r++) 
//	{
//		for (int c = 0; c < width; c++) 
//		{
//			float sum = 0.0f; 
//			for (int i = 0; i < width; i++) 
//			{
//				sum += (M[r][i]*N[i][c]); 
//			}
//			P[r][c] = sum; 
//		}
//	}
//}
//
//int main() 
//{
//	// set up memory and data transfer 
//	cudaMalloc((void**)&Ad, size); 
//	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice); 
//
//	cudaMalloc((void**)&Bd, size); 
//	cudaMemcpy((void**)&Bd, B, size, cudaMemcpyHostToDevice); 
//
//	cudaMalloc((void**)&Cdadd, size); 
//	cudaMalloc((void**)&Cdsub, size); 
//	cudaMalloc((void**)&Cdmult, size); 
//
//	// initialize matrices
//	initMat(); 
//
//	// add, subtract, and multiply matrices
//	mat_add(Ad, Bd, Cdadd); 
//	//mat_sub(Ad, Bd, Cdsub); 
//	//mat_mult(Ad, Bd, Cdmult); 
//
//	// serial add, subtract, multiply 
//	//mat_add_serial(A, B, Cadd); 
//	//mat_sub_serial(A, B, Csub); 
//	//mat_mult_serial(A, B, Cmult); 
//
//	// assert
//	assert(check_add); 
//	assert(check_sub); 
//	assert(check_mult); 
//
//	cudaMemcpy(Cadd, Cdadd, size, cudaMemcpyDeviceToHost); 
//	cudaMemcpy(Csub, Cdsub, size, cudaMemcpyDeviceToHost); 
//	cudaMemcpy(Cmult, Cdmult, size, cudaMemcpyDeviceToHost); 
//
//	int test = 101; 
//
//	cudaFree(Ad); 
//	cudaFree(Bd); 
//	cudaFree(Cdadd); 
//	cudaFree(Cdsub); 
//	cudaFree(Cdmult); 
//
//	return 0;
//}

/*float Ad[width][width]; 
float Bd[width][width];

float Cdadd[width][width]; 
float Cdsub[width][width]; 
float Cdmult[width][width]; */

//float A[width][width]; 
//float B[width][width]; 
//
//float Cadd[width][width]; 
//float Csub[width][width]; 
//float Cmult[width][width]; 
//
//float * A_ptr = &(A[0][0]); 
//float * B_ptr = &(B[0][0]); 
//float * Cadd_ptr = &(Cadd[0][0]); 
//float * Csub_ptr = &(Csub[0][0]); 
//float * Cmult_ptr = &(Cmult[0][0]); 
//
//float * Ad; 
//float * Bd; 
//float * Cdadd; 
//float * Cdsub; 
//float * Cdmult; 
//
//
//// initialize matrices A and B
//void initMat() 
//{
//	int index = 0; 
//	for (int r = 0; r < width; r++) 
//	{
//		for (int c = 0; c < width; c++) 
//		{
//			A[r][c] = index; 
//			B[r][c] = index; 
//			index++; 
//		}
//	}
//}
//
//__global__ void mat_add_kernel(float * M, float * N, float * P) 
//{
//	// 2D thread ID 
//	int tx = threadIdx.x; 
//	int ty = threadIdx.y; 
//	P[ty*width+tx] = M[ty*width+tx] + N[ty*width+tx]; 
//}
//
//void mat_add(float * M, float * N, float * P) 
//{
//	mat_add_kernel<<<numBlocks, threadsPerBlock>>>(M, N, P);
//	fprintf(stderr, "%f\n", M[0]); 
//	fprintf(stderr, "%f\n", M[1]);
//	fprintf(stderr, "%f\n", M[2]);
//	fprintf(stderr, "%f\n", M[3]);
//	fprintf(stderr, "%f\n", M[4]);
//}
//
//__global__ void mat_sub_kernel(float * M, float * N, float * P) 
//{
//	// 2D thread ID 
//	int tx = threadIdx.x; 
//	int ty = threadIdx.y; 
//
//	P[ty*width+tx] = M[ty*width+tx] - N[ty*width+tx]; 
//}
//
//void mat_sub(float * M, float * N, float * P) 
//{
//	mat_sub_kernel<<<numBlocks, threadsPerBlock>>>(M, N, P);
//}
//
//__global__ void mat_mult_kernel(float * M, float * N, float * P) 
//{
//	// 2D thread ID 
//	int tx = threadIdx.x; 
//	int ty = threadIdx.y; 
//
//
//	float sum = 0.0f; 
//	for (int i = 0; i < width; i++) 
//	{
//		sum += (M[ty*width+i]*N[i*width+tx]); 
//	}
//	P[ty*width+tx] = sum; 
//}
//
//void mat_mult(float * M, float * N, float * P) 
//{
//	mat_mult_kernel<<<numBlocks, threadsPerBlock>>>(M, N, P);
//}
//
//
//
///* serial versions of:  
//	mat_add 
//	mat_sub 
//	mat_mult */ 
//
//void mat_add_serial(float M[width][width], float N[width][width], float P[width][width]) 
//{
//	for (int r = 0; r < width; r++) 
//	{
//		for (int c = 0; c < width; c++) 
//		{
//			P[r][c] = M[r][c] + N[r][c]; 
//		}
//	}
//}
//
//void mat_sub_serial(float M[width][width], float N[width][width], float P[width][width]) 
//{
//	for (int r = 0; r < width; r++) 
//	{
//		for (int c = 0; c < width; c++) 
//		{
//			P[r][c] = M[r][c] - N[r][c]; 
//		}
//	}
//}
//
//void mat_mult_serial(float M[width][width], float N[width][width], float P[width][width]) 
//{
//	for (int r = 0; r < width; r++) 
//	{
//		for (int c = 0; c < width; c++) 
//		{
//			float sum = 0.0f; 
//			for (int i = 0; i < width; i++) 
//			{
//				sum += (M[r][i]*N[i][c]); 
//			}
//			P[r][c] = sum; 
//		}
//	}
//}
//
//int main() 
//{
//	// initialize matrices on CPU
//	initMat(); 
//
//	// set up memory and data transfer 
//	cudaMalloc((void**)&Ad, size); 
//	cudaMemcpy(Ad, A_ptr, size, cudaMemcpyHostToDevice); 
//
//	cudaMalloc((void**)&Bd, size); 
//	cudaMemcpy(Bd, B_ptr, size, cudaMemcpyHostToDevice); 
//
//	cudaMalloc((void**)&Cdadd, size); 
//	cudaMalloc((void**)&Cdsub, size); 
//	cudaMalloc((void**)&Cdmult, size); 
//
//	fprintf(stderr, "%f\n", Ad[1]); 
//
//	
//
//	// add, subtract, and multiply matrices
//	mat_add(Ad, Bd, Cdadd); 
//	//mat_sub(Ad, Bd, Cdsub); 
//	//mat_mult(Ad, Bd, Cdmult); 
//
//	// serial add, subtract, multiply 
//	//mat_add_serial(A, B, Cadd); 
//	//mat_sub_serial(A, B, Csub); 
//	//mat_mult_serial(A, B, Cmult); 
//
//	// assert
//	//assert(check_add); 
//	//assert(check_sub); 
//	//assert(check_mult); 
//
//	cudaMemcpy(Cadd, Cdadd, size, cudaMemcpyDeviceToHost); 
//	//cudaMemcpy(Csub, Cdsub, size, cudaMemcpyDeviceToHost); 
//	//cudaMemcpy(Cmult, Cdmult, size, cudaMemcpyDeviceToHost); 
//
//	cudaFree(Ad); 
//	cudaFree(Bd); 
//	cudaFree(Cdadd); 
//	cudaFree(Cdsub); 
//	cudaFree(Cdmult); 
//
//	return 0;
//}

float * Ad; 
float * Bd;

float * Cdadd; 
float * Cdsub; 
float * Cdmult; 

float * A; 
float * B; 

float * Cadd; 
float * Csub; 
float * Cmult; 


// initialize matrices A and B
void initMat() 
{
	int index = 0; 
	for (int r = 0; r < width; r++) 
	{
		for (int c = 0; c < width; c++) 
		{
			A[r*width+c] = index; 
			B[r*width+c] = index; 
			index++; 
		}
	}
}

__global__ void mat_add_kernel(float* M, float* N, float* P) 
{
	// 2D thread ID 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 

	P[ty*width+tx] = M[ty*width+tx] + N[ty*width+tx]; 
}

void mat_add(float* M, float* N, float* P) 
{
	mat_add_kernel<<<numBlocks, threadsPerBlock>>>(M, N, P);
	//fprintf(stderr, "%f\n", M[0]); 
}

__global__ void mat_sub_kernel(float* M, float* N, float* P) 
{
	// 2D thread ID 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 

	P[ty*width+tx] = M[ty*width+tx] - N[ty*width+tx]; 
}

void mat_sub(float* M, float* N, float* P) 
{
	mat_sub_kernel<<<numBlocks, threadsPerBlock>>>(M, N, P);
}

__global__ void mat_mult_kernel(float* M, float* N, float* P) 
{
	// 2D thread ID 
	int tx = threadIdx.x; 
	int ty = threadIdx.y; 


	float sum = 0.0f; 
	for (int i = 0; i < width; i++) 
	{
		sum += (M[ty*width+i]*N[i*width+tx]); 
	}
	P[ty*width+tx] = sum; 
}

void mat_mult(float* M, float* N, float* P) 
{
	mat_mult_kernel<<<numBlocks, threadsPerBlock>>>(M, N, P);
}



// serial versions of mat_add, mat_sub, and mat_mult  

void mat_add_serial(float* M, float* N, float* P) 
{
	for (int r = 0; r < width; r++) 
	{
		for (int c = 0; c < width; c++) 
		{
			P[r*width+c] = M[r*width+c] + N[r*width+c]; 
		}
	}
}

void mat_sub_serial(float* M, float* N, float* P) 
{
	for (int r = 0; r < width; r++) 
	{
		for (int c = 0; c < width; c++) 
		{
			P[r*width+c] = M[r*width+c] - N[r*width+c]; 
		}
	}
}

void mat_mult_serial(float* M, float* N, float* P) 
{
	for (int r = 0; r < width; r++) 
	{
		for (int c = 0; c < width; c++) 
		{
			float sum = 0.0f; 
			for (int i = 0; i < width; i++) 
			{
				sum += (M[r*width+i]*N[i*width+c]); 
			}
			P[r*width+c] = sum; 
		}
	}
}

bool check_add(float* M)
{
	float ep = 0.0000001f; 
	float M_add[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48}; 
	for (int i = 0; i < width*width; i++) 
	{
		if (M[i] - M_add[i] > ep) return false;
	}
	return true; 
}

bool check_sub(float* M)
{
	float ep = 0.0000001f; 
	for (int i = 0; i < width*width; i++) 
	{
		if (M[i] > ep) return false;
	}
	return true; 
}

bool check_mult(float* M)
{
	float ep = 0.0000001f; 
	float M_add[] = {150, 160, 170, 180, 190, 400, 435, 470, 505, 540, 650, 710, 770, 830, 890, 900, 985, 1070, 1155, 1240, 1150, 1260, 1370, 1480, 1590}; 
	for (int i = 0; i < width*width; i++) 
	{
		if (M[i] - M_add[i] > ep) return false;
	}
	return true; 
}

int main() 
{
	// set up memory and data transfer 
	A = new float[width*width];  
	B = new float[width*width];  
	Cadd = new float[width*width];  
	Csub = new float[width*width];  
	Cmult = new float[width*width];  

	cudaMalloc((void**)&Ad, size); 
	cudaMemcpy(&Ad, &A, size, cudaMemcpyHostToDevice); 

	cudaMalloc((void**)&Bd, size); 
	cudaMemcpy(&Bd, &B, size, cudaMemcpyHostToDevice); 

	cudaMalloc((void**)&Cdadd, size); 
	cudaMalloc((void**)&Cdsub, size); 
	cudaMalloc((void**)&Cdmult, size); 

	// initialize matrices
	initMat(); 

	// add, subtract, and multiply matrices
	mat_add(Ad, Bd, Cdadd); 
	//mat_sub(Ad, Bd, Cdsub); 
	//mat_mult(Ad, Bd, Cdmult); 

	// serial add, subtract, multiply 
	//mat_add_serial(A, B, Cadd); 
	//mat_sub_serial(A, B, Csub); 
	//mat_mult_serial(A, B, Cmult); 

	// assert
	assert(check_add); 
	assert(check_sub); 
	assert(check_mult); 

	cudaMemcpy(Cadd, Cdadd, size, cudaMemcpyDeviceToHost); 
	//cudaMemcpy(Csub, Cdsub, size, cudaMemcpyDeviceToHost); 
	//cudaMemcpy(Cmult, Cdmult, size, cudaMemcpyDeviceToHost); 

	cudaFree(Ad); 
	cudaFree(Bd); 
	cudaFree(Cdadd); 
	cudaFree(Cdsub); 
	cudaFree(Cdmult); 

	return 0;
}