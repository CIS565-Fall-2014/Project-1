
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

__global__ void mat_add(float *A,float *B,float *C,int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float M = A[ty*width + tx];
	float N = B[ty*width + tx];
	C[ty*width + tx] = M+N;
}

__global__ void mat_sub(float *A,float *B,float *C,int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float M = A[ty*width + tx];
	float N = B[ty*width + tx];
	C[ty*width + tx] = M-N;
}

__global__ void mat_mult(float *A,float *B,float *C,int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float Cvalue = 0;
	for(int i=0;i<width;i++)
	{
		float M = A[ty*width + i];
		float N = B[i*width + tx];
		Cvalue += M*N;
	}

	C[ty*width + tx] = Cvalue;
}


//Single Thread
void s_mat_add(float *A,float *B,float *C,int width)
{
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<width;j++)
		{
			float M = A[i*width + j];
	        float N = B[i*width + j];
			C[i*width + j] = M+N;
		}
	}	
}

void s_mat_sub(float *A,float *B,float *C,int width)
{
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<width;j++)
		{
			float M = A[i*width + j];
	        float N = B[i*width + j];
			C[i*width + j] = M-N;
		}
	}	
}

void s_mat_mul(float *A,float *B,float *C,int width)
{
		
	float Cvalue = 0.0f;
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<width;j++)
		{
			for(int k=0;k<width;k++)
			{
				float M = A[i*width + k];
	            float N = B[j + k*width];
			    Cvalue += M*N;
			}

			C[i*width + j] = Cvalue;
			Cvalue = 0.0f;
		}	
	}	

	
}

int main()
{
	//Matrix for test
	float *A,*B,*P;
	int width = 5;
	A = new float[width*width];
    B = new float[width*width];
	P = new float[width*width];

    for(int i=0;i<width*width;i++)
	{
		A[i] = i;
		B[i] = i;
	}

	std::cout<<"Input1:"<<std::endl;
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<width;j++)
		{
			std::cout<<A[i*width + j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;

	std::cout<<"Input2:"<<std::endl;
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<width;j++)
		{
			std::cout<<B[i*width + j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;

	float *Ad,*Bd,*Pd;
	cudaMalloc((void**)&Ad,sizeof(float)*width*width);
	cudaMalloc((void**)&Bd,sizeof(float)*width*width);
	cudaMalloc((void**)&Pd,sizeof(float)*width*width);
	cudaMemcpy(Ad, A, width*width * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B, width*width * sizeof(int), cudaMemcpyHostToDevice);
	dim3 dimBlock(width,width);
	dim3 dimGrid(1,1);

	//Add
	mat_add<<<dimGrid,dimBlock>>> (Ad,Bd,Pd,width);
	cudaMemcpy(P, Pd, width*width * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout<<"Add:"<<std::endl;
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<width;j++)
		{
			std::cout<<P[i*width + j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
	//Sub
	mat_sub<<<dimGrid,dimBlock>>> (Ad,Bd,Pd,width);
	cudaMemcpy(P, Pd, width*width * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout<<"Sub:"<<std::endl;
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<width;j++)
		{
			std::cout<<P[i*width + j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;

	//Mul
	mat_mult<<<dimGrid,dimBlock>>> (Ad,Bd,Pd,width);
	cudaMemcpy(P, Pd, width*width * sizeof(int), cudaMemcpyDeviceToHost);

	std::cout<<"Multiple:"<<std::endl;
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<width;j++)
		{
			std::cout<<P[i*width + j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;


	//Add
	std::cout<<"Single Thread:"<<std::endl;
	s_mat_add(A,B,P,width);
	std::cout<<"Add:"<<std::endl;
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<width;j++)
		{
			std::cout<<P[i*width + j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;

	//Sub
    s_mat_sub(A,B,P,width);
	std::cout<<"Sub:"<<std::endl;
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<width;j++)
		{
			std::cout<<P[i*width + j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;


	//Mul
    s_mat_mul(A,B,P,width);
	std::cout<<"Mul:"<<std::endl;
	for(int i=0;i<width;i++)
	{
		for(int j=0;j<width;j++)
		{
			std::cout<<P[i*width + j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;

	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Pd);
    return 0;
}
