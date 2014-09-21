
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <cuda.h>
#include <cmath>


//initialize 2 5 x 5 matrices represented as an array of floats 
//each of the entry is equal to its position (i.e. A_00 = 0, A_01 = 1, A_44 = 24)

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
//enumeration for matrix function
#define MATRIX_ADD        0
#define MATRIX_SUB        1
#define MATRIX_MUL        2

const int Width = 5;    //matrix dimension
const int TILE_WIDTH = 2;   //tile size
const int size = Width*Width*sizeof(float);   //memory size of a matrix

float * Md,* Nd,* Pd;



void checkCUDAError(const char *msg, int line = -1)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        if( line >= 0 )
        {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
        exit(EXIT_FAILURE); 
    }
} 





__global__ void parallel_matrix_add(float * Md, float * Nd, float * Ad)
{
  /*int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  float Avalue = 0;
  for (int k = 0; k < width; ++k)
    Avalue += Md[Row * width + k] * Nd[k * width + Col];

  Ad[Row * width + Col] = Avalue;*/

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	Ad[ty * Width + tx] = Md[ty * Width + tx ] + Nd[ty * Width + tx];
}

__global__ void parallel_matrix_sub(float * Md, float * Nd, float * Ad)
{
  /*int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  float Avalue = 0;
  for (int k = 0; k < width; ++k)
    Avalue += Md[Row * width + k] * Nd[k * width + Col];

  Ad[Row * width + Col] = Avalue;*/

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	Ad[ty * Width + tx] = Md[ty * Width + tx ] - Nd[ty * Width + tx];
}


__global__ void parallel_matrix_mul( float* Md, float* Nd, float* Pd)
{
  /*int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  float Pvalue = 0;
  for (int k = 0; k < Width; ++k)
    Pvalue += Md[Row * Width + k] * Nd[k * Width + Col];

  Pd[Row * Width + Col] = Pvalue;*/
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float Pvalue = 0;
	for(int k=0; k<Width; ++k){
		Pvalue += Md[ty * Width + k] * Nd[k * Width + tx];
	}
	Pd[ty * Width + tx] = Pvalue;
}







//CUDA Data Transfer & parallel versioin
void MatrixOnDevice(float *M, float *N, float *P, int width, int operation)
{
	//load Md and Nd to device memory
	cudaMalloc((void**)&Md, size);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("Kernel failed!");

	cudaMalloc((void**)&Nd, size);
	checkCUDAErrorWithLine("Kernel failed!");
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("Kernel failed!");

	//alocate Pon the device
	cudaMalloc((void**)&Pd, size);
	checkCUDAErrorWithLine("Kernel failed!");

	//kernel invocation
	//dim3 dimGrid(Width/ TILE_WIDTH, Width/ TILE_WIDTH);
	//dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(1,1);
	dim3 dimBlock(Width, Width);

	switch(operation){
	case MATRIX_ADD:
		parallel_matrix_add<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
		break;
	case MATRIX_SUB:
		parallel_matrix_sub<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
		break;
	case MATRIX_MUL:
		parallel_matrix_mul<<<dimGrid, dimBlock>>>(Md, Nd, Pd);
		break;
	}


	//read P from device
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("Kernel failed!");

	//free memory
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);

}


//serial version
void MatrixOnHost(float *M, float *N, float *P, int width, int operation)
{
	switch(operation){
	case MATRIX_ADD:
		 for( int i = 0; i< width; i++){
			 for(int j=0; j<width; j++){
				 P[i*width+j] = N[i*width+j] + N[i*width+j];
			 }
		 }
		break;
	case MATRIX_SUB:
		 for( int i = 0; i< width; i++){
			 for(int j=0; j<width; j++){
				 P[i*width+j] = N[i*width+j] - N[i*width+j];
			 }
		 }
		break;
	case MATRIX_MUL:
		 for( int i = 0; i< width; i++){
			 for(int j=0; j<width; j++){
				float sum = 0;
				for (int k=0; k<width; k++){
					sum += M[i*width+k] *N[k*width+j];
				}
				P[i*width+j] = sum;
			 }
		 }
		break;
	}
}

int main(int argc, char** argv)
{
	float * entry;   //pointer to the entry
	entry = new float[Width*Width];
	for(int k=0; k<Width*Width; k++){
		entry[k] = k;
		//std::cout<<k<<std::endl;
	}
	/*std::cout<<"************Initial: ************"<<std::endl;
	for(int i=0; i<Width; i++){
		std::stringstream ss;
		for(int j=0; j<Width; j++){
			ss << entry[i*Width+j] <<" ";
		}
		std::cout<<ss.str()<<std::endl;
	}*/

	float * result;   //pointer to the result
	result = new float[Width*Width];

	//host call for addition
	MatrixOnDevice(entry, entry, result, Width, MATRIX_ADD);
	std::cout<<"************Deivce Add Result: ************"<<std::endl;
	for(int i=0; i<Width; i++){
		std::stringstream ss;
		for(int j=0; j<Width; j++){
			ss << result[i*Width+j] <<" ";
		}
		std::cout<<ss.str()<<std::endl;
	}

	//host call for subtraction
	MatrixOnDevice(entry, entry, result, Width, MATRIX_SUB);
	std::cout<<"************Deivce Sub Result: ************"<<std::endl;
	for(int i=0; i<Width; i++){
		std::stringstream ss;
		for(int j=0; j<Width; j++){
			ss << result[i*Width+j] <<" ";
		}
		std::cout<<ss.str()<<std::endl;
	}

	//host call for multiply
	MatrixOnDevice(entry, entry, result, Width, MATRIX_MUL);
	std::cout<<"************Deivce Mul Result: ************"<<std::endl;
	for(int i=0; i<Width; i++){
		std::stringstream ss;
		for(int j=0; j<Width; j++){
			ss << result[i*Width+j] <<" ";
		}
		std::cout<<ss.str()<<std::endl;
	}



	MatrixOnHost(entry, entry, result, Width, MATRIX_ADD);
	std::cout<<"************Host Add Result: ************"<<std::endl;
	for(int i=0; i<Width; i++){
		std::stringstream ss;
		for(int j=0; j<Width; j++){
			ss << result[i*Width+j] <<" ";
		}
		std::cout<<ss.str()<<std::endl;
	}

	MatrixOnHost(entry, entry, result, Width, MATRIX_SUB);
	std::cout<<"************Host Sub Result: ************"<<std::endl;
	for(int i=0; i<Width; i++){
		std::stringstream ss;
		for(int j=0; j<Width; j++){
			ss << result[i*Width+j] <<" ";
		}
		std::cout<<ss.str()<<std::endl;
	}


	MatrixOnHost(entry, entry, result, Width, MATRIX_MUL);
	std::cout<<"************Host Mul Result: ************"<<std::endl;
	for(int i=0; i<Width; i++){
		std::stringstream ss;
		for(int j=0; j<Width; j++){
			ss << result[i*Width+j] <<" ";
		}
		std::cout<<ss.str()<<std::endl;
	}
	return 0;
}