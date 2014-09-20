#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include <iostream>

using namespace std;

//Initialize memory, update some globals
void initCuda(int N)
{   
	cudaThreadSynchronize();
}

__global__ void mat_add(int n, float * A, float * B, float * out){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < n){
	  out[index] = A[index] + B[index];
  }
}

__global__ void mat_sub(int n, float * A, float * B, float * out){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (index < n){
	  out[index] = A[index] - B[index];
  }
}

__global__ void mat_mult(int n, float * A, float * B, float * out){
  int row = (blockIdx.y * blockDim.y) + threadIdx.y;
  int col = (blockIdx.x * blockDim.x) + threadIdx.x;

  int singleDim = sqrt(float(n)); //since we can assume n is square

  if (row < singleDim && col < singleDim){
	  float outVal = 0;
	  for (int i=0; i<singleDim; i+=1){
		 outVal += A[row * singleDim + i] * B[i * singleDim + col];
	  }
	  out[row * singleDim + col] = outVal;

  }
}

void mat_add_serial(int n, float * A, float * B, float * out){
	for (int i=0; i<n; i+=1){
		out[i] = A[i] + B[i];
	}
}

void mat_sub_serial(int n, float * A, float * B, float * out){
	for (int i=0; i<n; i+=1){
		out[i] = A[i] - B[i];
	}
}

void mat_mult_serial(int n, float * A, float * B, float * out){
	int singleDim = sqrt(float(n));
	for (int row=0; row<singleDim; row+=1){
		for (int col=0; col<singleDim; col+=1){
			float outVal = 0;
			for (int i=0; i<singleDim; i+=1){
				outVal += A[row * singleDim + i] * B[i * singleDim + col];
			}
			out[row * singleDim + col] = outVal;
		}
	}
}

void printMat (float* toPrint, int dim){
	int index = 0;
	for (int i=0; i<dim; i+=1){
		for (int j=0; j<dim; j+=1){
			cout<<toPrint[index]<<",";
			index += 1;
		}
		cout<<endl;
	}
	cout<<endl;
}

int main(){

	float * myCPUArray1 = new float[25];
	float * myCPUArray2 = new float[25];
	float * outCPUArray = new float[25];

	for (int i=0; i<25; i+=1){
		myCPUArray1[i] = i;
		myCPUArray2[i] = i;
		outCPUArray[i] = i;
	}

	printMat (myCPUArray1, 5);
	printMat (myCPUArray2, 5);

	float * myGPUArray1;
	float * myGPUArray2;

	cudaMalloc ((void**)&myGPUArray1, 25*sizeof(float));
	cudaMemcpy( myGPUArray1, myCPUArray1, 25*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc ((void**)&myGPUArray2, 25*sizeof(float));
	cudaMemcpy( myGPUArray2, myCPUArray1, 25*sizeof(float), cudaMemcpyHostToDevice);
	
	float * outGPUArray;
	cudaMalloc ((void**)&outGPUArray, 25*sizeof(float));
	cudaMemcpy( outGPUArray, myCPUArray1, 25*sizeof(float), cudaMemcpyHostToDevice);

	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGridSingle(25);
	dim3 fullBlocksPerGridDouble(5, 5);

	mat_add<<<fullBlocksPerGridSingle, threadsPerBlock>>> (25, myGPUArray1, myGPUArray2, outGPUArray);
	cudaMemcpy( outCPUArray, outGPUArray, 25*sizeof(float), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	printMat (outCPUArray, 5);

	mat_sub<<<fullBlocksPerGridSingle, threadsPerBlock>>> (25, myGPUArray1, myGPUArray2, outGPUArray);
	cudaMemcpy( outCPUArray, outGPUArray, 25*sizeof(float), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	printMat (outCPUArray, 5);

	cudaMemcpy( outGPUArray, myCPUArray2, 25*sizeof(float), cudaMemcpyHostToDevice);
	mat_mult<<<fullBlocksPerGridDouble, threadsPerBlock>>> (25, myGPUArray1, myGPUArray2, outGPUArray);
	cudaMemcpy( outCPUArray, outGPUArray, 25*sizeof(float), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	printMat (outCPUArray, 5);

	delete [] myCPUArray1;
	delete [] myCPUArray2;
	cudaFree (myGPUArray1);
	cudaFree (myGPUArray2);
	cudaFree (outGPUArray);

	std::cin.ignore ();
	return 0;
}
