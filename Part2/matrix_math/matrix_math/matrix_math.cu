#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<stdio.h>
#include<stdlib.h>

void CPU_Matrix_Add(const int* A,const int* B, int* C, const int& size){
	for(int i=0;i<size;i++)
		C[i]=A[i]+B[i];
}

void CPU_Matrix_Minus(const int* A, const int* B, int* C, const int& size){
	for(int i=0;i<size;i++)
		C[i]=A[i]-B[i];
}

void CPU_Matrix_Multiply(const int* A,const int* B,int* C, const int& size){
	for(int i=0;i<size;i++){
		//int tmp=0;
		for(int j=0;j<size;j++){
			for(int k=0;k<size;k++)
				C[i*size+j]+=A[i*size+k]*B[k*size+j];
		}

	}
}


__global__ void GPU_Matrix_Add(const int* A,const int* B, int* C, const int& size){
	int id=threadIdx.x;
	C[id]=A[id]+B[id];
}
__global__ void GPU_Matrix_Minus(const int* A, const int* B, int* C, const int& size){
	int id=threadIdx.x;
	C[id]=A[id]-B[id];
}
__global__ void GPU_Matrix_Multiply(const int* A,const int* B,int* C, const int& size){
	int id=threadIdx.x;
	int i=id/size;
	int j=id%size;

	for(int k=0;k<size;k++)
		C[id]=A[i*size+k]*B[k*size+j];
}

void print(const int* m,const int& size){
	for(int i=0;i<size;i++){
		for(int j=0;j<size;j++)
			printf("%d ",m[i*size+j]);
		printf("\n");
	}
	printf("\n");
}

int main()
{
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

	const int size=5;
	const int matrixSize=size*size;
	const int a[matrixSize]={	1,0,0,0,0,
							0,1,0,0,0,
							0,0,1,0,0,
							0,0,0,1,0,
							0,0,0,0,1};

	const int b[matrixSize]={	1,2,3,4,5,
							0,1,0,0,0,
							0,0,1,0,0,
							0,0,0,1,0,
							0,0,0,0,1};
	int c[matrixSize]={0};
	int d[matrixSize]={0};
	int *ad=0,*bd=0,*cd=0,*dd=0;
	cudaError_t cudaStatus;
	
	
	cudaStatus=cudaSetDevice(0);
	//cudaStatus = cudaDeviceReset();
	if(!cudaStatus) printf("-1");

	cudaStatus=cudaMalloc((void**)&ad,matrixSize*sizeof(int));
	cudaStatus=cudaMalloc((void**)&bd,matrixSize*sizeof(int));
	cudaStatus=cudaMalloc((void**)&cd,matrixSize*sizeof(int));
	cudaStatus=cudaMalloc((void**)&dd,matrixSize*sizeof(int));
	if(!cudaStatus) printf("-2");
	cudaStatus=cudaMemcpy(ad,a,matrixSize*sizeof(int),cudaMemcpyHostToDevice);
	cudaStatus=cudaMemcpy(bd,b,matrixSize*sizeof(int),cudaMemcpyHostToDevice);
	cudaStatus=cudaMemcpy(cd,c,matrixSize*sizeof(int),cudaMemcpyHostToDevice);
	cudaStatus=cudaMemcpy(dd,d,matrixSize*sizeof(int),cudaMemcpyHostToDevice);
	if(!cudaStatus) printf("-3");
	CPU_Matrix_Add(a,b,c,matrixSize);
	print(c,size);
	CPU_Matrix_Minus(a,b,c,matrixSize);
	print(c,size);
	CPU_Matrix_Multiply(a,b,d,size);
	print(d,size);

	GPU_Matrix_Add<<<1,matrixSize>>>(ad,bd,cd,matrixSize);
	cudaMemcpy(c,cd,matrixSize*sizeof(int),cudaMemcpyDeviceToHost);
	print(c,size);
	GPU_Matrix_Minus<<<1,matrixSize>>>(ad,bd,cd,matrixSize);
	cudaMemcpy(c,cd,matrixSize*sizeof(int),cudaMemcpyDeviceToHost);
	print(c,size);
	GPU_Matrix_Multiply<<<1,matrixSize>>>(ad,bd,dd,size);
	cudaMemcpy(dd,d,matrixSize*sizeof(int),cudaMemcpyDeviceToHost);
	print(d,size);

	cudaFree(ad);
	cudaFree(bd);
	cudaFree(cd);
	cudaFree(dd);
	cudaStatus = cudaDeviceReset();
    // Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}

    return 0;
}
