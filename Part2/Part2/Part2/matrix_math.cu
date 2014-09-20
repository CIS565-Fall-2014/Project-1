#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
 
#define Matrix_Size 5
#define MatrixArray_Size 25


__global__ void MatrixAdd(float * Md, float * Nd, float * Pd){

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if(tx<Matrix_Size && ty<Matrix_Size){
		int index = ty*Matrix_Size + tx;
		Pd[index] = Md[index] + Nd[index];
	}
}


__global__ void MatrixSub(float * Md, float * Nd, float * Pd){

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if(tx<Matrix_Size && ty<Matrix_Size){
		int index = ty*Matrix_Size + tx;
		Pd[index] = Md[index] - Nd[index];
	}
}

__global__ void MatrixMul(float * Md, float * Nd, float * Pd){

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	if(tx<Matrix_Size && ty<Matrix_Size){
		
		float Pvalue = 0;
		for(int k =0; k<Matrix_Size; k++){
			float Melement = Md[ty*Matrix_Size + k];
			float Nelement = Nd[k*Matrix_Size + tx];
			Pvalue += Melement * Nelement;
		}

		Pd[ty*Matrix_Size + tx] = Pvalue;
	}
}

void addOnCPU(float * M, float * N, float * P){
	for(int i=0; i<MatrixArray_Size; i++){
		P[i] = M[i] + N[i];
		if(i>0 && i%5==0) printf("\n");
		printf("P[%d]=%f  ", i, P[i]);
	}

	printf("\n\n");
}

void subOnCPU(float * M, float * N, float * P){
	for(int i=0; i<MatrixArray_Size; i++){
		P[i] = M[i] - N[i];
		if(i>0 && i%5==0) printf("\n");
		printf("P[%d]=%f  ", i, P[i]);
	}

	printf("\n\n");
}


void mulOnCPU(float * M, float * N, float * P){
	for(int i = 0; i< Matrix_Size; i++){
		for(int j = 0; j<Matrix_Size; j++){


			float sum = 0;
			for(int k = 0; k<Matrix_Size; k++){
				float Melement = M[j*Matrix_Size + k];
				float Nelement = N[k*Matrix_Size + i];
				sum += Melement * Nelement;
			}

			P[i] = sum;
			if(i>0 && i%5==0) printf("\n");
			printf("P[%d]=%f  ", i, P[i]);

		}
	}

	printf("\n\n");
}

void main()
{
	printf("Welcome Karway!\n");


	float M [MatrixArray_Size];
	float N [MatrixArray_Size];

	for(int i =0; i<MatrixArray_Size; i++){
		M[i] = i;
		N[i] = i;
	}
	
	float P_add [MatrixArray_Size];
	float P_sub [MatrixArray_Size];
	float P_mul [MatrixArray_Size];

	float * Md;
	float * Nd;
	float * Pd;

	int size = MatrixArray_Size * sizeof(float);
 
	cudaMalloc(&Md, size); 
	cudaMalloc(&Nd, size); 
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice); 

	cudaMalloc(&Pd, size); 
	
	dim3 ThreadsPerBlock(Matrix_Size, Matrix_Size);
	dim3 numBlocks(1, 1);
	MatrixAdd<<<numBlocks, ThreadsPerBlock>>>(Md, Nd, Pd);
	cudaMemcpy(P_add, Pd, size, cudaMemcpyDeviceToHost); 


	MatrixSub<<<numBlocks, ThreadsPerBlock>>>(Md, Nd, Pd);
	cudaMemcpy(P_sub, Pd, size, cudaMemcpyDeviceToHost); 


	MatrixMul<<<numBlocks, ThreadsPerBlock>>>(Md, Nd, Pd);
	cudaMemcpy(P_mul, Pd, size, cudaMemcpyDeviceToHost); 

	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);

	//print Matrix Addition output
	printf("Output of Matrix Addition on GPU:\n");
	for(int i=0; i<MatrixArray_Size; i++){
		if(i>0 && i%5==0) printf("\n");
		printf("P[%d]=%f  ", i, P_add[i]);
	}

	printf("\n\n\n");


	//print Matrix Subtraction output
	printf("Output of Matrix Subtraction on GPU:\n");
	for(int i=0; i<MatrixArray_Size; i++){
		if(i>0 && i%5==0) printf("\n");
		printf("P[%d]=%f  ", i, P_sub[i]);
	}

	printf("\n\n\n");

	//print Matrix Multiplication output
	printf("Output of Matrix Multiplication on GPU:\n");
	for(int i=0; i<MatrixArray_Size; i++){
		if(i>0 && i%5==0) printf("\n");
		printf("P[%d]=%f  ", i, P_mul[i]);
	}
	
	printf("\n\n\n");


	/*Then Do Matrix Math On CPU*/

	printf("Then we do the matrix math on CPU:\n");

	printf("Matrix addition on CPU:\n");
	addOnCPU(M, N, P_add);
	printf("Matrix subtraction on CPU:\n");
	subOnCPU(M, N, P_sub);
	printf("Matrix multiplication on CPU:\n");
	mulOnCPU(M, N, P_mul);

	system("pause");
}