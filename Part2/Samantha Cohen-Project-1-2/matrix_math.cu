#include <cuda_runtime.h>
#include <stdio.h>

void CPU_mat_sub(float* M, float* N, float* P, int width) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			P[i*width+j] = M[i*width+j] - N[i*width+j];
		}
	}
}

void CPU_mat_add(float* M, float* N, float* P, int width) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			P[i*width+j] = M[i*width+j] + N[i*width+j];
		}
	}
}

void CPU_mat_mult(float* M, float* N, float* P, int width) {
	P = new float[25];
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			P[i*width+j] += M[i*width+j] * N[j*width+i];
		}
	}
}

__global__ void mat_sub_kernel(float* Md, float* Nd, float* Pd, int width) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float Pvalue = Md[ty*width+tx] - Nd[ty*width+tx];

	Pd[ty*width+tx] = Pvalue;
}

void mat_sub(float* M, float* N, float* P, int width) {
	int size = width * width * sizeof(float);
	float *Md;
	float *Nd;
	float *Pd;
	//Matrix 1
	cudaMalloc((void**)&Md, size);
	cudaMemcpy(Md,M,size,cudaMemcpyHostToDevice);
	//Matrix 2
	cudaMalloc((void**)&Nd, size);
	cudaMemcpy(Nd,N,size,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Pd, size);

	dim3 dimBlock(width, width);
	dim3 dimGrid(1, 1);

	mat_sub_kernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);

	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);
}

__global__ void mat_add_kernel(float* Md, float* Nd, float* Pd, int width) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float Pvalue = Md[ty*width+tx] + Nd[ty*width+tx];

	Pd[ty*width+tx] = Pvalue;
}

void mat_add(float* M, float* N, float* P, int width) {
	int size = width * width * sizeof(float);
	float *Md;
	float *Nd;
	float *Pd;
	//Matrix 1
	cudaMalloc((void**)&Md, size);
	cudaMemcpy(Md,M,size,cudaMemcpyHostToDevice);
	//Matrix 2
	cudaMalloc((void**)&Nd, size);
	cudaMemcpy(Nd,N,size,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Pd, size);

	dim3 dimBlock(width, width);
	dim3 dimGrid(1, 1);

	mat_add_kernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);

	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);
}
__global__ void mat_mult_kernel(float* Md, float* Nd, float* Pd, int width) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float Pvalue = 0;

	for (int k = 0; k < width; ++k) {
		float Mdelement = Md[ty * width + k];
		float Ndelement = Nd[k * width + tx];
		Pvalue += Mdelement * Ndelement;
	}

	Pd[ty*width+tx] = Pvalue;
}

void mat_mult(float* M, float* N, float* P, int width) {
	int size = width * width * sizeof(float);
	float *Md;
	float *Nd;
	float *Pd;
	//Matrix 1
	cudaMalloc((void**)&Md, size);
	cudaMemcpy(Md,M,size,cudaMemcpyHostToDevice);
	//Matrix 2
	cudaMalloc((void**)&Nd, size);
	cudaMemcpy(Nd,N,size,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Pd, size);

	dim3 dimBlock(width, width);
	dim3 dimGrid(1, 1);

	mat_mult_kernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, width);

	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);
}


int main () { //You need to fix linker issues so that your methods will be recognized, but in the meantime you can check out individual functions by putting them in main
	float * M = new float[25];
	M[0] = 2; M[1] = 1; M[2] = 1; M[3] = 1; M[4] = 1;
	M[5] = 1; M[6] = 3; M[7] = 1; M[8] = 1; M[9] = 1;
	M[10] = 1; M[11] = 1; M[12] = 4; M[13] = 1; M[14] = 1;
	M[15] = 1; M[16] = 1; M[17] = 1; M[18] = 5; M[19] = 1;
	M[20] = 1; M[21] = 1; M[22] = 1; M[23] = 1; M[24] = 6;
	float * N = new float[25];
	N[0] = 2; N[1] = 2; N[2] = 2; N[3] = 2; N[4] = 2;
	N[5] = 2; N[6] = 2; N[7] = 2; N[8] = 2; N[9] = 2;
	N[10] = 2; N[11] = 2; N[12] = 2; N[13] = 2; N[14] = 2;
	N[15] = 2; N[16] = 2; N[17] = 2; N[18] = 2; N[19] = 2;
	N[20] = 2; N[21] = 2; N[22] = 2; N[23] = 2; N[24] = 2;
	float * P = new float[25];
	P[0] = 0; P[1] = 0; P[2] = 0; P[3] = 0; P[4] = 0;
	P[5] = 0; P[6] = 0; P[7] = 0; P[8] = 0; P[9] = 0;
	P[10] = 0; P[11] = 0; P[12] = 0; P[13] = 0; P[14] = 0;
	P[15] = 0; P[16] = 0; P[17] = 0; P[18] = 0; P[19] = 0;
	P[20] = 0; P[21] = 0; P[22] = 0; P[23] = 0; P[24] = 0;

	printf("Matrix M:\n");
	printf("%f ",M[0]); printf("%f ",M[1]); printf("%f ",M[2]); printf("%f ",M[3]); printf("%f ",M[4]); printf("\n");
	printf("%f ",M[5]); printf("%f ",M[6]); printf("%f ",M[7]); printf("%f ",M[8]); printf("%f ",M[9]); printf("\n");
	printf("%f ",M[10]); printf("%f ",M[11]); printf("%f ",M[12]); printf("%f ",M[13]); printf("%f ",M[14]); printf("\n");
	printf("%f ",M[15]); printf("%f ",M[16]); printf("%f ",M[17]); printf("%f ",M[18]); printf("%f ",M[19]); printf("\n");
	printf("%f ",M[20]); printf("%f ",M[21]); printf("%f ",M[22]); printf("%f ",M[23]); printf("%f ",M[24]); printf("\n");
	printf("\n");

	printf("Matrix N:\n");
	printf("%f ",N[0]); printf("%f ",N[1]); printf("%f ",N[2]); printf("%f ",N[3]); printf("%f ",N[4]); printf("\n");
	printf("%f ",N[5]); printf("%f ",N[6]); printf("%f ",N[7]); printf("%f ",N[8]); printf("%f ",N[9]); printf("\n");
	printf("%f ",N[10]); printf("%f ",N[11]); printf("%f ",N[12]); printf("%f ",N[13]); printf("%f ",N[14]); printf("\n");
	printf("%f ",N[15]); printf("%f ",N[16]); printf("%f ",N[17]); printf("%f ",N[18]); printf("%f ",N[19]); printf("\n");
	printf("%f ",N[20]); printf("%f ",N[21]); printf("%f ",N[22]); printf("%f ",N[23]); printf("%f ",N[24]); printf("\n");
	printf("\n");

	int size = 5*5*sizeof(float);
	int numBlocks = 1;
	dim3 threadsPerBlock(5,5);

	mat_add(M,N,P,5);

	printf("GPU Matrix Add:\n");
	printf("%f ",P[0]); printf("%f ",P[1]); printf("%f ",P[2]); printf("%f ",P[3]); printf("%f ",P[4]); printf("\n");
	printf("%f ",P[5]); printf("%f ",P[6]); printf("%f ",P[7]); printf("%f ",P[8]); printf("%f ",P[9]); printf("\n");
	printf("%f ",P[10]); printf("%f ",P[11]); printf("%f ",P[12]); printf("%f ",P[13]); printf("%f ",P[14]); printf("\n");
	printf("%f ",P[15]); printf("%f ",P[16]); printf("%f ",P[17]); printf("%f ",P[18]); printf("%f ",P[19]); printf("\n");
	printf("%f ",P[20]); printf("%f ",P[21]); printf("%f ",P[22]); printf("%f ",P[23]); printf("%f ",P[24]); printf("\n");
	printf("\n");

	CPU_mat_add(M,N,P,5);

	printf("CPU Matrix Add:\n");
	printf("%f ",P[0]); printf("%f ",P[1]); printf("%f ",P[2]); printf("%f ",P[3]); printf("%f ",P[4]); printf("\n");
	printf("%f ",P[5]); printf("%f ",P[6]); printf("%f ",P[7]); printf("%f ",P[8]); printf("%f ",P[9]); printf("\n");
	printf("%f ",P[10]); printf("%f ",P[11]); printf("%f ",P[12]); printf("%f ",P[13]); printf("%f ",P[14]); printf("\n");
	printf("%f ",P[15]); printf("%f ",P[16]); printf("%f ",P[17]); printf("%f ",P[18]); printf("%f ",P[19]); printf("\n");
	printf("%f ",P[20]); printf("%f ",P[21]); printf("%f ",P[22]); printf("%f ",P[23]); printf("%f ",P[24]); printf("\n");
	printf("\n");

	mat_sub(M,N,P,5);

	printf("GPU Matrix Subtract:\n");
	printf("%f ",P[0]); printf("%f ",P[1]); printf("%f ",P[2]); printf("%f ",P[3]); printf("%f ",P[4]); printf("\n");
	printf("%f ",P[5]); printf("%f ",P[6]); printf("%f ",P[7]); printf("%f ",P[8]); printf("%f ",P[9]); printf("\n");
	printf("%f ",P[10]); printf("%f ",P[11]); printf("%f ",P[12]); printf("%f ",P[13]); printf("%f ",P[14]); printf("\n");
	printf("%f ",P[15]); printf("%f ",P[16]); printf("%f ",P[17]); printf("%f ",P[18]); printf("%f ",P[19]); printf("\n");
	printf("%f ",P[20]); printf("%f ",P[21]); printf("%f ",P[22]); printf("%f ",P[23]); printf("%f ",P[24]); printf("\n");
	printf("\n");

	CPU_mat_sub(M,N,P,5);

	printf("CPU Matrix Subtract:\n");
	printf("%f ",P[0]); printf("%f ",P[1]); printf("%f ",P[2]); printf("%f ",P[3]); printf("%f ",P[4]); printf("\n");
	printf("%f ",P[5]); printf("%f ",P[6]); printf("%f ",P[7]); printf("%f ",P[8]); printf("%f ",P[9]); printf("\n");
	printf("%f ",P[10]); printf("%f ",P[11]); printf("%f ",P[12]); printf("%f ",P[13]); printf("%f ",P[14]); printf("\n");
	printf("%f ",P[15]); printf("%f ",P[16]); printf("%f ",P[17]); printf("%f ",P[18]); printf("%f ",P[19]); printf("\n");
	printf("%f ",P[20]); printf("%f ",P[21]); printf("%f ",P[22]); printf("%f ",P[23]); printf("%f ",P[24]); printf("\n");
	printf("\n");

	mat_mult(M,N,P,5);

	printf("GPU Matrix Multiply:\n");
	printf("%f ",P[0]); printf("%f ",P[1]); printf("%f ",P[2]); printf("%f ",P[3]); printf("%f ",P[4]); printf("\n");
	printf("%f ",P[5]); printf("%f ",P[6]); printf("%f ",P[7]); printf("%f ",P[8]); printf("%f ",P[9]); printf("\n");
	printf("%f ",P[10]); printf("%f ",P[11]); printf("%f ",P[12]); printf("%f ",P[13]); printf("%f ",P[14]); printf("\n");
	printf("%f ",P[15]); printf("%f ",P[16]); printf("%f ",P[17]); printf("%f ",P[18]); printf("%f ",P[19]); printf("\n");
	printf("%f ",P[20]); printf("%f ",P[21]); printf("%f ",P[22]); printf("%f ",P[23]); printf("%f ",P[24]); printf("\n");
	printf("\n");

	CPU_mat_mult(M,N,P,5);

	printf("CPU Matrix Multiply:\n");
	printf("%f ",P[0]); printf("%f ",P[1]); printf("%f ",P[2]); printf("%f ",P[3]); printf("%f ",P[4]); printf("\n");
	printf("%f ",P[5]); printf("%f ",P[6]); printf("%f ",P[7]); printf("%f ",P[8]); printf("%f ",P[9]); printf("\n");
	printf("%f ",P[10]); printf("%f ",P[11]); printf("%f ",P[12]); printf("%f ",P[13]); printf("%f ",P[14]); printf("\n");
	printf("%f ",P[15]); printf("%f ",P[16]); printf("%f ",P[17]); printf("%f ",P[18]); printf("%f ",P[19]); printf("\n");
	printf("%f ",P[20]); printf("%f ",P[21]); printf("%f ",P[22]); printf("%f ",P[23]); printf("%f ",P[24]); printf("\n");
	printf("\n");

	getchar();

	return 0;
}