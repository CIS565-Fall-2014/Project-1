#include <cuda_runtime.h>
#include <stdio.h>

int main () { //You need to fix linker issues so that your methods will be recognized, but in the meantime you can check out individual functions by putting them in main
	float * M;//[5][5];
	float * N;
	float * P;
	int size = 5*5*sizeof(float);
	int numBlocks = 1;
	dim3 threadsPerBlock(5,5);
	mat_add(M,N,P,5);
	printf("SAMMMMMMMMMMMMMMMMMMMMMMMMMMMM");
	getchar();
	return 0;
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

