#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


const int width = 5;
void initiateMatrix(float *matrixM, float *matrixN){
	int val = 0;
	for(int i = 0; i < width; i++){
		for(int j = 0; j < width; j++){
			matrixM[i*width+j] = val;
			matrixN[i*width+j] = val;
			val++;
		}
	}

}

__global__ void MatrixAddKernel(float* M, float* N, float* P){
	//2D Thread ID
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	P[ty*width + tx] = M[ty*width + tx] + N[ty*width + tx];
}

__global__ void MatrixSubKernel(float* M, float* N, float* P){
	//2D Thread ID
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	P[ty*width + tx] = M[ty*width + tx] - N[ty*width + tx];
}

__global__ void MatrixMulKernel(float* M, float* N, float* P){
	//2D Thread ID
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float Pvalue = 0;

	for(int i = 0; i < width; i++){
		Pvalue += M[ty*width + i]*N[i*width + tx];
	}
	P[ty*width + tx] = Pvalue;

}
//single-threaded CPU version of mat_add, mat_sub and mat_mult
void MatrixAdd(float* M, float* N, float* P){
	for(int i = 0; i < width; i++){
		for(int j = 0; j < width; j++){
			P[i*width + j] = M[i*width + j] + N[i*width + j];
		}
	}
}
void MatrixSub(float* M, float* N, float* P){
	for(int i = 0; i < width; i++){
		for(int j = 0; j < width; j++){
			P[i*width + j] = M[i*width + j] - N[i*width + j];
		}
	}
}
void MatrixMul(float* M, float* N, float* P){
	for(int i = 0; i < width; i++){
		for(int j = 0; j < width; j++){
			float Pvalue = 0;
			for(int k = 0; k < width; k++){
				Pvalue += M[i*width + k] * N[k*width + j];
			}
			P[i*width + j] = Pvalue;
		}
	}
}
void printResult(float* add, float* sub, float* mul){
	std::cout<<"add:\n";
	std::cout<<add[0];std::cout<<" "; std::cout<<add[1];std::cout<<" "; std::cout<<add[2];std::cout<<" "; std::cout<<add[3];std::cout<<" "; std::cout<<add[4];std::cout<<"\n"; 
	std::cout<<add[5];std::cout<<" "; std::cout<<add[6];std::cout<<" "; std::cout<<add[7];std::cout<<" "; std::cout<<add[8];std::cout<<" "; std::cout<<add[9];std::cout<<"\n"; 
	std::cout<<add[10];std::cout<<" "; std::cout<<add[11];std::cout<<" "; std::cout<<add[12];std::cout<<" "; std::cout<<add[13];std::cout<<" "; std::cout<<add[14];std::cout<<"\n"; 
	std::cout<<add[15];std::cout<<" "; std::cout<<add[16];std::cout<<" "; std::cout<<add[17];std::cout<<" "; std::cout<<add[18];std::cout<<" "; std::cout<<add[19];std::cout<<"\n"; 
	std::cout<<add[20];std::cout<<" "; std::cout<<add[21];std::cout<<" "; std::cout<<add[22];std::cout<<" "; std::cout<<add[23];std::cout<<" "; std::cout<<add[24];std::cout<<"\n";
	std::cout<<"\n";
	std::cout<<"sub:\n";
	std::cout<<sub[0];std::cout<<" "; std::cout<<sub[1];std::cout<<" "; std::cout<<sub[2];std::cout<<" "; std::cout<<sub[3];std::cout<<" "; std::cout<<sub[4];std::cout<<"\n"; 
	std::cout<<sub[5];std::cout<<" "; std::cout<<sub[6];std::cout<<" "; std::cout<<sub[7];std::cout<<" "; std::cout<<sub[8];std::cout<<" "; std::cout<<sub[9];std::cout<<"\n"; 
	std::cout<<sub[10];std::cout<<" "; std::cout<<sub[11];std::cout<<" "; std::cout<<sub[12];std::cout<<" "; std::cout<<sub[13];std::cout<<" "; std::cout<<sub[14];std::cout<<"\n"; 
	std::cout<<sub[15];std::cout<<" "; std::cout<<sub[16];std::cout<<" "; std::cout<<sub[17];std::cout<<" "; std::cout<<sub[18];std::cout<<" "; std::cout<<sub[19];std::cout<<"\n"; 
	std::cout<<sub[20];std::cout<<" "; std::cout<<sub[21];std::cout<<" "; std::cout<<sub[22];std::cout<<" "; std::cout<<sub[23];std::cout<<" "; std::cout<<sub[24];std::cout<<"\n";
	std::cout<<"\n";
	std::cout<<"multi:\n";
	std::cout<<mul[0];std::cout<<" "; std::cout<<mul[1];std::cout<<" "; std::cout<<mul[2];std::cout<<" "; std::cout<<mul[3];std::cout<<" "; std::cout<<mul[4];std::cout<<"\n"; 
	std::cout<<mul[5];std::cout<<" "; std::cout<<mul[6];std::cout<<" "; std::cout<<mul[7];std::cout<<" "; std::cout<<mul[8];std::cout<<" "; std::cout<<mul[9];std::cout<<"\n"; 
	std::cout<<mul[10];std::cout<<" "; std::cout<<mul[11];std::cout<<" "; std::cout<<mul[12];std::cout<<" "; std::cout<<mul[13];std::cout<<" "; std::cout<<mul[14];std::cout<<"\n"; 
	std::cout<<mul[15];std::cout<<" "; std::cout<<mul[16];std::cout<<" "; std::cout<<mul[17];std::cout<<" "; std::cout<<mul[18];std::cout<<" "; std::cout<<mul[19];std::cout<<"\n"; 
	std::cout<<mul[20];std::cout<<" "; std::cout<<mul[21];std::cout<<" "; std::cout<<mul[22];std::cout<<" "; std::cout<<mul[23];std::cout<<" "; std::cout<<mul[24];std::cout<<"\n";
	std::cout<<"\n";

}
void main(){
	float *matrixM = new float[width*width];
	float *matrixN = new float[width*width];

	initiateMatrix(matrixM, matrixN);

	int size = width*width*sizeof(float);

	float *Md, *Nd, *Pd_add, *Pd_sub, *Pd_mul, *P_add, *P_sub, *P_mul;
	P_add = new float[width*width];
	P_sub = new float[width*width];
	P_mul = new float[width*width];

	cudaMalloc((void**)&Md,size);
	cudaMemcpy(Md, matrixM, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Nd,size);
	cudaMemcpy(Nd, matrixN, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Pd_add, size);
	cudaMalloc((void**)&Pd_sub, size);
	cudaMalloc((void**)&Pd_mul, size);

	dim3 dimBlock(width, width);
	dim3 dimGrid(1, 1);

	MatrixAddKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd_add);
	cudaMemcpy(P_add, Pd_add, size, cudaMemcpyDeviceToHost);

	MatrixSubKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd_sub);
	cudaMemcpy(P_sub, Pd_sub, size, cudaMemcpyDeviceToHost);

	MatrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd_mul);
	cudaMemcpy(P_mul, Pd_mul, size, cudaMemcpyDeviceToHost);

	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd_add);
	cudaFree(Pd_sub);
	cudaFree(Pd_mul);
	std::cout<<"cuda result: \n";
	printResult(P_add, P_sub, P_mul);
	
	MatrixAdd(matrixM, matrixN, P_add);
	MatrixSub(matrixM, matrixN, P_sub);
	MatrixMul(matrixM, matrixN, P_mul);
	std::cout<<"single-threaded CPU result: \n";
	printResult(P_add, P_sub, P_mul);
	
	delete[] matrixM;
	delete[] matrixN;
	delete[] P_add;
	delete[] P_sub;
	delete[] P_mul;



}