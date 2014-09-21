#include "matrix_math.h"
#define ROWS  5
#define COLS  5
int main (int argc, char** argv){
	float *a, *b, *result;
	a =      (float*)malloc( (ROWS * COLS) * sizeof(float));
	b =      (float*)malloc( (ROWS * COLS) * sizeof(float));
	result = (float*)malloc( (ROWS * COLS) * sizeof(float));
	// initialize matrices
	float x = 0.0;
	for(int i = 0; i < ROWS; i++){
		for(int j = 0; j < COLS; j++){
			a[(COLS * i) + j] = x;
			b[(COLS * i) + j] = x;
			result[(COLS * i) + j] = 0.0;
			x = x + 1.0;
		}
	}

	float* dev_a;
	float* dev_b; 
	float* dev_result;
	cudaMalloc( (void**)&dev_a, 25 * sizeof(float) );
	cudaMalloc( (void**)&dev_b, 25 * sizeof(float) );
	cudaMalloc( (void**)&dev_result, 25 * sizeof(float) );

	//add
	cudaMemcpy( dev_a, a, 25 * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, b, 25 * sizeof(float), cudaMemcpyHostToDevice );
	mat_add<<<ROWS,COLS>>>(dev_a, dev_b, dev_result, 5);
	cudaMemcpy( result, dev_result, 25 * sizeof(float), cudaMemcpyDeviceToHost );
	print_matrix(result,5);

	//subtract
	cudaMemcpy( dev_a, a, 25 * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, b, 25 * sizeof(float), cudaMemcpyHostToDevice );
	mat_sub<<<ROWS,COLS>>>(dev_a, dev_b, dev_result, 5);
	cudaMemcpy( result, dev_result, 25 * sizeof(float), cudaMemcpyDeviceToHost );
	print_matrix(result,5);

	//multiply
	cudaMemcpy( dev_a, a, 25 * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, b, 25 * sizeof(float), cudaMemcpyHostToDevice );
	mat_mul<<<ROWS,COLS>>>(dev_a, dev_b, dev_result, 5);
	cudaMemcpy( result, dev_result, 25 * sizeof(float), cudaMemcpyDeviceToHost );
	print_matrix(result,5);

	//sequential functions
	seq_mat_add( a,b,result,5);
	print_matrix(result,5);
	seq_mat_sub( a,b,result,5);
	print_matrix(result,5);
	seq_mat_mul( a,b,result,5);
	print_matrix(result,5);

	free(a);
	free(b);
	free(result);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_result);
	cin.get();
	return 0;
}

__global__ void mat_add( float *a, float *b, float *result, int arraySize){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[index] = a[index] + b[index];
}

__global__ void mat_sub( float *a, float *b, float *result, int arraySize){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[index] = a[index] - b[index];
}

__global__ void mat_mul( float *a, float *b, float *result, int arraySize){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	result[index ] = 0;
	
	for(int i = 0; i < arraySize; i++){
		result[index] += a[(i * arraySize) + (index % arraySize)] * b[index - (index % arraySize) + i];
	}
}

void seq_mat_add( float *a, float *b, float *result, int arraySize){
	for(int i = 0; i < arraySize; i++){
		for (int j = 0; j < arraySize; j++){
			result[(i * arraySize) + j] = a[(i * arraySize) + j] + b[(i * arraySize) + j];
		}
	}
}

void seq_mat_sub( float *a, float *b, float *result, int arraySize){
	for(int i = 0; i < arraySize; i++){
		for (int j = 0; j < arraySize; j++){
			result[(i * arraySize) + j] = a[(i * arraySize) + j] - b[(i * arraySize) + j];
		}
	}
}

void seq_mat_mul( float *a, float *b, float *result, int arraySize){
	for(int i = 0; i < arraySize; i++){
		for (int j = 0; j < arraySize; j++){
			result[(i * arraySize) + j] = 0;
			for (int k = 0; k < arraySize; k++){
				result[(i * arraySize) + j] += a[(i * arraySize) +k] * b[(k * arraySize) + j];
			}
		}
	}
}

void print_matrix(float *mat, int arraySize){
	for(int i = 0; i < arraySize; i++){
		for(int j = 0; j < arraySize; j++){
			std::cout << mat[(COLS * i) + j] << ",";
		}
		std::cout <<  "\n";
	}
}