
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(const float *a, const float *b, float *c, float *d,float *e,unsigned int size);
void serialMat_Mult(const float *a, const float *b, float *c, unsigned int width);
void serialMat_Add(const float *a, const float *b, float *c, unsigned int width);
void serialMat_Sub(const float *a, const float *b, float *c, unsigned int width);



int main()
{
    const int arraySize = 25;
	const float a[arraySize] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
	const float b[arraySize] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    float c_serialAdd[arraySize] = { 0 };
	float c_serialSub[arraySize] = { 0 };
	float c_serialMul[arraySize] = { 0 };
	float c_parallelAdd[arraySize] = { 0 };
	float c_parallelSub[arraySize] = { 0 };
	float c_parallelMul[arraySize] = { 0 };


	serialMat_Add(a, b, c_serialAdd, 5);
	serialMat_Sub(a, b, c_serialSub, 5);
	serialMat_Mult(a, b, c_serialMul, 5);
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(a, b, c_parallelAdd, c_parallelSub, c_parallelMul, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }



    printf("¡u 0,  1,  2,  3,  4¡U   ¡u 0,  1,  2,  3,  4¡U\n"
		   "¡U 5,  6,  7,  8,  9¡U   ¡U 5,  6,  7,  8,  9¡U\n"
	       "¡U10, 11, 12, 13, 14¡U + ¡U10, 11, 12, 13, 14¡U\n"
	       "¡U15, 16, 17, 18, 19¡U   ¡U15, 16, 17, 18, 19¡U\n"
	       "¡U20, 21, 22, 23, 24¡v   ¡U20, 21, 22, 23, 24¡v\n"
		   "= \n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n", 
		   c_serialAdd[0],  c_serialAdd[1],  c_serialAdd[2],  c_serialAdd[3],  c_serialAdd[4], 
		   c_serialAdd[5],  c_serialAdd[6],  c_serialAdd[7],  c_serialAdd[8],  c_serialAdd[9],
		   c_serialAdd[10], c_serialAdd[11], c_serialAdd[12], c_serialAdd[13], c_serialAdd[14], 
		   c_serialAdd[15], c_serialAdd[16], c_serialAdd[17], c_serialAdd[18], c_serialAdd[19],
		   c_serialAdd[20], c_serialAdd[21], c_serialAdd[22], c_serialAdd[23], c_serialAdd[24]);

	printf("¡u 0,  1,  2,  3,  4¡U   ¡u 0,  1,  2,  3,  4¡U\n"
		   "¡U 5,  6,  7,  8,  9¡U   ¡U 5,  6,  7,  8,  9¡U\n"
	       "¡U10, 11, 12, 13, 14¡U - ¡U10, 11, 12, 13, 14¡U\n"
	       "¡U15, 16, 17, 18, 19¡U   ¡U15, 16, 17, 18, 19¡U\n"
	       "¡U20, 21, 22, 23, 24¡v   ¡U20, 21, 22, 23, 24¡v\n"
		   "= \n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n", 
		   c_serialSub[0],  c_serialSub[1],  c_serialSub[2],  c_serialSub[3],  c_serialSub[4], 
		   c_serialSub[5],  c_serialSub[6],  c_serialSub[7],  c_serialSub[8],  c_serialSub[9],
		   c_serialSub[10], c_serialSub[11], c_serialSub[12], c_serialSub[13], c_serialSub[14], 
		   c_serialSub[15], c_serialSub[16], c_serialSub[17], c_serialSub[18], c_serialSub[19],
		   c_serialSub[20], c_serialSub[21], c_serialSub[22], c_serialSub[23], c_serialSub[24]);

	printf("¡u 0,  1,  2,  3,  4¡U   ¡u 0,  1,  2,  3,  4¡U\n"
		   "¡U 5,  6,  7,  8,  9¡U   ¡U 5,  6,  7,  8,  9¡U\n"
	       "¡U10, 11, 12, 13, 14¡U * ¡U10, 11, 12, 13, 14¡U\n"
	       "¡U15, 16, 17, 18, 19¡U   ¡U15, 16, 17, 18, 19¡U\n"
	       "¡U20, 21, 22, 23, 24¡v   ¡U20, 21, 22, 23, 24¡v\n"
		   "= \n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n", 
		   c_serialMul[0],  c_serialMul[1],  c_serialMul[2],  c_serialMul[3],  c_serialMul[4], 
		   c_serialMul[5],  c_serialMul[6],  c_serialMul[7],  c_serialMul[8],  c_serialMul[9],
		   c_serialMul[10], c_serialMul[11], c_serialMul[12], c_serialMul[13], c_serialMul[14], 
		   c_serialMul[15], c_serialMul[16], c_serialMul[17], c_serialMul[18], c_serialMul[19],
		   c_serialMul[20], c_serialMul[21], c_serialMul[22], c_serialMul[23], c_serialMul[24]);

	printf("-----------------------------------------------------------------\n");

	printf("¡u 0,  1,  2,  3,  4¡U   ¡u 0,  1,  2,  3,  4¡U\n"
		   "¡U 5,  6,  7,  8,  9¡U   ¡U 5,  6,  7,  8,  9¡U\n"
	       "¡U10, 11, 12, 13, 14¡U + ¡U10, 11, 12, 13, 14¡U\n"
	       "¡U15, 16, 17, 18, 19¡U   ¡U15, 16, 17, 18, 19¡U\n"
	       "¡U20, 21, 22, 23, 24¡v   ¡U20, 21, 22, 23, 24¡v\n"
		   "= \n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n", 
		  c_parallelAdd[0],  c_parallelAdd[1],  c_parallelAdd[2],  c_parallelAdd[3],  c_parallelAdd[4], 
		  c_parallelAdd[5],  c_parallelAdd[6],  c_parallelAdd[7],  c_parallelAdd[8],  c_parallelAdd[9],
		  c_parallelAdd[10], c_parallelAdd[11], c_parallelAdd[12], c_parallelAdd[13], c_parallelAdd[14], 
		  c_parallelAdd[15], c_parallelAdd[16], c_parallelAdd[17], c_parallelAdd[18], c_parallelAdd[19],
		  c_parallelAdd[20], c_parallelAdd[21], c_parallelAdd[22], c_parallelAdd[23], c_parallelAdd[24]);

	printf("¡u 0,  1,  2,  3,  4¡U   ¡u 0,  1,  2,  3,  4¡U\n"
		   "¡U 5,  6,  7,  8,  9¡U   ¡U 5,  6,  7,  8,  9¡U\n"
	       "¡U10, 11, 12, 13, 14¡U - ¡U10, 11, 12, 13, 14¡U\n"
	       "¡U15, 16, 17, 18, 19¡U   ¡U15, 16, 17, 18, 19¡U\n"
	       "¡U20, 21, 22, 23, 24¡v   ¡U20, 21, 22, 23, 24¡v\n"
		   "= \n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n", 
		   c_parallelSub[0],  c_parallelSub[1],  c_parallelSub[2],  c_parallelSub[3],  c_parallelSub[4], 
		   c_parallelSub[5],  c_parallelSub[6],  c_parallelSub[7],  c_parallelSub[8],  c_parallelSub[9],
		   c_parallelSub[10], c_parallelSub[11], c_parallelSub[12], c_parallelSub[13], c_parallelSub[14], 
		   c_parallelSub[15], c_parallelSub[16], c_parallelSub[17], c_parallelSub[18], c_parallelSub[19],
		   c_parallelSub[20], c_parallelSub[21], c_parallelSub[22], c_parallelSub[23], c_parallelSub[24]);

	printf("¡u 0,  1,  2,  3,  4¡U   ¡u 0,  1,  2,  3,  4¡U\n"
		   "¡U 5,  6,  7,  8,  9¡U   ¡U 5,  6,  7,  8,  9¡U\n"
	       "¡U10, 11, 12, 13, 14¡U * ¡U10, 11, 12, 13, 14¡U\n"
	       "¡U15, 16, 17, 18, 19¡U   ¡U15, 16, 17, 18, 19¡U\n"
	       "¡U20, 21, 22, 23, 24¡v   ¡U20, 21, 22, 23, 24¡v\n"
		   "= \n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n{%.2f,%.2f,%.2f,%.2f,%.2f}\n", 
		   c_parallelMul[0],  c_parallelMul[1],  c_parallelMul[2],  c_parallelMul[3],  c_parallelMul[4], 
		   c_parallelMul[5],  c_parallelMul[6],  c_parallelMul[7],  c_parallelMul[8],  c_parallelMul[9],
		   c_parallelMul[10], c_parallelMul[11], c_parallelMul[12], c_parallelMul[13], c_parallelMul[14], 
		   c_parallelMul[15], c_parallelMul[16], c_parallelMul[17], c_parallelMul[18], c_parallelMul[19],
		   c_parallelMul[20], c_parallelMul[21], c_parallelMul[22], c_parallelMul[23], c_parallelMul[24]);


	// cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	int i;
	scanf("%d", &i);
    return 0;
}

__global__ void mat_add (const float *Md, const float *Nd, float *c, int width)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
    c[j*width + i] = Md[j*width + i] + Nd[j*width + i];
}

__global__ void mat_sub (const float *Md, const float *Nd,float *c,  int width)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
    c[j*width + i] = Md[j*width + i] - Nd[j*width + i];
}

__global__ void mat_mult (const float *Md, const float *Nd, float *c, int width)
{
    int i = threadIdx.x;
	int j = threadIdx.y;


	float value = 0;
	for(int k = 0; k < width; ++k){
		float MdElement = Md[j * width + k];
		float NdElement = Nd[k * width + i];
		value +=  MdElement * NdElement;
	}
	c[j*width + i] = value;
}

void serialMat_Add(const float *a, const float *b, float *c, unsigned int width){
	for(int i = 0; i < width; ++i){
		for(int j = 0; j < width; ++j){
			c[i*width+j] = a[i*width+j] + b[i*width+j];
		}
	}
}

void serialMat_Sub(const float *a, const float *b, float *c, unsigned int width){
	for(int i = 0; i < width; ++i){
		for(int j = 0; j < width; ++j){
			c[i*width+j] = a[i*width+j] - b[i*width+j];
		}
	}
}

void serialMat_Mult(const float *a, const float *b, float *c, unsigned int width){
	for(int i = 0; i < width; ++i){
		for(int j = 0; j < width; ++j){
			float sum = 0;
			for(int k = 0 ;k < width; ++k){
				float m = a[i * width + k];
				float n = b[k * width + j];
				sum += m*n;
			}
			c[i*width+j] = sum;
		}
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(const float *a, const float *b, float *c, float *d, float *e, unsigned int size)
{
    float *dev_a = 0;
    float *dev_b = 0;
    float *dev_c = 0;
	float *dev_d = 0;
	float *dev_e = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_d, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_e, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
	dim3 threadPerBlock(5, 5);
    mat_add <<<1, threadPerBlock>>>(dev_a, dev_b, dev_c, 5);
	mat_sub <<<1, threadPerBlock>>>(dev_a, dev_b, dev_d, 5);
	mat_mult <<<1, threadPerBlock>>>(dev_a, dev_b, dev_e, 5);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(d, dev_d, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(e, dev_e, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:

    cudaFree(dev_a);
    cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFree(dev_d);
	cudaFree(dev_e);
    
    return cudaStatus;
}
