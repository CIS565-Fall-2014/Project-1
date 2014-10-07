
#include <stdio.h>
#include <cuda_runtime.h>
#define MatrixWidth 10


//GPU matrix add function
__global__ void mat_add(float *A,float *B,float *C,int N)
{
	int row=threadIdx.x;
	int col=threadIdx.y;
	C[row*MatrixWidth+col]=A[row*MatrixWidth+col]+B[row*MatrixWidth+col];
   
}

//GPU matrix sub function
__global__ void mat_sub(float *A,float *B,float *C,int N)
{
	int row=threadIdx.x;
	int col=threadIdx.y;
	C[row*MatrixWidth+col]=A[row*MatrixWidth+col]-B[row*MatrixWidth+col];
   
}

//GPU matrix mult function
__global__ void mat_mult(float *A,float *B,float *C,int N)
{
	int row=threadIdx.x;
	int col=threadIdx.y;
	for (int i=0;i<MatrixWidth;i++)
	{
	C[row*MatrixWidth+col]+=A[row*MatrixWidth+i]*B[i*MatrixWidth+col];
	}
   
}

//CPU matrix add function
void mat_add_serial(float *A,float *B,float *C,int N)
{
	for(int i=0;i<N;i++)
	{
		C[i]=A[i]+B[i];
	}
}

//CPU matrix sub function
void mat_sub_serial(float *A,float *B,float *C,int N)
{
	for(int i=0;i<N;i++)
	{
		C[i]=A[i]-B[i];
	}
}

//CPU matrix mult function
void mat_mult_serial(float *A,float *B,float *C)
{
	for(int i=0;i<MatrixWidth;i++)
	{
		for(int j=0;j<MatrixWidth;j++)
		{
			float sum=0;
			for(int k=0;k<MatrixWidth;k++)
			{
				sum+=A[i*MatrixWidth+k]*B[k*MatrixWidth+j];
			}
			C[i*MatrixWidth+j]=sum;
		}
	}

	
}


int main(){

	float *A, *B, *C;
	float *d_A, *d_B, *d_C;
	int Size=MatrixWidth*MatrixWidth;
	A=(float *)malloc(Size*sizeof(float));
	B=(float *)malloc(Size*sizeof(float));
	C=(float *)malloc(Size*sizeof(float));
	//Allocate input memory on GPU
	cudaMalloc(&d_A,Size*sizeof(float));
	cudaMalloc(&d_B,Size*sizeof(float));
	cudaMalloc(&d_C,Size*sizeof(float));

	//init A, B, and C
	for (int i=0;i<MatrixWidth;i++)
	{
		for (int j=0;j<MatrixWidth;j++)
		{
			A[i+j*MatrixWidth]=i+j*MatrixWidth;
			B[i+j*MatrixWidth]=i+j*MatrixWidth;
			C[i+j*MatrixWidth]=0;
		}
	}

	cudaMemcpy(d_A,A,Size*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,B,Size*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_C,C,Size*sizeof(float),cudaMemcpyHostToDevice);



	dim3 BlocksPerGrid(1,1);
	dim3 ThreadsPerBlock(MatrixWidth,MatrixWidth);

	//init for cuda timer
	cudaEvent_t start;
	cudaEvent_t end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start,0);

	//mat_add<<< BlocksPerGrid,ThreadsPerBlock>>>(d_A,d_B,d_C,Size);
	//mat_sub<<< BlocksPerGrid,ThreadsPerBlock>>>(d_A,d_B,d_C,Size);
	mat_mult<<< BlocksPerGrid,ThreadsPerBlock>>>(d_A,d_B,d_C,Size);
    //mat_add_serial(A,B,C,Size);
	//mat_sub_serial(A,B,C,Size);
	//mat_mult_serial(A,B,C);
	
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	float time;
	cudaEventElapsedTime(&time,start,end);
	

	//Read C from device
	cudaMemcpy(C,d_C,Size*sizeof(float),cudaMemcpyDeviceToHost);
	

	for (int i=0; i<MatrixWidth;i++)
	{
		for(int j=0;j<MatrixWidth;j++)
		{
		printf("%f ",C[i*MatrixWidth+j]);
		}
		printf("\n");
	}

	printf("Time: %f \n\n",time);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	free(A);
	free(B);
	free(C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	getchar();
	return 0;
}
