// matrix_cpu.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>

void printMat(float *a, int size)
{
	for(int i=0; i<size; ++i)
	{
		for(int j=0; j < size; ++j)
		{
			std::cout<<a[i*size +j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;
}
void mat_add(float *a, float *b, float *c, unsigned int size)
{
	for (int i=0; i< size;++i)
	{
		for (int j=0; j < size;++j)
		{
			c[i*size +j] = a[i*size +j] + b[i*size +j];
		}
	}
}

void mat_sub(float *a, float *b, float *c, unsigned int size)
{
	for (int i=0; i< size;++i)
	{
		for (int j=0; j < size;++j)
		{
			c[i*size +j] = a[i*size +j] - b[i*size +j];
		}
	}
}

void mat_mult(float *a, float *b, float *c, unsigned int size)
{
	for (int i=0; i< size;++i)
	{
		for (int j=0; j < size;++j)
		{
			float cValue = 0;
			for (int k=0; k < size; ++k)
			{
				cValue += a[i*size + k] * b[k*size + j];
			}
			c[i*size +j] = cValue;
		}
	}
}
int _tmain(int argc, _TCHAR* argv[])
{
	unsigned int size = 5;
	unsigned int len = size * size;
	float *mat1 = (float*) malloc(sizeof(float)* len);
	float *mat2 =  (float*) malloc(sizeof(float)* len);
	float count = 0.0f;
	for(int i=0; i< size; ++i){
		for(int j=0; j < size;++j){
			mat1[i * size +j] = count;
			mat2[i * size +j] = count;
			count += 1.0f;
		}
	}
	std::cout<<"This is CPU version"<<std::endl;
	printMat(mat1,size);
	printMat(mat2,size);
	float *mat3 = (float*) malloc(sizeof(float) * len);
	mat_add(mat1,mat2,mat3,size);
	printMat(mat3,size);
	mat_sub(mat1,mat2,mat3,size);
	printMat(mat3,size);
	mat_mult(mat1,mat2,mat3,size);
	printMat(mat3,size);
	free(mat1);
	free(mat2);
	free(mat3);
	return 0;
}

