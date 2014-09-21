#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

__global__ void mat_add(float*, float*, float*, int);
__global__ void mat_sub(float*, float*, float*, int);
__global__ void mat_mult(float*, float*, float*, int);

void print(float* mat, int size);
bool equals(float* A, float* B, int size);

#endif
