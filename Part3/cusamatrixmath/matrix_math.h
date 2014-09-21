#ifndef MAIN_H
#define MAIN_H

#include <stdlib.h>
#include <GL/glew.h>
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include "glslUtility.h"


using namespace std;


//-------------------------------
//----------CUDA STUFF-----------
//-------------------------------

int width=1000; int height=1000;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv);

//-------------------------------
//----------function declarations----------
//-------------------------------

__global__ void mat_add( float *a, float *b, float *result, int arraySize);
__global__ void mat_sub( float *a, float *b, float *result, int arraySize);
__global__ void mat_mul( float *a, float *b, float *result, int arraySize);
void seq_mat_add( float *a, float *b, float *result, int arraySize);
void seq_mat_sub( float *a, float *b, float *result, int arraySize);
void seq_mat_mul( float *a, float *b, float *result, int arraySize);
void print_matrix(float *mat, int arraySize);

#endif
