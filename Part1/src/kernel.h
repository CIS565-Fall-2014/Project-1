#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

#define blockSize 128
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define SHARED 0

void checkCUDAError(const char *msg, int line);
void cudaNBodyUpdateWrapper(float dt);
void initCuda(int N);
glm::vec3 universalForce(glm::vec4 pos1,glm::vec4 pos2);
void cudaUpdatePBO(float4 * pbodptr, int width, int height);
void cudaUpdateVBO(float * vbodptr, int width, int height);
#endif
