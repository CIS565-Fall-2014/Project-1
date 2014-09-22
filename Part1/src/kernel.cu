#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "kernel.h"

//GLOBALS
dim3 threadsPerBlock(blockSize);

int numObjects;
const float planetMass = 3e8;
const __device__ float starMass = 5e10;

const float scene_scale = 2e2; //size of the height map in simulation space

glm::vec4 * dev_pos;
glm::vec3 * dev_vel;
glm::vec3 * dev_acc;

void checkCUDAError(const char *msg, int line = -1)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        if( line >= 0 )
        {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
        exit(EXIT_FAILURE); 
    }
} 

__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//Function that generates static.
__host__ __device__ glm::vec3 generateRandomNumberFromThread(float time, int index)
{
    thrust::default_random_engine rng(hash(index*time));
    thrust::uniform_real_distribution<float> u01(0,1);

    return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}

//Generate randomized starting positions for the planets in the XY plane
//Also initialized the masses
__global__ void generateRandomPosArray(int time, int N, glm::vec4 * arr, float scale, float mass)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 rand = scale*(generateRandomNumberFromThread(time, index)-0.5f);
        arr[index].x = rand.x;
        arr[index].y = rand.y;
        arr[index].z = 0.0f;
        arr[index].w = mass;
    }
}

//Determine velocity from the distance from the center star. Not super physically accurate because 
//the mass ratio is too close, but it makes for an interesting looking scene
__global__ void generateCircularVelArray(int time, int N, glm::vec3 * arr, glm::vec4 * pos)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(index < N)
    {
        glm::vec3 R = glm::vec3(pos[index].x, pos[index].y, pos[index].z);
        float r = glm::length(R) + EPSILON;
        float s = sqrt(G * starMass / r);
        glm::vec3 D = glm::normalize(glm::cross(R / r, glm::vec3(0,0,1)));
        arr[index].x = s*D.x;
        arr[index].y = s*D.y;
        arr[index].z = s*D.z;
    }
}

__device__ glm::vec3 calcAccel(float m_o, glm::vec3 p_i, glm::vec3 p_o)
{
    glm::vec3 r = p_o - p_i;
    float R = glm::length(r);
    if (R < 1) {
        return glm::vec3();
    } else {
        return r * (m_o / (R * R * R));
    }
}

// Core force calc kernel global memory
//		 HINT : You may want to write a helper function that will help you 
//              calculate the acceleration contribution of a single body.
//		 REMEMBER : F = (G * m_a * m_b) / (r_ab ^ 2)
__device__  glm::vec3 accelerate(int N, glm::vec4 my_pos, glm::vec4 * their_pos)
{
    glm::vec3 p = glm::vec3(my_pos);
    glm::vec3 a;

    a += calcAccel(starMass, p, glm::vec3());
    for (int o = 0; o < N; ++o) {
        a += calcAccel(planetMass, p, glm::vec3(their_pos[o]));
    }

    float _G = G;
    return a * _G;
}

// update the acceleration of each body
__global__ void updateF(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    acc[i] = accelerate(N, pos[i], pos);
}

// update velocity and position using a simple Euler integration scheme
__global__ void updateS(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    vel[i] += dt * acc[i];
    pos[i] = glm::vec4(glm::vec3(pos[i]) + dt * vel[i], 1);
}

// Update the vertex buffer object
// (The VBO is where OpenGL looks for the positions for the planets)
__global__ void sendToVBO(int N, glm::vec4 * pos, float * vbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);

    float c_scale_w = -2.0f / s_scale;
    float c_scale_h = -2.0f / s_scale;

    if(index<N)
    {
        vbo[4*index+0] = pos[index].x*c_scale_w;
        vbo[4*index+1] = pos[index].y*c_scale_h;
        vbo[4*index+2] = 0;
        vbo[4*index+3] = 1;
    }
}

// Update the texture pixel buffer object
// (This texture is where openGL pulls the data for the height map)
// We will not be using this in this homework
__global__ void sendToPBO(int N, glm::vec4 * pos, float4 * pbo, int width, int height, float s_scale)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    int x = index % width;
    int y = index / width;
    float w2 = width / 2.0;
    float h2 = height / 2.0;

    float c_scale_w = width / s_scale;
    float c_scale_h = height / s_scale;

    glm::vec3 color(0.05, 0.15, 0.3);
    //glm::vec3 acc = accelerate(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), pos);

    if(x<width && y<height)
    {
        //float mag = sqrt(sqrt(acc.x*acc.x + acc.y*acc.y + acc.z*acc.z));
        float mag = 1.0f;
        
		// Each thread writes one pixel location in the texture (textel)
        pbo[index].w = (mag < 1.0f) ? mag : 1.0f;
    }
}

/*************************************
 * Wrappers for the __global__ calls *
 *************************************/

//Initialize memory, update some globals
void initCuda(int N)
{
    numObjects = N;
    dim3 fullBlocksPerGrid((int)ceil(float(N)/float(blockSize)));

    cudaMalloc((void**)&dev_pos, N*sizeof(glm::vec4));
    checkCUDAErrorWithLine("Kernel failed!");
    
	cudaMalloc((void**)&dev_vel, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");
    
	cudaMalloc((void**)&dev_acc, N*sizeof(glm::vec3));
    checkCUDAErrorWithLine("Kernel failed!");

    generateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects, dev_pos, scene_scale, planetMass);
    checkCUDAErrorWithLine("Kernel failed!");
    
	generateCircularVelArray<<<fullBlocksPerGrid, blockSize>>>(2, numObjects, dev_vel, dev_pos);
    checkCUDAErrorWithLine("Kernel failed!");
    
	cudaThreadSynchronize();
}

// Using the functions you wrote above, write a function that calls the CUDA kernels to update a single sim step
void cudaNBodyUpdateWrapper(float dt)
{
#if PERFTEST
    static bool uninit = true;
    static cudaEvent_t ev0;
    static cudaEvent_t ev1;
    static cudaEvent_t ev2;
    if (uninit) {
        uninit = false;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventCreate(&ev2);
    }

    static int i = 0;
    static int b = 8;

    if (b > 1024) {
        exit(0);
    }
#else
    static const int b = 64;
#endif

    int g = ceil(float(numObjects)/float(b));

#if PERFTEST
    cudaEventRecord(ev0, 0);
#endif
    updateF<<<g, b>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
#if PERFTEST
    cudaEventRecord(ev1, 0);
#endif
    updateS<<<g, b>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
#if PERFTEST
    cudaEventRecord(ev2, 0);

    static float tt01, tt12;
    float t01, t12;
    cudaEventSynchronize(ev2);
    cudaEventElapsedTime(&t01, ev0, ev1);
    cudaEventElapsedTime(&t12, ev1, ev2);
    tt01 += t01;
    tt12 += t12;

    if (i < 4) {
        i++;
    } else {
        i = 0;
        printf("%d %d %d %f %f\n", numObjects, g, b, tt01 / 4, tt12 / 4);
        b += 8;
        tt01 = tt12 = 0;
    }
#endif
}

void cudaUpdateVBO(float * vbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
    sendToVBO<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, vbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}

void cudaUpdatePBO(float4 * pbodptr, int width, int height)
{
    dim3 fullBlocksPerGrid((int)ceil(float(width*height)/float(blockSize)));
    sendToPBO<<<fullBlocksPerGrid, blockSize, blockSize*sizeof(glm::vec4)>>>(numObjects, dev_pos, pbodptr, width, height, scene_scale);
    cudaThreadSynchronize();
}


