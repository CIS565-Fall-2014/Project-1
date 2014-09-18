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

// TODO: Core force calc kernel global memory
//		 HINT : You may want to write a helper function that will help you 
//              calculate the acceleration contribution of a single body.
//		 REMEMBER : F = (G * m_a * m_b) / (r_ab ^ 2)
__device__  glm::vec3 accelerate(int N, glm::vec4 my_pos, glm::vec4 * their_pos)//Maybe ok??? :'(
{
	//You might need to take into account the acceleration of EVERY other body
	//So this method can call a helper method as was in the hint, and the helper method
	//will do the below calculations, but essentially you need to calculate the acceleration
	//due to all other bodies
	glm::vec3 r = glm::vec3(their_pos->x - my_pos.x, their_pos->y - my_pos.y, their_pos->z - my_pos.z);
	float r_ab1 = pow(r.x,2.f) + pow(r.y,2.f) + pow(r.z,2.f);
	float r_ab2 = r_ab1 * sqrt(r_ab1);
	glm::vec3 a = glm::vec3(3e8*starMass*r.x/r_ab2, 3e8*starMass*r.y/r_ab2, 3e8*starMass*r.z/r_ab2);
	//float f = (G * 3e8 * starMass) / pow(r_ab, 2.f);
    return a;
}

// TODO : update the acceleration of each body
__global__ void updateF(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc)
{
	// FILL IN HERE

	/*int width = 800;
	int height = 800;
	float s_scale = 2e2;

	int index = threadIdx.x + (blockIdx.x * blockDim.x);
    int x = index % width;
    int y = index / width;

	float w2 = width / 2.0;
    float h2 = height / 2.0;

    float c_scale_w = width / s_scale;
    float c_scale_h = height / s_scale;

    glm::vec3 a = accelerate(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), pos);
	acc->x = acc->x + a.x;
	acc->y = acc->y + a.y;
	acc->z = acc->z + a.z;*/

	
	int width = 800;
	int height = 800;
	float s_scale = 2e2;
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	for (int i = 0; i < N; i++) {
		int x = i % width;
		int y = i / width;
		float w2 = width / 2.0;
		float h2 = height / 2.0;
		float c_scale_w = width / s_scale;
		float c_scale_h = height / s_scale;
		glm::vec4 * p = new glm::vec4();
		p->x = pos[i].x;
		p->y = pos[i].y;
		p->z = pos[i].z;
		glm::vec3 a = accelerate(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), p);
		acc[i].x = acc[i].x + a.x;
		acc[i].y = acc[i].y + a.y;
		acc[i].z = acc[i].z + a.z;
	}

}

// TODO : update velocity and position using a simple Euler integration scheme
__global__ void updateS(int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc) //DONE
{
	/*pos->x = pos->x + vel->x * dt;
	pos->y = pos->y + vel->y * dt;
	pos->z = pos->z + vel->z * dt;
	pos->w = pos->w + dt;
					
	vel->x = vel->x + acc->x * dt;
	vel->y = vel->y + acc->y * dt;
	vel->z = vel->z + acc->z * dt;*/

	for (int i = 0; i < N; i++) {
		pos[i].x = pos[i].x + vel[i].x * dt;
		pos[i].y = pos[i].y + vel[i].y * dt;
		pos[i].z = pos[i].z + vel[i].z * dt;
		//pos[i].w += 1;//= pos[i].w + dt;
				
		vel[i].x = vel[i].x + acc[i].x * dt;
		vel[i].y = vel[i].y + acc[i].y * dt;
		vel[i].z = vel[i].z + acc[i].z * dt;
	}
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
    glm::vec3 acc = accelerate(N, glm::vec4((x-w2)/c_scale_w,(y-h2)/c_scale_h,0,1), pos);

    if(x<width && y<height)
    {
        float mag = sqrt(sqrt(acc.x*acc.x + acc.y*acc.y + acc.z*acc.z));
        
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

// TODO : Using the functions you wrote above, write a function that calls the CUDA kernels to update a single sim step
void cudaNBodyUpdateWrapper(float dt)
{
	// FILL IN HERE
	dim3 fullBlocksPerGrid((int)ceil(float(numObjects)/float(blockSize)));
	//updateF<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
	updateS<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
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


