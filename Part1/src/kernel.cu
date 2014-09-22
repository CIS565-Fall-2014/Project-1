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

// Helper function for accelerate().
// Compute the gravitational acceleration body2 imposes on body1.
__device__
glm::vec3 computeGravitationalAccelerationBetweenTwoBodies( glm::vec4 body1, glm::vec4 body2 )
{
	// NOTE: F = (G * m_a * m_b) / (r_ab ^ 2)
	//		 F = ma => a = F / m.
	//		 So, a = (G * m_a * m_b) / (m_a * r_ab ^ 2) => a = (G * m_b) / (r_ab ^ 2)

	// NOTE: The the 4th component (w) of each body is that body's mass.

	glm::vec3 r( body2.x - body1.x, body2.y - body1.y, body2.z - body1.z );

	// If two bodies have identical positions, immediately return a force of 0 to avoid dividing by 0.
	//if ( glm::length( r ) < EPSILON ) {
	if ( glm::length( r ) < 0.1f ) {
		return glm::vec3( 0.0f, 0.0f, 0.0f );
	}

	float numerator = ( float )( G * body2.w );
	float denominator = ( float )( glm::length( r ) * glm::length( r ) );

	return ( numerator / denominator ) * glm::normalize( r );
}

// TODO: Core force calc kernel global memory
//		 HINT : You may want to write a helper function that will help you 
//              calculate the acceleration contribution of a single body.
//		 REMEMBER : F = (G * m_a * m_b) / (r_ab ^ 2)
// N: Number of planets.
// my_pos: Position of current planet.
// their_pos: Array of planet positions.
__device__
glm::vec3 accelerate( int N, glm::vec4 my_pos, glm::vec4 *their_pos )
{
	// Danny was here.

	glm::vec3 acceleration( 0.0f, 0.0f, 0.0f );

	// Compute acceleration caused by the star at the center of the "solar system".
	acceleration += computeGravitationalAccelerationBetweenTwoBodies( my_pos, glm::vec4( 0.0f, 0.0f, 0.0f, starMass ) );

	// Compute acceleration caused by the other N-1 planets.
	for ( unsigned int i = 0; i < N; ++i ) {
		acceleration += computeGravitationalAccelerationBetweenTwoBodies( my_pos, their_pos[i] );
	}

	return acceleration;
}

// TODO : update the acceleration of each body
// N: Number of planets.
// dt: Time step.
// pos: Array of planet positions.
// vel: Array of planet velocities.
// acc: array of planet accelerations.
__global__
void updateF( int N, float dt, glm::vec4 * pos, glm::vec3 * vel, glm::vec3 * acc )
{
	// Danny was here.

	// NOTE: Blocks are a one-dimensional array in the grid (they only have an x-component).
	// NOTE: Threads are a one dimensional array in the current block (they only have an x-component).

	int planet_index = threadIdx.x + ( blockIdx.x * blockDim.x );

	// If number of planets do not fit perfectly inside the number of blocks specified,
	// then some thread indices will not correlate to a planet, so check bounds.
	if ( planet_index < N ) {
		acc[planet_index] = accelerate( N, pos[planet_index], pos );
	}
}

// TODO : update velocity and position using a simple Euler integration scheme
// N: Number of planets.
// dt: Time step.
// pos: Array of planet positions.
// vel: Array of planet velocities.
// acc: array of planet accelerations.
__global__
void updateS( int N, float dt, glm::vec4 *pos, glm::vec3 *vel, glm::vec3 *acc )
{
	// Danny was here.

	// NOTE: Blocks are a one-dimensional array in the grid (they only have an x-component).
	// NOTE: Threads are a one dimensional array in the current block (they only have an x-component).

	int planet_index = threadIdx.x + ( blockIdx.x * blockDim.x );

	// If number of planets do not fit perfectly inside the number of blocks specified,
	// then some thread indices will not correlate to a planet, so check bounds.
	if ( planet_index < N ) {
		vel[planet_index] += ( acc[planet_index] * dt );

		// Must separate into x-, y-, and z-components since pos is an array of glm::vec4s while vel is an array of glm::vec3s.
		pos[planet_index].x += ( vel[planet_index].x * dt );
		pos[planet_index].y += ( vel[planet_index].y * dt );
		pos[planet_index].z += ( vel[planet_index].z * dt );
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
	// Danny was here.

	// Compute grid dimensions to pass into kernel.
	// Block dimensions are already given in threadsPerBlock.
	dim3 blocks_per_grid( ( int )ceil( ( float )numObjects / ( float )blockSize ) );

	// dev_pos, dev_vel, and dev_acc are arrays of vectors that have already been allocated on the device.
	updateF<<<blocks_per_grid, threadsPerBlock>>>( numObjects, dt, dev_pos, dev_vel, dev_acc );
	checkCUDAErrorWithLine("Kernel failed!"); // Because initCuda() does this.

	// dev_pos, dev_vel, and dev_acc are arrays of vectors that have already been allocated on the device.
	updateS<<<blocks_per_grid, threadsPerBlock>>>(numObjects, dt, dev_pos, dev_vel, dev_acc);
	checkCUDAErrorWithLine("Kernel failed!"); // Because initCuda() does this.

	cudaThreadSynchronize(); // Because cudaThreadSynchronize() and cudaUpdatePBO() do this.
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