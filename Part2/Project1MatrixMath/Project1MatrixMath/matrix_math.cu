/* 
 * MICHAEL B LI
 * CIS 565 Project-1
 * 
 * "Hello world" made possible with support from http://www.brainlings.com/2011/11/hello-world-in-cuda/
 * 
 * 
 */

#include <iostream>

#define MAT_WIDTH 5


int main(int argc, char** argv) {

	//std::cout << "Hello there!" << std::endl;

	//std::cin.ignore();

	//return 0;

	//#################################################################################


	// CPU float array
	// I will just be using a 1D array to hold the floats.
	// MAT_WIDTH is defined to be 5 above; change it to test with something else. All matrices assumed to be SQUARE.

	float* M = new float[MAT_WIDTH*MAT_WIDTH]();
	for (int i = 0; i < MAT_WIDTH*MAT_WIDTH; i++) {
		M[i] = float(i);
	}

	for (int i = 0; i < 26; i++) {
		std::cout << M[i] << std::endl;
	}

	int size = MAT_WIDTH * MAT_WIDTH * sizeof(float);


	// will be using global memory only (cudaMalloc), since using __shared__ was not specified in the directions
	// and this project is already difficult enough such that I'd rather not give myself extra work.




	std::cin.ignore();
	return 0;
}
