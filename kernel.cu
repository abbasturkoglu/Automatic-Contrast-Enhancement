#include <stdio.h>
// Do not alter the preprocessor directives
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>
#include <cassert>
#include <cstdlib>


#define NUM_CHANNELS 1

#define BLOCKSIZE 512

#define MinVal(x, y) (((x) < (y)) ? (x) : (y))
#define MaxVal(x, y) (((x) > (y)) ? (x) : (y))



__global__ void minKernel(uint8_t *image, int size, uint8_t *odata)
{
	extern __shared__ volatile uint8_t sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(BLOCKSIZE)+tid;
	unsigned int gridSize = BLOCKSIZE * gridDim.x;

	uint8_t val = image[i];
	i += gridSize;
	while (i < size) {
		val = MinVal(image[i], val);
		i += gridSize;
	}
	sdata[tid] = val;
	__syncthreads();

	// Parallel reduction v5
	for (int i = (tid + 32); ((tid < 32) && (i < BLOCKSIZE)); i += 32)
		sdata[tid] = MinVal(sdata[tid], sdata[i]);

	if (tid < 16) sdata[tid] = MinVal(sdata[tid], sdata[tid + 16]);
	if (tid < 8)  sdata[tid] = MinVal(sdata[tid], sdata[tid + 8]);
	if (tid < 4)  sdata[tid] = MinVal(sdata[tid], sdata[tid + 4]);
	if (tid < 2)  sdata[tid] = MinVal(sdata[tid], sdata[tid + 2]);
	if (tid == 0) odata[blockIdx.x] = MinVal(sdata[tid], sdata[tid + 1]);
}



__global__ void maxKernel(uint8_t *image, int size, uint8_t *odata)
{
	extern __shared__ volatile uint8_t sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(BLOCKSIZE)+tid;
	unsigned int gridSize = BLOCKSIZE * gridDim.x;

	uint8_t val = image[i];
	i += gridSize;
	while (i < size) {
		val = MaxVal(image[i], val);
		i += gridSize;
	}
	sdata[tid] = val;
	__syncthreads();


	// Parallel reduction v5

	for (int i = (tid + 32); ((tid < 32) && (i < BLOCKSIZE)); i += 32)
		sdata[tid] = MaxVal(sdata[tid], sdata[i]);

	if (tid < 16) sdata[tid] = MaxVal(sdata[tid], sdata[tid + 16]);
	if (tid < 8)  sdata[tid] = MaxVal(sdata[tid], sdata[tid + 8]);
	if (tid < 4)  sdata[tid] = MaxVal(sdata[tid], sdata[tid + 4]);
	if (tid < 2)  sdata[tid] = MaxVal(sdata[tid], sdata[tid + 2]);
	if (tid == 0) odata[blockIdx.x] = MaxVal(sdata[tid], sdata[tid + 1]);
}

__global__ void SubKernel(uint8_t *image, uint8_t min_value)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	image[i] = image[i] - min_value;
}

__global__ void ScaleKernel(uint8_t *image, float scale_constant)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	image[i] = image[i] * scale_constant;
}

int main() {

	int width; //image width
	int height; //image height
	int bpp;  //bytes per pixel if the image was RGB (not used)



	// Load a grayscale bmp image to an unsigned integer array with its height and weight.
	//  (uint8_t is an alias for "unsigned char")
	uint8_t* image = stbi_load("./samples/1280x843.bmp", &width, &height, &bpp, NUM_CHANNELS);
	size_t image_size = width * height * sizeof(uint8_t);

	// Print for sanity check
	printf("Bytes per pixel: %d \n", bpp / 3); //Image is grayscale, so bpp / 3;
	printf("Height: %d \n", height);
	printf("Width: %d \n", width);


	//Start Counter
	cudaEvent_t start, stop;
	float elapsed_time_ms;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Device variables
	uint8_t* min_d;
	uint8_t* max_d;
	uint8_t* image_d;

	//Kernel execution sizes
	unsigned int GRIDSIZE = width / (BLOCKSIZE * 1.0);
	unsigned int GRIDSIZE2 = ceil((width * height) / (BLOCKSIZE * 1.0));

	//Allocate memory for host varables
	uint8_t* min_host = (uint8_t*)malloc(GRIDSIZE * sizeof(uint8_t));
	uint8_t* max_host = (uint8_t*)malloc(GRIDSIZE * sizeof(uint8_t));

	//CUDA allocate memory
	cudaMalloc((void**)&min_d, sizeof(uint8_t) * GRIDSIZE);
	cudaMalloc((void**)&max_d, sizeof(uint8_t) * GRIDSIZE);
	cudaMalloc((void**)&image_d, image_size);

	//Copy image values to device
	cudaMemcpy(image_d, image, image_size, cudaMemcpyHostToDevice);



	//kernels to find minimum and maximum values
	minKernel << <GRIDSIZE, BLOCKSIZE, sizeof(uint8_t)*BLOCKSIZE >> > (image_d, width* height, min_d);
	maxKernel << <GRIDSIZE, BLOCKSIZE, sizeof(uint8_t)*BLOCKSIZE >> > (image_d, width*height, max_d);


	//Get min and max values
	cudaMemcpy(min_host, min_d, sizeof(uint8_t) * GRIDSIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(max_host, max_d, sizeof(uint8_t) * GRIDSIZE, cudaMemcpyDeviceToHost);


	//Subtraction Kernel
	SubKernel << <GRIDSIZE2, BLOCKSIZE >> > (image_d, min_host[0]);


	float scale_constant = 255.0f / (max_host[0] - min_host[0]);

	//Scale Kernel
	ScaleKernel << <GRIDSIZE2, BLOCKSIZE >> > (image_d, scale_constant);

	//Copy image from device to host
	cudaMemcpy(image, image_d, image_size, cudaMemcpyDeviceToHost);


	//Stop timing
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	printf("\nTime to calculate results(GPU Time): %f ms.\n\n", elapsed_time_ms);

	printf("Minimum Pixel Value: %d\n", min_host[0]);
	printf("Maximum Pixel Value: %d\n", max_host[0]);

	// Write image array into a bmp file
	stbi_write_bmp("./samples/out_img.bmp", width, height, 1, image);
	printf("\nEnchanced image successfully saved.\n");




	//free memory
	free(min_host);
	free(max_host);
	cudaFree(min_d);
	cudaFree(max_d);
	cudaFree(image_d);

	return 0;
}
