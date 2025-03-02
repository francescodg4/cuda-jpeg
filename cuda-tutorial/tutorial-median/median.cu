#include <iostream>
#include <cstdint>
#include <cstdlib>

#define WIDTH 512
#define HEIGHT 512
#define SIZE ((WIDTH*HEIGHT*sizeof(unsigned char)))


__device__ void sort(unsigned char *arr, size_t n)
{
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {

			if (arr[i] <= arr[j])
				continue;
			
			//Swap the variables.
			char tmp = arr[i];
			arr[i] = arr[j];
			arr[j] = tmp;
		}
	}
}


__global__ void medianFilterKernel(unsigned char *inputImageKernel, unsigned char *outputImagekernel, int imageWidth, int imageHeight)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Define filter window
	unsigned char filterVector[WINDOW_SIZE*WINDOW_SIZE];
	memset(filterVector, 0, sizeof(filterVector));
	
	// Deal with boundry conditions
	
	// Setup the filtering window.
	
	sort(filterVector, WINDOW_SIZE*WINDOW_SIZE);

	//Set the output variables.
	// outputImagekernel[row*imageWidth+col] = 
}


int main()
{
	FILE *fpIn = fopen("lena.gray", "rb");

	if (!fpIn)
		return -1;

	unsigned char *imageIn = (unsigned char *) malloc(SIZE);
	unsigned char *imageOut = (unsigned char *) malloc(SIZE);

	fread(imageIn, SIZE, 1, fpIn);       

	// Prepare images in GPU memory
	unsigned char *deviceInputImage;
	unsigned char *deviceOutputImage;
	
	cudaMalloc((void**) &deviceInputImage, SIZE);
	cudaMalloc((void**) &deviceOutputImage, SIZE);

	cudaMemcpy(deviceInputImage, imageIn, SIZE, cudaMemcpyHostToDevice);

	dim3 dimBlock;
	dim3 dimGrid;

	std::cout << "threads.x " << dimBlock.x << ", "
		  << "threads.y " << dimBlock.y << ", "
		  << "grid.x " << dimGrid.x << ", "
		  << "grid.y " << dimGrid.y << std::endl;
		

	cudaEvent_t startEvent, stopEvent;
	
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);

	cudaEventRecord(startEvent);
	medianFilterKernel<<<dimGrid, dimBlock>>>(deviceInputImage, deviceOutputImage, WIDTH, HEIGHT);
	cudaEventRecord(stopEvent);

	cudaEventSynchronize(stopEvent);
	cudaDeviceSynchronize();

	// Copy image back in host memory
	cudaMemcpy(imageOut, deviceOutputImage, SIZE, cudaMemcpyDeviceToHost);


	// Show elapsed time
	float time = 0;
	cudaEventElapsedTime(&time, startEvent, stopEvent);
	printf("Elapsed time: %.4f ms\n", time); 

	// Save image for visualization
	FILE *fpOut = fopen("output.gray", "wb");
	fwrite(imageOut, SIZE, 1, fpOut);

	fclose(fpOut);
	fclose(fpIn);

	cudaFree(deviceOutputImage);
	cudaFree(deviceInputImage);
	cudaFree(startEvent);
	cudaFree(stopEvent);
	
	return 0;
}
