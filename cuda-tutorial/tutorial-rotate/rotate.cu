#include <iostream>
#include <cstdint>
#include <cstdlib>

#define WIDTH 512
#define HEIGHT 512
#define SIZE ((WIDTH*HEIGHT*sizeof(unsigned char)))


__global__ void rotate90cw(
	unsigned char *deviceInputImage,
	unsigned char *deviceOutputImage,
	int width, int height)
{
	/* ... */
}


__global__ void rotate90ccw(
	unsigned char *deviceInputImage,
	unsigned char *deviceOutputImage,
	int width, int height)
{
	/* ... */
}


int main(int argc, const char *argv[])
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

	/* Prepare buffers and copy */
	/* ... */
	
	/* Define dimBlock and dimGrid */
	/* ... */

	/* Launch kernel */
	/* ... */
	
	// Copy image back in host memory
	cudaMemcpy(imageOut, deviceOutputImage, SIZE, cudaMemcpyDeviceToHost);
	
	// Save image for visualization
	FILE *fpOut = fopen("output.gray", "wb");
	fwrite(imageOut, SIZE, 1, fpOut);

	fclose(fpOut);
	fclose(fpIn);

	/* Clean up */
	/* ... */
	
	return 0;
}
