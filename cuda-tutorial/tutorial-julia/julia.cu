#include <iostream>


__global__ void kernel_julia(unsigned char *dev_img)
{

}


int main()
{
	/* Allocate memory on GPU */
	/* Launch kernel */
	/* Copy back to CPU Memory */
	
	// Save output to file
	FILE *fp = fopen("output.gray", "wb");
	fclose(fp);

	/* Cleanup */
	
	return 0;
}
