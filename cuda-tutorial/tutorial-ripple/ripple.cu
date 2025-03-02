#include <iostream>
#include <string>

#define DIM 256


__global__ void ripple(unsigned char *dev_img, int tick)
{
	/* ... */
}


int main()
{
	unsigned char *dev_img;
	
	cudaMalloc(&dev_img, DIM*DIM);

	unsigned char *img = (unsigned char *) malloc(DIM*DIM*sizeof(unsigned char));
	
	for (int t = 0; t < 100; t++) {

		/* Kernel launch ripple(..., t) */
		
		cudaMemcpy(img, dev_img, DIM*DIM, cudaMemcpyDeviceToHost);

		// Save output to file
		std::string filename = "/tmp/ripple";
		filename += std::to_string(i);
		filename += ".gray";
		
		std::cout << filename << "\n";
			
		FILE *fp = fopen(filename.c_str(), "wb");

		fwrite(img, DIM*DIM, 1, fp);
		fclose(fp);
	}
	
	/* Cleanup */
	
	return 0;
}
