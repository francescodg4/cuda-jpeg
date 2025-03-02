#include <iostream>


__global__ void memcpy_example(void)
{
	uint8_t arr1[100];
	uint8_t arr2[100];
	
	/* Initialize arr1 to sequence 1...100 */
	/* ... */
	
	/* Initialize arr2 to constant value */
	/* ... */
	
	/* Copy arr1 to arr2 */
	/* ...*/
}


int main(int argc, const char *argv[])
{

	memcpy_example<<<1, 1>>>();
	cudaDeviceSynchronize();
	
	return 0;
}
