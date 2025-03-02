#include <numeric>
#include <iostream>


#define N 10000

__global__ void initialize(double *v)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	v[tid] = (tid + 1);
}


void cuda_stuff()
{
	double *dev_v;
	const size_t size = N*sizeof(double);
	
	cudaMalloc(&dev_v, size);

	initialize<<<dim3(625), dim3(16)>>>(dev_v);

	double *vout = (double*) malloc(size);	
	cudaMemcpy(vout, dev_v, size, cudaMemcpyDeviceToHost);
	
	double sum = std::accumulate(vout, vout+N, 0);

	std::cout << "Sum: " << sum << " == "<< (N*(N+1)/2) << "\n";
	
	cudaFree(dev_v);
	free(vout);
}
