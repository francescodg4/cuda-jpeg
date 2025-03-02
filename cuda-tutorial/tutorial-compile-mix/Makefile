all: a.out

cuda.o: cuda.cu
	nvcc -c cuda.cu

a.out: cuda.o main.o
	g++ -O$@ $^ -L/usr/local/cuda-10.0/lib64 -lcudart
