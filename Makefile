all: raytrace_cpu raytrace_cuda

raytrace_cpu: src/raytrace_cpu.cpp
	g++ -O3 -std=c++17 src/raytrace_cpu.cpp -o ./build/raytrace_cpu

raytrace_cuda: src/raytrace_cuda.cu
	nvcc -O2 -arch=sm_75 -lcudart -lcurand src/raytrace_cuda.cu -o ./build/raytrace_cuda

clean:
	rm -f raytrace_cpu raytrace_cuda

.PHONY: all clean
