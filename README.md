# Welcome to the CUDA Wiki! 
I'm excited to share my love for CUDA and its amazing possibilities with you. This is a personal project that I created to help beginners and enthusiasts alike to learn more about CUDA and its vast range of applications. Whether you're interested in image and video processing, scientific simulations, machine learning, or anything in between, CUDA has something for you. Through this project, I hope to encourage and inspire others to explore the many possibilities of this exciting technology, and to help build a community of advocates for CUDA development.

**CUDA** is a _parallel computing platform_ and _programming model_ developed by **NVIDIA** that enables developers to use GPUs (Graphics Processing Units) for general-purpose computing applications. It allows developers to write programs in a C-like language (CUDA C/C++) and then execute those programs on the GPU. CUDA makes it possible to harness the power of the GPU's massively parallel architecture, which can perform thousands of arithmetic operations simultaneously, to accelerate computationally intensive tasks such as image and video processing, scientific simulations, machine learning, and more. It also provides a set of libraries and tools that simplify the development process and make it easier to optimize the performance of CUDA applications.

## Understanding CUDA architecture
The CUDA architecture is based on a hierarchy of thread blocks, which are composed of threads that can execute in parallel on the GPU. You need to understand how data is transferred between the CPU and GPU, how kernel functions are executed, and how to optimize memory access for efficient performance. Here is an example of how a CUDA kernel function is executed on the GPU:
```c
__global__ void myKernel(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] *= 2;
    }
}

int main() {
    float* data;
    int size = 1000;
    cudaMalloc(&data, size * sizeof(float));
    myKernel<<<ceil(size/256.0), 256>>>(data, size);
    cudaFree(data);
    return 0;
}
```

## **Installing CUDA toolkit**: 
To develop and run CUDA applications, you need to install the CUDA toolkit, which includes the CUDA runtime, compiler, and other tools. You can download the toolkit from the NVIDIA website. To install the CUDA toolkit on Ubuntu, you can run the following commands in a terminal:
```bash
sudo apt-get update
sudo apt-get install cuda
```

## **Writing CUDA kernel functions** 
A kernel function is a function that runs on the GPU and can be executed in parallel by multiple threads. You need to write kernel functions in C or C++ and annotate them with the __global__ keyword to indicate that they should be executed on the GPU. Here is an example of a simple CUDA kernel function that adds two arrays together:
```c++
__global__ void addArrays(float* a, float* b, float* c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}
```
## **Compiling CUDA code**
To compile CUDA code, you need to use the NVIDIA CUDA compiler (nvcc), which can compile both C/C++ and CUDA code. The compiler generates both CPU and GPU code, and you can link the resulting object files with your application code. To compile a CUDA program using the nvcc compiler, you can run the following command:
```bash
nvcc myProgram.cu -o myProgram
```

## Allocating memory on the GPU
To allocate memory on the GPU, you need to use CUDA memory allocation functions, such as cudaMalloc(), cudaFree(), and cudaMemcpy(). You can allocate memory on the device (GPU) or the host (CPU), and you need to transfer data between the host and device memory as needed. Here is an example of how to allocate memory on the device (GPU) and copy data from the host (CPU):
```c++
float* a = new float[size];
float* b = new float[size];
float* c = new float[size];
float* dev_a, *dev_b, *dev_c;
cudaMalloc(&dev_a, size * sizeof(float));
cudaMalloc(&dev_b, size * sizeof(float));
cudaMalloc(&dev_c, size * sizeof(float));
cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
```

## Launching kernel functions
To launch a kernel function on the GPU, you need to specify the number of blocks and threads per block, which determine the total number of threads that will execute the kernel function in parallel. You can use the <<<...>>> syntax to specify the block and thread dimensions. Here is an example of how to launch a CUDA kernel function with 256 threads per block and 10 blocks:
```c++
int numBlocks = 10;
int numThreadsPerBlock = 256;
addArrays<<<numBlocks, numThreadsPerBlock>>>(dev_a, dev_b, dev_c, size);
```

## Debugging CUDA code
CUDA code can be debugged using NVIDIA's CUDA debugger (cuda-gdb), which provides features such as breakpoint, stepping, and memory inspection. You can also use third-party tools such as NVIDIA Nsight to profile and debug CUDA applications. To debug a CUDA program using cuda-gdb, you can run the following command:
```bash
cuda-gdb myProgram
```

## Optimizing CUDA code
Optimizing CUDA code involves reducing memory access latency, maximizing GPU utilization, and minimizing thread divergence. You can use techniques such as shared memory, memory coalescing, and loop unrolling to improve performance. Here is example code for optimizing the CUDA kernel function that calculates the dot product of two vectors using shared memory:
```c++
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define numThreadsPerBlock 256
#define numBlocks 64
#define size (numThreadsPerBlock * numBlocks)

__global__ void dotProduct(float* a, float* b, float* result, int n) {
    __shared__ float cache[numThreadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;
    while (tid < n) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
    cache[cacheIndex] = temp;
    __syncthreads();
    int j = blockDim.x / 2;
    while (j != 0) {
        if (cacheIndex < j) {
            cache[cacheIndex] += cache[cacheIndex + j];
        }
        __syncthreads();
        j /= 2;
    }
    if (cacheIndex == 0) {
        result[blockIdx.x] = cache[0];
    }
}

int main() {
    float* a = new float[size];
    float* b = new float[size];
    float* result = new float[numBlocks];
    float* dev_a, *dev_b, *dev_result;
    cudaMalloc(&dev_a, size * sizeof(float));
    cudaMalloc(&dev_b, size * sizeof(float));
    cudaMalloc(&dev_result, numBlocks * sizeof(float));
    cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);
    dotProduct<<<numBlocks, numThreadsPerBlock>>>(dev_a, dev_b, dev_result, size);
    cudaMemcpy(result, dev_result, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    float dotProduct = 0;
    for (int i = 0; i < numBlocks; i++) {
        dotProduct += result[i];
    }
    delete[] a;
    delete[] b;
    delete[] result;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
    return 0;
}
```
> This code creates a CUDA kernel function dotProduct() that calculates the dot product of two vectors using shared memory to reduce global memory accesses. The kernel function is launched with numBlocks blocks of numThreadsPerBlock threads each. The resulting dot product is calculated by summing the individual dot products calculated by each block on the CPU. The resulting program demonstrates how shared memory can be used to optimize a CUDA kernel function.

## CUDA Libraries
1. cuBLAS: provides linear algebra operations such as matrix multiplication, inversion, and eigendecomposition.
2. cuFFT: provides fast Fourier transform (FFT) operations on multi-dimensional arrays.
3. cuSPARSE: provides sparse matrix operations such as multiplication, addition, and conversion.
4. cuRAND: provides random number generation functions, including uniform, normal, and Poisson distributions.
5. cuDNN: provides deep neural network primitives, including convolution, pooling, activation functions, and normalization.
6. Thrust: provides a high-level template-based library for CUDA that supports operations such as sorting, searching, reduction, and scan.
7. NPP: provides a set of image and signal processing functions, including filtering, transformation, and feature detection.
8. NVGRAPH: provides graph algorithms such as shortest path, PageRank, and connected components.
9. NVTX: provides tools for profiling and visualizing CUDA applications, including range markers and event markers.

These libraries provide optimized implementations of common numerical and scientific computations, making it easier for developers to write high-performance CUDA applications without needing to implement these algorithms from scratch.
