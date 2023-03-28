#include <stdio.h>
#include <cuda.h>

#define N 1024
#define BLOCK_SIZE 32

__global__ void matrix_multiply(int *a, int *b, int *c)
{
    __shared__ int s_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int s_b[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    for (int i = 0; i < N / BLOCK_SIZE; i++) {
        s_a[ty][tx] = a[row * N + i * BLOCK_SIZE + tx];
        s_b[ty][tx] = b[(i * BLOCK_SIZE + ty) * N + col];
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum += s_a[ty][j] * s_b[j][tx];
        }
        __syncthreads();
    }

    c[row * N + col] = sum;
}

int main()
{
    int *a, *b, *c;
    int *dev_a, *dev_b, *dev_c;

    a = (int*)malloc(N*N*sizeof(int));
    b = (int*)malloc(N*N*sizeof(int));
    c = (int*)malloc(N*N*sizeof(int));

    for (int i = 0; i < N*N; i++) {
        a[i] = i % N;
        b[i] = i % N;
        c[i] = 0;
    }

    cudaMalloc((void**)&dev_a, N*N*sizeof(int));
    cudaMalloc((void**)&dev_b, N*N*sizeof(int));
    cudaMalloc((void**)&dev_c, N*N*sizeof(int));

    cudaMemcpy(dev_a, a, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N*N*sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(N / BLOCK_SIZE, N / BLOCK_SIZE);

    matrix_multiply<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, N*N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", c[i*N+j]);
        }
        printf("\n");
    }

    free(a);
    free(b);
    free(c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
