#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int n = 10;
    int a[n], b[n], c[n];
    int *dev_a, *dev_b, *dev_c;

    // Allocate memory on device
    cudaMalloc((void**)&dev_a, n * sizeof(int));
    cudaMalloc((void**)&dev_b, n * sizeof(int));
    cudaMalloc((void**)&dev_c, n * sizeof(int));

    // Initialize host arrays
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * i;
    }

    // Copy host arrays to device
    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch add kernel on device
    add<<<1, n>>>(dev_a, dev_b, dev_c);

    // Copy result from device to host
    cudaMemcpy(c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < n; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free memory on device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
