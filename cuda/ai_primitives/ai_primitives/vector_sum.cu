
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>

cudaError_t vectorSum(const int* a, const int* b, int *c, unsigned int size);

__global__ void vectorSumKernel(const int* a, const int* b, int *c,  int segment_size, int total_threads, int array_size)
{
    int i = threadIdx.x;
    // Edge case, when some threads does not need to do anything.
    if (total_threads > array_size) {
        if (i < array_size) {
            c[i] = a[i] + b[i];
        }
    }
    else {
      int start = i * segment_size;
      for (int index = start; index < min(start + segment_size, array_size); index++) {
          c[index] = a[index] + b[index];
      }
      
      // if this is the last thread, calculate the remaining items.
      if (i == blockDim.x - 1) {
          for (int index = start + segment_size; index < array_size; index++) {
              c[index] = a[index] + b[index];
          }
      }
    }
}

void printArray(const int* array, int length) {
    printf("{");
    for (int i = 0; i < length-1; i++) {
        printf("%d,", array[i]);
    }
    printf("%d", array[length-1]);
    printf("}");
}

int genRandom(int from, int to) {
    // Create a random device to seed the generator
    std::random_device rd;
    // Initialize Mersenne Twister engine with the seed
    std::mt19937 gen(rd());
    // Define the distribution range [1, 100]
    std::uniform_int_distribution<> dist(1, to);
    // Generate a random number
    int random_number = dist(gen);
    return random_number;
}

int main()
{
    constexpr int arraySize = 5;
    const int a[arraySize] = {1,2,3,4,5};
    const int b[arraySize] = {10, 20, 30, 40, 50};
    int c[arraySize];

    // Add vectors in parallel.
    cudaError_t cudaStatus = vectorSum(a, b, c, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "vectorSum failed!");
        return 1;
    }

    printArray(a, arraySize);
    printf(" + ");
    printArray(b, arraySize);
    printf(" = ");
    printArray(c, arraySize);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t vectorSum(const int* a, const int* b, int *c,  unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    int thread_count = genRandom(1, 10);
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    printf("thread_count: %d, segment_size: %d\n", thread_count, size / thread_count);
    // Launch a kernel on the GPU with one thread for each element.
    vectorSumKernel<< <1, thread_count>> > (dev_a, dev_b, dev_c, size / thread_count, thread_count, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
