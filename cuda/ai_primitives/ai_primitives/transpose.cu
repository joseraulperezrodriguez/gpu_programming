
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <tuple>
#include <vector>

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

#define TO_TARGET_POS( row, col, ld ) ( ( (col) * (ld) ) + (row) )

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

// Matrix traspose.
__global__ void Transpose(int length, int* input, int* output) {
    __shared__ float target_locations[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y];

    const int myRow = blockDim.x * blockIdx.x + threadIdx.x;
    const int myCol = blockDim.y * blockIdx.y + threadIdx.y;

    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;

    if (myRow < length && myCol < length) {
        target_locations[threadIdx.x][threadIdx.y] = input[TO_TARGET_POS(tileX + threadIdx.x, tileY + threadIdx.y, length)];
    }

    __syncthreads();

    if (myRow < length && myCol < length) {
        output[TO_TARGET_POS(tileY + threadIdx.x, tileX + threadIdx.y, length)] = target_locations[threadIdx.y][threadIdx.x];
    }
}

int main() {
    int SIZE = 4 * THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y;
    int length = THREADS_PER_BLOCK_X;
    int* input_host = (int*)malloc(SIZE * sizeof(int));

    for (int i = 0; i < SIZE; i++) {
        int cell_value = genRandom(1, 10);
        input_host[i] = cell_value;
    }

    int* input_device;
    cudaError_t error_status = cudaMalloc(&input_device, SIZE * sizeof(int));

    error_status = cudaMemcpy(input_device, input_host, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (error_status != cudaSuccess) {
        printf("Error cudaMemcpy dev_elements\n");
        return 1;
    }

    dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 dimGrid(2, 2);

    int* output_device;
    error_status = cudaMalloc(&output_device, SIZE * sizeof(int));

    Transpose<<<dimGrid, dimBlock>>> (length, input_device, output_device);

    int* result_host;
    error_status = cudaMalloc(&result_host, SIZE * sizeof(int));
    cudaError_t cudaStatus = cudaMemcpy(result_host, output_device, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("Error cudaMemcpy devide to host error matrix result_host");
        return 1;
    }

    return 0;
}
