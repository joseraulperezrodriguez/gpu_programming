
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <tuple>
#include <vector>

typedef struct {
    int width;
    int height;
    int stride;
    int* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix M, int row, int col) {
    return M.elements[row * M.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix M, int row, int col, float value) {
    M.elements[row * M.stride + col] = value;
}

// Thread block size
#define BLOCK_SIZE 2

// Matrix size
#define MATRIX_SIZE 4

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix M, int row, int col) {
    Matrix Msub;
    Msub.width = BLOCK_SIZE;
    Msub.height = BLOCK_SIZE;
    Msub.stride = M.stride;
    Msub.elements = &M.elements[M.stride * BLOCK_SIZE * row
        + BLOCK_SIZE * col];
    return Msub;
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

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
cudaError_t MatMul(const Matrix left, const Matrix right, Matrix result) {
    // Load left and right to device memory
    Matrix d_left;
    d_left.width = d_left.stride = left.width; d_left.height = left.height;
    size_t size = left.width * left.height * sizeof(float);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(left.width / dimBlock.x, left.height / dimBlock.y);

    cudaError_t error_status;

    error_status = cudaMalloc(&d_left.elements, size);
    if (error_status != cudaSuccess) {
        printf("Error allocating d_left matrix.\n");
        goto Error;
    }

    error_status = cudaMemcpy(d_left.elements, left.elements, size, cudaMemcpyHostToDevice);
    if (error_status != cudaSuccess) {
        printf("Error copying matriz 'left' Host To Device\n");
        goto Error;
    }

    Matrix d_right;
    d_right.width = d_right.stride = right.width; d_right.height = right.height;
    size = right.width * right.height * sizeof(float);
    error_status = cudaMalloc(&d_right.elements, size);
    if (error_status != cudaSuccess) {
        printf("Error allocating d_right matrix.\n");
        goto Error;
    }

    error_status = cudaMemcpy(d_right.elements, right.elements, size, cudaMemcpyHostToDevice);
    if (error_status != cudaSuccess) {
        printf("Error copying matriz 'right' Host To Device\n");
        goto Error;
    }

    // Allocate C in device memory
    Matrix d_result;
    d_result.width = d_result.stride = result.width; d_result.height = result.height;
    size = result.width * result.height * sizeof(float);
    error_status = cudaMalloc(&d_result.elements, size);
    if (error_status != cudaSuccess) {
        printf("Error allocating d_result matrix.\n");
        goto Error;
    }

    // Invoke kernel
    MatMulKernel << <dimGrid, dimBlock >> > (d_left, d_right, d_result);
    // Read C from device memory
    error_status = cudaMemcpy(result.elements, d_result.elements, size, cudaMemcpyDeviceToHost);
    if (error_status != cudaSuccess) {
        printf("Error copying matriz 'result' Device To Host\n");
        goto Error;
    }

  Error:
    // Free device memory
    cudaFree(d_left.elements);
    cudaFree(d_right.elements);
    cudaFree(d_result.elements);
    return error_status;
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix left, Matrix right, Matrix result) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix result_sub = GetSubMatrix(result, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float result_value = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (left.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix left_sub of left
        Matrix left_sub = GetSubMatrix(left, blockRow, m);
        // Get sub-matrix right_sub of right
        Matrix right_sub = GetSubMatrix(right, m, blockCol);
        // Shared memory used to store left_sub and right_sub respectively
        __shared__ float left_s[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float right_s[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        left_s[row][col] = GetElement(left_sub, row, col);
        right_s[row][col] = GetElement(right_sub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            result_value += left_s[row][e] * right_s[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of left and right in the next iteration
        __syncthreads();
    }
    // Write result_sub to device memory
    // Each thread writes one element
    SetElement(result_sub, row, col, result_value);
}

void ValidationMult(int* left, int* right, int* result) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            int index = (MATRIX_SIZE * i) + j;
            int sum = 0;
            for (int k = 0; k < MATRIX_SIZE; k++) {
                sum += left[(i*MATRIX_SIZE) + k] * right[(k*MATRIX_SIZE) + j];
            }
            result[index] = sum;
        }
    }
}

void PrintMatrix(int* matrix, std::string name) {
    printf("input matrix %s:\n", name.c_str());
    for (int i = 0; i < MATRIX_SIZE; i++) {
        printf("{");
        for (int j = 0; j < MATRIX_SIZE; j++) {
            int index = (i * MATRIX_SIZE) + j;
            printf("%d ", matrix[index]);
        }
        printf("}\n");
    }
    printf("\n");
}

int main() {
    Matrix left;
    left.width = MATRIX_SIZE;
    left.height = MATRIX_SIZE;
    left.stride = MATRIX_SIZE;
    left.elements = (int*)malloc(left.width * left.height * sizeof(int));

    Matrix right;
    right.width = MATRIX_SIZE;
    right.height = MATRIX_SIZE;
    right.stride = MATRIX_SIZE;
    right.elements = (int*)malloc(right.width * right.height * sizeof(int));

    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            int left_cell_value = genRandom(1, 10);
            left.elements[(left.width * i) + j] = left_cell_value;
            int right_cell_value = genRandom(1, 10);
            right.elements[(right.width * i) + j] = right_cell_value;
        }
    }

    int* validated = (int*)malloc(right.width * right.height * sizeof(int));
    ValidationMult(left.elements, right.elements, validated);

    Matrix left_dev;
    left_dev.width = left.width;
    left_dev.height = left.height;
    left_dev.stride = left.stride;
    cudaError_t error_status = cudaMalloc(&left_dev.elements, left_dev.height * left_dev.width * sizeof(int));
    if (error_status != cudaSuccess) {
        printf("Error cudaMalloc array left_dev\n");
        return 1;
    }
    error_status = cudaMemcpy(left_dev.elements, left.elements, left_dev.height * left_dev.width * sizeof(int), cudaMemcpyHostToDevice);
    if (error_status != cudaSuccess) {
        printf("Error cudaMemcpy left_dev\n");
        return 1;
    }

    Matrix right_dev;
    right_dev.width = right.width;
    right_dev.height = right.height;
    right_dev.stride = right.stride;
    error_status = cudaMalloc(&right_dev.elements, right_dev.height * right_dev.width * sizeof(int));
    if (error_status != cudaSuccess) {
        printf("Error cudaMalloc array right_dev\n");
        return 1;
    }
    error_status = cudaMemcpy(right_dev.elements, right.elements, right_dev.height * right_dev.width * sizeof(int), cudaMemcpyHostToDevice);
    if (error_status != cudaSuccess) {
        printf("Error cudaMmcpy array right_dev\n");
        return 1;
    }

    Matrix result;
    result.width = MATRIX_SIZE;
    result.height = MATRIX_SIZE;
    result.stride = MATRIX_SIZE;
    error_status = cudaMalloc(&result.elements, result.width * result.height * sizeof(int));
    if (error_status != cudaSuccess) {
        fprintf(stderr, "Error allocating memory for matrix result\n");
        return 1;
    }
    // Add vectors in parallel.
    cudaError_t cudaStatus = MatMul(left_dev,right_dev,result);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "MatMult failed\n");
        return 1;
    }

    int result_host[MATRIX_SIZE * MATRIX_SIZE];
    cudaStatus = cudaMemcpy(result_host, result.elements, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("Error cudaMemcpy devide to host error matrix result_host");
        return 1;
    }

    PrintMatrix(left.elements, "left_device");
    PrintMatrix(right.elements, "right_device");
    PrintMatrix(result_host, "result_host");

    // Check diff with validated matrix
    std::vector<std::tuple<int, int>> not_equals;
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            int index = (i * left.width) + j;
            if (validated[index] != result_host[index]) {
                not_equals.push_back(std::make_tuple(i, j));
            }
        }
    }

    PrintMatrix(validated, "validated matrix");

    if (!not_equals.empty()) {
        printf("missmatch in the following indexes\n:");
        for (int i = 0; i < not_equals.size(); i++) {
            printf("%d, %d\n", std::get<0>(not_equals[i]), std::get<1>(not_equals[i]));
        }
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}
