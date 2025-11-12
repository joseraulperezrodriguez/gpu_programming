
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>

#include <stdio.h>
#include <random>

// Distributed Shared memory histogram kernel
__global__ void clusterHist_kernel(int* bins, const int nbins, const int bins_per_block, const int* __restrict__ input,
    size_t array_size) {
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    // Cluster initialization, size and calculating local bin offsets.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
        smem[i] = 0; //Initialize shared memory histogram to zeros
    }

    // cluster synchronization ensures that shared memory is initialized to zero in
    // all thread blocks in the cluster. It also ensures that all thread blocks
    // have started executing and they exist concurrently.
    cluster.sync();

    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x) {
        int ldata = input[i];

        //Find the right histogram bin.
        int binid = ldata;
        if (ldata < 0)
            binid = 0;
        else if (ldata >= nbins)
            binid = nbins - 1;

        //Find destination block rank and offset for computing
        //distributed shared memory histogram
        int dst_block_rank = (int)(binid / bins_per_block);
        int dst_offset = binid % bins_per_block;

        //Pointer to target block shared memory
        int* dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

        //Perform atomic update of the histogram bin
        atomicAdd(dst_smem + dst_offset, 1);
    }

    // cluster synchronization is required to ensure all distributed shared
    // memory operations are completed and no thread block exits while
    // other thread blocks are still accessing distributed shared memory
    cluster.sync();

    // Perform global memory histogram, using the local distributed memory histogram
    int* lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
        atomicAdd(&lbins[i], smem[i]);
    }
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

int main() {

    uint64_t threads_per_block = 4;
    uint64_t array_size = 16;
    uint64_t nbins = 16;

    int* bins = (int*)malloc(array_size * sizeof(int));
    int* __restrict__ input = (int*)malloc(array_size * sizeof(int));

    for (int i = 0; i < array_size; i++) {
        bins[i] = genRandom(1, 100);
        input[i] = bins[i];
    }

    cudaLaunchConfig_t config = { 0 };
    config.gridDim = array_size / threads_per_block;
    config.blockDim = threads_per_block;

    // cluster_size depends on the histogram size.
    // ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
    int cluster_size = 2; // size 2 is an example here
    int nbins_per_block = nbins / cluster_size;

    //dynamic shared memory size is per block.
    //Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
    config.dynamicSmemBytes = nbins_per_block * sizeof(int);

    cudaError_t cudaStatus = ::cudaFuncSetAttribute((void*)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes);
    if (cudaStatus != cudaSuccess) {
        printf(cudaGetErrorString(cudaStatus));
        return 1;
    }

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cluster_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attribute;

    cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}
