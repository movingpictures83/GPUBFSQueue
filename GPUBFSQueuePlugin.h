#ifndef GPUBFSQUEUEPLUGIN_H
#define GPUBFSQUEUEPLUGIN_H

#include "Plugin.h"
#include "Tool.h"
#include "PluginProxy.h"
#include <string>
#include <map>

class GPUBFSQueuePlugin : public Plugin, public Tool {

	public:
		void input(std::string file);
		void run();
		void output(std::string file);
	private:
                std::string inputfile;
		std::string outputfile;
 //               std::map<std::string, std::string> parameters;
};


#include <stdio.h>

#define BLOCK_SIZE 512
// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

#define gpuTKCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      gpuTKLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Global queuing stub
__global__ void gpu_global_queuing_kernel(
    int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
    int *currLevelNodes, int *nextLevelNodes,
    const unsigned int numCurrLevelNodes, int *numNextLevelNodes) {

  //@@ Insert Global Queuing Code Here

  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Loop over all nodes in the curent level
  for (unsigned int idx = tid; idx < numCurrLevelNodes;
       idx += gridDim.x * blockDim.x) {
    const unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
         ++nbrIdx) {
      const unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      const unsigned int visited = atomicExch(&(nodeVisited[neighbor]), 1);
      if (!visited) {
        // Add it to the global queue (already marked in the exchange)
        const unsigned int gQIdx = atomicAdd(numNextLevelNodes, 1);
        nextLevelNodes[gQIdx]    = neighbor;
      }
    }
  }
}

// Block queuing stub
__global__ void gpu_block_queuing_kernel(
    int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
    int *currLevelNodes, int *nextLevelNodes,
    const unsigned int numCurrLevelNodes, int *numNextLevelNodes) {

  //@@ INSERT KERNEL CODE HERE

  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int numCurrLevelNodes_reg = numCurrLevelNodes;

  // Initialize shared memory queue
  __shared__ int bQueue[BQ_CAPACITY];
  __shared__ int bQueueCount, gQueueStartIdx;
  if (threadIdx.x == 0) {
    bQueueCount = 0;
  }
  __syncthreads();

  // Loop over all nodes in the curent level
  for (unsigned int idx = tid; idx < numCurrLevelNodes_reg;
       idx += gridDim.x * blockDim.x) {
    const unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
         ++nbrIdx) {
      const unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      const unsigned int visited = atomicExch(&(nodeVisited[neighbor]), 1);
      if (!visited) {
        // Add it to the block queue
        const unsigned int bQueueIdx = atomicAdd(&bQueueCount, 1);
        if (bQueueIdx < BQ_CAPACITY) {
          bQueue[bQueueIdx] = neighbor;
        } else { // If full, add it to the global queue
          bQueueCount                  = BQ_CAPACITY;
          const unsigned int gQueueIdx = atomicAdd(numNextLevelNodes, 1);
          nextLevelNodes[gQueueIdx]    = neighbor;
        }
      }
    }
  }
  __syncthreads();

  // Allocate space for block queue to go into global queue
  if (threadIdx.x == 0) {
    gQueueStartIdx = atomicAdd(numNextLevelNodes, bQueueCount);
  }
  __syncthreads();

  // Store block queue in global queue
  for (unsigned int bQueueIdx = threadIdx.x; bQueueIdx < bQueueCount;
       bQueueIdx += blockDim.x) {
    nextLevelNodes[gQueueStartIdx + bQueueIdx] = bQueue[bQueueIdx];
  }
}

#endif
