#include "GPUBFSQueuePlugin.h"

// Host function for global queuing invocation
void gpu_global_queuing(int *nodePtrs, int *nodeNeighbors,
                        int *nodeVisited, int *currLevelNodes,
                        int *nextLevelNodes,
                        unsigned int numCurrLevelNodes,
                        int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(
      nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
      numCurrLevelNodes, numNextLevelNodes);
}

// Host function for block queuing invocation
void gpu_block_queuing(int *nodePtrs, int *nodeNeighbors, int *nodeVisited,
                       int *currLevelNodes, int *nextLevelNodes,
                       unsigned int numCurrLevelNodes,
                       int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel<<<numBlocks, BLOCK_SIZE>>>(
      nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
      numCurrLevelNodes, numNextLevelNodes);
}

void GPUBFSQueuePlugin::input(std::string file) {
 inputfile = file;
 readParameterFile(file);
}

void GPUBFSQueuePlugin::run() {}




void GPUBFSQueuePlugin::output(std::string file) {
  // Variables
  int numNodes;
  int *nodePtrs_h;
  int *nodeNeighbors_h;
  int *nodeVisited_h;
  int numTotalNeighbors_h;
  int *currLevelNodes_h;
  int *nextLevelNodes_h;
  int numCurrLevelNodes;
  int numNextLevelNodes_h;
  int *nodePtrs_d;
  int *nodeNeighbors_d;
  int *nodeVisited_d;
  int *currLevelNodes_d;
  int *nextLevelNodes_d;
  int *numNextLevelNodes_d;


  enum Mode { GPU_GLOBAL_QUEUE = 2, GPU_BLOCK_QUEUE };
  Mode mode = (Mode) atoi(myParameters["mode"].c_str());
  
  std::ifstream myinput((std::string(PluginManager::prefix())+myParameters["nodes"]).c_str(), std::ios::in);
  numNodes = atoi(myParameters["N"].c_str());
  nodePtrs_h = (int*) malloc(numNodes*sizeof(int));
 for (int i = 0; i < numNodes; ++i) {
        float k;
        myinput >> k;
        nodePtrs_h[i] = k;
 }
  std::ifstream myinput2((std::string(PluginManager::prefix())+myParameters["neighbors"]).c_str(), std::ios::in);
 numTotalNeighbors_h = atoi(myParameters["M"].c_str());
 nodeNeighbors_h = (int*) malloc(numTotalNeighbors_h*sizeof(int));
  for (int i = 0; i < numTotalNeighbors_h; ++i) {
        float k;
        myinput2 >> k;
        nodeNeighbors_h[i] = k;
 }
  std::ifstream myinput3((std::string(PluginManager::prefix())+myParameters["visited"]).c_str(), std::ios::in);
 numNodes = atoi(myParameters["V"].c_str());
 nodeVisited_h = (int*) malloc(numNodes*sizeof(int));
  for (int i = 0; i < numNodes; ++i) {
        float k;
        myinput3 >> k;
        nodeVisited_h[i] = k;
 }
  std::ifstream myinput4((std::string(PluginManager::prefix())+myParameters["currlevel"]).c_str(), std::ios::in);
  numCurrLevelNodes = atoi(myParameters["L"].c_str());
  currLevelNodes_h = (int*) malloc(numCurrLevelNodes*sizeof(int));
 for (int i = 0; i < numCurrLevelNodes; ++i) {
        float k;
        myinput4 >> k;
        currLevelNodes_h[i] = k;
 }

  /*

  gpuTKArg_t args = gpuTKArg_read(argc, argv);
  Mode mode    = (Mode)gpuTKImport_flag(gpuTKArg_getInputFile(args, 0));

  nodePtrs_h =
      (int *)gpuTKImport(gpuTKArg_getInputFile(args, 1), &numNodes, "Integer");
  nodeNeighbors_h = (int *)gpuTKImport(gpuTKArg_getInputFile(args, 2),
                                    &numTotalNeighbors_h, "Integer");

  nodeVisited_h =
      (int *)gpuTKImport(gpuTKArg_getInputFile(args, 3), &numNodes, "Integer");
  currLevelNodes_h = (int *)gpuTKImport(gpuTKArg_getInputFile(args, 4),
                                     &numCurrLevelNodes, "Integer");
*/
  // (do not modify) Datasets should be consistent

  // (do not modify) Prepare next level containers (i.e. output variables)
  numNextLevelNodes_h = 0;
  nextLevelNodes_h    = (int *)malloc((numNodes) * sizeof(int));


  cudaMalloc((void **)&nodePtrs_d, (numNodes + 1) * sizeof(int));
  cudaMalloc((void **)&nodeVisited_d, numNodes * sizeof(int));
  cudaMalloc((void **)&nodeNeighbors_d,
                     nodePtrs_h[numNodes] * sizeof(int));
  cudaMalloc((void **)&currLevelNodes_d,
                     numCurrLevelNodes * sizeof(int));
  cudaMalloc((void **)&numNextLevelNodes_d, sizeof(int));
  
      cudaMalloc((void **)&nextLevelNodes_d, (numNodes) * sizeof(int));
  cudaDeviceSynchronize();

  // (do not modify) Copy host variables to device --------------------


  cudaMemcpy(nodePtrs_d, nodePtrs_h, (numNodes + 1) * sizeof(int),
                     cudaMemcpyHostToDevice);
  cudaMemcpy(nodeVisited_d, nodeVisited_h, numNodes * sizeof(int),
                     cudaMemcpyHostToDevice);
  cudaMemcpy(nodeNeighbors_d, nodeNeighbors_h,
                     nodePtrs_h[numNodes] * sizeof(int),
                     cudaMemcpyHostToDevice);
  cudaMemcpy(currLevelNodes_d, currLevelNodes_h,
                     numCurrLevelNodes * sizeof(int),
                     cudaMemcpyHostToDevice);
  cudaMemset(numNextLevelNodes_d, 0, sizeof(int));
  cudaDeviceSynchronize();

  // (do not modify) Launch kernel ----------------------------------------

  printf("Launching kernel ");

  if (mode == GPU_GLOBAL_QUEUE) {
    gpu_global_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                       currLevelNodes_d, nextLevelNodes_d,
                       numCurrLevelNodes, numNextLevelNodes_d);
    cudaDeviceSynchronize();
  } else if (mode == GPU_BLOCK_QUEUE) {
    gpu_block_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
                      currLevelNodes_d, nextLevelNodes_d,
                      numCurrLevelNodes, numNextLevelNodes_d);
    cudaDeviceSynchronize();
  } else {
    exit(0);
  }

  // (do not modify) Copy device variables from host ----------------------


  cudaMemcpy(&numNextLevelNodes_h, numNextLevelNodes_d,
                     sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(nextLevelNodes_h, nextLevelNodes_d,
                     numNodes * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(nodeVisited_h, nodeVisited_d, numNodes * sizeof(int),
                     cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // (do not modify) Verify correctness
  // -------------------------------------
  // Only check that the visited nodes match the reference implementation

  //gpuTKSolution(args, nodeVisited_h, numNodes);

  // (do not modify) Free memory
  // ------------------------------------------------------------
  free(nodePtrs_h);
  free(nodeVisited_h);
  free(nodeNeighbors_h);
  free(currLevelNodes_h);
  free(nextLevelNodes_h);
  cudaFree(nodePtrs_d);
  cudaFree(nodeVisited_d);
  cudaFree(nodeNeighbors_d);
  cudaFree(currLevelNodes_d);
  cudaFree(numNextLevelNodes_d);
  cudaFree(nextLevelNodes_d);

}

PluginProxy<GPUBFSQueuePlugin> GPUBFSQueuePluginProxy = PluginProxy<GPUBFSQueuePlugin>("GPUBFSQueue", PluginManager::getInstance());

