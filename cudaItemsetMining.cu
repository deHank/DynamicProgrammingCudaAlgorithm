#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuco/dynamic_map.cuh>
#include <unordered_map>
#include <iostream>
#include <string.h>
#include <time.h>

#define MAX_NODES 6000  // Maximum nodes in the FP-Tree
#define EMPTY -1

typedef struct {
    int id; 
    int processed; // 1 signifies it was processed
    int itemSet;
    int count; 
    int parent;
    int nextSibling; 
    int firstChild; 
} Node; 

// Calculates the distance between two instances
__device__ float generateItemSet(float* instance_A, float* instance_B, int num_attributes) {
    float sum = 0;
    
    for (int i = 0; i < num_attributes-1; i++) {
        float diff = instance_A[i] - instance_B[i];
        //printf("instance a and b were %.3f %.3f\n", instance_A[i] ,instance_B[i]);
        sum += diff*diff;
    }
    //printf("sum was %.3f\n,", sum);
    return sqrt(sum);
}

__global__ void processItemSets(char *inData, int minimumSetNum, int *d_Offsets, int totalRecords, Node* d_fpSubtrees, int* d_treeSizes, int blocksPerGrid) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory is treated as a single contiguous block
    extern __shared__ int sharedMemory[];

    // Assign sections of shared memory to arrays
    int* mutex = &sharedMemory[0];
    int* itemsInNode = &sharedMemory[1];                       // Item at each node
    int* counts = &sharedMemory[2 + MAX_NODES];                    // Support count at each node
    int* parents = &sharedMemory[2 + 2 * MAX_NODES];               // Parent index for each node
    int* firstChild = &sharedMemory[2 + 3 * MAX_NODES];            // First child index for each node
    int* nextSibling = &sharedMemory[2 + 4 * MAX_NODES];           // Next sibling index for each node
    int* nodeCounter = &sharedMemory[2 + 5 * MAX_NODES];           // Counter for nodes in the block (single value)
    

    // Initialize the shared memory (done by thread 0 in each block)
    if (threadIdx.x == 0) {
        //printf("FP-Tree for Block %d:\n", blockIdx.x);
        *mutex = 0;
        *nodeCounter = 1;  // Root node is index 0
        for (int i = 0; i < MAX_NODES; i++) {
            itemsInNode[i] = EMPTY;
            counts[i] = 0;
            parents[i] = EMPTY;
            firstChild[i] = EMPTY;
            nextSibling[i] = EMPTY;
        }
    }
    __syncthreads();

    // Parse the input and build the FP-Tree
    if (tid < totalRecords) {
    // if(blockIdx.x == 0){
        // Parse the transaction data
        char* line = inData + d_Offsets[tid];
        int items[32];  // Local array to store items in the transaction
        int itemCount = 0;
        int number = 0;
        bool inNumber = false;

        // Extract items from the input line
        for (char* current = line; *current != '\n' && *current != '\0'; current++) {
            if (*current >= '0' && *current <= '9') {
                number = number * 10 + (*current - '0');
                inNumber = true;
            } else if (inNumber) {
                items[itemCount++] = number;
                number = 0;
                inNumber = false;
            }
        }
        if (inNumber) {
            items[itemCount++] = number;
        }

        // Generate all subsets and insert into the FP-Tree
        int totalSubsets = 1 << itemCount;  // 2^itemCount
        for (int mask = 0; mask < totalSubsets; mask++) {
            int parentNode = 0;  // Start at the root for each subset
            for (int i = 0; i < itemCount; i++) {
                if (mask & (1 << i)) {
                    int item = items[i];

                    bool isSet = false; 
                    do 
                    {
                        if (isSet = atomicCAS(mutex, 0, 1) == 0) {
                            // Search for the child node with this item
                            int childNode = firstChild[parentNode];
                            while (childNode != EMPTY && itemsInNode[childNode] != item) {
                                childNode = nextSibling[childNode];
                            }
                            
                            // If the item doesn't exist, create a new node
                            if (childNode == EMPTY) {
                                int newNodeIndex = atomicAdd(nodeCounter, 1);
                                if (newNodeIndex < MAX_NODES) {
                                    //itemsInNode[newNodeIndex] = item;
                                    atomicExch(&itemsInNode[newNodeIndex], item);
                                    //counts[newNodeIndex] = 1;  // Initialize count
                                    atomicExch(&counts[newNodeIndex], 1);
                                    // parents[newNodeIndex] = parentNode;
                                    atomicExch(&parents[newNodeIndex], parentNode);
                                    // firstChild[newNodeIndex] = EMPTY;
                                    atomicExch(&firstChild[newNodeIndex], EMPTY);
                                    // nextSibling[newNodeIndex] = EMPTY;
                                    atomicExch(&nextSibling[newNodeIndex], EMPTY);

                                    // Link to parent's child list
                                    if (atomicCAS(&firstChild[parentNode], EMPTY, newNodeIndex) != EMPTY) {
                                        int sibling = firstChild[parentNode];
                                        while (atomicCAS(&nextSibling[sibling], EMPTY, newNodeIndex) != EMPTY) {
                                            sibling = nextSibling[sibling];
                                        }
                                    }
                                    childNode = newNodeIndex;
                                }
                            } else {
                                // Increment the count for an existing node
                                atomicAdd(&counts[childNode], 1);
                            }

                            // Move to the child node
                            parentNode = childNode;
                        }
                        if (isSet) 
                        {
                            atomicExch(mutex, 0);
                        }
                     } 
                while (!isSet); 
                }
            }
        }
        __syncthreads();

        // if (tid == 0) {
        //     printf("FP-Tree for Block %d:\n", blockIdx.x);
        //     for (int i = 0; i < *nodeCounter; i++) {
        //         printf("Node %d: Item=%d, Count=%d, Parent=%d, FirstChild=%d, NextSibling=%d\n",
        //             i, itemsInNode[i], counts[i], parents[i], firstChild[i], nextSibling[i]);
        //     }
        // }

        // send each block's subtree to the cpu 
        int max = MAX_NODES;
        if (threadIdx.x == 0) { // first thread of every block delegates global writes
            int numOfNodes = 0; 
           //printf("I am block %d\n", blockIdx.x);
            for (int i = 0; i < *nodeCounter; i++) {
                d_fpSubtrees[max * blockIdx.x + numOfNodes].itemSet = itemsInNode[i];
                d_fpSubtrees[max * blockIdx.x + numOfNodes].count = counts[i];
                d_fpSubtrees[max * blockIdx.x + numOfNodes].parent = parents[i];
                d_fpSubtrees[max * blockIdx.x + numOfNodes].firstChild = firstChild[i];
                d_fpSubtrees[max * blockIdx.x + numOfNodes].nextSibling = nextSibling[i];
                d_fpSubtrees[max * blockIdx.x + numOfNodes].id = i; // set id of node

                // printf("DP Node %d: Item=%d, Count=%d, Parent=%d, FirstChild=%d, NextSibling=%d\n",
                //     i, d_fpSubtrees[max * blockIdx.x + numOfNodes].itemSet, d_fpSubtrees[max * blockIdx.x + numOfNodes].count, d_fpSubtrees[max * blockIdx.x + numOfNodes].parent, 
                //     d_fpSubtrees[max * blockIdx.x + numOfNodes].firstChild, d_fpSubtrees[max * blockIdx.x + numOfNodes].nextSibling);

                numOfNodes++; 
            }

            // write the size of the block's fp subtree
             d_treeSizes[blockIdx.x] = numOfNodes;
        }
        __syncthreads();
    }


    
}

// Implements a threaded kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int KNN() {   
    clock_t cpu_start_withSetup = clock();
    
    clock_t setupTimeStart = clock();
    int lineCountInDataset = 55012;
    const char* inDataFilePath = "sortedDataBase.txt";

    FILE* file = fopen(inDataFilePath, "r");

    // Get the file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    rewind(file);

    char* h_buffer = (char*)malloc(file_size);
    fread(h_buffer, 1, file_size, file);
    

    // Count the number of lines and create offsets
    int* h_offsets = (int*)malloc((file_size + 1) * sizeof(int));
    int lineCount = 0;
    h_offsets[lineCount++] = 0; // First line starts at the beginning
    
    for (size_t i = 0; i < file_size; i++) {
        //printf("are we in size?");
        if (h_buffer[i] == '\n') {
            //printf("we are in the newline stuff");
            h_offsets[lineCount++] = i + 1; // Next line starts after '\n'
            
        }
    }
    
    // Allocate memory to hold the file contents
    char* h_text = (char*)malloc(file_size);

    // Read the file into the host buffer
    fread(h_text, 1, file_size, file);
    //fclose(file);
    size_t sharedMemSize = (6 * MAX_NODES) * sizeof(int) +  1 * sizeof(int) ;  // 5 arrays + nodeCounter
    
    // Allocate memory on the GPU
    char* d_text;
    int* d_offsets; 
    cudaMalloc(&d_text, file_size);
    cudaMalloc(&d_offsets, lineCountInDataset * sizeof(int));

    // Copy the file contents to the GPU
    cudaMemcpy(d_text, h_buffer, file_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, lineCountInDataset * sizeof(int), cudaMemcpyHostToDevice);
    int threadsPerBlock = 32;
    int blocksPerGrid = ((lineCountInDataset + threadsPerBlock) - 1) /  threadsPerBlock; //how do we know how many blocks we need to use?
    //printf("BlocksPerGrid = %d\n", blocksPerGrid);
    //printf("number of threads is roughly %d\n", threadsPerBlock*blocksPerGrid);

    // allocate space for the fpSubstree node list in GPU
    Node* d_fpSubtrees;
    int* d_treeSizes; 
    int maxNodesPerBlock = MAX_NODES; // Adjust if necessary
    cudaMalloc(&d_fpSubtrees, blocksPerGrid * maxNodesPerBlock * sizeof(Node));
    cudaMalloc(&d_treeSizes, blocksPerGrid * sizeof(int));
    
    //1_692_082 lineCount of Sorted DataBase
    int minItemCount = 3; //setting the minimum # of items to be considered an itemset

    //here I would want to generate all itemsets
    cudaFuncSetAttribute(processItemSets, cudaFuncAttributeMaxDynamicSharedMemorySize, 164 * 1024);
    clock_t setupTimeEnd = clock();

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float cudaElapsedTime;

    
    cudaEventRecord(startEvent);
    processItemSets<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_text, minItemCount, d_offsets, lineCountInDataset, d_fpSubtrees, d_treeSizes, blocksPerGrid);
    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    // Print the elapsed time (milliseconds)
    cudaEventElapsedTime(&cudaElapsedTime, startEvent, stopEvent);
    printf("CUDA Kernel Execution Time: %.3f ms\n", cudaElapsedTime);

    // ensure there are no kernel errors
    cudaError_t cudaError = cudaGetLastError();
    if(cudaError != cudaSuccess) {
        fprintf(stderr, "processItemSets cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    clock_t retrieveGPUResultsStart = clock();
    // copy each block's fp subtree nodes back to host (CPU)
    Node* h_fpSubtrees = (Node*)malloc(blocksPerGrid * maxNodesPerBlock * sizeof(Node));
    int* h_treeSizes = (int*)malloc(blocksPerGrid * sizeof(int));

    // copy the blocks' subtrees + subtree counts to host 
    cudaMemcpy(h_fpSubtrees, d_fpSubtrees, blocksPerGrid * maxNodesPerBlock * sizeof(Node), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_treeSizes, d_treeSizes, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
    clock_t retrieveGPUResultsEnd = clock();

    // global reduction will be written to file
    FILE *resultsFile = fopen("cudaItemSetMiningResults.txt", "w");
    if (resultsFile == NULL) {
        perror("Error opening results file");
        return 1;
    }

    //create hash table (<itemset : count>)
    std::unordered_map<std::string, int> itemsetMap;
    int totalNodes = 0;
    
    clock_t cpu_start = clock();
    // traverse all nodes
   for (int i = 0; i < blocksPerGrid; i++) {
        int numNodesInBlock = h_treeSizes[i];
        totalNodes += numNodesInBlock; 
        //printf("Block %d has %d nodes\n", i, numNodesInBlock);
        for (int j = 0; j < numNodesInBlock; j++) {

            // if is an unporcessed leaf, follow back up to the parent and print the itemsets on one line. make sure to keep track of count
            while(h_fpSubtrees[maxNodesPerBlock * i + j].count > 0 &&  h_fpSubtrees[maxNodesPerBlock * i + j].firstChild == -1) {
                int count = 1; 
                h_fpSubtrees[maxNodesPerBlock * i + j].count += -1;
                char buffer[512] = ""; // will be used to build the itemset string
                int currParentId = h_fpSubtrees[maxNodesPerBlock * i + j].parent; // stores id of parent node 

                // include the leaf itself
                if (h_fpSubtrees[maxNodesPerBlock * i + j].itemSet != -1) {
                    char itemToAdd[32];
                    sprintf(itemToAdd, "%d ", h_fpSubtrees[maxNodesPerBlock * i + j].itemSet);
                    strcat(buffer, itemToAdd);
                }

                // follow back up until we reach the root of this path that led to leaf
                while (currParentId != -1) { 
                    //h_fpSubtrees[maxNodesPerBlock * i + currParentId].count += 1;
                    int nodeItemSet = h_fpSubtrees[maxNodesPerBlock * i + currParentId].itemSet;
                    
                    //count+= h_fpSubtrees[maxNodesPerBlock * i + currParentId].count - 1;
                    h_fpSubtrees[maxNodesPerBlock * i + currParentId].count += -1;
                    if (nodeItemSet != -1) { // don't print empty set 
                        char itemToAdd[32]; 
                        sprintf(itemToAdd, "%d ", nodeItemSet); 
                        strcat(buffer, itemToAdd);

                        

                        //fprintf(resultsFile, "%d ", nodeItemSet);
                    } 

                    // mark as processed and move to next parent
                    h_fpSubtrees[maxNodesPerBlock * i + currParentId].processed = 1;
                     h_fpSubtrees[maxNodesPerBlock * i + currParentId].count += -1;
                    currParentId = h_fpSubtrees[maxNodesPerBlock * i + currParentId].parent;
                } 

                // finally print the count on the line 
                if (count != 0) { // avoids the empty set count edge case 
                    // if (count > 2) {
                        // add count to the end of the itemset line string we will write
                        char countToAdd[32]; 
                        //sprintf(countToAdd, "(%d)\n", count); 
                        //strcat(buffer, countToAdd);

                        //fputs(buffer, resultsFile);

                        // Store the itemset in the map
                        std::string itemsetKey(buffer);
                        // If the itemset already exists, increment its count; otherwise, insert it
                        if (itemsetMap.find(itemsetKey) != itemsetMap.end()) {
                            itemsetMap[itemsetKey]++;
                        } else {
                            itemsetMap[itemsetKey] = 1;
                        }


                        //fprintf(resultsFile, "(%d)\n", count);
                    //}
                }
            }

            // printf("Node %d: Item=%d, Count=%d, Parent=%d, FirstChild=%d, NextSibling=%d\n",
            //     j, h_fpSubtrees[maxNodesPerBlock * i + j].itemSet, 
            //     h_fpSubtrees[maxNodesPerBlock * i + j].count, 
            //     h_fpSubtrees[maxNodesPerBlock * i + j].parent, 
            //     h_fpSubtrees[maxNodesPerBlock * i + j].firstChild, 
            //     h_fpSubtrees[maxNodesPerBlock * i + j].nextSibling);    
        }
    }

    clock_t cpu_end = clock();
    // Filter and process the hashmap
    for (const auto& [itemset, freq] : itemsetMap) {
        if (freq >= 3) {
            // Print to console
            //std::cout << "Itemset: " << itemset << " -> Count: " << freq << "\n";
            
            // Save to file
            fprintf(resultsFile, "Itemset: %s -> Count: %d\n", itemset.c_str(), freq);
        }
    }

    fclose(resultsFile);
    

    // Record end time
    clock_t cpu_end_withSetup = clock();
    // Calculate elapsed time in milliseconds
    float cpuElapsedTime = ((float)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    float cpuElapsedTimeSetup = ((float)(cpu_end_withSetup - cpu_start_withSetup)) / CLOCKS_PER_SEC * 1000.0;
    float setupTime = ((float)(setupTimeEnd - setupTimeStart)) / CLOCKS_PER_SEC * 1000.0;
    float gpuRetrievalTime = ((float)(retrieveGPUResultsEnd - retrieveGPUResultsStart)) / CLOCKS_PER_SEC * 1000.0;

    printf("CPU Execution Time: %.3f ms\n", cpuElapsedTime);
    printf("Total Runtime: %.3f ms\n", cudaElapsedTime + cpuElapsedTime);
    printf("Total Runtime (with setup/file write): %.3f ms\n", cpuElapsedTimeSetup);
    printf("Total Setup Time: %.3f ms\n", setupTime);
    printf("Total GPU Results Retrieval Time: %.3f ms\n", gpuRetrievalTime);
    //printf("Proccessed %d nodes\n", totalNodes);
    // // Print the aggregated counts (if has no child then follow up to the parent)
    // printf("{ ");
    // for (const auto& [itemSet, count] : map) {
    //     std::cout << itemSet << ": " << count << '\n';
    // } printf("}");
    return 1;
}

int main(int argc, char *argv[])
{
    

    int x = KNN();
    return -1;  
}