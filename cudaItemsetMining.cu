#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuco/dynamic_map.cuh>


#define MAX_NODES 6000  // Maximum nodes in the FP-Tree
#define EMPTY -1


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

__global__ void processItemSets(char *inData, int minimumSetNum, int *d_Offsets, int totalRecords) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory is treated as a single contiguous block
    extern __shared__ int sharedMemory[];

    // Assign sections of shared memory to arrays
    int* mutex = &sharedMemory[0];
    int* itemsInNode = &sharedMemory[1];                       // Item at each node
    int* counts = &sharedMemory[1 + MAX_NODES];                    // Support count at each node
    int* parents = &sharedMemory[1 + 2 * MAX_NODES];               // Parent index for each node
    int* firstChild = &sharedMemory[1 + 3 * MAX_NODES];            // First child index for each node
    int* nextSibling = &sharedMemory[1 + 4 * MAX_NODES];           // Next sibling index for each node
    int* nodeCounter = &sharedMemory[1 + 5 * MAX_NODES];           // Counter for nodes in the block (single value)
    

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
    // if (tid < totalRecords) {
    if(blockIdx.x == 0){
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

        // Debugging: Print the FP-Tree for this block (only one thread does this)
        if (tid == 0) {
            printf("FP-Tree for Block %d:\n", blockIdx.x);
            for (int i = 0; i < *nodeCounter; i++) {
                printf("Node %d: Item=%d, Count=%d, Parent=%d, FirstChild=%d, NextSibling=%d\n",
                    i, itemsInNode[i], counts[i], parents[i], firstChild[i], nextSibling[i]);
            }
        }

        __syncthreads();
    }


    
}



__global__ void printStuff(float *test_matrix, float *train_matrix, int numElements, int train_num_instances, int k, int num_attributes, int num_classes, int *predictionsGlobal, int testNumInstances, int stream, int testNumInstancePerStream){
    
    //threadid within this block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < testNumInstancePerStream){
        int idx = stream * testNumInstancePerStream + tid;
        int classCounts[10];  // Each thread gets its own copy of 'localArray'
        // Declare a single shared memory block
        
        // Assign parts of sharedMemory to different arrays
        float candidates[6];                           // First part: candidates
        //int* predictions = (int*)&sharedMemory;     // Second part: predictions
        //int* classCounts = (int*)&sharedMemory[2 * k * blockDim.x + testNumInstances]; // Third part: class counts

        for(int i = 0; i < 6; i++){ candidates[i] = FLT_MAX; }

        //__shared__ float candidates[k*2 * blockDim.x]; //we need to do k*2 * blocksize (blockDim.x)
        //__shared__ int predictions[numElements]; //predicitons array in shared memory (no penalty)
        
        
        
        

    
        //so i know this thread index is less than the number of test elements
        //meaning i would want to run the KNN algorithm for this section 
        //I could call this the outer loop of the KNN Algorithm
        //printf("test\n");
    
    
        //__shared__ float predictions[numElements];

        int queryIndex = idx;
        //and obviously we want to get stride to = 1 
        //But we can use shared memory to compensate for bad strides
        //so is the best way to share the entire training array
        //and that way each test point begins at 1 in the training array
        //stride can be 1?
        //for each thread,

        
        for(int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
                float dist = 1; 
            //float dist = distance(&test_matrix[queryIndex*num_attributes], &train_matrix[keyIndex*num_attributes], num_attributes);
            
            if(idx == 41){
                //printf("getting the distance at idx = %d and it was %.3f and total train size is %d num attributes is %d key index is %d\n",idx, dist, testNumInstances, num_attributes, keyIndex);
                //printf("train num instances is %d\n", train_num_instances);
            }
            
            
            //printf("our distance was %.3f, num classes is %d\n", dist, num_classes);
            // Add to our candidates
            for(int c = 0; c < k; c++){
                if(0 < candidates[2*c]) {
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for(int x = k-2; x >= c; x--) {
                        candidates[2*x+2] = candidates[2*x];
                        candidates[2*x+3] = candidates[2*x+1];
                    }
                    
                    // Set key vector as potential k NN
                    candidates[2*c ] = dist;
                    candidates[2*c+1 ] = train_matrix[keyIndex*num_attributes + num_attributes - 1]; // class value

                    break;
                }
            }
        }

        // Bincount the candidate labels and pick the most common
        for(int i = 0; i < k; i++) {
            classCounts[(int)candidates[2*i+1 ] ] += 1;
            //printf("class counts here are %d\n", classCounts[(int)candidates[2*i+1 + candidatesOffset] + classCountsOffset]);
        }
        
        int max_value = -1;
        int max_class = 0;
        for(int i = 0; i < num_classes; i++) {
            if(classCounts[i] > max_value) {
                max_value = classCounts[i ];
                max_class = i;
                if(idx < 10){
                    //printf("max class at 41 was %d and the classCountOfset is %d \n", max_class, classCountsOffset);
                    //printf("my thread is is %d\n", threadIdx.x);
                }
                //printf("class count at i= %d is %d\n", i, classCounts[i + classCountsOffset]);
            }
            
        }

        //printf("Thread %d: max_class = %d\n", idx, max_class);
        //printf("predictions %d is %d\n", idx, predictions[idx]);
        __syncthreads();
        
        
        
        predictionsGlobal[idx] = max_class;
    
        // Make prediction with 
        
        
        //printf("max class was %d\n, and predictions at %d was %d\n", max_class, idx, predictions[idx]);


        
        
        
        //printf("max class was %d\n, and predictionsGlobal at %d was %d\n", max_class, idx, predictionsGlobal[idx]);


    }
}

// Implements a threaded kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int KNN() {     
    int lineCountInDataset = 1692082;
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
    size_t sharedMemSize = (6 * MAX_NODES) * sizeof(int) +  2 * sizeof(int) ;  // 5 arrays + nodeCounter
    // Allocate memory on the GPU
    char* d_text;
    int* d_offsets; 
    cudaMalloc(&d_text, file_size);
    cudaMalloc(&d_offsets, lineCountInDataset * sizeof(int));
    // Copy the file contents to the GPU
    cudaMemcpy(d_text, h_buffer, file_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, lineCountInDataset * sizeof(int), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = ((lineCountInDataset + threadsPerBlock) - 1) /  threadsPerBlock; //how do we know how many blocks we need to use?
    printf("number of threads is roughly %d\n", threadsPerBlock*blocksPerGrid);
    //1_692_082 lineCount of Sorted DataBase
    int minItemCount = 3; //setting the minimum # of items to be considered an itemset

    //here I would want to generate all itemsets
    cudaFuncSetAttribute(processItemSets, cudaFuncAttributeMaxDynamicSharedMemorySize, 164 * 1024);

    processItemSets<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_text, minItemCount, d_offsets, lineCountInDataset);
    cudaDeviceSynchronize();
    return 1;



    //so essentially my goal is to produce all of these itemssets 
    //i want to take a previous gpu solution and use NVIDIA's hardware based 
    //dynamic programming APIs 
    //we send the transactional database to the GPU
    
    //1. We want to have a hashtable (key, value pairs)
    //our keys will be our itsemset, and our value will be the count 
    //we will do this for each itsemset and transaction
    
    //each thread will then reduce the hashtable into the shared memory
    //maybe we should begin with a pure list?
    //


    // int* predictions = (int*)malloc(test->num_instances() * sizeof(int));

    // int num_classes = train->num_classes();
    // int num_attributes = train->num_attributes();
    // int train_num_instances = train->num_instances();

    // int test_num_instances = test->num_instances();
    // int testNumClasses = test->num_classes();
    // int testNumAttributes = test-> num_attributes();
    

    // float *d_testSet;
    // float *d_trainSet;

    // int *d_predictions;

    // int numElements = num_attributes * train_num_instances;
    // int testNumElements = testNumAttributes * test_num_instances;
    // // Pointers representing the dataset as a 2D matrix size num_instances x num_attributes
    // float *train_matrix = train->get_dataset_matrix();  // Use directly
    // float *test_matrix = test->get_dataset_matrix();    // Use directly

    
    // int numStreams = 4;

    // //need to cudamalloc to hold the training data
    // //CUDA MALLOC for the train array
    // cudaMalloc(&d_trainSet, (numElements) * sizeof(float));
    // cudaMalloc(&d_testSet,(testNumElements) *  sizeof(float));
    // cudaMalloc(&d_predictions, (test_num_instances * sizeof(int)));

    // cudaStream_t *streams = (cudaStream_t*) malloc (numStreams * sizeof(cudaStream_t));

    // for (int i = 0; i < numStreams; i++){
    //     cudaStreamCreate(&streams[i]);
    // }

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("CUDA error: %s\n", cudaGetErrorString(err));
    // }

    // //int numberTrainElementsPerStream = (numElements + numStreams - 1) / numStreams;
    // int numberTestElementsPerStream = (testNumElements + numStreams - 1)/numStreams;
    // int numTestInstancesPerStream = (test_num_instances + numStreams - 1)/numStreams;
    
    
    
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // float milliseconds = 0;
    
    // int threadsPerBlockDim = 64;

    // // Calculate the size of shared memory needed
    // int sharedMemSize = 0;                  // For classCounts
    // printf("size of memory usage is %d\n", sharedMemSize);
    // int gridDimSize = (numTestInstancesPerStream + threadsPerBlockDim - 1) / threadsPerBlockDim;

    // cudaEventRecord(start);
    // printf("before we call the kernel\n");
    // //cudaMemcpy(d_trainSet, train_matrix, (numElements)*sizeof(float), cudaMemcpyHostToDevice);
    // printf("size of numberTestElementsPerStream is %d\n", numberTestElementsPerStream);
    // printf("size of numberTestInstancePerStream is %d\n", numTestInstancesPerStream);
    // for (int i = 0; i < numStreams; i++)
    // {
    //     //copying the train set to our device (GPU)
    //     //0.041824 ms each way for 2 things... so why is performance with streams so poor?
    //     cudaMemcpyAsync(d_trainSet, train_matrix, (numElements)*sizeof(float),cudaMemcpyHostToDevice, streams[i]);
    //     cudaMemcpyAsync(&d_testSet[i*numberTestElementsPerStream], &test_matrix[i*numberTestElementsPerStream], numberTestElementsPerStream  *  sizeof(float), cudaMemcpyHostToDevice, streams[i]);
    //     //cudaMemcpyAsync(&d_predictions[i*numTestInstancesPerStream], &predictions[i*numTestInstancesPerStream], numTestInstancesPerStream * sizeof(int), cudaMemcpyHostToDevice, streams[i]);

    //     printStuff<<<gridDimSize, threadsPerBlockDim, sharedMemSize, streams[i]>>>(d_testSet, d_trainSet, numberTestElementsPerStream, train_num_instances, k, num_attributes, num_classes, d_predictions, test_num_instances, i, numTestInstancesPerStream);
        
    //     cudaMemcpyAsync(&predictions[i*numTestInstancesPerStream], &d_predictions[i*numTestInstancesPerStream], numTestInstancesPerStream * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);

    // }

    // cudaDeviceSynchronize();  // Ensure kernel finishes THIS IS SO IMPORTANT 
    // //OTHERWISE THE PROGRAM WILL JUST END BEFORE THE KERNELS EVEN GET A CHANCE TO LAUNCH

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("v1 GPU option time to sum the matrixes %f ms\n", milliseconds);

    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, 0);

    // printf("GPU %d: %s\n", 0, deviceProp.name);
    
    

    // printf("after we call the kernel\n");
    
    // cudaFree(d_trainSet);
    // cudaFree(d_testSet);
    // free(streams);
    // cudaFree(d_predictions);

    

    // return predictions;
}

int main(int argc, char *argv[])
{
    

    int x = KNN();
    return -1;  
}

/*  // Example to print the test dataset
    float* test_matrix = test->get_dataset_matrix();
    for(int i = 0; i < test->num_instances(); i++) {
        for(int j = 0; j < test->num_attributes(); j++)
            printf("%.0f, ", test_matrix[i*test->num_attributes() + j]);
        printf("\n");
    }
*/
