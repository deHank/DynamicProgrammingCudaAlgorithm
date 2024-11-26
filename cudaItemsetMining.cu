    #include <stdio.h>
    #include <stdlib.h>
    #include <pthread.h>
    #include <bits/stdc++.h>
    #include <cuda_runtime.h>
    #include "libarff/arff_parser.h"
    #include "libarff/arff_data.h"


    struct ThreadData{
        ArffData *train;  // Pointer to the first array
        ArffData *test;  // Pointer to the second array
        int* predictions;    // Start index
        int TRstart;  //start index of array
        int TRend;      // End index
        int TTstart;
        int TTend;
        int k;  
    } ;



    // Calculates the distance between two instances
    __device__ float distance(float* instance_A, float* instance_B, int num_attributes) {
        float sum = 0;
        
        for (int i = 0; i < num_attributes-1; i++) {
            float diff = instance_A[i] - instance_B[i];
            //printf("instance a and b were %.3f %.3f\n", instance_A[i] ,instance_B[i]);
            sum += diff*diff;
        }
        //printf("sum was %.3f\n,", sum);
        return sqrt(sum);
    }



    __global__ void printStuff(float *test_matrix, float *train_matrix, 
    int numElements, int train_num_instances, int k, int num_attributes, int num_classes, int *predictionsGlobal, int testNumInstances, int stream, int testNumInstancePerStream){
        
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
                   
                float dist = distance(&test_matrix[queryIndex*num_attributes], &train_matrix[keyIndex*num_attributes], num_attributes);
                
                if(idx == 41){
                    //printf("getting the distance at idx = %d and it was %.3f and total train size is %d num attributes is %d key index is %d\n",idx, dist, testNumInstances, num_attributes, keyIndex);
                    //printf("train num instances is %d\n", train_num_instances);
                }
                
                
                //printf("our distance was %.3f, num classes is %d\n", dist, num_classes);
                // Add to our candidates
                for(int c = 0; c < k; c++){
                    if(dist < candidates[2*c]) {
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
    int* KNN(ArffData* train, ArffData* test, int k, int num_threads) {     
        
        const char* inDataFilePath = "sortedDataBase.txt";

        FILE* file = fopen(inDataFilePath, "r");

        // Get the file size
        fseek(file, 0, SEEK_END);
        size_t file_size = ftell(file);
        rewind(file);
        

            // Allocate memory to hold the file contents
        char* h_text = (char*)malloc(file_size);

        // Read the file into the host buffer
        fread(h_text, 1, file_size, file);
        fclose(file);

        // Allocate memory on the GPU
        char* d_text;
        cudaMalloc((void**)&d_text, file_size);

        // Copy the file contents to the GPU
        cudaMemcpy(d_text, h_text, file_size, cudaMemcpyHostToDevice);

        

        //so essentially my goal is to produce all of these itemssets 
        //i want to take a previous gpu solution and use NVIDIA's hardware based 
        //dynamic programming APIs 
        //we send the transactional database to the GPU
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

    int* computeConfusionMatrix(int* predictions, ArffData* dataset)
    {
        int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses
        
        for(int i = 0; i < dataset->num_instances(); i++) { // for each instance compare the true class and predicted class    
            int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
            int predictedClass = predictions[i];
            //printf("predictions[%d] = %d , true class was %d\n", i, predictions[i], trueClass);
            confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
        }
        
    
        return confusionMatrix;
    }

    float computeAccuracy(int* confusionMatrix, ArffData* dataset)
    {
        int successfulPredictions = 0;
        
        for(int i = 0; i < dataset->num_classes(); i++) {
            successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
        }
        
        return 100 * successfulPredictions / (float) dataset->num_instances();
    }

    int main(int argc, char *argv[])
    {
        if(argc != 5)
        {
            printf("Usage: ./program datasets/train.arff datasets/test.arff k num_threads");
            exit(0);
        }

        // k value for the k-nearest neighbors
        int k = strtol(argv[3], NULL, 10);
        int num_threads = strtol(argv[4], NULL, 10);

        // Open the datasets
        ArffParser parserTrain(argv[1]);
        ArffParser parserTest(argv[2]);
        ArffData *train = parserTrain.parse();
        ArffData *test = parserTest.parse();
        
        struct timespec start, end;
        int* predictions = NULL;
        
        // Initialize time measurement
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        
        predictions = KNN(train, test, k, num_threads);
        
        // Stop time measurement
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        // Compute the confusion matrix
        int* confusionMatrix = computeConfusionMatrix(predictions, test);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, test);

        uint64_t time_difference = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

        printf("The %i-NN classifier for %lu test instances and %lu train instances required %llu ms CPU time for threaded with %d threads. Accuracy was %.2f\%\n", k, test->num_instances(), train->num_instances(), (long long unsigned int) time_difference, accuracy, num_threads);

        // Print the predictions array
        // printf("Predictions array after cudaMemcpy:\n");
        // for (int i = 0; i < test->num_instances(); i++) {
        //     printf("predictions[%d] = %d\n", i, predictions[i]);
        // }

        free(predictions);
        free(confusionMatrix);
    }

    /*  // Example to print the test dataset
        float* test_matrix = test->get_dataset_matrix();
        for(int i = 0; i < test->num_instances(); i++) {
            for(int j = 0; j < test->num_attributes(); j++)
                printf("%.0f, ", test_matrix[i*test->num_attributes() + j]);
            printf("\n");
        }
    */