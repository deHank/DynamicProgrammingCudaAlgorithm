todo: cudaItemsets
cudaItemsets: cudaItemsetMining.cu
	nvcc --expt-extended-lambda --ptxas-options=-v -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -IcuCollections/include -Icccl/libcudacxx/include -Icccl/cub -Icccl/thrust cudaItemsetMining.cu -o cudaItems
clean:
	rm cudaItemsets
