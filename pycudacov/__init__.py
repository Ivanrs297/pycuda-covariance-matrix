import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import sys
from sys import getsizeof
import time
import numpy as np
import pandas as pd

# Define CUDA function
mod = SourceModule(
    """

__global__ void get_cov(int cols, int total_rows, float *d_A, float *d_means, float *d_covariances) {

	extern __shared__ float s_sum[];  					// sum of the values per row per block
	int tid = threadIdx.x;  							// Local: Thread ID
	unsigned int g_tid = 0;								// Global ID of the first feature
	unsigned int g_tid_2 = 0; 							// Global ID of the second feature
	unsigned int b_id = 0;								// Block ID

	for (int c = 0; c < __float2uint_ru((float)cols/(float)gridDim.x); c++) {
		b_id = (c * gridDim.x) + blockIdx.x;
		if (b_id < cols) {

			// First, get the means:
			for ( int i = 0; i < __float2uint_ru((float)total_rows/(float)blockDim.x); i++) {
				g_tid = (i * blockDim.x + tid) + (b_id * total_rows);
				s_sum[tid] = d_A[g_tid];
				__syncthreads();

				// Inclusive Scan
				float temp = 0.0;
				for (int j = 1; j < blockDim.x; j *= 2 ) {
					if ( (tid - j) >= 0)
						temp = s_sum[tid - j];
					__syncthreads();
					if ( (tid - j) >= 0)
						s_sum[tid] += temp;
					__syncthreads();
				}

				if(tid == blockDim.x - 1) {
					d_means[b_id] += s_sum[tid];
				}
				__syncthreads();		
			}

			// Save the result of Feature-Block on global memory
			if(tid == blockDim.x - 1)
				d_means[b_id] /= total_rows;
			__syncthreads();	
		}
	}	

	// Then, compute the covariance:
	for (int c = 0; c < __float2uint_ru((float)cols/(float)gridDim.x); c++) {
			// Iterate over features, starting from the actual feature
			b_id = (c * gridDim.x) + blockIdx.x;
			if (b_id < cols) {
				for (int i = b_id; i < cols; i++ ){

					// Index of cell i,j
					int index = (b_id * cols) + i;

					// index of cell j,i
					int index_2 = (i * cols) + b_id;
					
					// Iterate over the size of samples
					for ( int k = 0; k < __float2uint_ru((float)total_rows/(float)blockDim.x); k++) {

						// Calculate mapped indexes
						g_tid = (k * blockDim.x + tid) + (b_id * total_rows);			
						g_tid_2 = (k * blockDim.x + tid) + (total_rows * i);

						// Compute the covariance
						s_sum[tid] = (d_A[g_tid] - d_means[b_id]) * (d_A[g_tid_2] - d_means[i]);
						__syncthreads();

						// Inclusive scan
						float temp;
						for (int j = 1; j < blockDim.x; j *= 2 ){
							if ( (tid - j) >= 0)
								temp = s_sum[tid - j];
							__syncthreads();
							if ( (tid - j) >= 0)
								s_sum[tid] += temp;
							__syncthreads();
						}

						if(tid == blockDim.x - 1) {
							d_covariances[index] += s_sum[tid];

						}
						__syncthreads();
					}

					// Save the result of Feature-Block on global memory
					if(tid == blockDim.x - 1) {
						float aux = d_covariances[index] / total_rows;
						d_covariances[index] = aux;

						// Symmetric Cell
						// if not diagonal
						if ( index % (cols+1) != 0 ) {
							d_covariances[index_2] = aux;
						}
					}
				}
			}
	}

}

"""
)

func = mod.get_function("get_cov")


def get_cov(A, blocks=None, threads=None):

    rows, cols = A.shape
    rows = int(rows)
    cols = int(cols)

    # Assign block and thread size
    if blocks and threads and blocks <= 1024 and threads <= 1024:
        blockCount = blocks
        threadCount = threads
    else:
        # Number of threads per block
        if rows >= 1024:
            threadCount = 1024
        else:
            threadCount = rows

        # Number of blocks per grid
        if cols >= 1024:
            blockCount = 1024
        else:
            blockCount = cols

    # Host Memory
    means = np.zeros(cols)
    means = means.astype(np.float32)

    covariances = np.zeros(cols * cols)
    covariances = covariances.astype(np.float32)

    # Allocate on device
    d_A = cuda.mem_alloc(A.size * A.dtype.itemsize)
    d_means = cuda.mem_alloc(means.size * means.dtype.itemsize)
    d_covariances = cuda.mem_alloc(covariances.size * covariances.dtype.itemsize)

    # Copy from host to device
    cuda.memcpy_htod(d_A, A)
    cuda.memcpy_htod(d_means, means)
    cuda.memcpy_htod(d_covariances, covariances)

    # # Number of threads per block
    # if rows >= 1024:
    #     threadCount = 1024
    # else:
    #     threadCount = rows

    # # Number of blocks per grid
    # blockCount = cols

    # Start GPU time
    start = cuda.Event()
    end = cuda.Event()
    start.record()

    # Run Kernel
    func(
        np.int32(cols),
        np.int32(rows),
        d_A,
        d_means,
        d_covariances,
        block=(threadCount, 1, 1),
        grid=(blockCount, 1),
        shared=threadCount * A.dtype.itemsize,
    )

    # End GPU time
    end.record()
    end.synchronize()
    ms = start.time_till(end)

    # Copy result to host
    cuda.memcpy_dtoh(covariances, d_covariances)

    # Return Covariance Matrix
    return np.resize(covariances, (cols, cols)), ms


# Test funciton
def hello():
    print("Hello Ivan and World")

