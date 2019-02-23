#include "memory.h"

#include <assert.h>

void
cuda_malloc_clear(void** ptr, size_t bytes)
{
    cudaError_t err;
    
    // Malloc to device, check for errors
    err = cudaMalloc(ptr, bytes);
    assert(err == cudaSuccess);

    // Set val to 0, check for errors
    err = cudaMemset(*ptr, 0, bytes);
    assert(err == cudaSuccess);
}
