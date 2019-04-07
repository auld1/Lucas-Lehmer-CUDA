#include "compare.h"

#include "bigint.h"
#include "memory.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


#define COMPARE_BLOCK_SIZE (128)

__global__ void
ge_word(const unsigned int* a,
        unsigned int word,
        bool* ret)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i == 0)
    {
        if (a[0] >= word)
        {
            *ret = true;
        }
    } else {
        if (a[i] > 0)
        {
            *ret = true;
        }
    }
}

__global__ void
cuda_eq(const unsigned int* a,
        const unsigned int* b,
        bool* ret)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (a[i] != b[i])
    {
        *ret = false;
    }
}


__global__ void
eq_word_low(const unsigned int* a,
            unsigned int word,
            bool* ret)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i == 0)
    {
        if (a[0] == word)
        {
            *ret = true;
        }
    }
}

__global__ void
eq_word_high(const unsigned int* a,
             unsigned int word,
             bool* ret)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i > 0)
    {
        if (a[i] != 0)
        {
            *ret = false;
        }
    }
}

bool
greater_or_equal(CudaBigInt& a, unsigned int b)
{
    bool* ret_cuda;
    bool ret_local;
    cudaError_t err;
    
    cuda_malloc_clear((void**) &ret_cuda, sizeof(bool));
    
    ge_word<<<(a.word_len/COMPARE_BLOCK_SIZE), COMPARE_BLOCK_SIZE>>>(a.val, b, ret_cuda);
    
    
    
    err = cudaMemcpy(&ret_local, ret_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    
    cuda_malloc_free(ret_cuda);
    
    return ret_local;
}

bool
equalu(CudaBigInt& a, unsigned int b)
{
    bool* ret_cuda;
    bool ret_local;
    cudaError_t err;
    
    cuda_malloc_clear((void**) &ret_cuda, sizeof(bool));
    
    eq_word_low<<<(a.word_len/COMPARE_BLOCK_SIZE), COMPARE_BLOCK_SIZE>>>(a.val, b, ret_cuda);
    eq_word_high<<<(a.word_len/COMPARE_BLOCK_SIZE), COMPARE_BLOCK_SIZE>>>(a.val, b, ret_cuda);
    
    
    
    err = cudaMemcpy(&ret_local, ret_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    
    cuda_malloc_free(ret_cuda);
    
    return ret_local;
}

bool
equal(CudaBigInt& a, CudaBigInt& b)
{
    bool* ret_cuda;
    bool ret_local;
    cudaError_t err;
    
    cuda_malloc_clear((void**) &ret_cuda, sizeof(bool));
    
    err = cudaMemset(ret_cuda, true, sizeof(bool));
    assert(err == cudaSuccess);
    
    cuda_eq<<<(a.word_len/COMPARE_BLOCK_SIZE), COMPARE_BLOCK_SIZE>>>(a.val, b.val, ret_cuda);
    
    
    err = cudaMemcpy(&ret_local, ret_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    
    cuda_malloc_free(ret_cuda);
    
    return ret_local;
}
