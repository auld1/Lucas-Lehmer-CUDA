#include "subtract.h"

#include "bigint.h"
#include "carry.h"
#include "memory.h"
#include "rand.h"
#include "compare.h"

#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <gmp.h>

#define SUB_BLOCK_SIZE (128)

/*
void
set_mpz_uint(mpz_t t, unsigned int* val, int len)
{
    mpz_import(t, len, -1, sizeof(unsigned int), -1, 0, val);
}*/


__global__
void cuda_sub(const unsigned int* a,
              unsigned int b,
              unsigned int* c,
              unsigned char* carry_out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (i == 0)
    {
        unsigned int tmp = a[i];
        c[i] = a[i] - b;
        
        if (c[i] > tmp)
        {
            carry_out[i]++;
        }
    } else {
        c[i] = a[i];
    }
}

void subu(CudaBigInt& a, unsigned int b, CudaBigInt& c)
{
    unsigned char* byte_carry1;
    unsigned char* byte_carry2;
    bool* should_carry_cuda;
    bool should_carry_host;
    cudaError_t err;
    
    if (equalu(a, b))
    {
        err = cudaMemset(c.val, 0, c.word_len * sizeof(*c.val));
        assert(err == cudaSuccess);
        return;
    }
    
    cuda_malloc_clear((void**) &byte_carry1, c.word_len * sizeof(*byte_carry1));
    cuda_malloc_clear((void**) &byte_carry2, c.word_len * sizeof(*byte_carry2));
    cuda_malloc_clear((void**) &should_carry_cuda, sizeof(bool));
    
    cuda_sub<<<(a.word_len/SUB_BLOCK_SIZE), SUB_BLOCK_SIZE>>>(a.val, b, c.val, byte_carry1);
    
    do
    {
        cuda_negative_byte_carry<<<(a.word_len/SUB_BLOCK_SIZE), SUB_BLOCK_SIZE>>>(c.val, byte_carry1, byte_carry2, should_carry_cuda);
    
        err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);
        
        err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
        assert(err == cudaSuccess);
        
        unsigned char* temp = byte_carry1;
        byte_carry1 = byte_carry2;
        byte_carry2 = temp;
    } while (should_carry_host);
    
    cuda_malloc_free(byte_carry1);
    cuda_malloc_free(byte_carry2);
    cuda_malloc_free(should_carry_cuda);
    
}
