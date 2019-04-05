#include "mod.h"

#include "add.h"
#include "bigint.h"
#include "carry.h"
#include "memory.h"
#include "rand.h"
#include "compare.h"

#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <gmp.h>

#define MOD_BLOCK_SIZE (128)

__global__
void bit_set(const unsigned int* a,
             int bit, 
             bool* __restrict__ set)
{
    int word = bit / 32;
    int offset = bit % 32;
    
    *set = (((a[word] >> offset) & 1) == 1);
}

__global__
void clear_bit(unsigned int* a,
               int bit)
{
    int word = bit / 32;
    int offset = bit % 32;
    
    a[word] ^= (1<<offset);
}

__global__
void cuda_shiftr(const unsigned int* a,
                 unsigned int s,
                 unsigned int* c,
                 unsigned int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    int big_shift = s / 32;
    int small_shift = s % 32;
    
    c[i] = 0;
    
    if (i + big_shift < N)
    {
        c[i] = a[i+big_shift] >> small_shift;
        if (i + big_shift + 1 < N)
        {
            c[i] |= (a[i+big_shift+1] << (32-small_shift));
        }
    }
}

__global__
void cuda_mers_and(const unsigned int* a,
                   const unsigned int m,
                   unsigned int* c)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    c[i] = 0;
    
    if (i*32 < m)
    {
        c[i] = a[i];
        if ((i+1) * 32 > m)
        {
            c[i] &= ((1 << (m%32)) - 1);
        }
    }
}

__global__
void cuda_mod(const unsigned int* a,
              const unsigned int m,
              unsigned int* c,
              unsigned char* __restrict__ carry_out,
              const int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    
    int big_shift = m / 32;
    int small_shift = m % 32;
    
    c[i] = 0;
    carry_out[i] = 0;
    
    if ((i)*32 < (int)m)
    {
        unsigned int w1 = a[i+big_shift] >> small_shift;
        unsigned int w2 = 0;
        if ((i+1)*32 < m && small_shift != 0)
        {
            w2 = a[i+big_shift+1] << (32-small_shift);
        }
        
        c[i] = a[i] + w1 + w2;
    
        if (c[i] < a[i])
        {
            carry_out[i] = 1;
        }
    }
}

void cuda_shr(CudaBigInt& a, unsigned int m, CudaBigInt& c)
{
    cuda_shiftr<<<(c.word_len/MOD_BLOCK_SIZE), MOD_BLOCK_SIZE>>>(a.val, m, c.val, a.word_len);
}

void cuda_and(CudaBigInt& a, unsigned int m, CudaBigInt& c)
{
    cuda_mers_and<<<(c.word_len/MOD_BLOCK_SIZE), MOD_BLOCK_SIZE>>>(a.val, m, c.val);
}

void mod(CudaBigInt& a, unsigned int m, CudaBigInt& p, CudaBigInt& c)
{
    CudaBigInt shifted_a(c.word_len*32);
    CudaBigInt xor_a(c.word_len*32);
    unsigned char* byte_carry1;
    unsigned char* byte_carry2;
    bool* should_carry_cuda;
    bool should_carry_host;
    cudaError_t err;
    
    
    cuda_malloc_clear((void**) &byte_carry1, c.word_len * sizeof(*byte_carry1));
    cuda_malloc_clear((void**) &byte_carry2, c.word_len * sizeof(*byte_carry2));
    cuda_malloc_clear((void**) &should_carry_cuda, sizeof(bool));
     
    cuda_mers_and<<<(c.word_len/MOD_BLOCK_SIZE), MOD_BLOCK_SIZE>>>(a.val, m, xor_a.val);
    cuda_shiftr<<<(c.word_len/MOD_BLOCK_SIZE), MOD_BLOCK_SIZE>>>(a.val, m, shifted_a.val, a.word_len);
    
    add(xor_a, shifted_a, c);
    
    
    bit_set<<<1,1>>>(c.val, m, should_carry_cuda);
    
    err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
        
    err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
    assert(err == cudaSuccess);
    
    while (should_carry_host)
    {
        clear_bit<<<1,1>>>(c.val, m);
        addu(c, 1, c);
        
        bit_set<<<1,1>>>(c.val, m, should_carry_cuda);
        
        err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);
        
        err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
        assert(err == cudaSuccess);
    }
        
    if (equal(c, p))
    {
        err = cudaMemset(c.val, 0, c.word_len * sizeof(*c.val));
        assert(err == cudaSuccess);
    }
    
    
    cuda_malloc_free(byte_carry1);
    cuda_malloc_free(byte_carry2);
    cuda_malloc_free(should_carry_cuda);
}
