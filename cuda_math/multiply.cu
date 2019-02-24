#include "multiply.h"

#include "bigint.h"
#include "carry.h"
#include "memory.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

unsigned int rand_uint32(void) {
  unsigned int r = 0;
  for (int i=0; i<32; i += 30) {
    r = r*((unsigned int)RAND_MAX + 1) + rand();
  }
  return r;
}

__device__ unsigned int
cuda_multiply_gradeschool_digit(const unsigned int* __restrict__ a,
                                const unsigned int* __restrict__ b,
                                int digit,
                                unsigned long long * __restrict__ carry_out,
                                unsigned long long carry_in,
                                const int N)
{
    unsigned long long carry = 0;
    unsigned int temp = 0;
    unsigned int result = 0;
    int a_i = 0;
    int b_i = 0;

    result = carry_in & 0xffffffff;
    for (b_i = max(0, digit - N/2 + 1); b_i <= min(digit, N/2 - 1); b_i++)
    {
        a_i = digit - b_i;
        temp = a[a_i] * b[b_i];
        result += temp;
        if (result < temp)
        {
            carry++;
        }
        carry += __umulhi(a[a_i], b[b_i]);
    }
    carry += (carry_in >> 32) & 0xffffffff;
    *carry_out = carry;
    return result;
}

__global__ void
cuda_multiply_gradeschool(const unsigned int* __restrict__ a,
                          const unsigned int* __restrict__ b,
                          unsigned int* __restrict__ c,
                          unsigned long long* __restrict__ carry_out,
                          const int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    c[i] = cuda_multiply_gradeschool_digit(a, b, i, &carry_out[i], 0, N);
}

__global__ void
cuda_multiply_gradeschool_4(const unsigned int* __restrict__ a,
                            const unsigned int* __restrict__ b,
                            unsigned int* __restrict__ c,
                            unsigned long long* __restrict__ carry_out,
                            const int N)
{
    int i = (blockIdx.x*blockDim.x + threadIdx.x) * 4;
    unsigned long long carry = 0;
    
    c[i] = cuda_multiply_gradeschool_digit(a, b, i, &carry, 0, N);
    c[i+1] = cuda_multiply_gradeschool_digit(a, b, i+1, &carry, carry, N);
    c[i+2] = cuda_multiply_gradeschool_digit(a, b, i+2, &carry, carry, N);
    c[i+3] = cuda_multiply_gradeschool_digit(a, b, i+3, &carry, carry, N);
    carry_out[i/4] = carry;
}

void
multiply(CudaBigInt a, CudaBigInt b, CudaBigInt c)
{
    unsigned long long* long_carry;
    unsigned char* byte_carry1;
    unsigned char* byte_carry2;
    bool* should_carry_cuda;
    bool should_carry_host;
    cudaError_t err;
    
    assert(a.word_len == b.word_len);
    assert(a.word_len + b.word_len == c.word_len);
    
    cuda_malloc_clear((void**) &long_carry, c.word_len * sizeof(*long_carry));
    cuda_malloc_clear((void**) &byte_carry1, c.word_len * sizeof(*byte_carry1));
    cuda_malloc_clear((void**) &byte_carry2, c.word_len * sizeof(*byte_carry2));
    cuda_malloc_clear((void**) &should_carry_cuda, sizeof(bool));
    
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
     
    cuda_multiply_gradeschool_4<<<512, (c.word_len/512)>>>(a.val, b.val, c.val, long_carry, c.word_len);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
    
    cuda_long_carry_4<<<512, (c.word_len/512)>>>(c.val, long_carry, byte_carry1, should_carry_cuda);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
    
    err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    
    err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
    assert(err == cudaSuccess);
    
    while (should_carry_host)
    {
        cuda_byte_carry_4<<<512, (c.word_len/512)>>>(c.val, byte_carry1, byte_carry2, should_carry_cuda);
    
        err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);
        
        err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
        assert(err == cudaSuccess);
        
        unsigned char* temp = byte_carry1;
        byte_carry1 = byte_carry2;
        byte_carry2 = temp;
    }
    c.sign = a.sign*b.sign;
}

int
main()
{
    CudaBigInt a(1024*1024*8);
    CudaBigInt b(1024*1024*8);
    CudaBigInt c(1024*1024*8*2);
    
    unsigned int a_host[a.word_len];
    unsigned int b_host[b.word_len];
    unsigned int c_host[c.word_len];
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    int i = 0;
    
    for(i = 0; i < c.word_len; i++)
    {
        c_host[i] = 0;
    }
    
    for(i = 0; i < a.word_len; i++)
    {
        a_host[i] = 0;
        b_host[i] = 0;
    }
    
    for(i = 0; i < a.word_len; i++)
    {
        a_host[i] = rand_uint32();
        b_host[i] = rand_uint32();
    }
    
    
    for(i = 0; i < c.word_len; i++)
    {
        c_host[i] = 0;
    }
    
    
    cudaMemcpy(a.val, a_host, a.word_len * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(b.val, b_host, b.word_len * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(c.val, c_host, c.word_len * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    multiply(a, b, c);
    
    
    cudaMemcpy(c_host, c.val, c.word_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    

    return 0;
}
