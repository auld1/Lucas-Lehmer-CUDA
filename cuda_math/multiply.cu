#include "multiply.h"

#include "bigint.h"
#include "carry.h"
#include "memory.h"
#include "rand.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <gmp.h>

#define MULTIPLY_BLOCK_SIZE (64)

void
set_mpz_uint(mpz_t t, unsigned int* val, int len)
{
    mpz_import(t, len, -1, sizeof(unsigned int), -1, 0, val);
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

void
multiply(CudaBigInt& a, CudaBigInt& b, CudaBigInt& c)
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
     
    cuda_multiply_gradeschool<<<(c.word_len/MULTIPLY_BLOCK_SIZE), MULTIPLY_BLOCK_SIZE>>>(a.val, b.val, c.val, long_carry, c.word_len);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
    
    cuda_long_carry<<<(c.word_len/MULTIPLY_BLOCK_SIZE), MULTIPLY_BLOCK_SIZE>>>(c.val, long_carry, byte_carry1, should_carry_cuda);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
    
    err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    
    err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
    assert(err == cudaSuccess);
    
    while (should_carry_host)
    {
        cuda_byte_carry<<<(c.word_len/MULTIPLY_BLOCK_SIZE), MULTIPLY_BLOCK_SIZE>>>(c.val, byte_carry1, byte_carry2, should_carry_cuda);
    
        err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);
        
        err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
        assert(err == cudaSuccess);
        
        unsigned char* temp = byte_carry1;
        byte_carry1 = byte_carry2;
        byte_carry2 = temp;
    }
    c.sign = a.sign*b.sign;
    
    cudaFree(long_carry);
    cudaFree(byte_carry1);
    cudaFree(byte_carry2);
    cudaFree(should_carry_cuda);
}

int
test()
{
    CudaBigInt a(1024*1024*4);
    CudaBigInt b(1024*1024*4);
    CudaBigInt c(1024*1024*4*2);
    
    mpz_t a_gmp;
    mpz_t b_gmp;
    mpz_t c_gmp;
    mpz_t mul_gmp;
    
    mpz_init2(a_gmp, a.word_len*32);
    mpz_init2(b_gmp, b.word_len*32);
    mpz_init2(c_gmp, c.word_len*32);
    mpz_init2(mul_gmp, c.word_len*32);
    
    unsigned int* a_host;
    unsigned int* b_host;
    unsigned int c_host[c.word_len];
    
    srand(time(NULL));
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    a_host = get_random_array(a.val, a.word_len);
    b_host = get_random_array(b.val, a.word_len);
    
    set_mpz_uint(a_gmp, a_host, a.word_len);
    set_mpz_uint(b_gmp, b_host, b.word_len);
    
    multiply(a, b, c);
    
    cudaMemcpy(c_host, c.val, c.word_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    set_mpz_uint(c_gmp, c_host, c.word_len);
    mpz_mul(mul_gmp, a_gmp, b_gmp);
    
    assert(0 == mpz_cmp(mul_gmp, c_gmp));
    
    return 0;
}


int
main(void)
{
    int i = 0;
    printf("Testing 1000 iterations of multiply on random digits\n");
    for (i = 0; i < 1000; i++)
    {
        test();
    }
    printf("Passed\n");
}
