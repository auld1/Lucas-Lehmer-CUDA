#include "add.h"

#include "bigint.h"
#include "carry.h"
#include "memory.h"
#include "rand.h"

#include <assert.h>
#include <stdio.h>
#include <time.h>

#include <gmp.h>

#define ADD_BLOCK_SIZE (128)

void
set_mpz_uint(mpz_t t, unsigned int* val, int len)
{
    mpz_import(t, len, -1, sizeof(unsigned int), -1, 0, val);
}

__device__
unsigned int cuda_add_digit(const unsigned int* __restrict__ a,
                            const unsigned int* __restrict__ b,
                            int digit,
                            unsigned char* __restrict__ carry_out,
                            unsigned int carry_in,
                            const int N)
{
    unsigned char carry = 0;
    unsigned int result = 0;
    unsigned int temp = 0;
    
    result = carry_in;
    temp = a[digit] + b[digit];
    if (temp < a[digit])
    {
        carry++;
    }
    
    result += temp;
    if (result < temp)
    {
        carry++;
    }
    
    *carry_out = carry;
    return result;
}

__global__
void cuda_add(const unsigned int* __restrict__ a,
              const unsigned int* __restrict__ b,
              unsigned int* __restrict__ c,
              unsigned char* __restrict__ carry_out,
              const int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    c[i] = cuda_add_digit(a, b, i, &carry_out[i], 0, N);
}

void add(CudaBigInt& a, CudaBigInt& b, CudaBigInt& c)
{
    unsigned char* byte_carry1;
    unsigned char* byte_carry2;
    bool* should_carry_cuda;
    bool should_carry_host;
    cudaError_t err;
    
    assert(a.word_len == b.word_len);
    assert(a.word_len + 1 <= c.word_len);
    
    if (a.sign == -1)
    {
        if (b.sign == 1)
        {
            //subtract(b, a, c);
            return;
        }
        // Both signs are -1, we will add them together as positives
        // but change the sign of c
        c.sign = -1;
    } else if (b.sign == -1)
    {
        //subtract(a, b, c);
        return;
    }
    
    cuda_malloc_clear((void**) &byte_carry1, c.word_len * sizeof(*byte_carry1));
    cuda_malloc_clear((void**) &byte_carry2, c.word_len * sizeof(*byte_carry2));
    cuda_malloc_clear((void**) &should_carry_cuda, sizeof(bool));
    
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
     
    cuda_add<<<(a.word_len/ADD_BLOCK_SIZE), ADD_BLOCK_SIZE>>>(a.val, b.val, c.val, byte_carry1, a.word_len);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
    
    do
    {
        cuda_byte_carry<<<(c.word_len/ADD_BLOCK_SIZE), ADD_BLOCK_SIZE>>>(c.val, byte_carry1, byte_carry2, should_carry_cuda);
    
        err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);
        
        err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
        assert(err == cudaSuccess);
        
        unsigned char* temp = byte_carry1;
        byte_carry1 = byte_carry2;
        byte_carry2 = temp;
    } while (should_carry_host);
    
    cudaFree(byte_carry1);
    cudaFree(byte_carry2);
    cudaFree(should_carry_cuda);
    
}

int
test()
{
    CudaBigInt a(1024*1024);
    CudaBigInt b(1024*1024);
    CudaBigInt c(1024*1024*2);
    
    mpz_t a_gmp;
    mpz_t b_gmp;
    mpz_t c_gmp;
    mpz_t add_gmp;
    
    mpz_init2(a_gmp, a.word_len*32);
    mpz_init2(b_gmp, b.word_len*32);
    mpz_init2(c_gmp, c.word_len*32);
    mpz_init2(add_gmp, c.word_len*32);
    
    unsigned int* a_host;
    unsigned int* b_host;
    unsigned int c_host[c.word_len];
    
    srand(time(NULL));
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    a_host = get_random_array(a.val, a.word_len);
    b_host = get_random_array(b.val, a.word_len);
    
    set_mpz_uint(a_gmp, a_host, a.word_len);
    set_mpz_uint(b_gmp, b_host, b.word_len);
    
    add(a, b, c);
    
    cudaMemcpy(c_host, c.val, c.word_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    set_mpz_uint(c_gmp, c_host, c.word_len);
    mpz_add(add_gmp, a_gmp, b_gmp);
    
    assert(0 == mpz_cmp(add_gmp, c_gmp));
    
    return 0;
}


int
main(void)
{
    int i = 0;
    printf("Testing 10 iterations of add on random digits\n");
    for (i = 0; i < 10; i++)
    {
        test();
    }
    printf("Passed\n");
}
