#include <cufft.h>
#include <gmp.h>
#include <stdio.h>

#include "bigint.h"
#include "carry.h"
#include "rand.h"


void
set_mpz_uint(mpz_t t, unsigned int* val, int len)
{
    mpz_import(t, len, -1, sizeof(unsigned int), -1, 0, val);
}
__global__
void pointwise(cufftComplex* c)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    float2 cu = c[idx];
    c[idx].x = cu.x * cu.x - cu.y * cu.y;
    c[idx].y = cu.x * cu.y + cu.y * cu.x;
}

__global__
void split(const unsigned int* a, cufftComplex* c)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    for (int i = 0; i < 32; i += 1)
    {
        c[idx*32+i].x = (float)((a[idx] >> i) & 0x1);
        c[idx*32+i].y = 0;
    }
}

__global__
void combine(cufftComplex* a, unsigned int* c, unsigned long long* carry)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned long long ret = 0;
    for(int i = 31; i >= 0; i--)
    {
        ret <<= 1;
        ret += (unsigned int)(a[idx*32+i].x + .25);
    }
    
    c[idx] = (unsigned int)(ret & 0xffffffff);
    carry[idx] = (unsigned int)(ret >> 32);
}

#define BLOCK_SIZE 32
int
main(void)
{
    cudaError_t err;
    CudaBigInt a(1024*4);
    CudaBigInt c(a.word_len*32*2);
    
    mpz_t a_gmp;
    mpz_t c_gmp;
    
    mpz_t ct_gmp;
    
    mpz_init2(a_gmp, a.word_len*32);
    mpz_init2(c_gmp, c.word_len*32);
    
    mpz_init2(ct_gmp, c.word_len*32);
    
    unsigned int *a_host = (unsigned int *) malloc(a.word_len * sizeof(*a.val));
    unsigned int *c_host = (unsigned int *) malloc(c.word_len * sizeof(*c.val));
    
    cufftHandle plan;
    cufftComplex *data;
    
    unsigned long long* long_carry;
    unsigned char* byte_carry1;
    unsigned char* byte_carry2;
    bool* should_carry_cuda;
    bool should_carry_host;
    
    cudaMalloc((void**) &long_carry, c.word_len * sizeof(*long_carry));
    cudaMalloc((void**) &byte_carry1, c.word_len * sizeof(*byte_carry1));
    cudaMalloc((void**) &byte_carry2, c.word_len * sizeof(*byte_carry2));
    cudaMalloc((void**) &should_carry_cuda, sizeof(bool));
    
    err = cudaMalloc((void**)&data, sizeof(cufftComplex)*c.word_len*32);
    assert(err == cudaSuccess);
    
    assert(cufftPlan1d(&plan, c.word_len*32, CUFFT_C2C, 1) == CUFFT_SUCCESS);
    
    for(int i = 0; i < 100; i++)
    {
    
        err = cudaMemset(c.val, 0, sizeof(*c.val)*c.word_len);
        assert(err == cudaSuccess);
        
        err = cudaMemset(data, 0, sizeof(cufftComplex)*c.word_len*32);
        assert(err == cudaSuccess);
        err = cudaMemset(long_carry, 0, sizeof(*long_carry)*c.word_len);
        assert(err == cudaSuccess);
        err = cudaMemset(byte_carry1, 0, sizeof(*byte_carry1)*c.word_len);
        assert(err == cudaSuccess);
        err = cudaMemset(byte_carry2, 0, sizeof(*byte_carry2)*c.word_len);
        assert(err == cudaSuccess);
        err = cudaMemset(should_carry_cuda, 0, sizeof(*should_carry_cuda));
        assert(err == cudaSuccess);
        
        get_random_array(a.val, a.word_len);
    
        err = cudaMemcpy(a_host, a.val, a.word_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);
        
        set_mpz_uint(a_gmp, a_host, a.word_len);
        
        split<<<(a.word_len/BLOCK_SIZE), BLOCK_SIZE>>>(a.val, data);
        
        assert(cufftExecC2C(plan, data, data, CUFFT_FORWARD) == CUFFT_SUCCESS);
        
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess);
        
        pointwise<<<(c.word_len*32/BLOCK_SIZE), BLOCK_SIZE>>>(data);
        
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess);
        
        assert(cufftExecC2C(plan, data, data, CUFFT_INVERSE) == CUFFT_SUCCESS);
        
        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess);
        
        combine<<<(c.word_len/BLOCK_SIZE), BLOCK_SIZE>>>(data, c.val, long_carry);
    
    
        cuda_long_carry<<<(c.word_len/BLOCK_SIZE), BLOCK_SIZE>>>(c.val, long_carry, byte_carry1, should_carry_cuda);
    
        err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);
    
        err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
        assert(err == cudaSuccess);
        
        while (should_carry_host)
        {
            cuda_byte_carry<<<(c.word_len/BLOCK_SIZE), BLOCK_SIZE>>>(c.val, byte_carry1, byte_carry2, should_carry_cuda);
    
            err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
            assert(err == cudaSuccess);
        
            err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
            assert(err == cudaSuccess);
        
            unsigned char* temp = byte_carry1;
            byte_carry1 = byte_carry2;
            byte_carry2 = temp;
        }
        mpz_mul(c_gmp, a_gmp, a_gmp);
    
        cudaMemcpy(c_host, c.val, c.word_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        set_mpz_uint(ct_gmp, c_host, c.word_len);
        
        //mpz_cdiv_q_ui(ct_gmp, ct_gmp, 2);
    
        if (0 != mpz_cmp(c_gmp, ct_gmp))
        {
            gmp_printf("%Zx\n\n\n", c_gmp);
            gmp_printf("%Zx\n", ct_gmp);
            assert(0);
        }
    
        printf("Done %d\n", i+1);
    }
        

}






















