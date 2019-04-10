#include "fft_multiply.h"

#include "bigint.h"
#include "carry.h"
#include "memory.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


#include <cuComplex.h>
#include <cufft.h>


#define FFT_BLOCK_SIZE (128)

#define FFT_USE_DOUBLE

#ifdef FFT_USE_DOUBLE

#define BITS_PER_FLOAT 8
#define cuComplex cuDoubleComplex
#define make_cuComplex make_cuDoubleComplex
#define floatType double
#define cuAdd cuCadd
#define cuSub cuCsub
#define cuMul cuCmul
#define cuDiv cuCdiv

#else

#define BITS_PER_FLOAT 4
#define cuComplex cuFloatComplex
#define make_cuComplex make_cuFloatComplex
#define floatType float
#define cuAdd cuCaddf
#define cuSub cuCsubf
#define cuMul cuCmulf
#define cuDiv cuCdivf

#endif


#define FLOATS_PER_WORD (32/BITS_PER_FLOAT)


__global__ void
split(const unsigned int* __restrict__ in,
      cuComplex* __restrict__ out)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x);
    
    for(int i = 0; i < FLOATS_PER_WORD; i++)
    {
        out[idx*FLOATS_PER_WORD+i].x = (floatType) ((in[idx] >> (i*BITS_PER_FLOAT)) & ((1<<BITS_PER_FLOAT) - 1));
        out[idx*FLOATS_PER_WORD+i].y = 0;
    }
}

__global__ void
complex_to_complex_bitreverse(cuComplex* __restrict__ out,
                              int bitlen)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int rev_idx = (__brev(idx) >> (32-bitlen));
    if (rev_idx < idx)
    {
        cuComplex tmp = out[rev_idx];
        out[rev_idx] = out[idx];
        out[idx] = tmp;
    }
}

__global__ void
cooley_tukey_complex_fft(cuComplex* __restrict__ A,
                         int s,
                         int exp_sign,
                         cuComplex wn,
                         int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int m = (1 << s);
    int k = idx / (m/2);
    k *= m;
    int j = idx % (m/2);
    cuComplex w = make_cuComplex(cos(wn.y*j), sin(wn.y*j));
    cuComplex t, u;
    
    t = cuMul(w, A[k + j + m/2]);
    u = A[k + j];
    
    A[k + j] = cuAdd(u, t);
    A[j + k + m/2] = cuSub(u, t);
    
    if (m == N && exp_sign == 1)
    {
        A[k + j] = cuDiv(A[k + j], make_cuComplex((floatType)N, 0));
        A[k + j + m/2] = cuDiv(A[k + j + m/2], make_cuComplex((floatType)N, 0));
    }
}

__global__ void
pointwise_square(cuComplex* __restrict__ A)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    A[idx] = cuMul(A[idx], A[idx]);
}


void
cooley_tukey_fft(cuComplex* a, int len)
{
    assert(isPow2(len));
    
    complex_to_complex_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_fft<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s, -1, make_cuComplex(0, ((floatType)-2.0) * M_PI / (1<<s)), len);
    }
}

void
cooley_tukey_ifft(cuComplex* a, int len)
{
    assert(isPow2(len));
    
    complex_to_complex_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_fft<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s, 1, make_cuComplex(0, ((floatType)2.0) * M_PI / (1<<s)), len);
    }
}

__global__ void
cuda_combine(cuComplex* a, unsigned int* c, unsigned long long* carry)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    unsigned long long result = 0;
    /*
    unsigned int w1 = (unsigned int) (a[idx*4].x + .5);
    unsigned int w2 = (unsigned int) (a[idx*4+1].x + .5);
    unsigned int w3 = (unsigned int) (a[idx*4+2].x + .5);
    unsigned int w4 = (unsigned int) (a[idx*4+3].x + .5);
    
    result = w4;
    result <<= 8;
    result += w3;
    result <<= 8;
    result += w2;
    result <<= 8;
    result += w1;
    */
    
    for (int i = FLOATS_PER_WORD-1; i >= 0; i--)
    {
        result <<= BITS_PER_FLOAT;
        result += (a[idx*FLOATS_PER_WORD+i].x + .5);
    }
    c[idx] = result & 0xffffffff;
    carry[idx] = (result >> 32);
    
}

void
combine(cuComplex* a, CudaBigInt& c)
{
    unsigned long long* long_carry;
    unsigned char* byte_carry1;
    unsigned char* byte_carry2;
    bool* should_carry_cuda;
    bool should_carry_host;
    cudaError_t err;
    
    cuda_malloc_clear((void**) &long_carry, c.word_len * sizeof(*long_carry));
    cuda_malloc_clear((void**) &byte_carry1, c.word_len * sizeof(*byte_carry1));
    cuda_malloc_clear((void**) &byte_carry2, c.word_len * sizeof(*byte_carry2));
    cuda_malloc_clear((void**) &should_carry_cuda, sizeof(bool));
    
    cuda_combine<<<(c.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, c.val, long_carry);
    
    cuda_long_carry<<<(c.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(c.val, long_carry, byte_carry1, should_carry_cuda);
    
    err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess);
    
    err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
    assert(err == cudaSuccess);
    
    while (should_carry_host)
    {
        cuda_byte_carry<<<(c.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(c.val, byte_carry1, byte_carry2, should_carry_cuda);
    
        err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);
        
        err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
        assert(err == cudaSuccess);
        
        unsigned char* temp = byte_carry1;
        byte_carry1 = byte_carry2;
        byte_carry2 = temp;
    }
    
    
    cuda_malloc_free(long_carry);
    cuda_malloc_free(byte_carry1);
    cuda_malloc_free(byte_carry2);
    cuda_malloc_free(should_carry_cuda);
}


void
fft_square(CudaBigInt& a, CudaBigInt& c)
{
    cuComplex* cuda_a;
    
    cuda_malloc_clear((void**) &cuda_a, sizeof(*cuda_a)*a.word_len*FLOATS_PER_WORD*2);
    
    split<<<(a.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a.val, cuda_a);
    
    cooley_tukey_fft(cuda_a, a.word_len*FLOATS_PER_WORD*2);
    pointwise_square<<<(a.word_len*FLOATS_PER_WORD*2/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_a);
    cooley_tukey_ifft(cuda_a, a.word_len*FLOATS_PER_WORD*2);
    
    combine(cuda_a, c);
    
    cuda_malloc_free(cuda_a);
}
