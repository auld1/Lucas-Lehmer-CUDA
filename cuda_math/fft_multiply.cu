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


#define FFT_BLOCK_SIZE (128)

__global__ void
split8(const unsigned int* __restrict__ in,
      cuDoubleComplex* __restrict__ out)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x);
    
    out[idx*4].x = (double) (in[idx] & 0xff);
    out[idx*4].y = 0;
    
    out[idx*4+1].x = (double) ((in[idx] >> 8) & 0xff);
    out[idx*4+1].y = 0;
    
    out[idx*4+2].x = (double) ((in[idx] >> 16) & 0xff);
    out[idx*4+2].y = 0;
    
    out[idx*4+3].x = (double) ((in[idx] >> 24) & 0xff);
    out[idx*4+3].y = 0;
}

__global__ void
split4(const unsigned int* __restrict__ in,
      cuDoubleComplex* __restrict__ out)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x);
    
    out[idx*8].x = (double) (in[idx] & 0xf);
    out[idx*8].y = 0;
    
    out[idx*8+1].x = (double) ((in[idx] >> 4) & 0xf);
    out[idx*8+1].y = 0;
    
    out[idx*8+2].x = (double) ((in[idx] >> 8) & 0xf);
    out[idx*8+2].y = 0;
    
    out[idx*8+3].x = (double) ((in[idx] >> 12) & 0xf);
    out[idx*8+3].y = 0;
    
    out[idx*8+4].x = (double) ((in[idx] >> 16) & 0xf);
    out[idx*8+4].y = 0;
    
    out[idx*8+5].x = (double) ((in[idx] >> 20) & 0xf);
    out[idx*8+5].y = 0;
    
    out[idx*8+6].x = (double) ((in[idx] >> 24) & 0xf);
    out[idx*8+6].y = 0;
    
    out[idx*8+7].x = (double) ((in[idx] >> 28) & 0xf);
    out[idx*8+7].y = 0;
}

__global__ void
split2(const unsigned int* __restrict__ in,
      cuDoubleComplex* __restrict__ out)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x);
    
    out[idx*16].x = (double) (in[idx] & 0x3);
    out[idx*16].y = 0;
    
    out[idx*16+1].x = (double) ((in[idx] >> 2) & 0x3);
    out[idx*16+1].y = 0;
    
    out[idx*16+2].x = (double) ((in[idx] >> 4) & 0x3);
    out[idx*16+2].y = 0;
    
    out[idx*16+3].x = (double) ((in[idx] >> 6) & 0x3);
    out[idx*16+3].y = 0;
    
    out[idx*16+4].x = (double) ((in[idx] >> 8) & 0x3);
    out[idx*16+4].y = 0;
    
    out[idx*16+5].x = (double) ((in[idx] >> 10) & 0x3);
    out[idx*16+5].y = 0;
    
    out[idx*16+6].x = (double) ((in[idx] >> 12) & 0x3);
    out[idx*16+6].y = 0;
    
    out[idx*16+7].x = (double) ((in[idx] >> 14) & 0x3);
    out[idx*16+7].y = 0;
    
    out[idx*16+8].x = (double) ((in[idx] >> 16) & 0x3);
    out[idx*16+8].y = 0;
    
    out[idx*16+9].x = (double) ((in[idx] >> 18) & 0x3);
    out[idx*16+9].y = 0;
    
    out[idx*16+10].x = (double) ((in[idx] >> 20) & 0x3);
    out[idx*16+10].y = 0;
    
    out[idx*16+11].x = (double) ((in[idx] >> 22) & 0x3);
    out[idx*16+11].y = 0;
    
    out[idx*16+12].x = (double) ((in[idx] >> 24) & 0x3);
    out[idx*16+12].y = 0;
    
    out[idx*16+13].x = (double) ((in[idx] >> 26) & 0x3);
    out[idx*16+13].y = 0;
    
    out[idx*16+14].x = (double) ((in[idx] >> 28) & 0x3);
    out[idx*16+14].y = 0;
    
    out[idx*16+15].x = (double) ((in[idx] >> 30) & 0x3);
    out[idx*16+15].y = 0;
}

__global__ void
complex_to_complex_bitreverse(cuDoubleComplex* __restrict__ out,
                              int bitlen)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int rev_idx = (__brev(idx) >> (32-bitlen));
    if (rev_idx < idx)
    {
        cuDoubleComplex tmp = out[rev_idx];
        out[rev_idx] = out[idx];
        out[idx] = tmp;
    }
}

// Found at https://devtalk.nvidia.com/default/topic/814159/additional-cucomplex-functions-cucnorm-cucsqrt-cucexp-and-some-complex-double-functions-/
/*__host__ __device__ static __inline__ cuDoubleComplex
cuCexp(cuDoubleComplex x)
{
	double factor = exp(x.x);
	return make_cuDoubleComplex(factor * cos(x.y), factor * sin(x.y));
}*/

__global__ void
cooley_tukey_complex_fft(cuDoubleComplex* __restrict__ A,
                         int s,
                         int exp_sign,
                         cuDoubleComplex wn,
                         int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int m = (1 << s);
    int k = idx / (m/2);
    k *= m;
    int j = idx % (m/2);
    cuDoubleComplex w = make_cuDoubleComplex(cos(wn.y*j), sin(wn.y*j));
    cuDoubleComplex t, u;
    
    t = cuCmul(w, A[k + j + m/2]);
    u = A[k + j];
    
    A[k + j] = cuCadd(u, t);
    A[j + k + m/2] = cuCsub(u, t);
    
    if (m == N && exp_sign == 1)
    {
        A[k + j] = cuCdiv(A[k + j], make_cuDoubleComplex((double)N, 0));
        A[k + j + m/2] = cuCdiv(A[k + j + m/2], make_cuDoubleComplex((double)N, 0));
    }
}

__global__ void
pointwise_square(cuDoubleComplex* __restrict__ A)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    A[idx] = cuCmul(A[idx], A[idx]);
}

void
cooley_tukey_fft(cuDoubleComplex* a, int len)
{
    assert(isPow2(len));
    
    complex_to_complex_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_fft<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s, -1, make_cuDoubleComplex(0, ((double)-2.0) * M_PI / (1<<s)), len);
    }
}

void
cooley_tukey_ifft(cuDoubleComplex* a, int len)
{
    assert(isPow2(len));
    
    complex_to_complex_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_fft<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s, 1, make_cuDoubleComplex(0, ((double)2.0) * M_PI / (1<<s)), len);
    }
    
    /*
    for (int i = 0; i < len; i++)
    {
        out[i] = (unsigned int) (device_out[i].x + .5);
    }*/
}

__global__ void
cuda_combine8(cuDoubleComplex* a, unsigned int* c, unsigned long long* carry)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    unsigned long long result = 0;
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
    
    c[idx] = result & 0xffffffff;
    carry[idx] = (result >> 32);
}

__global__ void
cuda_combine4(cuDoubleComplex* a, unsigned int* c, unsigned long long* carry)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    unsigned long long result = 0;
    unsigned int w1 = (unsigned int) (a[idx*8].x + .5);
    unsigned int w2 = (unsigned int) (a[idx*8+1].x + .5);
    unsigned int w3 = (unsigned int) (a[idx*8+2].x + .5);
    unsigned int w4 = (unsigned int) (a[idx*8+3].x + .5);
    unsigned int w5 = (unsigned int) (a[idx*8+4].x + .5);
    unsigned int w6 = (unsigned int) (a[idx*8+5].x + .5);
    unsigned int w7 = (unsigned int) (a[idx*8+6].x + .5);
    unsigned int w8 = (unsigned int) (a[idx*8+7].x + .5);
    
    result = w8;
    result <<= 4;
    result += w7;
    result <<= 4;
    result += w6;
    result <<= 4;
    result += w5;
    result <<= 4;
    result += w4;
    result <<= 4;
    result += w3;
    result <<= 4;
    result += w2;
    result <<= 4;
    result += w1;
    
    c[idx] = result & 0xffffffff;
    carry[idx] = (result >> 32);
}

__global__ void
cuda_combine2(cuDoubleComplex* a, unsigned int* c, unsigned long long* carry)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    unsigned long long result = 0;
    unsigned int w1 = (unsigned int) (a[idx*16].x + .5);
    unsigned int w2 = (unsigned int) (a[idx*16+1].x + .5);
    unsigned int w3 = (unsigned int) (a[idx*16+2].x + .5);
    unsigned int w4 = (unsigned int) (a[idx*16+3].x + .5);
    unsigned int w5 = (unsigned int) (a[idx*16+4].x + .5);
    unsigned int w6 = (unsigned int) (a[idx*16+5].x + .5);
    unsigned int w7 = (unsigned int) (a[idx*16+6].x + .5);
    unsigned int w8 = (unsigned int) (a[idx*16+7].x + .5);
    unsigned int w9 = (unsigned int) (a[idx*16+8].x + .5);
    unsigned int w10 = (unsigned int) (a[idx*16+9].x + .5);
    unsigned int w11 = (unsigned int) (a[idx*16+10].x + .5);
    unsigned int w12 = (unsigned int) (a[idx*16+11].x + .5);
    unsigned int w13 = (unsigned int) (a[idx*16+12].x + .5);
    unsigned int w14 = (unsigned int) (a[idx*16+13].x + .5);
    unsigned int w15 = (unsigned int) (a[idx*16+14].x + .5);
    unsigned int w16 = (unsigned int) (a[idx*16+15].x + .5);
    
    result = w16;
    result <<= 2;
    result += w15;
    result <<= 2;
    result += w14;
    result <<= 2;
    result += w13;
    result <<= 2;
    result += w12;
    result <<= 2;
    result += w11;
    result <<= 2;
    result += w10;
    result <<= 2;
    result += w9;
    result <<= 2;
    result += w8;
    result <<= 2;
    result += w7;
    result <<= 2;
    result += w6;
    result <<= 2;
    result += w5;
    result <<= 2;
    result += w4;
    result <<= 2;
    result += w3;
    result <<= 2;
    result += w2;
    result <<= 2;
    result += w1;
    
    c[idx] = result & 0xffffffff;
    carry[idx] = (result >> 32);
}

void
combine8(cuDoubleComplex* a, CudaBigInt& c)
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
    
    cuda_combine8<<<(c.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, c.val, long_carry);
    
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
combine4(cuDoubleComplex* a, CudaBigInt& c)
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
    
    cuda_combine4<<<(c.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, c.val, long_carry);
    
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
combine2(cuDoubleComplex* a, CudaBigInt& c)
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
    
    cuda_combine2<<<(c.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, c.val, long_carry);
    
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
    cuDoubleComplex* cuda_a;
    
    cuda_malloc_clear((void**) &cuda_a, sizeof(*cuda_a)*a.word_len*32);
    
    split2<<<(a.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a.val, cuda_a);
    
    cooley_tukey_fft(cuda_a, a.word_len*32);
    pointwise_square<<<(a.word_len*32/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_a);
    cooley_tukey_ifft(cuda_a, a.word_len*32);
    
    combine2(cuda_a, c);
    
    cuda_malloc_free(cuda_a);
}

