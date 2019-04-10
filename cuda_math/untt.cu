#include "untt.h"

#include "bigint.h"
#include "carry.h"
#include "memory.h"
#include "multiply.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


#define FFT_BLOCK_SIZE (128)

// This prime is of the form k*2^n+1 = 15*(2^27)+1
#define NTT_PRIME ((unsigned int) 0x78000001) // 2013265921

__device__
unsigned int primitive_roots[] = {
    1,
    2013265920,
    1728404513,
    1592366214,
    196396260,
    760005850,
    1721589904,
    397765732,
    1732600167,
    1753498361,
    341742893,
    1340477990,
    1282623253,
    298008106,
    1657000625,
    2009781145,
    1421947380,
    1286330022,
    1559589183,
    1049899240,
    195061667,
    414040701,
    570250684,
    1267047229,
    1003846038,
    1149491290,
    975630072,
    440564289,
    1340157138,
    29791,
    31
};

__device__
unsigned int inverse_roots[] = {
    1,
    2013265920,
    284861408,
    1801542727,
    567209306,
    1273220281,
    662200255,
    1856545343,
    1611842161,
    1861675199,
    774513262,
    449056851,
    1255670133,
    1976924129,
    106301669,
    1411306935,
    1540942033,
    1043440885,
    173207512,
    463443832,
    1021415956,
    1574319791,
    953617870,
    987386499,
    1469248932,
    165179394,
    1498740239,
    1713844692,
    627186708,
    1477021247,
    64944062
};

__device__
unsigned int inverse_mod[] = {
    1,
    1006632961,
    1509949441,
    1761607681,
    1887436801,
    1950351361,
    1981808641,
    1997537281,
    2005401601,
    2009333761,
    2011299841,
    2012282881,
    2012774401,
    2013020161,
    2013143041,
    2013204481,
    2013235201,
    2013250561,
    2013258241,
    2013262081,
    2013264001,
    2013264961,
    2013265441,
    2013265681,
    2013265801,
    2013265861,
    2013265891,
    2013265906,
    1006632953,
    1509949437,
    1761607679
};

__global__ void
bitreverse(unsigned int* out,
           int bitlen)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int rev_idx = (__brev(idx) >> (32-bitlen));
    if (rev_idx < idx)
    {
        unsigned int tmp = out[rev_idx];
        out[rev_idx] = out[idx];
        out[idx] = tmp;
    }
}

__device__ unsigned int
modmul(unsigned int a, unsigned int b)
{
    /*unsigned long long ret = (a % NTT_PRIME);
    ret *= (b % NTT_PRIME);
    return (unsigned int) (ret % ((unsigned long long)NTT_PRIME));*/
    return (((unsigned long long) a) * b) % NTT_PRIME;
}

__device__ unsigned int
modpow(unsigned int b, unsigned int e)
{
    unsigned int result = 1;
    //b = b % NTT_PRIME;
    while (e > 0)
    {
        if (e % 2 == 1)
        {
           result = modmul(result, b);
        }
        e >>= 1;
        b = modmul(b, b);
    }
    return result;
}

__global__ void
cooley_tukey_complex_fft(unsigned int* __restrict__ A,
                         int s)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int m = (1 << s);
    int k = idx / (m/2);
    k *= m;
    int j = idx % (m/2);
    unsigned int wn = primitive_roots[s];
    unsigned int w = modpow(wn, (unsigned int)j);
    unsigned int t, u;
    
    t = modmul(w, A[k + j + m/2]);
    u = A[k + j];
    
    A[k + j] = (u + t) % NTT_PRIME;
    A[j + k + m/2] = (u + NTT_PRIME - t) % NTT_PRIME;
}

__global__ void
cooley_tukey_complex_ifft(unsigned int* __restrict__ A,
                         int s,
                         int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int m = (1 << s);
    int k = idx / (m/2);
    k *= m;
    int j = idx % (m/2);
    unsigned int wn = inverse_roots[s];
    unsigned int w = modpow(wn, (unsigned int)j);
    unsigned int t, u;
    
    t = modmul(w, A[k + j + m/2]);
    u = A[k + j];
    
    A[k + j] = (u + t) % NTT_PRIME;
    A[j + k + m/2] = (u + NTT_PRIME - t) % NTT_PRIME;
    
    if (m == N)
    {
        A[k + j] = modmul(A[k + j], inverse_mod[s]);
        A[k + j + m/2] = modmul(A[k + j + m/2], inverse_mod[s]);
    }
}

void
cooley_tukey_fft(unsigned int* a, int len)
{
    assert(isPow2(len));
    
    bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_fft<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s);
    }
}

void
cooley_tukey_ifft(unsigned int* a, int len)
{
    assert(isPow2(len));
    
    bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_ifft<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s, len);
    }
}





__global__ void
cuda_combine2(unsigned int* a, unsigned int* c, unsigned long long* carry)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    unsigned long long result = 0;
    unsigned int w1 = a[idx*16];
    unsigned int w2 = a[idx*16+1];
    unsigned int w3 = a[idx*16+2];
    unsigned int w4 = a[idx*16+3];
    unsigned int w5 = a[idx*16+4];
    unsigned int w6 = a[idx*16+5];
    unsigned int w7 = a[idx*16+6];
    unsigned int w8 = a[idx*16+7];
    unsigned int w9 = a[idx*16+8];
    unsigned int w10 = a[idx*16+9];
    unsigned int w11 = a[idx*16+10];
    unsigned int w12 = a[idx*16+11];
    unsigned int w13 = a[idx*16+12];
    unsigned int w14 = a[idx*16+13];
    unsigned int w15 = a[idx*16+14];
    unsigned int w16 = a[idx*16+15];
    
    result += w16;
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
    
    c[idx] = (unsigned int) (result & 0xffffffff);
    carry[idx] = (result >> 32);
}

__global__ void
split2(const unsigned int* __restrict__ in,
      unsigned int* __restrict__ out)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x);
    
    out[idx*16] = (unsigned int) (in[idx] & 0x3);
    out[idx*16+1] = (unsigned int) ((in[idx] >> 2) & 0x3);
    out[idx*16+2] = (unsigned int) ((in[idx] >> 4) & 0x3);
    out[idx*16+3] = (unsigned int) ((in[idx] >> 6) & 0x3);
    out[idx*16+4] = (unsigned int) ((in[idx] >> 8) & 0x3);
    out[idx*16+5] = (unsigned int) ((in[idx] >> 10) & 0x3);
    out[idx*16+6] = (unsigned int) ((in[idx] >> 12) & 0x3);
    out[idx*16+7] = (unsigned int) ((in[idx] >> 14) & 0x3);
    out[idx*16+8] = (unsigned int) ((in[idx] >> 16) & 0x3);
    out[idx*16+9] = (unsigned int) ((in[idx] >> 18) & 0x3);
    out[idx*16+10] = (unsigned int) ((in[idx] >> 20) & 0x3);
    out[idx*16+11] = (unsigned int) ((in[idx] >> 22) & 0x3);
    out[idx*16+12] = (unsigned int) ((in[idx] >> 24) & 0x3);
    out[idx*16+13] = (unsigned int) ((in[idx] >> 26) & 0x3);
    out[idx*16+14] = (unsigned int) ((in[idx] >> 28) & 0x3);
    out[idx*16+15] = (unsigned int) ((in[idx] >> 30) & 0x3);
}


__global__ void
cuda_combine4(unsigned int* a, unsigned int* c, unsigned long long* carry)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    unsigned long long result = 0;
    unsigned int w1 = a[idx*8];
    unsigned int w2 = a[idx*8+1];
    unsigned int w3 = a[idx*8+2];
    unsigned int w4 = a[idx*8+3];
    unsigned int w5 = a[idx*8+4];
    unsigned int w6 = a[idx*8+5];
    unsigned int w7 = a[idx*8+6];
    unsigned int w8 = a[idx*8+7];

    result += w8;
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
    
    c[idx] = (unsigned int) (result & 0xffffffff);
    carry[idx] = (result >> 32);
}

__global__ void
split4(const unsigned int* __restrict__ in,
      unsigned int* __restrict__ out)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x);
    
    out[idx*8] = (unsigned int) (in[idx] & 0xf);
    out[idx*8+1] = (unsigned int) ((in[idx] >> 4) & 0xf);
    out[idx*8+2] = (unsigned int) ((in[idx] >> 8) & 0xf);
    out[idx*8+3] = (unsigned int) ((in[idx] >> 12) & 0xf);
    out[idx*8+4] = (unsigned int) ((in[idx] >> 16) & 0xf);
    out[idx*8+5] = (unsigned int) ((in[idx] >> 20) & 0xf);
    out[idx*8+6] = (unsigned int) ((in[idx] >> 24) & 0xf);
    out[idx*8+7] = (unsigned int) ((in[idx] >> 28) & 0xf);
}


__global__ void
cuda_combine8(unsigned int* a, unsigned int* c, unsigned long long* carry)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    unsigned long long result = 0;
    unsigned int w1 = a[idx*4];
    unsigned int w2 = a[idx*4+1];
    unsigned int w3 = a[idx*4+2];
    unsigned int w4 = a[idx*4+3];
    
    result += w4;
    result <<= 8;
    result += w3;
    result <<= 8;
    result += w2;
    result <<= 8;
    result += w1;
    
    c[idx] = (unsigned int) (result & 0xffffffff);
    carry[idx] = (result >> 32);
}

__global__ void
split8(const unsigned int* __restrict__ in,
      unsigned int* __restrict__ out)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x);
    
    out[idx*4] = (unsigned int) (in[idx] & 0xff);
    out[idx*4+1] = (unsigned int) ((in[idx] >> 8) & 0xff);
    out[idx*4+2] = (unsigned int) ((in[idx] >> 16) & 0xff);
    out[idx*4+3] = (unsigned int) ((in[idx] >> 24) & 0xff);
}

void
combine(unsigned int* a, CudaBigInt& c, int bitsize)
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
    
    if (bitsize == 2)
    {
        cuda_combine2<<<(c.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, c.val, long_carry);
    } else if (bitsize == 4) {
        cuda_combine4<<<(c.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, c.val, long_carry);
    } else if (bitsize == 8) {
        cuda_combine8<<<(c.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, c.val, long_carry);
    } else {
        assert(0);
    }
    
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

__global__ void
pointwise_square(unsigned int* __restrict__ A)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    A[idx] = modmul(A[idx], A[idx]);
}

void
square(CudaBigInt& a, CudaBigInt& c)
{
    unsigned int* cuda_a;
    
    if (a.word_len*32 <= (1<<15))
    {
        multiply(a, a, c);
    } else if (a.word_len*32 <= (1<<19))
    {
        cuda_malloc_clear((void**) &cuda_a, sizeof(*cuda_a)*a.word_len*8);
        
        split8<<<(a.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a.val, cuda_a);
        
        cooley_tukey_fft(cuda_a, a.word_len*8);
        pointwise_square<<<(a.word_len*8/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_a);
        cooley_tukey_ifft(cuda_a, a.word_len*8);
        
        combine(cuda_a, c, 8);
        
        cuda_malloc_free(cuda_a);
    } else if (a.word_len*32 <= (1<<23))
    {
        cuda_malloc_clear((void**) &cuda_a, sizeof(*cuda_a)*a.word_len*16);
        
        split4<<<(a.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a.val, cuda_a);
        
        cooley_tukey_fft(cuda_a, a.word_len*16);
        pointwise_square<<<(a.word_len*16/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_a);
        cooley_tukey_ifft(cuda_a, a.word_len*16);
        
        combine(cuda_a, c, 4);
        
        cuda_malloc_free(cuda_a);
    } else if (a.word_len*32 <= (1<<27))
    {
        cuda_malloc_clear((void**) &cuda_a, sizeof(*cuda_a)*a.word_len*32);
        
        split2<<<(a.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a.val, cuda_a);
        
        cooley_tukey_fft(cuda_a, a.word_len*32);
        pointwise_square<<<(a.word_len*32/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_a);
        cooley_tukey_ifft(cuda_a, a.word_len*32);
        
        combine(cuda_a, c, 2);
        
        cuda_malloc_free(cuda_a);
    }
}
