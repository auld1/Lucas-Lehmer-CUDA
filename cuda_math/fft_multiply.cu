#include "fft_multiply.h"

#include "bigint.h"
#include "carry.h"
#include "memory.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuComplex.h>

#include <gmp.h>

unsigned int rand_uint32(void) {
  unsigned int r = 0;
  for (int i=0; i<32; i += 30) {
    r = r*((unsigned int)RAND_MAX + 1) + rand();
  }
  return r;
}

__global__ void
split(const unsigned int* __restrict__ in,
      unsigned int* __restrict__ out,
      int in_words_per_part,
      int out_words_per_part)
{
    int in_idx = (blockIdx.x*blockDim.x + threadIdx.x) * in_words_per_part;
    int out_idx =  (blockIdx.x*blockDim.x + threadIdx.x) * out_words_per_part;
    int i = 0;
    
    for (i = 0; i < in_words_per_part; i++)
    {
        out[out_idx + i] = in[in_idx + i];
    }
    
    for (; i < out_words_per_part; i++)
    {
        out[out_idx + i] = 0;
    }
}

__global__ void
ssa_fft(unsigned int* __restrict__ A,
        int words_per_part,
        int total_parts,
        int root_of_unity_shift,
        int m,
        unsigned int* __restrict__ scratch1,
        unsigned int* __restrict__ scratch2)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int k = 2 * (idx/m);
    k *= m;
    int j = idx % (m/2);
    (void) k;
    (void) j;
}

__global__ void
ints_to_complex_bitreverse(const unsigned int* __restrict__ in,
                           cuDoubleComplex* __restrict__ out,
                           int bitlen)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    out[(__brev(idx) >> (32-bitlen))] = make_cuDoubleComplex((double)in[idx], 0);
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

__global__ void
ints_to_complex_bitreverse_split(const unsigned int* __restrict__ in,
                                 cuDoubleComplex* __restrict__ out,
                                 int bitlen)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    out[(__brev(idx*2) >> (32-(bitlen+1)))] = make_cuDoubleComplex((double)(in[idx]&0xffff), 0);
    out[(__brev(idx*2+1) >> (32-(bitlen+1)))] = make_cuDoubleComplex((double)((in[idx]>>16)&0xffff), 0);
}

__global__ void
words_to_complex(const unsigned int* __restrict__ in,
                 cuDoubleComplex* __restrict__ out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    out[idx*2] = make_cuDoubleComplex((double)(in[idx] & 0xffff), 0);
    out[idx*2+1] = make_cuDoubleComplex((double)((in[idx] >> 16) & 0xffff), 0);
}

// Found at https://devtalk.nvidia.com/default/topic/814159/additional-cucomplex-functions-cucnorm-cucsqrt-cucexp-and-some-complex-double-functions-/
__host__ __device__ static __inline__ cuDoubleComplex
cuCexp(cuDoubleComplex x)
{
	double factor = exp(x.x);
	return make_cuDoubleComplex(factor * cos(x.y), factor * sin(x.y));
}

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

#define FFT_BLOCK_SIZE (128)
void
cooley_tukey_fft(unsigned int* in, cuDoubleComplex* out, int len)
{
    cuDoubleComplex* cuda_out;
    unsigned int* cuda_in;
    
    assert(isPow2(len));
    
    cuda_malloc_clear((void**) &cuda_out, len * sizeof(cuDoubleComplex));
    cuda_malloc_clear((void**) &cuda_in, len * sizeof(unsigned int));
    
    cudaMemcpy(cuda_in, in, len * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    ints_to_complex_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_in, cuda_out, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_fft<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_out, s, -1, make_cuDoubleComplex(0, ((double)-2.0) * M_PI / (1<<s)), len);
    }
    
    cudaMemcpy(out, cuda_out, len * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    cudaFree(cuda_out);
    cudaFree(cuda_in);
}

void
cooley_tukey_ifft(cuDoubleComplex* in, unsigned int* out, int len)
{
    cuDoubleComplex* cuda_in;
    cuDoubleComplex* cuda_out;
    cuDoubleComplex* device_out = (cuDoubleComplex*) malloc(len * sizeof(cuDoubleComplex));
    
    assert(isPow2(len));
    
    cuda_malloc_clear((void**) &cuda_out, len * sizeof(cuDoubleComplex));
    
    cudaMemcpy(cuda_out, in, len * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    complex_to_complex_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_out, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_fft<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_out, s, 1, make_cuDoubleComplex(0, ((double)2.0) * M_PI / (1<<s)), len);
    }
    
    cudaMemcpy(device_out, cuda_out, len * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < len; i++)
    {
        out[i] = (unsigned int) (device_out[i].x + .5);
    }
    
    free(device_out);
    cudaFree(cuda_out);
}

void
fft_square(CudaBigInt& a, CudaBigInt& c)
{
    cuDoubleComplex* cuda_out;
    int cuda_out_word_len = c.word_len*2;
    
    assert(isPow2(a.word_len));
    assert(c.word_len = 2*a.word_len);
    
    cuda_malloc_clear((void**) &cuda_out, cuda_out_word_len * sizeof(cuDoubleComplex));
    
    ints_to_complex_bitreverse_split<<<(a.word_len/32), 32>>>(a.val, cuda_out, log2(c.word_len));

    
    for (int m = 2; m <= c.word_len*2; m*=2)
    {
        //cooley_tukey_complex_fft<<<(cuda_out_word_len/64), 32>>>(cuda_out, m, -1, cuda_out_word_len);
    }
    
    pointwise_square<<<(cuda_out_word_len/32), 32>>>(cuda_out);
    
    
}


int
main(void)
{
    cuDoubleComplex* host_out;
    unsigned int* host_in;
    int len = 1<<26;
    
    host_out = (cuDoubleComplex*) malloc(len * sizeof(*host_out));
    host_in = (unsigned int*) malloc(len * sizeof(*host_in));
    
    for (int i = 0; i < len; i++)
    {
        host_in[i] = i;
    }
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    cooley_tukey_fft(host_in, host_out, len);
    
    for (int i = 0; i < len; i++)
    {
        host_in[i] = 0;
    }
    
    cooley_tukey_ifft(host_out, host_in, len);
    
    for (int i = 0; i < len; i++)
    {
        assert(host_in[i] == i);
        /*
        if (host_out[i].y >= 0)
        {
            printf("%lf+%lfi\n", host_out[i].x, host_out[i].y);
        } else {
            printf("%lf%lfi\n", host_out[i].x, host_out[i].y);
        }*/
    }
}














