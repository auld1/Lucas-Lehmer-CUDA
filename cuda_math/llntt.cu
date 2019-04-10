#include "llntt.h"

#include "bigint.h"
#include "carry.h"
#include "memory.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


#define FFT_BLOCK_SIZE (128)

// This prime is of the form k*n + 1 =  ((2^31)-7)*2^32+1
// therefore, our max transform length is 2^32 (in reality 2^30 to
// avoid overflow with 16 bit inputs).  Because 2^32 is highly
// composite, every power of two length transform less than it has
// a primitive root
#define NTT_PRIME 0x7ffffff900000001ULL // 9223372006790004737

__device__
unsigned long long primitive_roots[] = {
    1ULL,
    9223372006790004736ULL,
    5755924619642695216ULL,
    4769699149823068835ULL,
    1942127889539119125ULL,
    3782048346210197788ULL,
    1270679814011126415ULL,
    7265870340368527970ULL,
    3828317218034052165ULL,
    2835482956611806098ULL,
    4429726914657305271ULL,
    6729197123921240845ULL,
    8470770063660059814ULL,
    1259775548357471949ULL,
    1886203001709905978ULL,
    8551568661743327187ULL,
    559870612843557680ULL,
    4380295187905642524ULL,
    3279344313026961415ULL,
    7501605496247997781ULL,
    2076037129982507684ULL,
    5018156356764108893ULL,
    2287809993268449646ULL,
    4283269332146465754ULL,
    890417586068990129ULL,
    6074655204156143209ULL,
    1442549083741876711ULL,
    1077732879817165149ULL,
    4521737409507229998ULL,
    1601543172978583857ULL,
    7065375509131717369ULL
};

__device__
unsigned long long inverse_roots[] = {
    1ULL,
    9223372006790004736ULL,
    3467447387147309521ULL,
    6851455013668542986ULL,
    3362766400017373277ULL,
    429292554160982884ULL,
    1641660664282386704ULL,
    6070162760184377345ULL,
    5213051889972836346ULL,
    3063650602234021557ULL,
    1428254981365260835ULL,
    5774179872364407492ULL,
    3039652001753857781ULL,
    8894232283185975283ULL,
    4423020116378691237ULL,
    7272603243271965854ULL,
    4181120089695106629ULL,
    237770818864091407ULL,
    3999420630608685535ULL,
    1544078994390720807ULL,
    8173098041952696521ULL,
    3728509672789315694ULL,
    5178408059680766236ULL,
    6956117554446149628ULL,
    2963619937174409050ULL,
    62243420894161466ULL,
    7901279738766610461ULL,
    4510859787583909548ULL,
    8828377337343191099ULL,
    8316491991072566619ULL,
    7189316364534173387ULL,
};

__device__
unsigned long long inverse_mod[] = {
    1ULL,
    4611686003395002369ULL,
    6917529005092503553ULL,
    8070450505941254145ULL,
    8646911256365629441ULL,
    8935141631577817089ULL,
    9079256819183910913ULL,
    9151314412986957825ULL,
    9187343209888481281ULL,
    9205357608339243009ULL,
    9214364807564623873ULL,
    9218868407177314305ULL,
    9221120206983659521ULL,
    9222246106886832129ULL,
    9222809056838418433ULL,
    9223090531814211585ULL,
    9223231269302108161ULL,
    9223301638046056449ULL,
    9223336822418030593ULL,
    9223354414604017665ULL,
    9223363210697011201ULL,
    9223367608743507969ULL,
    9223369807766756353ULL,
    9223370907278380545ULL,
    9223371457034192641ULL,
    9223371731912098689ULL,
    9223371869351051713ULL,
    9223371938070528225ULL,
    9223371972430266481ULL,
    9223371989610135609ULL,
    9223371998200070173ULL
};

__global__ void
bitreverse(unsigned long long* __restrict__ out,
           int bitlen)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int rev_idx = (__brev(idx) >> (32-bitlen));
    if (rev_idx < idx)
    {
        unsigned long long tmp = out[rev_idx];
        out[rev_idx] = out[idx];
        out[idx] = tmp;
    }
}

// Code found at https://www.geeksforgeeks.org/how-to-avoid-overflow-in-modular-multiplication/
// Note, because our prime is 63 bits long, any mod of it is at most 63 bits.
// Therefore, we can multiply by 2 or add two members and they will be at
// most 64 bits, allowing reduction to 63 bits again without overflow
__device__ unsigned long long
modmul(unsigned long long a, unsigned long long b)
{ 
    unsigned long long res = 0; // Initialize result 
    a = a % NTT_PRIME; 
    while (b > 0) 
    { 
        // If b is odd, add 'a' to result 
        if (b % 2 == 1) 
            res = (res + a) % NTT_PRIME; 
  
        // Multiply 'a' with 2 
        a = (a * 2) % NTT_PRIME; 
  
        // Divide b by 2 
        b /= 2; 
    } 
  
    // Return result 
    return res % NTT_PRIME; 
}

__device__ unsigned long long
modpow(unsigned long long b, unsigned long long e)
{
    unsigned long long result = 1;
    b = b % NTT_PRIME;
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
cooley_tukey_complex_fft(unsigned long long* __restrict__ A,
                         int s)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int m = (1 << s);
    int k = idx / (m/2);
    k *= m;
    int j = idx % (m/2);
    unsigned long long wn = primitive_roots[s];
    unsigned long long w = modpow(wn, (unsigned long long)j);
    unsigned long long t, u;
    
    t = modmul(w, A[k + j + m/2]);
    u = A[k + j];
    
    A[k + j] = (u + t) % NTT_PRIME;
    if (u >= t)
    {
        A[j + k + m/2] = (u - t) % NTT_PRIME;
    } else {
        A[j + k + m/2] = ((u + NTT_PRIME) - t) % NTT_PRIME;
    }
}

__global__ void
cooley_tukey_complex_ifft(unsigned long long* __restrict__ A,
                         int s,
                         int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int m = (1 << s);
    int k = idx / (m/2);
    k *= m;
    int j = idx % (m/2);
    unsigned long long wn = inverse_roots[s];
    unsigned long long w = modpow(wn, (unsigned long long)j);
    unsigned long long t, u;
    
    t = modmul(w, A[k + j + m/2]);
    u = A[k + j];
    
    A[k + j] = (u + t) % NTT_PRIME;
    if (u >= t)
    {
        A[j + k + m/2] = (u - t) % NTT_PRIME;
    } else {
        A[j + k + m/2] = ((u + NTT_PRIME) - t) % NTT_PRIME;
    }
    
    if (m == N)
    {
        A[k + j] = modmul(A[k + j], inverse_mod[s]);
        A[k + j + m/2] = modmul(A[k + j + m/2], inverse_mod[s]);
    }
}

void
cooley_tukey_fft(unsigned long long* a, int len)
{
    assert(isPow2(len));
    
    bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_fft<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s);
    }
}

void
cooley_tukey_ifft(unsigned long long* a, int len)
{
    assert(isPow2(len));
    
    bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_ifft<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s, len);
    }
}





__global__ void
cuda_combine(unsigned long long* a, unsigned int* c, unsigned long long* carry)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    unsigned long long result = 0;
    unsigned long long w1 = a[idx*2];
    unsigned long long w2 = a[idx*2+1];
    unsigned long long tmp = 0;

    result += w2;
    tmp = (result >> 48) & 0xffff;
    result <<= 16; // Watch the top bits
    result += w1;
    if (result < w1)
    {
        tmp += 1; // Handle overflow
    }
    
    c[idx] = result & 0xffffffff;
    carry[idx] = (result >> 32) + (tmp << 32);
}

__global__ void
split(const unsigned int* __restrict__ in,
      unsigned long long* __restrict__ out)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x);
    
    out[idx*2] = in[idx] & 0xffff;
    out[idx*2+1] = (in[idx] >> 16) & 0xffff;
}

void
combine(unsigned long long* a, CudaBigInt& c)
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

__global__ void
pointwise_square(unsigned long long* __restrict__ A)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    A[idx] = modmul(A[idx], A[idx]);
}

void
ntt_square(CudaBigInt& a, CudaBigInt& c)
{
    unsigned long long* cuda_a;
    
    cuda_malloc_clear((void**) &cuda_a, sizeof(*cuda_a)*a.word_len*4);
    
    split<<<(a.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a.val, cuda_a);
    
    cooley_tukey_fft(cuda_a, a.word_len*4);
    pointwise_square<<<(a.word_len*4/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_a);
    cooley_tukey_ifft(cuda_a, a.word_len*4);
    
    combine(cuda_a, c);
    
    cuda_malloc_free(cuda_a);
}
