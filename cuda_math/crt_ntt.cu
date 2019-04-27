#include "crt_ntt.h"

#include "bigint.h"
#include "carry.h"
#include "memory.h"
#include "multiply.h"

#define FFT_BLOCK_SIZE (128)

// This prime is of the form k*2^n+1 = 15*(2^27)+1
#define NTT_PRIME1 ((unsigned int) 0x78000001) // 2013265921

__device__
unsigned int primitive_roots1[] = {
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
unsigned int inverse_roots1[] = {
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
unsigned int inverse_mod1[] = {
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







// This prime is of the form k*2^n+1 = 27*(2^26)+1
#define NTT_PRIME2 ((unsigned int) 0x6c000001) // 1811939329

__device__
unsigned int primitive_roots2[] = {
    1,
    1811939328,
    1416949424,
    1452317833,
    659408637,
    860297611,
    1472666535,
    1109739630,
    18116277,
    209403217,
    69915711,
    1154101769,
    606837284,
    1489399950,
    465083369,
    517598978,
    1456252962,
    1784331046,
    1330130053,
    1138266161,
    971241113,
    121895319,
    579520204,
    388825445,
    1762019879,
    209208363,
    72705542,
    388053258,
    4826809,
    2197,
    13,
    1
};


__device__
unsigned int inverse_roots2[] = {
    1,
    1811939328,
    394989905,
    1756022077,
    1368643352,
    1681104208,
    1594001182,
    842788380,
    576638474,
    770487725,
    1682986047,
    488136043,
    450492458,
    669232625,
    1720000667,
    110413286,
    1537158106,
    1125316264,
    1363276908,
    925937071,
    311798718,
    1506675331,
    1748414954,
    1510677230,
    964549597,
    461327191,
    801700081,
    645050696,
    477859254,
    743909547,
    696899742,
    1
};

__device__
unsigned int inverse_mod2[] = {
    1,
    905969665,
    1358954497,
    1585446913,
    1698693121,
    1755316225,
    1783627777,
    1797783553,
    1804861441,
    1808400385,
    1810169857,
    1811054593,
    1811496961,
    1811718145,
    1811828737,
    1811884033,
    1811911681,
    1811925505,
    1811932417,
    1811935873,
    1811937601,
    1811938465,
    1811938897,
    1811939113,
    1811939221,
    1811939275,
    1811939302,
    905969651,
    1358954490,
    679477245,
    1245708287,
    1528823808
};


// The following are twiddle factors that will be precalculated
unsigned int *twiddle_factors1[26];
unsigned int *twiddle_factors2[26];
unsigned int *twiddle_factors_inv1[26];
unsigned int *twiddle_factors_inv2[26];
int highest_twiddle_calculated = -1;


#define NTT_PRIMES (((unsigned long long) NTT_PRIME1) * NTT_PRIME2)
#define NTT_T1 ((unsigned long long) 10)
#define NTT_T2 ((unsigned long long) 1811939320)

__global__ void
crt_bitreverse(unsigned int* out,
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

/*
    Arithmetic costs for Nvidia GPU 3.0 compute:
    
    add (32 bit)      - ~1.2
    multiply (32 bit) - ~6.0
    bitwise (32 bit)  - ~1.2
    shift (32 bit)    - ~6.0
    bitrev (32 bit)   - ~6.0
    
    add (64 bit)      - ~3
    multiply (64 bit) - ~24
    bitwise (64 bit)  - ~3
    shift (64 bit)    - ~12
    

*/

// Approx. 30 clock cycles
__device__ unsigned int
modmul1(unsigned int a, unsigned int b)
{
    return (((unsigned long long) a) * b) % NTT_PRIME1;
}

__device__ unsigned int
modmul2(unsigned int a, unsigned int b)
{
    return (((unsigned long long) a) * b) % NTT_PRIME2;
}

__device__ unsigned int
modpow1(unsigned int b, unsigned int e)
{
    unsigned int result = 1;
    while (e > 0)
    {
        if (e % 2 == 1)
        {
           result = modmul1(result, b);
        }
        e >>= 1;
        b = modmul1(b, b);
    }
    return result;
}

__device__ unsigned int
modpow2(unsigned int b, unsigned int e)
{
    unsigned int result = 1;
    while (e > 0)
    {
        if (e % 2 == 1)
        {
           result = modmul2(result, b);
        }
        e >>= 1;
        b = modmul2(b, b);
    }
    return result;
}

// Procompute twiddle factors
__global__ void
compute_twiddles(int s, unsigned int* __restrict__ t1,
                 unsigned int* __restrict__ it1, unsigned int* __restrict__ t2,
                 unsigned int* __restrict__ it2)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    t1[idx] = modpow1(primitive_roots1[s], idx);
    it1[idx] = modpow1(inverse_roots1[s], idx);
    t2[idx] = modpow2(primitive_roots2[s], idx);
    it2[idx] = modpow2(inverse_roots2[s], idx);
}



__global__ void
cooley_tukey_complex_fft1_1(unsigned int* __restrict__ A, unsigned int* tf)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int k = idx;
    k <<= 1;
    int j = 0;
    unsigned int w1 = tf[j];
    unsigned int t, u;
    
    u = A[k + j];
    t = modmul1(w1, A[k + j + 1]);
    
    A[k + j] = (u + t) % NTT_PRIME1;
    A[j + k + 1] = (u + NTT_PRIME1 - t) % NTT_PRIME1;
}

__global__ void
cooley_tukey_complex_fft2_1(unsigned int* __restrict__ A, unsigned int* tf)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int k = idx;
    k <<= 1;
    int j = 0;// % (m/2);
    unsigned int w2 = tf[j];
    unsigned int t, u;
    
    t = modmul2(w2, A[k + j + 1]);
    u = A[k + j];
    
    A[k + j] = (u + t) % NTT_PRIME2;
    A[j + k + 1] = (u + NTT_PRIME2 - t) % NTT_PRIME2;
}

__global__ void
cooley_tukey_complex_ifft1_1(unsigned int* __restrict__ A,
                             unsigned int* tf)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int k = idx;
    k <<= 1;
    int j = 0;// % (m/2);
    unsigned int w1 = tf[j];
    unsigned int t, u;
    
    t = modmul1(w1, A[k + j + 1]);
    u = A[k + j];
    
    A[k + j] = (u + t) % NTT_PRIME1;
    A[j + k + 1] = (u + NTT_PRIME1 - t) % NTT_PRIME1;
}

__global__ void
cooley_tukey_complex_ifft2_1(unsigned int* __restrict__ A,
                             unsigned int* tf)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int k = idx;
    k <<= 1;
    int j = 0;// % (m/2);
    unsigned int w2 = tf[j];
    unsigned int t, u;
    
    t = modmul2(w2, A[k + j + 1]);
    u = A[k + j];
    
    A[k + j] = (u + t) % NTT_PRIME2;
    A[j + k + 1] = (u + NTT_PRIME2 - t) % NTT_PRIME2;
}




/*
    Arithmetic costs for Nvidia GPU 3.0 compute:
    
    add (32 bit)      - ~1.2
    multiply (32 bit) - ~6.0
    bitwise (32 bit)  - ~1.2
    shift (32 bit)    - ~6.0
    bitrev (32 bit)   - ~6.0
    
    add (64 bit)      - ~3
    multiply (64 bit) - ~24
    bitwise (64 bit)  - ~3
    shift (64 bit)    - ~12
    

*/
__global__ void
cooley_tukey_complex_fft1(unsigned int* __restrict__ A,
                         int s, unsigned int* tf)
{
    // ~7 cycles
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    // ~6 cycles
    int m = (1 << s);
    // ~12 cycles
    int k = idx / (m/2);
    // ~6 cycles
    k <<= s;
    // ~12 cycles
    int j = idx & ((1 << (s-1)) - 1);// % (m/2);
    unsigned int w1 = tf[j];
    unsigned int t, u;
    
    u = A[k + j];
    // ~30 cycles
    t = modmul1(w1, A[k + j + m/2]);
    
    // ~7 cycles
    A[k + j] = (u + t) % NTT_PRIME1;
    // ~9 cycles
    A[j + k + m/2] = (u + NTT_PRIME1 - t) % NTT_PRIME1;
}

__global__ void
cooley_tukey_complex_fft2(unsigned int* __restrict__ A,
                         int s, unsigned int* tf)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int m = (1 << s);
    int k = idx / (m/2);
    k <<= s;
    int j = idx & ((1 << (s-1)) - 1);// % (m/2);
    unsigned int w2 = tf[j];
    unsigned int t, u;
    
    t = modmul2(w2, A[k + j + m/2]);
    u = A[k + j];
    
    A[k + j] = (u + t) % NTT_PRIME2;
    A[j + k + m/2] = (u + NTT_PRIME2 - t) % NTT_PRIME2;
}

__global__ void
cooley_tukey_complex_ifft1(unsigned int* __restrict__ A,
                           int s, unsigned int* tf,
                           int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int m = (1 << s);
    int k = idx / (m/2);
    k <<= s;
    int j = idx & ((1 << (s-1)) - 1);// % (m/2);
    unsigned int w1 = tf[j];
    unsigned int t, u;
    
    t = modmul1(w1, A[k + j + m/2]);
    u = A[k + j];
    
    A[k + j] = (u + t) % NTT_PRIME1;
    A[j + k + m/2] = (u + NTT_PRIME1 - t) % NTT_PRIME1;
    
    if (m == N)
    {
        A[k + j] = modmul1(A[k + j], inverse_mod1[s]);
        A[k + j + m/2] = modmul1(A[k + j + m/2], inverse_mod1[s]);
    }
}

__global__ void
cooley_tukey_complex_ifft2(unsigned int* __restrict__ A,
                           int s, unsigned int* tf,
                           int N)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int m = (1 << s);
    int k = idx / (m/2);
    k <<= s;
    int j = idx & ((1 << (s-1)) - 1);// % (m/2);
    unsigned int w2 = tf[j];
    unsigned int t, u;
    
    t = modmul2(w2, A[k + j + m/2]);
    u = A[k + j];
    
    A[k + j] = (u + t) % NTT_PRIME2;
    A[j + k + m/2] = (u + NTT_PRIME2 - t) % NTT_PRIME2;
    
    if (m == N)
    {
        A[k + j] = modmul2(A[k + j], inverse_mod2[s]);
        A[k + j + m/2] = modmul2(A[k + j + m/2], inverse_mod2[s]);
    }
}


__global__ void
split(const unsigned int* __restrict__ in,
      unsigned int* __restrict__ out)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x);
    
    out[idx*2] = (in[idx] & 0xffff);
    out[idx*2+1] = ((in[idx] >> 16) & 0xffff);
}


__global__ void
cuda_combine(unsigned int* in1, unsigned int* in2, unsigned int* out, unsigned long long* carry)
{
    int idx = (blockIdx.x*blockDim.x + threadIdx.x);
    
    unsigned long long crt1 = 0;
    unsigned long long crt2 = 0;
    unsigned long long y11 = in1[idx*2];
    unsigned long long y21 = in2[idx*2];
    unsigned long long y12 = in1[idx*2+1];
    unsigned long long y22 = in2[idx*2+1];
    
    y11 *= NTT_T1;
    y21 *= NTT_T2;
    
    y11 %= NTT_PRIME1;
    y21 %= NTT_PRIME2;
    
    y11 *= NTT_PRIME2;
    y21 *= NTT_PRIME1;
    
    crt1 = (y11 + y21) % NTT_PRIMES;
    
    y12 *= NTT_T1;
    y22 *= NTT_T2;
    
    y12 %= NTT_PRIME1;
    y22 %= NTT_PRIME2;
    
    y12 *= NTT_PRIME2;
    y22 *= NTT_PRIME1;
    
    crt2 = (y12 + y22) % NTT_PRIMES;
    
    out[idx] = (crt2 & 0xffff) << 16;
    out[idx] += crt1 & 0xffffffff;
    if (out[idx] < (crt1 & 0xffffffff))
    {
        carry[idx] = 1;
    }
    
    carry[idx] += crt1 >> 32;
    carry[idx] += crt2 >> 16;
}

void
combine(unsigned int* in1, unsigned int* in2, CudaBigInt& c)
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
    
    cuda_combine<<<(c.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(in1, in2, c.val, long_carry);
    
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
pointwise_square1(unsigned int* __restrict__ A)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    A[idx] = modmul1(A[idx], A[idx]);
}

__global__ void
pointwise_square2(unsigned int* __restrict__ A)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    A[idx] = modmul2(A[idx], A[idx]);
}

/*
void
cooley_tukey_fft1(unsigned int* a, int len)
{
    assert(isPow2(len));
    
    crt_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_fft1<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s);
    }
}

void
cooley_tukey_ifft1(unsigned int* a, int len)
{
    assert(isPow2(len));
    
    crt_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_ifft1<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s, len);
    }
}


void
cooley_tukey_fft2(unsigned int* a, int len)
{
    assert(isPow2(len));
    
    crt_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_fft2<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s);
    }
}

void
cooley_tukey_ifft2(unsigned int* a, int len)
{
    assert(isPow2(len));
    
    crt_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        cooley_tukey_complex_ifft2<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a, s, len);
    }
}*/


void
cooley_tukey_ffts(unsigned int* a1, unsigned int* a2, int len)
{
    assert(isPow2(len));
    
    cudaError_t err;
    
    crt_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a1, log2(len));
    crt_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a2, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        if (highest_twiddle_calculated < s)
        {
            cuda_malloc_clear((void**)&twiddle_factors1[s], sizeof(unsigned int) * (1<<s)/2);
            cuda_malloc_clear((void**)&twiddle_factors_inv1[s], sizeof(unsigned int) * (1<<s)/2);
            cuda_malloc_clear((void**)&twiddle_factors2[s], sizeof(unsigned int) * (1<<s)/2);
            cuda_malloc_clear((void**)&twiddle_factors_inv2[s], sizeof(unsigned int) * (1<<s)/2);
            
            if ((1<<s)/2 < FFT_BLOCK_SIZE)
            {
                compute_twiddles<<<1, (1<<s)/2>>>(s, twiddle_factors1[s], twiddle_factors_inv1[s], twiddle_factors2[s], twiddle_factors_inv2[s]);
            } else {
                compute_twiddles<<<((1<<s)/2)/FFT_BLOCK_SIZE, FFT_BLOCK_SIZE>>>(s, twiddle_factors1[s], twiddle_factors_inv1[s], twiddle_factors2[s], twiddle_factors_inv2[s]);
            }
            highest_twiddle_calculated = s;
        }
        if (s == 1)
        {
            cooley_tukey_complex_fft1_1<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a1, twiddle_factors1[s]);
            cooley_tukey_complex_fft2_1<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a2, twiddle_factors2[s]);
        } else {
            cooley_tukey_complex_fft1<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a1, s, twiddle_factors1[s]);
            cooley_tukey_complex_fft2<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a2, s, twiddle_factors2[s]);
        }
    }
}

void
cooley_tukey_iffts(unsigned int* a1, unsigned int* a2, int len)
{
    assert(isPow2(len));
    
    cudaError_t err;
    
    crt_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE, 0>>>(a1, log2(len));
    crt_bitreverse<<<(len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE, 0>>>(a2, log2(len));

    
    for (int s = 1; s <= log2(len); s++)
    {
        if (highest_twiddle_calculated < s)
        {
            cuda_malloc_clear((void**)&twiddle_factors1[s], sizeof(unsigned int) * (1<<s)/2);
            cuda_malloc_clear((void**)&twiddle_factors_inv1[s], sizeof(unsigned int) * (1<<s)/2);
            cuda_malloc_clear((void**)&twiddle_factors2[s], sizeof(unsigned int) * (1<<s)/2);
            cuda_malloc_clear((void**)&twiddle_factors_inv2[s], sizeof(unsigned int) * (1<<s)/2);
            
            if ((1<<s)/2 < FFT_BLOCK_SIZE)
            {
                compute_twiddles<<<1, (1<<s)/2>>>(s, twiddle_factors1[s], twiddle_factors_inv1[s], twiddle_factors2[s], twiddle_factors_inv2[s]);
            } else {
                compute_twiddles<<<((1<<s)/2)/FFT_BLOCK_SIZE, FFT_BLOCK_SIZE>>>(s, twiddle_factors1[s], twiddle_factors_inv1[s], twiddle_factors2[s], twiddle_factors_inv2[s]);
            }
            highest_twiddle_calculated = s;
        }
        if (s == 1)
        {
            cooley_tukey_complex_ifft1_1<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a1,twiddle_factors_inv1[s]);
            cooley_tukey_complex_ifft2_1<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a2,twiddle_factors_inv2[s]);
        } else {
            cooley_tukey_complex_ifft1<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a1, s, twiddle_factors_inv1[s], len);
            cooley_tukey_complex_ifft2<<<((len/2)/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a2, s, twiddle_factors_inv2[s], len);
        }
    }
}


void
crt_square(CudaBigInt& a, CudaBigInt& c)
{
    unsigned int* cuda_a1;
    unsigned int* cuda_a2;
    
    if (a.word_len*32 <= (1<<15))
    {
        multiply(a, a, c);
    } else {
        cuda_malloc_clear((void**) &cuda_a1, sizeof(*cuda_a1)*a.word_len*4);
        cuda_malloc_clear((void**) &cuda_a2, sizeof(*cuda_a2)*a.word_len*4);
        
        split<<<(a.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a.val, cuda_a1);
        split<<<(a.word_len/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(a.val, cuda_a2);
        
        //cooley_tukey_fft1(cuda_a1, a.word_len*4);
        //cooley_tukey_fft2(cuda_a2, a.word_len*4);
        cooley_tukey_ffts(cuda_a1, cuda_a2, a.word_len*4);
        pointwise_square1<<<(a.word_len*4/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_a1);
        pointwise_square2<<<(a.word_len*4/FFT_BLOCK_SIZE), FFT_BLOCK_SIZE>>>(cuda_a2);
        //cooley_tukey_ifft1(cuda_a1, a.word_len*4);
        //cooley_tukey_ifft2(cuda_a2, a.word_len*4);
        cooley_tukey_iffts(cuda_a1, cuda_a2, a.word_len*4);
        
        combine(cuda_a1, cuda_a2, c);
        
        cuda_malloc_free(cuda_a1);
        cuda_malloc_free(cuda_a2);
    }
}





