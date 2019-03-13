#include "bigint.h"
#include "carry.h"
#include "memory.h"

// Residue Number System
typedef struct rns
{
    unsigned int w1;
    unsigned int w2;
    unsigned int w3;
} rns;

// unsigned 96 bit int
typedef struct uint96_t
{
    unsigned int low64;
    unsigned long long high32;
} uint96_t;


//1912602623
#define RNS_P1 (57*(1<<25)-1)

//2147483647
#define RNS_P2 (64*(1<<25)-1)

//2046820351
#define RNS_P3 (61*(1<<25)-1)

// bezout coefficients for P1 w.r.t P2
#define RNS_M1 (306783369L)
// bezout coefficients for P2 w.r.t P1
// THIS IS ACTUALLY NEGATIVE
#define RNS_M2 (273228938L)


// bezout coefficients for P1 * P2 w.r.t P3
// THIS IS ACTUALLY NEGATIVE
#define RNS_M1M2 (852842123L)

// bezout coefficients for P3 w.r.t P1 * P2
#define RNS_M3   (1711368478942472564L)

// small and large factors of RNS_M3
#define RNS_M3_SMALL (670343332L)
#define RNS_M3_LARGE (2552973077L)

__device__ uint96_t
rns_to_int(rns a)
{
    uint96_t ret;
    
    unsigned long long int low;
    unsigned long long int high;
    
    unsigned long long int t1;
    unsigned long long int t2;
    
    
    // (RNS_P2-1) * RNS_P1 * RNS_M1 < 2^91
    low = rns.w2 * RNS_P1 * RNS_M1;
    high = _mul64hi(rns.w2, RNS_P1 * RNS_M1);
    
    // (RNS_P1-1) * RNS_P2 * RNS_M2 < 2^90 (actually negative)
    t1 = rns.w1 * RNS_P2 * RNS_M2;
    t2 = _mul64hi(rns.w1, RNS_P2 * RNS_M2);
    
    t1 = low - t1;
    high = high - t2 - (t1 > low);
    low = t1;
    
    // First pairwise complete for CRT
    
    t1 = 
    
}


typedef struct int128_t
{
    unsigned long long low64;
    unsigned long long high64;
    int sign;
} int128_t;

__device__ int128_t
mul_int128_t(long long a, long long b)
{
    int128_t ret;
    
    sign = 1;
    if (a < 0)
    {
        sign *= -1;
        a *= -1;
    }
    
    if (b < 0)
    {
        sign *= -1;
        b *= -1;
    }
    
    ret.low64 = a * b;
    ret.high64 = _umul64hi(a, b);
    ret.sign = sign;
    
    return ret;
}

__device__ int128_t
_add_int128_t(int128_t a, int128_t b)
{
    int128_t ret;
    ret.low64 = a.low64 + b.low64;
    ret.high64 = a.high64 + b.high64 + (ret.low64 < a.low64);
    //if (ret.high64 < a.high64)
    //{
        // may be due to carry, undefined behavior if overflow
    //}
    
    return ret;
}

__device__ int128_t
_sub_int128_t(int128_t a, int128_t b)
{
    int128_t ret;
    
    ret.low64 = a - b;
    ret.high64 = a - b - (ret.low64 > a.low64);
    //if (ret.high64 > a.high64)
    //{
        // may be due to carry, undefined behavior if underflow
    //}
    
    return ret;
}

// Not a complete implementation, ignores sign.
__device__ bool
_ge_int128_t(int128_t a, int128_t b)
{
    if (a.high64 >= b.high64)
    {
        return true;
    }
    if (a.high64 < b.high64)
    {
        return false;
    }
    return a.low64 >= b.low64;
}

__device__ int128_t
add_int128_t(int128_t a, int128_t b)
{
    int128_t ret;
    
    if (a.sign == -1)
    {
        if (b.sign == -1)
        {
            ret = _add_int128_t(a, b);
            ret.sign = -1;
        } else {
            if (_ge_int128_t(a, b))
            {
                ret = _sub_int128_t(a, b);
                ret.sign = -1;
            } else {
                ret = _sub_int128_t(b, a);
                ret.sign = 1;
            }
        }
    } else {
        if (b.sign == -1)
        {
            if (_ge_int128_t(a, b))
            {
                ret = _sub_int128_t(a, b);
                ret.sign = 1;
            } else {
                ret = _sub_int128_t(b, a);
                ret.sign = -1;
            }
        } else {
            ret = _add_int128_t(a, b);
        }
    }
    
    if (ret.high64 == 0 && ret.low64 == 0)
    {
        ret.sign = 1;
    }
    
    return ret;
}

__global__ void
int_to_rns(const unsigned int* __restrict__ in,
           rns* __restrict__ out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    out[idx].w1 = in[idx] % RNS_P1;
    out[idx].w2 = in[idx] % RNS_P2;
    out[idx].w3 = in[idx] % RNS_P3;
}

__global__ void
rns_to_uint96_t(const rns* __restrict__ in,
                uint96_t* __restrict__ out)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    out[idx].w1 = in[idx] % RNS_P1;
    out[idx].w2 = in[idx] % RNS_P2;
    out[idx].w3 = in[idx] % RNS_P3;
}

__global__ void
rns_in_place_bitreverse(rns* __restrict__ out,
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
