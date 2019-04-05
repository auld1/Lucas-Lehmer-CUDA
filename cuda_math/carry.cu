#include "carry.h"

#include <stdbool.h>

__global__
void cuda_long_carry(unsigned int*   c,
                     const unsigned long long*   carry_in,
                     unsigned char*   carry_out,
                     bool*   needs_carry)
{
    int c_i = blockIdx.x*blockDim.x + threadIdx.x;
    
    carry_out[c_i] = 0;
    if (c_i - 1 >= 0)
    {
        unsigned int temp = carry_in[c_i - 1] & 0xffffffff;
        c[c_i] += temp;
        if (c[c_i] < temp)
        {
            carry_out[c_i]++;
            *needs_carry = true;
        }
    }
    
    if (c_i - 2 >= 0)
    {
        unsigned int temp = (carry_in[c_i - 2] >> 32) & 0xffffffff;
        c[c_i] += temp;
        if (c[c_i] < temp)
        {
            carry_out[c_i]++;
            *needs_carry = true;
        }
    }
}

__global__
void cuda_int_carry(unsigned int*   c,
                    const unsigned int*   carry_in,
                    unsigned char*   carry_out,
                    bool*   needs_carry)
{
    int c_i = blockIdx.x*blockDim.x + threadIdx.x;
    
    carry_out[c_i] = 0;
    if (c_i - 1 >= 0)
    {
        c[c_i] += carry_in[c_i - 1];
        if (c[c_i] < carry_in[c_i - 1])
        {
            carry_out[c_i]++;
            *needs_carry = true;
        }
    }
}

__global__
void cuda_byte_carry(unsigned int*   c,
                     const unsigned char*   carry_in,
                     unsigned char*   carry_out,
                     bool*   needs_carry)
{
    int c_i = blockIdx.x*blockDim.x + threadIdx.x;
    
    carry_out[c_i] = 0;
    if (c_i - 1 >= 0)
    {
        unsigned int temp = c[c_i];
        c[c_i] += carry_in[c_i - 1];
        if (c[c_i] < temp)
        {
            carry_out[c_i] = 1;
            *needs_carry = true;
        }
    }
}

__global__
void cuda_negative_byte_carry(unsigned int*   c,
                              const unsigned char*   carry_in,
                              unsigned char*   carry_out,
                              bool*   needs_carry)
{
    int c_i = blockIdx.x*blockDim.x + threadIdx.x;
    
    carry_out[c_i] = 0;
    if (c_i - 1 >= 0)
    {
        unsigned int temp = c[c_i];
        c[c_i] -= carry_in[c_i - 1];
        if (c[c_i] > temp)
        {
            carry_out[c_i] = 1;
            *needs_carry = true;
        }
    }
}
