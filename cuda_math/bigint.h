#ifndef BIGINT_H
#define BIGINT_H

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#define WORD_BIT_LEN (sizeof(unsigned int) * 8)


static bool
isPow2(unsigned int val)
{
    while (val != 1)
    {
        if (val % 2 != 0)
        {
            return false;
        }
        val /= 2;
    }
    return true;
}


static unsigned int
nextPow2(unsigned int val)
{
    unsigned int ret = 1;
    
    while (ret < val)
    {
        ret *= 2;
    }
    return ret;
}


class CudaBigInt
{
    public:
    CudaBigInt()
    {
        cudaError_t err;

        sign     = 1;                    // Default positive
        word_len = 65536 / WORD_BIT_LEN; // Default to 65536 bits

        // Malloc to device, check for errors
        err = cudaMalloc(&val, word_len * sizeof(*val));
        assert(err == cudaSuccess);

        // Set val to 0, check for errors
        err = cudaMemset(val, 0, word_len * sizeof(*val));
        assert(err == cudaSuccess);
    }
    
    CudaBigInt(unsigned int bit_len)
    {
        cudaError_t err;
        
        if (bit_len < 4096)
        {
            bit_len = 4096; // make sure there are at least 2048 words
        }
        
        bit_len = nextPow2(bit_len); // make sure the length is a power of 2
        
        sign     = 1;                      // Default positive
        word_len = bit_len / WORD_BIT_LEN; // Set word_len based on bit_len

        // Malloc to device, check for errors
        err = cudaMalloc(&val, word_len * sizeof(*val));
        assert(err == cudaSuccess);

        // Set val to 0, check for errors
        err = cudaMemset(val, 0, word_len * sizeof(*val));
        assert(err == cudaSuccess);
    }
    
    ~CudaBigInt()
    {
        cudaFree(val);
    }
    
    int           sign;
    unsigned int  word_len;
    unsigned int * val;


    private:
};

#endif // BIGINT_H
