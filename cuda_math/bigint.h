#ifndef BIGINT_H
#define BIGINT_H

#include <assert.h>

#define WORD_BIT_LEN (sizeof(unsigned int) * 8)

class CudaBigInt
{
    public:
    CudaBigInt()
    {
        cudaError_t err;

        sign     = 1;                   // Default positive
        word_len = 2048 / WORD_BIT_LEN; // Default to 1024 bits

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

        sign     = 1;                      // Default positive
        word_len = bit_len / WORD_BIT_LEN; // Set word_len based on bit_len

        // Malloc to device, check for errors
        err = cudaMalloc(&val, word_len * sizeof(*val));
        assert(err == cudaSuccess);

        // Set val to 0, check for errors
        err = cudaMemset(val, 0, word_len * sizeof(*val));
        assert(err == cudaSuccess);
    }
    
    int           sign;
    unsigned int  word_len;
    unsigned int * val;


    private:
};

#endif // BIGINT_H
