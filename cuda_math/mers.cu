#include "mers.h"
#include "bigint.h"

#define MERS_BLOCK_SIZE (128)

__global__
void cuda_mers(unsigned int m,
               unsigned int* __restrict__ c)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int mers_big = m / 32;
    int mers_small = m % 32;
    
    if (i < mers_big)
    {
        c[i] = 0xffffffff;
    }
    else if (i == mers_big)
    {
        c[i] = (1 << mers_small) - 1;
    } else {
        c[i] = 0;
    }
}

void mers(unsigned int m, CudaBigInt& c)
{
    cuda_mers<<<(c.word_len/MERS_BLOCK_SIZE), MERS_BLOCK_SIZE>>>(m, c.val);
}
