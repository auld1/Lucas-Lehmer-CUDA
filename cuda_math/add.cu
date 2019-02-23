#include "add.h"

#include "bigint.h"
#include "carry.h"
#include "memory.h"

#include <assert.h>

__device__
unsigned int cuda_add_digit(const unsigned int* __restrict__ a,
                            const unsigned int* __restrict__ b,
                            int digit,
                            unsigned char* __restrict__ carry_out,
                            unsigned int carry_in,
                            const int N)
{
    unsigned char carry = 0;
    unsigned int result = 0;
    unsigned int temp = 0;
    
    result = carry_in;
    temp = a[digit] + b[digit];
    if (temp < a[digit])
    {
        carry++;
    }
    
    result += temp;
    if (result < temp)
    {
        carry++;
    }
    
    *carry_out = carry;
    return result;
}

__global__
void cuda_add(const unsigned int* __restrict__ a,
              const unsigned int* __restrict__ b,
              unsigned int* __restrict__ c,
              unsigned char* __restrict__ carry_out,
              const int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    c[i] = cuda_add_digit(a, b, i, &carry_out[i], 0, N);
}

void add(CudaBigInt a, CudaBigInt b, CudaBigInt c)
{
    unsigned char* byte_carry1;
    unsigned char* byte_carry2;
    bool* should_carry_cuda;
    bool should_carry_host;
    cudaError_t err;
    
    
    if (a.sign == -1)
    {
        if (b.sign == 1)
        {
            //subtract(b, a, c);
            return;
        }
        // Both signs are -1, we will add them together as positives
        // but change the sign of c
        c.sign = -1;
    } else if (b.sign == -1)
    {
        //subtract(a, b, c);
        return;
    }
    
    cuda_malloc_clear((void**) &byte_carry1, c.word_len * sizeof(*byte_carry1));
    cuda_malloc_clear((void**) &byte_carry2, c.word_len * sizeof(*byte_carry2));
    cuda_malloc_clear((void**) &should_carry_cuda, sizeof(bool));
    
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
     
    cuda_add<<<64, c.word_len/64>>>(a.val, b.val, c.val, byte_carry1, c.word_len);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
    
    do
    {
        cuda_byte_carry<<<64, c.word_len/64>>>(c.val, byte_carry1, byte_carry2, should_carry_cuda);
    
        err = cudaMemcpy(&should_carry_host, should_carry_cuda, sizeof(bool), cudaMemcpyDeviceToHost);
        assert(err == cudaSuccess);
        
        err = cudaMemset(should_carry_cuda, 0, sizeof(bool));
        assert(err == cudaSuccess);
        
        unsigned char* temp = byte_carry1;
        byte_carry1 = byte_carry2;
        byte_carry2 = temp;
    } while (should_carry_host);
    
}
