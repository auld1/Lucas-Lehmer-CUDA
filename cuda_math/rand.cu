#include "rand.h"

#include "memory.h"

#include <assert.h>
#include <sys/time.h>

__global__
void random_uint(unsigned int* __restrict__ a,
                 unsigned int seed)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int state = (48271 * seed) % 0x7fffffff;
    state += idx;
    int i = 0;
    
    for(i = 0; i < 5; i++)
    {
        state = (48271 * state) % 0x7fffffff;
    }
    
    a[idx] = state & 0xffff;
    
    for(i = 0; i < 7; i++)
    {
        state = (48271 * state) % 0x7fffffff;
    }
    
    a[idx] += (state << 16);
}


void
get_random_array(unsigned int* cuda_arr, unsigned int word_len)
{
    struct timeval tv;
    unsigned long long usec;
    
    gettimeofday(&tv,NULL);
    usec = tv.tv_sec;
    usec *= 1000000;
    usec += tv.tv_usec;
    
    random_uint<<<(word_len/128), 128>>>(cuda_arr, usec);
}
