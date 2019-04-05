#include "memory.h"

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#define CACHED


#ifdef CACHED
#define CACHELEN 128

typedef struct _tup
{
    size_t bytes;
    void* ptr;
    bool free;
} _tup;

_tup cache[CACHELEN];
bool initialized = false;
#endif // CACHED


void
cuda_malloc_clear(void** ptr, size_t bytes)
{
    cudaError_t err;
    
#ifdef CACHED
    if (!initialized)
    {
        for (int i = 0; i < CACHELEN; i++)
        {
            cache[i].bytes = 0;
            cache[i].ptr = 0;
            cache[i].free = true;
        }
        initialized = true;
    }
    
    for (int i = 0; i < CACHELEN; i++)
    {
        if (cache[i].free && cache[i].bytes == bytes)
        {
            // We don't have to remalloc, we already have a valid free ptr
            cache[i].free = false;
            *ptr = cache[i].ptr;
            err = cudaMemset(*ptr, 0, bytes);
            assert(err == cudaSuccess);
            return;
        }
    }
#endif // CACHED

    // Malloc to device, check for errors
    err = cudaMalloc(ptr, bytes);
    assert(err == cudaSuccess);

    // Set val to 0, check for errors
    err = cudaMemset(*ptr, 0, bytes);
    assert(err == cudaSuccess);

#ifdef CACHED
    for (int i = 0; i < CACHELEN; i++)
    {
        if (cache[i].free && cache[i].ptr == 0)
        {
            cache[i].free = false;
            cache[i].ptr = *ptr;
            cache[i].bytes = bytes;
            return;
        }
    }
#endif // CACHED
}

void
cuda_malloc_free(void* ptr)
{
#ifdef CACHED
    for (int i = 0; i < CACHELEN; i++)
    {
        if (cache[i].ptr == ptr)
        {
            cache[i].free = true;
            return;
        }
    }
#endif // CACHED

    cudaFree(ptr);
}
