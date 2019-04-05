#ifndef MEMORY_H
#define MEMORY_H

void
cuda_malloc_clear(void** ptr, size_t bytes);

void
cuda_malloc_free(void* ptr);

#endif // MEMORY_H
