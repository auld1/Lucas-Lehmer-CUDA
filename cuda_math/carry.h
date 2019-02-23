#ifndef CARRY_H
#define CARRY_H

#include <stdbool.h>

__global__
void cuda_long_carry(unsigned int* __restrict__ c,
                     const unsigned long long* __restrict__ carry_in,
                     unsigned char* __restrict__ carry_out,
                     bool* __restrict__ needs_carry);

__global__
void cuda_int_carry(unsigned int* __restrict__ c,
                    const unsigned int* __restrict__ carry_in,
                    unsigned char* __restrict__ carry_out,
                    bool* __restrict__ needs_carry);

__global__
void cuda_byte_carry(unsigned int* __restrict__ c,
                     const unsigned char* __restrict__ carry_in,
                     unsigned char* __restrict__ carry_out,
                     bool* __restrict__ needs_carry);

#endif // CARRY_H
