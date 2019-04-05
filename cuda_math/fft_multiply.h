#ifndef FFT_MULTIPLY_H
#define FFT_MULTIPLY_H

#include <cuComplex.h>
#include "bigint.h"

void
fft_square(CudaBigInt& a, CudaBigInt& c);

#endif // FFT_MULTIPLY_H
