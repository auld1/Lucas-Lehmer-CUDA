#ifndef FFT_MULTIPLY_H
#define FFT_MULTIPLY_H

#include <cuComplex.h>

void
cooley_tukey_fft(unsigned int* in, cuDoubleComplex* out, int len);

#endif // FFT_MULTIPLY_H
