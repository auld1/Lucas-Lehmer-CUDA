#ifndef COMPARE_H
#define COMPARE_H

#include "bigint.h"

bool
greater_or_equal(CudaBigInt& a, unsigned int b);

bool
equalu(CudaBigInt& a, unsigned int b);

bool
equal(CudaBigInt& a, CudaBigInt& b);

#endif // COMPARE_H
