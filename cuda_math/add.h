#ifndef ADD_H
#define ADD_H

#include "bigint.h"

void
add(CudaBigInt& a, CudaBigInt& b, CudaBigInt& c);

void
addu(CudaBigInt& a, unsigned int b, CudaBigInt& c);

#endif // ADD_H
