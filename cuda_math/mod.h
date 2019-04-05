#ifndef MOD_H
#define MOD_H

#include "bigint.h"

void
mod(CudaBigInt& a, unsigned int m, CudaBigInt& p, CudaBigInt& c);

#endif // MOD_H
