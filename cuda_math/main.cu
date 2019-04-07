#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <gmp.h>


#include "compare.h"
#include "add.h"
#include "subtract.h"
#include "multiply.h"
#include "mers.h"
#include "mod.h"
#include "bigint.h"
#include "fft_multiply.h"
#include "rand.h"
#include "llntt.h"
#include "untt.h"

unsigned int mers_prime_exps[] = {
    2,
    3,
    5,
    7,
    13,
    17,
    19,
    31,
    61,
    89,
    107,
    127,
    521,
    607,
    1279,
    2203,
    2281,
    3217,
    4253,
    4423,
    9689,
    9941,
    11213,
    19937,
    21701,
    23209,
    44497,
    86243,
    110503,
    132049,
    216091,
    756839,
    859433,
    1257787,
    1398269,
    2976221,
    3021377,
    6972593,
    13466917,
    20996011,
    24036583,
    25964951,
    30402457,
    32582657,
    37156667,
    42643801,
    43112609,
    57885161,
    74207281,
    77232917,
    82589933
};

void
set_mpz_uint(mpz_t t, unsigned int* val, int len)
{
    mpz_import(t, len, -1, sizeof(unsigned int), -1, 0, val);
}

void
test(unsigned int size)
{
    CudaBigInt a(size);
    CudaBigInt c(a.word_len*32*2);
    
    mpz_t a_gmp;
    mpz_t c_gmp;
    
    mpz_t ct_gmp;
    
    mpz_init2(a_gmp, a.word_len*32);
    mpz_init2(c_gmp, c.word_len*32);
    
    mpz_init2(ct_gmp, c.word_len*32);
    
    unsigned int *a_host = (unsigned int *) malloc(a.word_len * sizeof(*a.val));
    unsigned int *c_host = (unsigned int *) malloc(c.word_len * sizeof(*c.val));
    
    get_random_array(a.val, a.word_len);
    
    cudaMemcpy(a_host, a.val, a.word_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    set_mpz_uint(a_gmp, a_host, a.word_len);
    
    square(a, c);
    mpz_mul(c_gmp, a_gmp, a_gmp);
    
    cudaMemcpy(c_host, c.val, c.word_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    set_mpz_uint(ct_gmp, c_host, c.word_len);
    
    
    
    if (0 != mpz_cmp(c_gmp, ct_gmp))
    {
        //gmp_printf("%Zd\n\n\n", c_gmp);
        //gmp_printf("%Zd\n", ct_gmp);
        assert(0);
    }

    free(a_host);
    free(c_host);
    
    mpz_clear(a_gmp);
    mpz_clear(c_gmp);
    mpz_clear(ct_gmp);
}

int
isMersPrime(unsigned int p)
{
    CudaBigInt a(p);
    CudaBigInt m(p);
    CudaBigInt c(a.word_len * 32 * 2);
    
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    
    mers(p, m);
    
    addu(a, 4, a);
    
    for (int i = 1; i <= p-2; i++)
    {
        printf("Iteration %d of %d\n", i, p-2);
        
        //multiply(a, a, c);
        //fft_square(a, c);
        //ntt_square(a, c);
        square(a,c);
        
        if (!greater_or_equal(c, 2))
        {
            add(c, m, c);
        }
        
        subu(c, 2, c);
        
        mod(c, p, m, a);
    }
    
    return equalu(a, 0);
}


int
main(void)
{
    for (int i = 1; i < 51; i++)
    {
        assert(isMersPrime(mers_prime_exps[i]));
    }
/*
    if (isMersPrime(859433))
    {
        printf("Worked!\n");
    } else {
        printf("Problem!\n");
    }
/*
    for (int i = 1; i < 28; i++)
    {
        test(1<<i);
        printf("Passed %d\n", i);
    }*/
}

