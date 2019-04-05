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

void
set_mpz_uint(mpz_t t, unsigned int* val, int len)
{
    mpz_import(t, len, -1, sizeof(unsigned int), -1, 0, val);
}

void
test(void)
{
    CudaBigInt a(1024*64);
    CudaBigInt c(1024*64*2);
    
    mpz_t a_gmp;
    mpz_t c_gmp;
    
    mpz_t ct_gmp;
    
    mpz_init2(a_gmp, a.word_len*32);
    mpz_init2(c_gmp, c.word_len*32);
    
    mpz_init2(ct_gmp, c.word_len*32);
    
    unsigned int a_host[a.word_len];
    unsigned int c_host[c.word_len];
    
    get_random_array(a.val, a.word_len);
    
    cudaMemcpy(a_host, a.val, a.word_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    set_mpz_uint(a_gmp, a_host, a.word_len);
    
    fft_square(a, c);
    mpz_mul(c_gmp, a_gmp, a_gmp);
    
    cudaMemcpy(c_host, c.val, c.word_len * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    set_mpz_uint(ct_gmp, c_host, c.word_len);
    
    
    gmp_printf("%Zx\n\n\n\n", c_gmp);
    gmp_printf("%Zx\n", ct_gmp);
    
    
    assert(0 == mpz_cmp(c_gmp, ct_gmp));

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
        fft_square(a, c);
        
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
    if (isMersPrime(3021377))
    {
        printf("Worked!\n");
    } else {
        printf("Problem!\n");
    }

    //test();
}
