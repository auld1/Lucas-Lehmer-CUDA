/* int128.c - 128-bit integer arithmetic for C++, by Robert Munafo
   
   modified by Zachary Auld for CUDA support

   (and copied to rhtf/.../int128.c.txt by proj/.../MCS.per)

LICENSE

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


The license (GPL version 3) for this source code is at:
mrob.com/pub/ries/GPL-3.txt

INSTRUCTIONS

  To add 128-bit integers to an existing program:

  Pass -DI128_COMPAT_PLUS_EQUAL to the compiler (unless on a really
old compiler)
  Rename the program to "foo.cxx", recompile, deal with error messages
  Include .../int128.c
  Declare variables as s128

REVISION HISTORY

 20040416 Fix bug in div1: Its answer was one too low when remainder is 0

 20070812 Add +=, etc. operators
 20070818 Add demotion to long long, and lots of operators that take s64, u64 or s32 on the RHS
 20070820 Massively faster multiply algorithm
 20070822 Add conversion from double
 20070825 Fix bug in operator - (const int128_t &, s32)

 20090619 Fix bug in div1: Its answer was one too low when quotient is
a power of 2 and remainder is 0 (apparently a different case of the
bug I fixed on 20040416)
 20091122 Fix bug in mult1: Sign handling trickery failed on multiplying
-1 * 0. For now it's back to the simple "treat it as unsigned" method.

 20130410 Add GPL license header

 20170215 Add #define switch I128_COMPAT_PLUS_EQUAL to avoid errors
"'int128_t::int128_t' names the constructor, not the type"

 */

#include "int128.h"

__device__ int128_t
add128(int128_t lhs, int128_t rhs)
{
  int128_t res;

  res.high = lhs.high + rhs.high;
  res.low = lhs.low + rhs.low;
  if (res.low < rhs.low) {
    // carry
    (res.high)++;
  }

  return res;
}

__device__ int128_t
sub128(int128_t lhs, int128_t rhs)
{
  int128_t res;

  res.high = lhs.high - rhs.high;
  res.low = lhs.low - rhs.low;
  if (res.low > lhs.low) {
    // borrow
    (res.high)--;
  }
  
  return res;
}

__device__ int128_t
negate128(int128_t x )
{
  int128_t res;

  res.high = ~(x.high);
  res.low = ~(x.low);
  res.low += 1;
  if (res.low == 0) {
    res.high += 1;
  }
  
  return res;
}

__device__ int
lt128(int128_t lhs, int128_t rhs )
{
  if (lhs.high < rhs.high)
    return 1;
  if (rhs.high < lhs.high)
    return 0;
  // high components are equal
  if (lhs.low < rhs.low)
    return 1;
  return 0;
}

__device__ int
le128(int128_t lhs, int128_t rhs )
{
  if ((lhs.high == rhs.high) && (lhs.low == rhs.low))
    return 1;

  if (lhs.high < rhs.high)
    return 1;
  if (rhs.high < lhs.high)
    return 0;
  // high components are equal
  if (lhs.low < rhs.low)
    return 1;
  return 0;
}

__device__ int
eq128(int128_t lhs, int128_t rhs )
{
  if (lhs.high != rhs.high)
    return 0;
  if (lhs.low != rhs.low)
    return 0;
  return 1;
}

__device__ int
ne128(int128_t lhs, int128_t rhs )
{
  if (lhs.high != rhs.high)
    return 1;
  if (lhs.low != rhs.low)
    return 1;
  return 0;
}

__device__ int
gt128(int128_t lhs, int128_t rhs )
{
  if (lhs.high > rhs.high)
    return 1;
  if (rhs.high > lhs.high)
    return 0;
  // high components are equal
  if (lhs.low > rhs.low)
    return 1;
  return 0;
}

__device__ int
ge128(int128_t lhs, int128_t rhs )
{
  if ((lhs.high == rhs.high) && (lhs.low == rhs.low))
    return 1;

  if (lhs.high > rhs.high)
    return 1;
  if (rhs.high > lhs.high)
    return 0;
  // high components are equal
  if (lhs.low > rhs.low)
    return 1;
  return 0;
}


// Support routines for multiply, divide and modulo

__device__ int128_t
s128_shr(int128_t x)
{
  int128_t rv;

//  printf("%.16llX %016llX  >> ", x.high, x.low);
  rv.low = (x.low >> 1) | (x.high << 63);
  rv.high = x.high >> 1;

//  printf("%.16llX %016llX\n", rv.high, rv.low);

  return rv;
}

__device__ int128_t
s128_shl(int128_t x)
{
  int128_t rv;

  rv.high = (x.high << 1) | (x.low >> 63);
  rv.low = x.low << 1;

  return rv;
}

__device__ int128_t
mul128(int128_t xi, int128_t yi)
{
  int128_t rv2;
  unsigned long long int acc, ac2, carry, o1, o2;
  unsigned long long int a, b, c, d, e, f, g, h;


  /*            x      a  b  c  d
                y      e  f  g  h
                   ---------------
                      ah bh ch dh
                      bg cg dg
                      cf df
                      de
        --------------------------
                      -o2-- -o1--
                                  */

  d = xi.low & LO_WORD;
  c = (xi.low & HI_WORD) >> 32LL;
  b = xi.high & LO_WORD;
  a = (xi.high & HI_WORD) >> 32LL;

  h = yi.low & LO_WORD;
  g = (yi.low & HI_WORD) >> 32LL;
  f = yi.high & LO_WORD;
  e = (yi.high & HI_WORD) >> 32LL;
  
  acc = d * h;
  o1 = acc & LO_WORD;
  acc >>= 32LL;
  carry = 0;
  ac2 = acc + c * h; if (ac2 < acc) { carry++; }
  acc = ac2 + d * g; if (acc < ac2) { carry++; }
  rv2.low = o1 | (acc << 32LL);
  ac2 = (acc >> 32LL) | (carry << 32LL); carry = 0;

  acc = ac2 + b * h; if (acc < ac2) { carry++; }
  ac2 = acc + c * g; if (ac2 < acc) { carry++; }
  acc = ac2 + d * f; if (acc < ac2) { carry++; }
  o2 = acc & LO_WORD;
  ac2 = (acc >> 32LL) | (carry << 32LL);

  acc = ac2 + a * h;
  ac2 = acc + b * g;
  acc = ac2 + c * f;
  ac2 = acc + d * e;
  rv2.high = (ac2 << 32LL) | o2;
  
  return rv2;
}

__device__ int128_t
mul128(int128_t xi, unsigned long long int yi)
{
  int128_t rv2;
  unsigned long long int acc, ac2, carry, o1, o2;
  unsigned long long int a, b, c, d, g, h;


  /*            x      a  b  c  d
                y      0  0  g  h
                   ---------------
                      ah bh ch dh
                      bg cg dg
                      c0 d0
                      d0
        --------------------------
                      -o2-- -o1--
                                  */

  d = xi.low & LO_WORD;
  c = (xi.low & HI_WORD) >> 32LL;
  b = xi.high & LO_WORD;
  a = (xi.high & HI_WORD) >> 32LL;

  h = yi & LO_WORD;
  g = (yi & HI_WORD) >> 32LL;
  
  acc = d * h;
  o1 = acc & LO_WORD;
  acc >>= 32LL;
  carry = 0;
  ac2 = acc + c * h; if (ac2 < acc) { carry++; }
  acc = ac2 + d * g; if (acc < ac2) { carry++; }
  rv2.low = o1 | (acc << 32LL);
  ac2 = (acc >> 32LL) | (carry << 32LL); carry = 0;

  acc = ac2 + b * h; if (acc < ac2) { carry++; }
  ac2 = acc + c * g; if (ac2 < acc) { carry++; }
  acc = ac2;
  o2 = acc & LO_WORD;
  ac2 = (acc >> 32LL) | (carry << 32LL);

  acc = ac2 + a * h;
  ac2 = acc + b * g;
  rv2.high = (ac2 << 32LL) | o2;
  
  return rv2;
}

__device__ int128_t
mul128(int128_t xi, unsigned int yi)
{
  int128_t rv2;
  unsigned long long int acc, ac2, carry, o1, o2;
  unsigned long long int a, b, c, d, h;


  /*            x      a  b  c  d
                y      0  0  0  h
                   ---------------
                      ah bh ch dh
                      b0 c0 d0
                      c0 d0
                      d0
        --------------------------
                      -o2-- -o1--
                                  */

  d = xi.low & LO_WORD;
  c = (xi.low & HI_WORD) >> 32LL;
  b = xi.high & LO_WORD;
  a = (xi.high & HI_WORD) >> 32LL;

  h = yi;
  
  acc = d * h;
  o1 = acc & LO_WORD;
  acc >>= 32LL;
  carry = 0;
  ac2 = acc + c * h; if (ac2 < acc) { carry++; }
  acc = ac2;
  rv2.low = o1 | (acc << 32LL);
  ac2 = (acc >> 32LL) | (carry << 32LL); carry = 0;

  acc = ac2 + b * h; if (acc < ac2) { carry++; }
  ac2 = acc;
  acc = ac2;
  o2 = acc & LO_WORD;
  ac2 = (acc >> 32LL) | (carry << 32LL);

  acc = ac2 + a * h;
  ac2 = acc;
  rv2.high = (ac2 << 32LL) | o2;
  
  return rv2;
}

__device__ int128_t
div128(int128_t x, int128_t d)
{
  int s;
  int128_t d1, p2, rv;

//printf("divide %.16llX %016llX / %.16llX %016llX\n", x.high, x.low, d.high, d.low);

  /* check for divide by zero */
  if ((d.low == 0) && (d.high == 0)) {
    rv.low = x.low / d.low; /* This will cause runtime error */
  }

  s = 1;
  if (x.high < 0) {
    // notice that MININT will be unchanged, this is used below.
    s = - s;
    x = negate128(x);
  }
  if (d.high < 0) {
    s = - s;
    d = negate128(d);
  }

  if (d.low == 1 && d.high == 0) {
    /* This includes the overflow case MININT/-1 */
    rv = x;
    x.high = 0;
    x.low = 0;
  } else if (lt128(x, d)) {
    /* x < d, so quotient is 0 and x is remainder */
    rv.high = 0;
    rv.low = 0;
  } else {
    rv.high = 0;
    rv.low = 0;

    /* calculate biggest power of 2 times d that's <= x */
    p2.low = 1;
    p2.high = 0;
    d1 = d;
    x = sub128(x, d1);
    while(ge128(x, d1)) {
      x = sub128(x, d1);
      d1 = add128(d1, d1);
      p2 = add128(p2, p2);
    }
    x = add128(x, d1);

    while(p2.high != 0 && p2.low != 0) {
//printf("x %.16llX %016llX d1 %.16llX %016llX\n", x.high, x.low, d1.high, d1.low);
      if (ge128(x, d1)) {
        x = sub128(x, d1);
        rv = add128(rv, p2);
//printf("`.. %.16llX %016llX\n", rv.high, rv.low);
      }
      p2 = s128_shr(p2);
      d1 = s128_shr(d1);
    }

    /* whatever is left in x is the remainder */
  }

  /* Put sign in result */
  if (s < 0) {
    rv = negate128(rv);
  }

  return rv;
}

__device__ int128_t
mod128(int128_t x, int128_t d)
{
  int s;
  int128_t d1, p2, rv;

  /* check for divide by zero */
  if ((d.low == 0) && (d.high == 0)) {
    rv.low = x.low / d.low; /* This will cause runtime error */
  }

  rv = x;
  s = 1;
  if (rv.high < 0) {
    // notice that MININT will be unchanged, this is used below.
    s = - s;
    rv = negate128(rv);
  }
  if (d.high < 0) {
    s = - s;
    d = negate128(d);
  }

  if (d.low == 1 && d.high == 0) {
    /* This includes the overflow case MININT/-1 */
    rv.high = 0;
    rv.low = 0;
  } else if (lt128(x, d)) {
    /* x < d, so quotient is 0 and x is remainder */
  } else {
    /* calculate biggest power of 2 times d that's <= x */
    p2.low = 1;
    p2.high = 0;
    d1 = d;
    rv = sub128(rv, d1);
    while(ge128(rv, d1)) {
      rv = sub128(rv, d1);
      d1 = add128(d1, d1);
      p2 = add128(p2, p2);
    }
    rv = add128(rv, d1);

    while(p2.high != 0 && p2.low != 0) {
      if (ge128(x, d1)) {
        rv = sub128(rv, d1);
      }
      p2 = s128_shr(p2);
      d1 = s128_shr(d1);
    }

    /* whatever is left in x is the remainder */
  }
  
  return rv;
}


int
main(void)
{
    return 1;
}

/* end of int128.cu */

