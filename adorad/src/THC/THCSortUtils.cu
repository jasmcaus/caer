#include <THC/THCSortUtils.cuh>

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
uint64_t nextHighestPowerOf2(uint64_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
#ifndef _MSC_VER
  n |= n >> 32;
#endif
  n++;

  return n;
}
