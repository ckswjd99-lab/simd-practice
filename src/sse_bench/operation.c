#include "operation.h"

void memcpy_sisd(char* dest, char* source, size_t count) {
  count /= 8;
  for (size_t i = 0; i < count; i++)
    *((long long int*)dest + i) = *((long long int*)source + i);
}

void memcpy_simd(char* dest, char* source, size_t count) {
  count /= 0x10;
  for (size_t i = 0; i< count; i++) {
    __m128i temp = _mm_loadu_si128((__m128i*)source+i);
    _mm_storeu_si128((__m128i*)dest + i, temp);
  }
}

void add_sisd(char* dest, char* src1, char* src2, size_t count) {
  count /= 2;
  for (size_t i = 0; i < count; i++) {
    *((short*)dest + i) = *((short*)src1 + i) + *((short*)src2 + i);
  }
}

void add_simd(char* dest, char* src1, char* src2, size_t count) {
  count /= 0x10;
  for (size_t i = 0; i< count; i++) {
    __m128i temp1 = _mm_loadu_si128((__m128i*)src1+i);
    __m128i temp2 = _mm_loadu_si128((__m128i*)src2+i);
    *((__m128i*)dest+i) = _mm_add_epi16(temp1, temp2);
  }
}

void sub_sisd(char* dest, char* src1, char* src2, size_t count) {
  count /= 2;
  for (size_t i = 0; i < count; i++) {
    *((short*)dest + i) = *((short*)src1 + i) - *((short*)src2 + i);
  }
}

void sub_simd(char* dest, char* src1, char* src2, size_t count) {
  count /= 0x10;
  for (size_t i = 0; i< count; i++) {
    __m128i temp1 = _mm_loadu_si128((__m128i*)src1+i);
    __m128i temp2 = _mm_loadu_si128((__m128i*)src2+i);
    *((__m128i*)dest+i) = _mm_sub_epi16(temp1, temp2);
  }
}

void mul_sisd(char* dest, char* src1, char* src2, size_t count) {
  count /= 4;
  for (size_t i = 0; i < count; i++) {
    *((float*)dest + i) = *((float*)src1 + i) * *((float*)src2 + i);
  }
}

void mul_simd(char* dest, char* src1, char* src2, size_t count) {
  count /= 0x10;
  for (size_t i = 0; i< count; i++) {
    __m128 temp1 = _mm_load_ps((__m128*)src1 + i);
    __m128 temp2 = _mm_load_ps((__m128*)src2 + i);
    *((__m128*)dest+i) = _mm_mul_ps(temp1, temp2);
  }
}

void div_sisd(char* dest, char* src1, char* src2, size_t count) {
  count /= 4;
  for (size_t i = 0; i < count; i++) {
    *((float*)dest + i) = *((float*)src1 + i) / *((float*)src2 + i);
  }
}

void div_simd(char* dest, char* src1, char* src2, size_t count) {
  count /= 0x10;
  for (size_t i = 0; i< count; i++) {
    __m128 temp1 = _mm_load_ps((__m128*)src1 + i);
    __m128 temp2 = _mm_load_ps((__m128*)src2 + i);
    *((__m128*)dest+i) = _mm_div_ps(temp1, temp2);
  }
}

void max_sisd(char* dest, char* src1, char* src2, size_t count) {
  count /= 2;
  for (size_t i = 0; i < count; i++) {
    short temp1 = *((short*)src1 + i);
    short temp2 = *((short*)src2 + i);
    *((short*)dest + i) = temp1 > temp2 ? temp1 : temp2;
  }
}

void max_simd(char* dest, char* src1, char* src2, size_t count) {
  count /= 0x10;
  for (size_t i = 0; i< count; i++) {
    __m128i temp1 = _mm_loadu_si128((__m128i*)src1+i);
    __m128i temp2 = _mm_loadu_si128((__m128i*)src2+i);
    *((__m128i*)dest+i) = _mm_max_epi16(temp1, temp2);
  }
}

void min_sisd(char* dest, char* src1, char* src2, size_t count) {
  count /= 2;
  for (size_t i = 0; i < count; i++) {
    short temp1 = *((short*)src1 + i);
    short temp2 = *((short*)src2 + i);
    *((short*)dest + i) = temp1 > temp2 ? temp2 : temp1;
  }
}

void min_simd(char* dest, char* src1, char* src2, size_t count) {
  count /= 0x10;
  for (size_t i = 0; i < count; i++) {
    __m128i temp1 = _mm_loadu_si128((__m128i*)src1+i);
    __m128i temp2 = _mm_loadu_si128((__m128i*)src2+i);
    *((__m128i*)dest+i) = _mm_min_epi16(temp1, temp2);
  }
}

void andnot_sisd(char* dest, char* src1, char* src2, size_t count) {
  count /= 8;
  for (size_t i = 0; i < count; i++) {
    *((long long int *)dest + i) = ~*((long long int *)src1 + i) & *((long long int *)src2 + i);
  }
}

void andnot_simd(char* dest, char* src1, char* src2, size_t count) {
  count /= 0x10;
  for (size_t i = 0; i< count; i++) {
    __m128i temp1 = _mm_loadu_si128((__m128i*)src1+i);
    __m128i temp2 = _mm_loadu_si128((__m128i*)src2+i);
    *((__m128i*)dest+i) = _mm_andnot_si128(temp1, temp2);
  }
}