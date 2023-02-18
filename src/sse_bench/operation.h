#include <stdio.h>
#include <emmintrin.h>

void memcpy_sisd(char* dest, char* source, size_t count);
void memcpy_simd(char* dest, char* source, size_t count);

void add_sisd(char* dest, char* src1, char* src2, size_t count);
void add_simd(char* dest, char* src1, char* src2, size_t count);

void sub_sisd(char* dest, char* src1, char* src2, size_t count);
void sub_simd(char* dest, char* src1, char* src2, size_t count);

void mul_sisd(char* dest, char* src1, char* src2, size_t count);
void mul_simd(char* dest, char* src1, char* src2, size_t count);

void div_sisd(char* dest, char* src1, char* src2, size_t count);
void div_simd(char* dest, char* src1, char* src2, size_t count);

void max_sisd(char* dest, char* src1, char* src2, size_t count);
void max_simd(char* dest, char* src1, char* src2, size_t count);

void min_sisd(char* dest, char* src1, char* src2, size_t count);
void min_simd(char* dest, char* src1, char* src2, size_t count);