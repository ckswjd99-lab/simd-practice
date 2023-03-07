#include <math.h>
#include <stdio.h>
#include <arm_neon.h>
#include "utils.h"

void mm_simd(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k);
// Matrices are in size of A[n, k], B[k, m], C[n, m]

void mmT_simd(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k);
// Matrices are in size of A[n, k], B[m, k], C[n, m], transpose of B (B.T) will be used

void mTm_simd(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k);
// Matrices are in size of A[k, n], B[k, m], C[n, m], transpose of A (A.T) will be used

void dotaddq_lane_f32(float32x4_t *src1, float32x4_t *src2, float32x4_t *dst, int lane);
// Dot product sources, then store at a lane of dst

void softmax_inplace_simd(float32_t *src, uint32_t n);
// Vector is in size of src[n]

void scale_inplace_simd(float32_t *src, float32_t scale, uint32_t n);
// Vector(or matrix) is in size of src[n]

void transpose(float32_t *src, float32_t *dst, uint32_t rows, uint32_t cols);
// Transpose data alignment in src