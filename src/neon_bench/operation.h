#include <arm_neon.h>

void mm_sisd(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k);
void mm_4x4_simd(float32_t *A, float32_t *B, float32_t *C);
void mm_simd(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k);