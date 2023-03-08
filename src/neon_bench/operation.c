#include "operation.h"

void mm_sisd(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t p) {
  for (int i=0; i<n; i++) {       // A row selection
    for (int j=0; j<m; j++) {     // B column selection
      C[n*j + i] = 0;             // C element init
      for (int k=0; k<p; k++) {   // iterate elements in vectors
        C[n*j + i] += A[n*k + i] * B[p*j + k];
      }
    }
  }
}

void mm_4x4_simd(float32_t *A, float32_t *B, float32_t *C) {
  // columns from A
  float32x4_t A0 = vld1q_f32(A);
  float32x4_t A1 = vld1q_f32(A+4);
  float32x4_t A2 = vld1q_f32(A+8);
  float32x4_t A3 = vld1q_f32(A+12);

  // columns from B
  float32x4_t B0 = vld1q_f32(B);
  float32x4_t B1 = vld1q_f32(B+4);
  float32x4_t B2 = vld1q_f32(B+8);
  float32x4_t B3 = vld1q_f32(B+12);

  // initializing C
  float32x4_t C0 = vmovq_n_f32(0);
  float32x4_t C1 = vmovq_n_f32(0);
  float32x4_t C2 = vmovq_n_f32(0);
  float32x4_t C3 = vmovq_n_f32(0);

  // MAC
  C0 = vfmaq_laneq_f32(C0, A0, B0, 0);  // mult A0[3:0] & B0[0], then add to C0
  C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
  C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
  C0 = vfmaq_laneq_f32(C0, A3, B0, 3);
  vst1q_f32(C, C0);                     // move C0 to C[3:0]

  C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
  C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
  C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
  C1 = vfmaq_laneq_f32(C1, A3, B1, 3);
  vst1q_f32(C+4, C0);

  C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
  C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
  C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
  C2 = vfmaq_laneq_f32(C2, A3, B2, 3);
  vst1q_f32(C+8, C0);

  C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
  C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
  C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
  C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
  vst1q_f32(C+12, C0);
}

void mm_simd(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
  
}