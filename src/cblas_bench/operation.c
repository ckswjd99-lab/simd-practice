#include "operation.h"

void mm_simd(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
  // Size of matrices are A[n, k], B[k, m], C[n, m]

  int A_idx, B_idx, C_idx;
  float32x4_t A0, A1, A2, A3;
  float32x4_t B0, B1, B2, B3;
  float32x4_t C0, C1, C2, C3;

  for (int i_idx=0; i_idx<n; i_idx+=4) {
    for (int j_idx=0; j_idx<m; j_idx+=4) {
      // gonna calculate C[i_idx ~ i_idx+3, j_idx ~ j_idx+3]
      C0 = vmovq_n_f32(0);  // C[i_idx ~ i_idx+3, j_idx]
      C1 = vmovq_n_f32(0);  // C[i_idx ~ i_idx+3, j_idx+1]
      C2 = vmovq_n_f32(0);  // C[i_idx ~ i_idx+3, j_idx+2]
      C3 = vmovq_n_f32(0);  // C[i_idx ~ i_idx+3, j_idx+3]

      for (int k_idx=0; k_idx<k; k_idx+=4) {
        // lets iterate through k_idx
        A_idx = ROWCOL2IDX(n, i_idx, k_idx);  // index pointing [i_idx, k_idx]
        B_idx = ROWCOL2IDX(k, k_idx, j_idx);  // index pointing [k_idx, j_idx]

        A0 = vld1q_f32(A + A_idx);             // A0: A[i_idx ~ i_idx+3, k_idx]
        A1 = vld1q_f32(A + A_idx + n);         // A1: A[i_idx ~ i_idx+3, k_idx+1]
        A2 = vld1q_f32(A + A_idx + 2*n);       // A2: A[i_idx ~ i_idx+3, k_idx+2]
        A3 = vld1q_f32(A + A_idx + 3*n);       // A3: A[i_idx ~ i_idx+3, k_idx+3]

        B0 = vld1q_f32(B + B_idx);             // B0: B[k_idx ~ k_idx+3, j_idx]
        B1 = vld1q_f32(B + B_idx + k);         // B1: B[k_idx ~ k_idx+3, j_idx+1]
        B2 = vld1q_f32(B + B_idx + 2*k);       // B2: B[k_idx ~ k_idx+3, j_idx+2]
        B3 = vld1q_f32(B + B_idx + 3*k);       // B3: B[k_idx ~ k_idx+3, j_idx+3]

        // use B0 all
        C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
        C0 = vfmaq_laneq_f32(C0, A1, B0, 1);
        C0 = vfmaq_laneq_f32(C0, A2, B0, 2);
        C0 = vfmaq_laneq_f32(C0, A3, B0, 3);

        // use B1 all
        C1 = vfmaq_laneq_f32(C1, A0, B1, 0);
        C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
        C1 = vfmaq_laneq_f32(C1, A2, B1, 2);
        C1 = vfmaq_laneq_f32(C1, A3, B1, 3);
        
        // use B2 all
        C2 = vfmaq_laneq_f32(C2, A0, B2, 0);
        C2 = vfmaq_laneq_f32(C2, A1, B2, 1);
        C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
        C2 = vfmaq_laneq_f32(C2, A3, B2, 3);
        
        // use B3 all
        C3 = vfmaq_laneq_f32(C3, A0, B3, 0);
        C3 = vfmaq_laneq_f32(C3, A1, B3, 1);
        C3 = vfmaq_laneq_f32(C3, A2, B3, 2);
        C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
      }

      C_idx = ROWCOL2IDX(n, i_idx, j_idx);    // index pointing C[i_idx, j_idx]
      vst1q_f32(C + C_idx, C0);               // update C[i_idx ~ i_idx+3, j_idx]
      vst1q_f32(C + C_idx + n, C1);           // update C[i_idx ~ i_idx+3, j_idx+1]
      vst1q_f32(C + C_idx + 2*n, C2);         // update C[i_idx ~ i_idx+3, j_idx+2]
      vst1q_f32(C + C_idx + 3*n, C3);         // update C[i_idx ~ i_idx+3, j_idx+3]
    }
  }
}

void mmT_simd(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
  // Size of matrices are A[n, k], B[m, k], C[n, m]

  int A_idx, B_idx, C_idx;
  float32x4_t A0, A1, A2, A3;
  float32x4_t B0, B1, B2, B3;
  float32x4_t C0, C1, C2, C3;

  for (int i_idx=0; i_idx<n; i_idx+=4) {
    for (int j_idx=0; j_idx<m; j_idx+=4) {
      // gonna calculate C[i_idx ~ i_idx+3, j_idx ~ j_idx+3]
      C0 = vmovq_n_f32(0);  // C[i_idx ~ i_idx+3, j_idx]
      C1 = vmovq_n_f32(0);  // C[i_idx ~ i_idx+3, j_idx+1]
      C2 = vmovq_n_f32(0);  // C[i_idx ~ i_idx+3, j_idx+2]
      C3 = vmovq_n_f32(0);  // C[i_idx ~ i_idx+3, j_idx+3]

      for (int k_idx=0; k_idx<k; k_idx+=4) {
        // lets iterate through k_idx
        A_idx = ROWCOL2IDX(n, i_idx, k_idx);  // index pointing [i_idx, k_idx]
        B_idx = ROWCOL2IDX(m, j_idx, k_idx);  // index pointing [j_idx, k_idx]

        A0 = vld1q_f32(A + A_idx);             // A0: A[i_idx ~ i_idx+3, k_idx]
        A1 = vld1q_f32(A + A_idx + n);         // A1: A[i_idx ~ i_idx+3, k_idx+1]
        A2 = vld1q_f32(A + A_idx + 2*n);       // A2: A[i_idx ~ i_idx+3, k_idx+2]
        A3 = vld1q_f32(A + A_idx + 3*n);       // A3: A[i_idx ~ i_idx+3, k_idx+3]

        B0 = vld1q_f32(B + B_idx);             // B0: B[j_idx ~ j_idx+3, k_idx]
        B1 = vld1q_f32(B + B_idx + m);         // B1: B[j_idx ~ j_idx+3, k_idx+1]
        B2 = vld1q_f32(B + B_idx + 2*m);       // B2: B[j_idx ~ j_idx+3, k_idx+2]
        B3 = vld1q_f32(B + B_idx + 3*m);       // B3: B[j_idx ~ j_idx+3, k_idx+3]

        // finish computing C0
        C0 = vfmaq_laneq_f32(C0, A0, B0, 0);
        C0 = vfmaq_laneq_f32(C0, A1, B1, 0);
        C0 = vfmaq_laneq_f32(C0, A2, B2, 0);
        C0 = vfmaq_laneq_f32(C0, A3, B3, 0);

        // finish computing C1
        C1 = vfmaq_laneq_f32(C1, A0, B0, 1);
        C1 = vfmaq_laneq_f32(C1, A1, B1, 1);
        C1 = vfmaq_laneq_f32(C1, A2, B2, 1);
        C1 = vfmaq_laneq_f32(C1, A3, B3, 1);

        // finish computing C2
        C2 = vfmaq_laneq_f32(C2, A0, B0, 2);
        C2 = vfmaq_laneq_f32(C2, A1, B1, 2);
        C2 = vfmaq_laneq_f32(C2, A2, B2, 2);
        C2 = vfmaq_laneq_f32(C2, A3, B3, 2);

        // finish computing C3
        C3 = vfmaq_laneq_f32(C3, A0, B0, 3);
        C3 = vfmaq_laneq_f32(C3, A1, B1, 3);
        C3 = vfmaq_laneq_f32(C3, A2, B2, 3);
        C3 = vfmaq_laneq_f32(C3, A3, B3, 3);
      }

      C_idx = ROWCOL2IDX(n, i_idx, j_idx);    // index pointing C[i_idx, j_idx]
      vst1q_f32(C + C_idx, C0);               // update C[i_idx ~ i_idx+3, j_idx]
      vst1q_f32(C + C_idx + n, C1);           // update C[i_idx ~ i_idx+3, j_idx+1]
      vst1q_f32(C + C_idx + 2*n, C2);         // update C[i_idx ~ i_idx+3, j_idx+2]
      vst1q_f32(C + C_idx + 3*n, C3);         // update C[i_idx ~ i_idx+3, j_idx+3]
    }
  }
};

void dotaddq_lane_f32(float32x4_t *src1, float32x4_t *src2, float32x4_t *dst, const int lane) {
  float32_t sum, added;
  float32x4_t temp = vmovq_n_f32(0);

  temp = vmlaq_f32(temp, *src1, *src2);
  sum = vaddvq_f32(temp);

  switch (lane) {
    case 0: added = vgetq_lane_f32(*dst, 0) + sum; *dst = vld1q_lane_f32(&added, *dst, 0); break;
    case 1: added = vgetq_lane_f32(*dst, 1) + sum; *dst = vld1q_lane_f32(&added, *dst, 1); break;
    case 2: added = vgetq_lane_f32(*dst, 2) + sum; *dst = vld1q_lane_f32(&added, *dst, 2); break;
    case 3: added = vgetq_lane_f32(*dst, 3) + sum; *dst = vld1q_lane_f32(&added, *dst, 3); break;
  }
}

void mTm_simd(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
  // Matrices are in size of A[k, n], B[k, m], C[n, m], transpose of A (A.T) will be used
  
  int A_idx, B_idx, C_idx;
  float32x4_t A0, A1, A2, A3;
  float32x4_t B0, B1, B2, B3;
  float32x4_t C0, C1, C2, C3;

  for (int i_idx=0; i_idx<n; i_idx+=4) {
    for (int j_idx=0; j_idx<m; j_idx+=4) {
      // gonna calculate C[i_idx ~ i_idx+3, j_idx ~ j_idx+3]
      C0 = vmovq_n_f32(0);
      C1 = vmovq_n_f32(0);
      C2 = vmovq_n_f32(0);
      C3 = vmovq_n_f32(0);

      for (int k_idx=0; k_idx<k; k_idx+=4) {
        // lets iterate through k_idx

        // Compute base index to 4x4 block
        A_idx = ROWCOL2IDX(k, k_idx, i_idx);  // index pointing [k_idx, i_idx]
        B_idx = ROWCOL2IDX(k, k_idx, j_idx);  // index pointing [k_idx, j_idx]
        
        A0 = vld1q_f32(A+A_idx);      // A0: A[k_idx ~ k_idx+3, i_idx]
        A1 = vld1q_f32(A+A_idx+k);    // A1: A[k_idx ~ k_idx+3, i_idx+1]
        A2 = vld1q_f32(A+A_idx+2*k);  // A2: A[k_idx ~ k_idx+3, i_idx+2]
        A3 = vld1q_f32(A+A_idx+3*k);  // A3: A[k_idx ~ k_idx+3, i_idx+3]
        
        B0 = vld1q_f32(B+B_idx);      // B0: B[k_idx ~ k_idx+3, j_idx]
        B1 = vld1q_f32(B+B_idx+k);    // B1: B[k_idx ~ k_idx+3, j_idx+1]
        B2 = vld1q_f32(B+B_idx+2*k);  // B2: B[k_idx ~ k_idx+3, j_idx+2]
        B3 = vld1q_f32(B+B_idx+3*k);  // B3: B[k_idx ~ k_idx+3, j_idx+3]

        // finish computing C0
        dotaddq_lane_f32(&A0, &B0, &C0, 0);
        dotaddq_lane_f32(&A1, &B0, &C0, 1);
        dotaddq_lane_f32(&A2, &B0, &C0, 2);
        dotaddq_lane_f32(&A3, &B0, &C0, 3);
        
        // finish computing C1
        dotaddq_lane_f32(&A0, &B1, &C1, 0);
        dotaddq_lane_f32(&A1, &B1, &C1, 1);
        dotaddq_lane_f32(&A2, &B1, &C1, 2);
        dotaddq_lane_f32(&A3, &B1, &C1, 3);
        
        // finish computing C2
        dotaddq_lane_f32(&A0, &B2, &C2, 0);
        dotaddq_lane_f32(&A1, &B2, &C2, 1);
        dotaddq_lane_f32(&A2, &B2, &C2, 2);
        dotaddq_lane_f32(&A3, &B2, &C2, 3);

        // finish computing C3
        dotaddq_lane_f32(&A0, &B3, &C3, 0);
        dotaddq_lane_f32(&A1, &B3, &C3, 1);
        dotaddq_lane_f32(&A2, &B3, &C3, 2);
        dotaddq_lane_f32(&A3, &B3, &C3, 3);
      }
      // Compute base index for stores
      C_idx = ROWCOL2IDX(n, i_idx, j_idx);    // i_idx-th row, j_idx-th column
      vst1q_f32(C+C_idx, C0);     // (j_idx ~ j_idx+3) row, i_idx-th column
      vst1q_f32(C+C_idx+n, C1);   // (j_idx ~ j_idx+3) row, (i_idx+1)-th column
      vst1q_f32(C+C_idx+2*n, C2); // (j_idx ~ j_idx+3) row, (i_idx+2)-th column
      vst1q_f32(C+C_idx+3*n, C3); // (j_idx ~ j_idx+3) row, (i_idx+3)-th column
    }
  }
}

void mm_T_simd(float32_t *A, float32_t *B, float32_t *C, uint32_t n, uint32_t m, uint32_t k) {
  // Matrices are in size of A[n, k], B[k, m], C[m, n] transpose of C (C.T) will be used

  float32_t *temp = malloc(sizeof(float32_t) * n * m);

  mm_simd(A, B, temp, n, m, k);
  transpose(temp, C, n, m);

  free(temp);
}


void softmax_inplace_simd(float32_t *src, uint32_t n) {
  // Vector is in size of src[n]

  // find max
  float32_t max = 0, temp;
  float32x4_t scope;
  for (int i=0; i<n; i+=4) {
    scope = vld1q_f32(src+i);
    temp = vmaxvq_f32(scope);
    max = max > temp ? max : temp;
  }
  
  // normalize (substract) by max and exponent
  float32x4_t maxes;
  maxes = vld1q_dup_f32(&max);
  for (int i=0; i<n; i+=4) {
    scope = vld1q_f32(src+i);
    scope = vsubq_f32(scope, maxes);
    vst1q_f32(src+i, scope);
  }

  // exponent and sum
  float32_t sumexp = 0;
  for (int i=0; i<n; i+=1) {
    src[i] = exp(src[i]);
    sumexp += src[i];
  }

  sumexp = 1/sumexp;

  // scale by sum
  scale_inplace_simd(src, sumexp, n);

}

void scale_inplace_simd(float32_t *src, float32_t scale, uint32_t n) {
  // Vector(or matrix) is in size of src[n]
  float32x4_t scope;
  float32x4_t scaler = vld1q_dup_f32(&scale);

  for (int i=0; i<n; i+=4) {
    scope = vld1q_f32(src+i);
    scope = vmulxq_laneq_f32(scope, scaler, 0);
    vst1q_f32(src+i, scope);
  }
}

void relu_inplace_simd(float32_t *src, uint32_t n) {
  // Vector(or matrix) is in size of src[n]
  float32x4_t scope;
  float32x4_t zeros = vmovq_n_f32(0);

  for (int i=0; i<n; i+=4) {
    scope = vld1q_f32(src+i);
    scope = vmaxq_f32(scope, zeros);
    vst1q_f32(src+i, scope);
  }
}

void addi_inplace_simd(float32_t *imm, float32_t *dst, uint32_t n) {
  // Matrices are in size of [n, m]. Elements in dst will be increased by imm, respectively.
  float32x4_t dst_scope;
  float32x4_t imm_scope;

  for (int i=0; i<n; i+=4) {
    dst_scope = vld1q_f32(dst+i);
    imm_scope = vld1q_f32(imm+i);
    dst_scope = vaddq_f32(imm_scope, dst_scope);
    vst1q_f32(dst+i, dst_scope);
  }  
}

void normalize_inplace_simd(float32_t *src, uint32_t n) {
  // Vector(or matrix) is in size of src[n].
  float32_t mean = 0, sigma=0;
  float32x4_t scope, buffer, means, sigmas;

  for (int i=0; i<n; i+=4) {
    scope = vld1q_f32(src+i);
    mean += vaddvq_f32(scope);
  }
  mean /= n;
  means = vld1q_dup_f32(&mean);

  for (int i=0; i<n; i+=4) {
    scope = vld1q_f32(src+i);
    buffer = vsubq_f32(scope, means);
    buffer = vmulq_f32(buffer, buffer);
    sigma += vaddvq_f32(buffer);
  }
  sigma = sqrt(sigma/n);
  sigmas = vld1q_dup_f32(&sigma);

  for (int i=0; i<n; i+=4) {
    scope = vld1q_f32(src+i);
    scope = vsubq_f32(scope, means);
    scope = vdivq_f32(scope, sigmas);
    vst1q_f32(src+i, scope);
  }

}

void transpose(float32_t *src, float32_t *dst, uint32_t rows, uint32_t cols) {
  for (int i=0; i<rows; i++) {
    for (int j=0; j<cols; j++) {
      dst[ROWCOL2IDX(cols, j, i)] = src[ROWCOL2IDX(rows, i, j)];
    }
  }
}

void transpose_4x4_inplace(float32_t *src, uint32_t spacing) {
  // Transpose 4x4 matrix inplace

  float32_t temp;

  for (int i=0; i<4; i++) {
    for (int j=0; j<i; j++) {
      temp = *(src + i * spacing + j);
      *(src + i * spacing + j) = *(src + j * spacing + i);
      *(src + j * spacing + i) = temp;
    }
  }
}

void transpose_4x4_inplace_simd(float32_t *src, uint32_t spacing) {
  // Transpose 4x4 matrix inplace

  float32x4_t D0 = vld1q_f32(src);
  float32x4_t D1 = vld1q_f32(src + spacing);
  float32x4_t D2 = vld1q_f32(src + 2*spacing);
  float32x4_t D3 = vld1q_f32(src + 3*spacing);

  float32x4x2_t buffer1_4x2, buffer2_4x2;
  float32x4_t buffer1_4, buffer2_4;
  
  buffer1_4x2 = vtrnq_f32(D0, D1);
  buffer2_4x2 = vtrnq_f32(D2, D3);
  
  D0 = buffer1_4x2.val[0];
  D1 = buffer1_4x2.val[1];
  D2 = buffer2_4x2.val[0];
  D3 = buffer2_4x2.val[1];
  
  buffer1_4 = vcombine_f32(vget_low_f32(D0), vget_low_f32(D2));
  buffer2_4 = vcombine_f32(vget_high_f32(D0), vget_high_f32(D2));
  vst1q_f32(src, buffer1_4);
  vst1q_f32(src + 2*spacing, buffer2_4);

  buffer1_4 = vcombine_f32(vget_low_f32(D1), vget_low_f32(D3));
  buffer2_4 = vcombine_f32(vget_high_f32(D1), vget_high_f32(D3));
  vst1q_f32(src + spacing, buffer1_4);
  vst1q_f32(src + 3*spacing, buffer2_4);
}

void transpose_inplace(float32_t *src, uint32_t rows, uint32_t cols) {
  // TODO
}