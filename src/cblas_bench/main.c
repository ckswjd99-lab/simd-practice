#include <stdio.h>
#include <cblas.h>
#include <time.h>

#include "utils.h"
#include "operation.h"

// void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
// 		 OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);


#define d_model     512
#define token_num   256
#define d_k         64
#define d_v         64
#define caching_row 32

float Q   [token_num * d_k];    // matrix
float K_T [d_k * token_num];    // matrix
float V_T [d_v * token_num];    // matrix
float W_O [d_model * d_v];      // matrix
float singleAttention1  [d_v * token_num];      // matrix
float singleAttention2  [d_v * token_num];      // matrix

float calcbuffer1 [token_num * token_num];
float calcbuffer2 [token_num * caching_row];

int single_attention_lbl(const float* _Q, const float* _K_T, const float* _V_T, float* _output, float* _calcbuffer) {
  // _calcbuffer: matrix[token_num, token_num]

  matrix_init_zero(_calcbuffer, token_num * token_num);

  cblas_sgemm(
    CblasColMajor, CblasNoTrans, CblasNoTrans, token_num, token_num, d_k, 1.0, _Q, token_num, _K_T, d_k, 1.0, _calcbuffer, token_num
  );

  scale_inplace_simd(_calcbuffer, sqrt(d_k), token_num * token_num);

  for (int i=0; i<token_num; i++) {
    softmax_inplace_simd(_calcbuffer + i * token_num, token_num);
  }

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_v, token_num, token_num, 1.0, _V_T, d_v, _calcbuffer, token_num, 1.0, _output, d_v);


}

int single_attention_fumm(const float* _Q, const float* _K_T, const float* _V_T, float* _output, float* _calcbuffer) {
  // _calcbuffer: matrix[token_num, caching_row]

  for (int i=0; i<token_num; i+=caching_row) {
    matrix_init_zero(_calcbuffer, token_num * caching_row);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, token_num, caching_row, d_k, 1.0, _Q, token_num, _K_T + i * d_k, d_k, 1.0, _calcbuffer, token_num);
    
    scale_inplace_simd(_calcbuffer, sqrt(d_k), token_num * caching_row);
    
    for (int j=0; j<caching_row; j++) {
      softmax_inplace_simd(_calcbuffer + j * token_num, token_num);
    }

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_v, caching_row, token_num, 1.0, _V_T, d_v, _calcbuffer, token_num, 1.0, _output + i * d_v, d_v);
  }

}

int main() {

  matrix_init_rand(Q, token_num * d_k);
  matrix_init_rand(K_T, d_k * token_num);
  matrix_init_rand(V_T, d_v * token_num);
  matrix_init_rand(W_O, d_model * d_v);
  matrix_init_zero(singleAttention1, d_v * token_num);
  matrix_init_zero(singleAttention2, d_v * token_num);

  int test1_num = 1024;
  int test1_score = 0;

  for (int i=0; i<test1_num; i++) {
    single_attention_lbl(Q, K_T, V_T, singleAttention1, calcbuffer1);
    single_attention_fumm(Q, K_T, V_T, singleAttention2, calcbuffer2);

    test1_score += matrix_comp(singleAttention1, singleAttention2, d_v, token_num);
  }
  printf("[Test1 - correctness] score %d/%d\n", test1_score, test1_num);

  int test2_num = 1024;
  int test2_score_lbl = 0;
  int test2_score_fumm = 0;
  clock_t start_time, end_time;

  start_time = clock();
  for (int i=0; i<test2_num; i++) {
    single_attention_lbl(Q, K_T, V_T, singleAttention1, calcbuffer1);
  }
  end_time = clock();

  test2_score_lbl += end_time - start_time;

  start_time = clock();
  for (int i=0; i<test2_num; i++) {
    single_attention_fumm(Q, K_T, V_T, singleAttention1, calcbuffer2);
  }
  end_time = clock();

  test2_score_fumm += end_time - start_time;

  printf("[Test2 - time] LBL: %d, FuMM: %d\n", test2_score_lbl, test2_score_fumm);

  return 0;
}

