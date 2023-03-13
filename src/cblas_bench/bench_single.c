#include <stdio.h>
#include <cblas.h>
#include <time.h>

#include "utils.h"
#include "operation.h"


#define d_model     512
#define token_num   256
#define d_k         64
#define d_v         64
#define head        8
#define caching_col 16

// Data align: row-wise. Element in next index is the next element in the same row.
// Notation: [#cols, #rows]

// For Single Benchmark
float Q_single    [d_k * token_num];    // matrix
float K_T_single  [token_num * d_k];    // matrix
float V_T_single  [token_num * d_v];    // matrix
float W_O_single  [d_v * d_model];      // matrix
float singleAttention1  [token_num * d_v];      // matrix
float singleAttention2  [token_num * d_v];      // matrix

float multiAttention1   [token_num * d_model];      // matrix
float multiAttention2   [token_num * d_model];      // matrix

float buffer1_sa [token_num * token_num];
float buffer1_ma [d_v * token_num];

float buffer2_sa [token_num * caching_col];
float buffer2_ma [d_v * caching_col];

int single_attention_lbl(const float* _Q, const float* _K_T, const float* _V_T, float* _output, float* _buffer_sa) {
  // _buffer_sa: matrix[token_num, token_num]

  matrix_init_zero(_buffer_sa, token_num * token_num);

  // cblas_sgemm(
  //   CblasColMajor, CblasNoTrans, CblasNoTrans, token_num, token_num, d_k, 1.0, _Q, token_num, _K_T, d_k, 1.0, _buffer_sa, token_num
  // );
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasNoTrans, token_num, token_num, d_k, 1.0, _K_T, d_k, _Q, token_num, 1.0, _buffer_sa, token_num
  );

  scale_inplace(_buffer_sa, 1/sqrt(d_k), token_num * token_num);

  for (int i=0; i<token_num; i++) {
    softmax_inplace(_buffer_sa + i * token_num, token_num);
  }

  // cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_v, token_num, token_num, 1.0, _V_T, d_v, _buffer_sa, token_num, 1.0, _output, d_v);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, token_num, d_v, token_num, 1.0, _buffer_sa, token_num, _V_T, d_v, 1.0, _output, d_v);


}

int single_attention_fumm(const float* _Q, const float* _K_T, const float* _V_T, float* _output, float* _buffer_sa) {
  // _buffer_sa: matrix[token_num, caching_col]

  for (int i=0; i<token_num; i+=caching_col) {
    matrix_init_zero(_buffer_sa, token_num * caching_col);

    // cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, token_num, caching_row, d_k, 1.0, _Q, token_num, _K_T + i * d_k, d_k, 1.0, _buffer_sa, token_num);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, caching_col, token_num, d_k, 1.0, _K_T + i * d_k, d_k, _Q, token_num, 1.0, _buffer_sa, token_num);
    
    scale_inplace(_buffer_sa, 1/sqrt(d_k), token_num * caching_col);

    for (int j=0; j<caching_col; j++) {
      softmax_inplace(_buffer_sa + j * token_num, token_num);
    }

    // cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_v, caching_row, token_num, 1.0, _V_T, d_v, _buffer_sa, token_num, 1.0, _output + i * d_v, d_v);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, caching_col, d_v, token_num, 1.0, _buffer_sa, token_num, _V_T, d_v, 1.0, _output + i * d_v, d_v);
  }

}

int main(int argc, char *argv[]) {

  matrix_init_rand(Q_single, token_num * d_k);
  matrix_init_rand(K_T_single, d_k * token_num);
  matrix_init_rand(V_T_single, d_v * token_num);
  matrix_init_rand(W_O_single, d_model * d_v);
  matrix_init_zero(singleAttention1, d_v * token_num);
  matrix_init_zero(singleAttention2, d_v * token_num);

  int test1_num = 256;
  int test1_score = 0;

  for (int i=0; i<test1_num; i++) {
    single_attention_lbl(Q_single, K_T_single, V_T_single, singleAttention1, buffer1_sa);
    single_attention_fumm(Q_single, K_T_single, V_T_single, singleAttention2, buffer2_sa);

    test1_score += matrix_comp(singleAttention1, singleAttention2, d_v, token_num);
  }
  printf("[Test1 - correctness]\n\tACCURACY: %d/%d\n", test1_score, test1_num);


  printf("[Test2 - Single head attention, time]\n");

  int test2_num = 1024;
  int test2_score_lbl = 0;
  int test2_score_fumm = 0;
  clock_t start_time, end_time;

  if (argc >= 2) {
    test2_num = atoi(argv[1]);
  }


  start_time = clock();
  for (int i=0; i<test2_num; i++) {
    single_attention_fumm(Q_single, K_T_single, V_T_single, singleAttention2, buffer2_sa);
  }
  end_time = clock();
  test2_score_fumm = end_time - start_time;

  start_time = clock();
  for (int i=0; i<test2_num; i++) {
    single_attention_lbl(Q_single, K_T_single, V_T_single, singleAttention1, buffer1_sa);
  }
  end_time = clock();
  test2_score_lbl = end_time - start_time;


  printf("\tPARAMS\n");
  printf("\trepeated %'d times, d_model: %'d, token_num: %'d, d_k: %'d, d_v: %'d, caching_col: %'d\n", test2_num, d_model, token_num, d_k, d_v, caching_col);
  printf("\tRESULTS\n");
  printf("\tLBL: %'d, FuMM: %'d\n", test2_score_lbl, test2_score_fumm);
  printf("\n");

  return 0;
}

