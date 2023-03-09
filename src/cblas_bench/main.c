#include <stdio.h>
#include <cblas.h>
#include <time.h>

#include "utils.h"
#include "operation.h"


#define d_model     512
#define token_num   256
#define d_k         64
#define d_v         64
#define caching_row 4

float Q   [token_num * d_k];    // matrix
float K_T [d_k * token_num];    // matrix
float V_T [d_v * token_num];    // matrix
float W_O [d_model * d_v];      // matrix
float singleAttention1  [d_v * token_num];      // matrix
float singleAttention2  [d_v * token_num];      // matrix
float multiAttention1  [d_model * token_num];      // matrix
float multiAttention2  [d_model * token_num];      // matrix

float buffer1_sa [token_num * token_num];
float buffer1_ma [d_v * token_num];

float buffer2_sa [token_num * caching_row];
float buffer2_ma [d_v * caching_row];

int single_attention_lbl(const float* _Q, const float* _K_T, const float* _V_T, float* _output, float* _buffer_sa) {
  // _buffer_sa: matrix[token_num, token_num]

  matrix_init_zero(_buffer_sa, token_num * token_num);

  cblas_sgemm(
    CblasColMajor, CblasNoTrans, CblasNoTrans, token_num, token_num, d_k, 1.0, _Q, token_num, _K_T, d_k, 1.0, _buffer_sa, token_num
  );

  scale_inplace_simd(_buffer_sa, sqrt(d_k), token_num * token_num);

  for (int i=0; i<token_num; i++) {
    softmax_inplace_simd(_buffer_sa + i * token_num, token_num);
  }

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_v, token_num, token_num, 1.0, _V_T, d_v, _buffer_sa, token_num, 1.0, _output, d_v);


}

int single_attention_fumm(const float* _Q, const float* _K_T, const float* _V_T, float* _output, float* _buffer_sa) {
  // _buffer_sa: matrix[token_num, caching_row]

  for (int i=0; i<token_num; i+=caching_row) {
    matrix_init_zero(_buffer_sa, token_num * caching_row);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, token_num, caching_row, d_k, 1.0, _Q, token_num, _K_T + i * d_k, d_k, 1.0, _buffer_sa, token_num);
    
    scale_inplace_simd(_buffer_sa, sqrt(d_k), token_num * caching_row);
    
    for (int j=0; j<caching_row; j++) {
      softmax_inplace_simd(_buffer_sa + j * token_num, token_num);
    }

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_v, caching_row, token_num, 1.0, _V_T, d_v, _buffer_sa, token_num, 1.0, _output + i * d_v, d_v);
  }

}

int multi_attention_lbl(const float* _Q, const float* _K_T, const float* _V_T, const float* _W_O, float* _output, float* _buffer_sa, float* _buffer_ma) {
  // _buffer_sa: matrix[token_num, token_num]
  // _buffer_ma: matrix[d_model, token_num]

  matrix_init_zero(_buffer_sa, token_num * token_num);

  cblas_sgemm(
    CblasColMajor, CblasNoTrans, CblasNoTrans, token_num, token_num, d_k, 1.0, _Q, token_num, _K_T, d_k, 1.0, _buffer_sa, token_num
  );

  scale_inplace_simd(_buffer_sa, sqrt(d_k), token_num * token_num);

  for (int i=0; i<token_num; i++) {
    softmax_inplace_simd(_buffer_sa + i * token_num, token_num);
  }

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_v, token_num, token_num, 1.0, _V_T, d_v, _buffer_sa, token_num, 1.0, _buffer_ma, d_v);

  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_model, token_num, d_v, 1.0, _W_O, d_model, _buffer_ma, d_v, 1.0, _output, d_model);

}

int multi_attention_fumm(const float* _Q, const float* _K_T, const float* _V_T, const float* _W_O, float* _output, float* _buffer_sa, float* _buffer_ma) {
  // _buffer_sa: matrix[token_num, caching_row]
  // _buffer_sa: matrix[d_model, caching_row]

  for (int i=0; i<token_num; i+=caching_row) {
    matrix_init_zero(_buffer_sa, token_num * caching_row);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, token_num, caching_row, d_k, 1.0, _Q, token_num, _K_T + i * d_k, d_k, 1.0, _buffer_sa, token_num);
    
    scale_inplace_simd(_buffer_sa, sqrt(d_k), token_num * caching_row);
    
    for (int j=0; j<caching_row; j++) {
      softmax_inplace_simd(_buffer_sa + j * token_num, token_num);
    }

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_v, caching_row, token_num, 1.0, _V_T, d_v, _buffer_sa, token_num, 1.0, _buffer_ma, d_v);

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, d_model, caching_row, d_v, 1.0, _W_O, d_model, _buffer_ma, d_v, 1.0, _output + i * d_model, d_model);
  }

}

int main(int argc, char *argv[]) {

  matrix_init_rand(Q, token_num * d_k);
  matrix_init_rand(K_T, d_k * token_num);
  matrix_init_rand(V_T, d_v * token_num);
  matrix_init_rand(W_O, d_model * d_v);
  matrix_init_zero(singleAttention1, d_v * token_num);
  matrix_init_zero(singleAttention2, d_v * token_num);
  matrix_init_zero(multiAttention1, d_model * token_num);
  matrix_init_zero(multiAttention2, d_model * token_num);

  // int test1_num = 256;
  // int test1_score = 0;

  // for (int i=0; i<test1_num; i++) {
  //   single_attention_lbl(Q, K_T, V_T, singleAttention1, buffer1_sa);
  //   single_attention_fumm(Q, K_T, V_T, singleAttention2, buffer2_sa);

  //   test1_score += matrix_comp(singleAttention1, singleAttention2, d_v, token_num);
  // }
  // printf("[Test1 - correctness] score %d/%d\n", test1_score, test1_num);


  printf("[Test2 - Single head attention, time]\n");

  int test2_num = 128;
  int test2_score_lbl = 0;
  int test2_score_fumm = 0;
  clock_t start_time, end_time;

  if (argc >= 2) {
    test2_num = atoi(argv[1]);
  }

  for (int i=0; i<test2_num; i++) {
    start_time = clock();
    single_attention_lbl(Q, K_T, V_T, singleAttention1, buffer1_sa);
    end_time = clock();
    test2_score_lbl += end_time - start_time;

  }
  for (int i=0; i<test2_num; i++) {

    start_time = clock();
    single_attention_fumm(Q, K_T, V_T, singleAttention2, buffer2_sa);
    end_time = clock();
    test2_score_fumm += end_time - start_time;
  }

  printf("\tPARAMS\n");
  printf("\trepeated %'d times, d_model: %'d, token_num: %'d, d_k: %'d, d_v: %'d, caching_row: %'d\n", test2_num, d_model, token_num, d_k, d_v, caching_row);
  printf("\tRESULTS\n");
  printf("\tLBL: %'d, FuMM: %'d\n", test2_score_lbl, test2_score_fumm);
  printf("\n");


  printf("[Test3 - Multi head attention, time]\n");
  
  int test3_num = 128;
  int test3_score_lbl = 0;
  int test3_score_fumm = 0;

  if (argc >= 2) {
    test3_num = atoi(argv[1]);
  }

  for (int i=0; i<test3_num; i++) {
    start_time = clock();
    multi_attention_lbl(Q, K_T, V_T, W_O, multiAttention1, buffer1_sa, buffer1_ma);
    end_time = clock();
    test3_score_lbl += end_time - start_time;

  }
  for (int i=0; i<test3_num; i++) {

    start_time = clock();
    multi_attention_fumm(Q, K_T, V_T, W_O, multiAttention2, buffer2_sa, buffer2_ma);
    end_time = clock();
    test3_score_fumm += end_time - start_time;
  }

  printf("\tPARAMS\n");
  printf("\trepeated %'d times, d_model: %'d, token_num: %'d, d_k: %'d, d_v: %'d, caching_row: %'d\n", test2_num, d_model, token_num, d_k, d_v, caching_row);
  printf("\tRESULTS\n");
  printf("\tLBL: %'d, FuMM: %'d\n", test3_score_lbl, test3_score_fumm);
  printf("\n");

  return 0;
}

