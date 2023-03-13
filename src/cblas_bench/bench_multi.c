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
#define caching_row 4

// Data align: row-wise. Element in next index is the next element in the same row.
// Notation: [#cols, #rows]

// For Multi Benchmark
float embedded    [d_model * token_num];
float W_Q         [d_k * head * d_model];
float W_K_T       [d_model * d_k * head];
float W_V         [d_v * head * d_model];
float W_O         [d_model * d_v * head];
float output_hlbl [d_model * token_num];
float output_slbl [d_model * token_num];
float output_fumm [d_model * token_num];

int multi_attention_hardlbl(
  const float *_embedded, const float *_W_Q, const float *_W_K_T, const float *_W_V, const float *_W_O, float *_output,
  const int _token_num, const int _d_model, const int _d_k, const int _d_v, const int _head,
  float *_Q, float *_K_T, float *_V, float **_queried, float *_single_attentions
) {
  
  // create Q, K, V
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _d_k * _head, _d_model, 1.0, _embedded, _d_model, _W_Q, _d_k * _head, 0.0, _Q, _d_k * _head);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, _d_k * _head, _token_num, _d_model, 1.0, _W_K_T, _d_model, _embedded, _d_model, 0.0, _K_T, _token_num);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _d_v * _head, _d_model, 1.0, _embedded, _d_model, _W_V, _d_v * _head, 0.0, _V, _d_v * _head);

  // calc queried
  for (int i=0; i<_head; i++) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _token_num, _d_k, 1.0, _Q + i * _d_k, _d_k * _head, _K_T + i * _token_num * _d_k, _token_num, 0.0, _queried[i], _token_num);
  }

  // scale & softmax
  for (int i=0; i<_head; i++) {
    for (int j=0; j<_token_num; j++) {
      scale_inplace(_queried[i] + j * _token_num, 1/sqrt(_d_k), _token_num);
      softmax_inplace(_queried[i] + j * _token_num, _token_num);
    }
  }

  // calc single attentions
  for (int i=0; i<_head; i++) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _d_v, _token_num, 1.0, _queried[i], _token_num, _V + i * _d_v, _d_v * _head, 0.0, _single_attentions + i * _d_v, _d_v * _head);
  }

  // calc multi attention
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _d_model, _d_v * _head, 1.0, _single_attentions, _d_v * _head, _W_O, _d_model, 1.0, _output, _d_model);

  // residual connection & layer normalization
  addi_inplace_simd(_embedded, _output, _d_model * _token_num);
  normalize_inplace_simd(_output, _d_model * _token_num);

}

int multi_attention_softlbl(
  const float *_embedded, const float *_W_Q, const float *_W_K_T, const float *_W_V, const float *_W_O, float *_output,
  const int _token_num, const int _d_model, const int _d_k, const int _d_v, const int _head,
  float *_Q, float *_K_T, float *_V, float *_queried, float *_single_attentions
) {

  // create Q, K, V
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _d_k * _head, _d_model, 1.0, _embedded, _d_model, _W_Q, _d_k * _head, 0.0, _Q, _d_k * _head);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, _d_k * _head, _token_num, _d_model, 1.0, _W_K_T, _d_model, _embedded, _d_model, 0.0, _K_T, _token_num);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _d_v * _head, _d_model, 1.0, _embedded, _d_model, _W_V, _d_v * _head, 0.0, _V, _d_v * _head);

  // calc each single attention
  for (int i=0; i<_head; i++) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _token_num, _d_k, 1.0, _Q + i * _d_k, _d_k * _head, _K_T + i * _token_num * _d_k, _token_num, 0.0, _queried, _token_num);
    for (int j=0; j<_token_num; j++) {
      scale_inplace(_queried + j * _token_num, 1/sqrt(_d_k), _token_num);
      softmax_inplace(_queried + j * _token_num, _token_num);
    }
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _d_v, _token_num, 1.0, _queried, _token_num, _V + i * _d_v, _d_v * _head, 0.0, _single_attentions + i * _d_v, _d_v * _head);

  }
  
  // calc multi attention
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _d_model, _d_v * _head, 1.0, _single_attentions, _d_v * _head, _W_O, _d_model, 1.0, _output, _d_model);

  // residual connection & layer normalization
  addi_inplace_simd(_embedded, _output, _d_model * _token_num);
  normalize_inplace_simd(_output, _d_model * _token_num);
}

int multi_attention_fumm(
  const float *_embedded, const float *_W_Q, const float *_W_K_T, const float *_W_V, const float *_W_O, const float *_output,
  const int _token_num, const int _d_model, const int _d_k, const int _d_v, const int _head,
  float *_Q, float *_K_T, float *_V, float *_queried, float *_single_attentions, const int _caching_rows
) {
  
  // init memories
  matrix_init_zero(_queried, _token_num * _caching_rows);
  matrix_init_zero(_single_attentions, _d_v * _caching_rows);

  // create Q, K, V
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _d_k * _head, _d_model, 1.0, _embedded, _d_model, _W_Q, _d_k * _head, 0.0, _Q, _d_k * _head);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, _d_k * _head, _token_num, _d_model, 1.0, _W_K_T, _d_model, _embedded, _d_model, 0.0, _K_T, _token_num);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _token_num, _d_v * _head, _d_model, 1.0, _embedded, _d_model, _W_V, _d_v * _head, 0.0, _V, _d_v * _head);

  // calc each single attention
  for (int i=0; i<_head; i++) {
    for (int j=0; j<_token_num; j+=_caching_rows) {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _caching_rows, _token_num, _d_k, 1.0, _Q + i * _d_k + j * _d_k * _head, _d_k * _head, _K_T + j * _token_num, _token_num, 0.0, _queried, _token_num);
      for (int k=0; k<_caching_rows; k++) {
        scale_inplace(_queried + k * _token_num, 1/sqrt(_d_k), _token_num);
        softmax_inplace(_queried + k * _token_num, _token_num);
      }

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _caching_rows, _d_v, _token_num, 1.0, _queried, _token_num, _V + i * _d_v, _d_v * _head, 0.0, _single_attentions, _d_v);

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, _caching_rows, _d_model, _d_v, 1.0, _single_attentions, _d_v, _W_O + i * _d_v * _d_model, _d_model, 1.0, _output + j * _d_model, _d_model);

    }

  }

  // residual connection & layer normalization
  addi_inplace_simd(_embedded, _output, _d_model * _token_num);
  normalize_inplace_simd(_output, _d_model * _token_num);
}

int main(int argc, char *argv[]) {

  // buffers for inner calc
  float *buf_Q = malloc(sizeof(float) * d_k * head * token_num);
  float *buf_K_T = malloc(sizeof(float) * token_num * d_k * head);
  float *buf_V = malloc(sizeof(float) * d_v * head * token_num);

  float **buf_queried_mh = malloc(sizeof(float *) * head);
  float *buf_queried_sh = malloc(sizeof(float) * token_num * token_num);
  float *buf_single_attentions = malloc(sizeof(float) * d_v * head * token_num);

  float *buf_queried_fumm = malloc(sizeof(float) * caching_row * token_num);
  float *buf_single_attentions_fumm = malloc(sizeof(float) * caching_row * d_v);

  for (int i=0; i<head; i++) {
    buf_queried_mh[i] = malloc(sizeof(float) * token_num * token_num);
  }

  // init input params
  matrix_init_rand(embedded, d_model * token_num);
  matrix_init_rand(W_Q, d_k * head * d_model);
  matrix_init_rand(W_K_T, d_model * d_k * head);
  matrix_init_rand(W_V, d_v * head * d_model);
  matrix_init_rand(W_O, d_model * d_v * head);
  
  clock_t start_time, end_time;

  /******************** CORRECTNESS TEST ********************/

  printf("[Test1 - Correctness among functions]\n");
  
  int test1_num = 32;
  int test1_score1 = 0;
  int test1_score2 = 0;

  for (int i=0; i<test1_num; i++) {
    // init memories
    matrix_init_rand(embedded, d_model * token_num);
    matrix_init_rand(W_Q, d_k * head * d_model);
    matrix_init_rand(W_K_T, d_model * d_k * head);
    matrix_init_rand(W_V, d_v * head * d_model);
    matrix_init_rand(W_O, d_model * d_v * head);
    matrix_init_zero(output_hlbl, d_model * token_num);
    matrix_init_zero(output_slbl, d_model * token_num);
    matrix_init_zero(output_fumm, d_model * token_num);

    // calc
    multi_attention_hardlbl(
      embedded, W_Q, W_K_T, W_V, W_O, output_hlbl, 
      token_num, d_model, d_k, d_v, head, 
      buf_Q, buf_K_T, buf_V, buf_queried_mh, buf_single_attentions
    );

    multi_attention_softlbl(
      embedded, W_Q, W_K_T, W_V, W_O, output_slbl, 
      token_num, d_model, d_k, d_v, head,
      buf_Q, buf_K_T, buf_V, buf_queried_sh, buf_single_attentions
    );

    multi_attention_fumm(
      embedded, W_Q, W_K_T, W_V, W_O, output_fumm,
      token_num, d_model, d_k, d_v, head,
      buf_Q, buf_K_T, buf_V, buf_queried_fumm, buf_single_attentions_fumm, caching_row
    );


    test1_score1 += matrix_comp(output_hlbl, output_slbl, token_num, d_model);
    test1_score2 += matrix_comp(output_fumm, output_slbl, token_num, d_model);
  }

  printf("\tPARAMS\n");
  printf("\trepeated %'d times, d_model: %'d, token_num: %'d, d_k: %'d, d_v: %'d, caching_row: %'d\n", test1_num, d_model, token_num, d_k, d_v, caching_row);
  printf("\tRESULTS\n");
  printf(
    "\tHard LBL vs. Soft LBL: %d/%d"\
    "\tSoft LBL vs. FuMM: %d/%d\n",
    test1_score1, test1_num, test1_score2, test1_num
  );
  printf("\n");



  /******************** TIME TEST ********************/
  printf("[Test3 - Multi head attention, time]\n");
  
  int test3_num = 18;
  int test3_score_hlbl = 0;
  int test3_score_slbl = 0;
  int test3_score_fumm = 0;

  if (argc >= 2) {
    test3_num = atoi(argv[1]);
  }

  for (int i=0; i<test3_num; i++) {
    matrix_init_zero(output_hlbl, d_model * token_num);
    
    start_time = clock();
    multi_attention_hardlbl(
      embedded, W_Q, W_K_T, W_V, W_O, output_hlbl, 
      token_num, d_model, d_k, d_v, head, 
      buf_Q, buf_K_T, buf_V, buf_queried_mh, buf_single_attentions
    );
    end_time = clock();
    test3_score_hlbl += end_time - start_time;

  }
  for (int i=0; i<test3_num; i++) {
    matrix_init_zero(output_slbl, d_model * token_num);

    start_time = clock();
    multi_attention_softlbl(
      embedded, W_Q, W_K_T, W_V, W_O, output_slbl, 
      token_num, d_model, d_k, d_v, head,
      buf_Q, buf_K_T, buf_V, buf_queried_sh, buf_single_attentions
    );
    end_time = clock();
    test3_score_slbl += end_time - start_time;

  }
  for (int i=0; i<test3_num; i++) {
    matrix_init_zero(output_fumm, d_model * token_num);

    start_time = clock();
    multi_attention_fumm(
      embedded, W_Q, W_K_T, W_V, W_O, output_fumm,
      token_num, d_model, d_k, d_v, head,
      buf_Q, buf_K_T, buf_V, buf_queried_fumm, buf_single_attentions, caching_row
    );
    end_time = clock();
    test3_score_fumm += end_time - start_time;
  }

  printf("\tPARAMS\n");
  printf("\trepeated %'d times, d_model: %'d, token_num: %'d, d_k: %'d, d_v: %'d, caching_row: %'d\n", test3_num, d_model, token_num, d_k, d_v, caching_row);
  printf("\tRESULTS (avg)\n");
  printf(
    "\tHard LBL: %f(sec), Soft LBL: %f(sec), FuMM: %f(sec)\n", 
    (float)test3_score_hlbl / test3_num / CLOCKS_PER_SEC, 
    (float)test3_score_slbl / test3_num / CLOCKS_PER_SEC, 
    (float)test3_score_fumm / test3_num / CLOCKS_PER_SEC
  );
  printf("\n");

  // free buffers
  free(buf_Q);
  free(buf_K_T);
  free(buf_V);
  free(buf_queried_sh);
  for (int i=0; i<head; i++) free(buf_queried_mh[i]);
  free(buf_queried_mh);
  free(buf_single_attentions);

  return 0;
}

