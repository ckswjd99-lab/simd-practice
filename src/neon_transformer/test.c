#include "utils.h"
#include "operation.h"
#include "encoder.h"
#include "decoder.h"

#include <time.h>

int main() {
    uint32_t token_num = 64;
    uint32_t d_model = 512;
    uint32_t d_k = 64;
    uint32_t d_v = 64;
    uint32_t h = 8;
    uint32_t d_ff = 2048;
    
    float32_t embedded[d_model * token_num];
    float32_t W_Q[d_model * d_k * h];
    float32_t W_K[d_model * d_k * h];
    float32_t W_V[d_model * d_v * h];
    float32_t W_O[d_model * d_v * h];
    float32_t *FC1 = malloc(sizeof(float32_t)* d_model * d_ff);
    float32_t bias1[d_ff];
    float32_t *FC2 = malloc(sizeof(float32_t) * d_ff * d_model);
    float32_t bias2[d_model];

    matrix_init_rand(embedded, d_model * token_num);
    matrix_init_rand(W_Q, d_model * d_k * h);
    matrix_init_rand(W_K, d_model * d_k * h);
    matrix_init_rand(W_V, d_model * d_v * h);
    matrix_init_rand(W_O, d_model * d_v * h);
    matrix_init_rand(FC1, d_model * d_ff);
    matrix_init_rand(bias1, d_ff);
    matrix_init_rand(FC2, d_ff * d_model);
    matrix_init_rand(bias2, d_model);

    encoder_lbl(embedded, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);
    encoder_lbl(embedded, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);
    encoder_lbl(embedded, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);
    encoder_lbl(embedded, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);
    encoder_lbl(embedded, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);
    encoder_lbl(embedded, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);

    decoder_lbl(embedded, embedded, W_Q, W_K, W_V, W_O, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);
    decoder_lbl(embedded, embedded, W_Q, W_K, W_V, W_O, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);
    decoder_lbl(embedded, embedded, W_Q, W_K, W_V, W_O, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);
    decoder_lbl(embedded, embedded, W_Q, W_K, W_V, W_O, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);
    decoder_lbl(embedded, embedded, W_Q, W_K, W_V, W_O, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);
    decoder_lbl(embedded, embedded, W_Q, W_K, W_V, W_O, W_Q, W_K, W_V, W_O, FC1, bias1, FC2, bias2, embedded, token_num, d_model, d_k, d_v, h, d_ff);

    free(FC1);
    free(FC2);

    /* CHECK MM CORRECTNESS */
    /*
    uint32_t d1 = 512;
    uint32_t d2 = 128;
    uint32_t d3 = 256;


    float32_t *src1     = malloc(sizeof(float32_t) * d1 * d2);
    float32_t *src2     = malloc(sizeof(float32_t) * d2 * d3);
    float32_t *src1_T   = malloc(sizeof(float32_t) * d2 * d1);
    float32_t *src2_T   = malloc(sizeof(float32_t) * d3 * d2);
    
    float32_t *dst1     = malloc(sizeof(float32_t) * d1 * d3);
    float32_t *dst2     = malloc(sizeof(float32_t) * d1 * d3);

    matrix_init_rand(src1, d1 * d2);
    matrix_init_rand(src2, d2 * d3);

    transpose(src1, src1_T, d1, d2);
    transpose(src2, src2_T, d2, d3);

    mm_simd(src1, src2, dst1, d1, d3, d2);
    mmT_simd(src1, src2_T, dst2, d1, d3, d2);
    
    printf("Compare:\n");
    printf("  src1 @ src2\n");
    printf("  src1 @T src2_T\n");

    printf("> same? : %d\n\n", matrix_comp(dst1, dst2, d1, d3));

    printf("Compare:\n");
    printf("  src1 @ src2\n");
    printf("  src1_T T@ src2\n");

    mTm_simd(src1_T, src2, dst2, d1, d3, d2);
    printf("> same? : %d\n\n", matrix_comp(dst1, dst2, d1, d3));
    */


    /*
    float32_t data[16];
    uint32_t test_loop = 1024*1024*16;
    
    clock_t start_time, end_time;
    
    start_time = clock();
    for (int i=0; i<test_loop; i++) {
        matrix_init_rand(data, 16);
        transpose_4x4_inplace(data, 4);
    }
    end_time = clock();
    printf("SISD: %ld\n", end_time - start_time);

    start_time = clock();
    for (int i=0; i<test_loop; i++) {
        matrix_init_rand(data, 16);
        transpose_4x4_inplace_simd(data, 4);
    }
    end_time = clock();
    printf("SIMD: %ld\n", end_time - start_time);
    */

}