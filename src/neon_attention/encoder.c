#include "encoder.h"

void encoder_lbl(
    float32_t* input,       // [d_model, token_num]
    float32_t* W_Q,         // [d_model, d_k * h]
    float32_t* W_K,         // [d_model, d_k * h]
    float32_t* W_V,         // [d_model, d_v * h]
    float32_t* W_O,         // [d_model, d_v * h]
    float32_t* FC1,         // [d_model, d_ff]
    float32_t *bias1,       // [d_ff]
    float32_t* FC2,         // [d_ff, d_model]
    float32_t *bias2,       // [d_model]
    float32_t *output,       // [d_model, token_num]
    uint32_t token_num, uint32_t d_model, uint32_t d_k, uint32_t d_v, uint32_t h, uint32_t d_ff
) {
    float32_t *Q, *K, *QKT, *V, *attention, *mh_attention, *sub1, *ff_hidden, *sub2;
    Q = malloc(sizeof(float32_t) * token_num * d_k * h);            // (token_num, d_k * h) matrix
    K = malloc(sizeof(float32_t) * token_num * d_k * h);            // (token_num, d_k * h) matrix
    QKT = malloc(sizeof(float32_t) * token_num * token_num * h);    // (token_num, token_num * h) matrix
    V = malloc(sizeof(float32_t) * token_num * d_v * h);            // (token_num, d_v * h) matrix
    attention = malloc(sizeof(float32_t) * token_num * d_v * h);    // (token_num, d_v * h) matrix
    mh_attention = malloc(sizeof(float32_t) * token_num * d_model); // (token_num, d_model) matrix

    float32_t **Q_heads = malloc(sizeof(float32_t *) * h);
    float32_t **K_heads = malloc(sizeof(float32_t *) * h);
    float32_t **QKT_heads = malloc(sizeof(float32_t *) * h);
    float32_t **V_heads = malloc(sizeof(float32_t *) * h);
    float32_t **attention_heads = malloc(sizeof(float32_t *) * h);

    for (int i=0; i<h; i++) {
        Q_heads[i] = Q + i * token_num * d_k;
        K_heads[i] = K + i * token_num * d_k;
        QKT_heads[i] = QKT + i * token_num * token_num;
        V_heads[i] = V + i * token_num * d_v;
        attention_heads[i] = attention + i * token_num * d_v;
    }

    /* Generate Q, K, V */
    mTm_simd(input, W_Q, Q, token_num, d_k * h, d_model);
    mTm_simd(input, W_K, K, token_num, d_k * h, d_model);
    mTm_simd(input, W_V, V, token_num, d_v * h, d_model);

    /* Dot product queries & keys */
    for (int i=0; i<h; i++) {
        mmT_simd(Q_heads[i], K_heads[i], QKT_heads[i], token_num, token_num, d_k);
    }

    /* Softmax and scale */
    for (int i=0; i<d_k * h; i++) {
        uint32_t column_idx = i * token_num;
        softmax_inplace_simd(QKT + column_idx, token_num);
        scale_inplace_simd(QKT + column_idx, sqrt(d_k), token_num);
    }

    /* Weighted(QKT) sum of values */
    for (int i=0; i<h; i++) {
        mTm_simd(V_heads[i], QKT_heads[i], attention_heads[i], d_v, token_num, token_num);
    }

    /* Integrate to multi-head attention */
    mmT_simd(attention, W_O, mh_attention, token_num, d_model, d_v * h);

    /* Add & Normalize */
    sub1 = mh_attention;
    addi_inplace_simd(input, sub1, token_num * d_model);
    normalize_inplace_simd(sub1, token_num * d_model);

    /* FFN */
    ff_hidden = malloc(sizeof(float32_t) * d_ff * token_num);   // (d_ff, token_num) matrix

    mm_T_simd(sub1, FC1, ff_hidden, token_num, d_ff, d_model);
    // sub1:      [token_num, d_model] matrix
    // FC1:       [d_model, d_ff] matrix
    // ff_hidden: [d_ff, token_num] matrix

    relu_inplace_simd(ff_hidden, d_ff * token_num);

    mTm_simd(FC2, ff_hidden, output, d_model, token_num, d_ff);


    /* Clear memories */
    free(Q);
    free(K);
    free(QKT);
    free(V);
    free(attention);
    free(mh_attention);
    free(Q_heads);
    free(K_heads);
    free(QKT_heads);
    free(V_heads);
    free(attention_heads);
    free(ff_hidden);

}

void attention_lbl(
    float32_t* Q, float32_t* K, float32_t* V,
    uint32_t token_num, uint32_t d_k, uint32_t d_v
) {

}