#include "decoder.h"

void decoder_lbl(
    float32_t *input_embedded, float32_t *input_encoder,
    float32_t *W1_Q, float32_t *W1_K, float32_t *W1_V, float32_t *W1_O,
    float32_t *W2_Q, float32_t *W2_K, float32_t *W2_V, float32_t *W2_O,
    float32_t *FC1, float32_t *bias1, float32_t *FC2, float32_t *bias2,
    float32_t *output, 
    uint32_t token_num, uint32_t d_model, uint32_t d_k, uint32_t d_v, uint32_t h, uint32_t d_ff
) {
    float32_t *Q, *K, *QKT, *V, *attention, *mh_attention, *sub1_output, *sub2_output, *ff_hidden;
    Q = malloc(sizeof(float32_t) * token_num * d_k * h);            // (token_num, d_k * h) matrix
    K = malloc(sizeof(float32_t) * token_num * d_k * h);            // (token_num, d_k * h) matrix
    QKT = malloc(sizeof(float32_t) * token_num * token_num * h);    // (token_num, token_num * h) matrix
    V = malloc(sizeof(float32_t) * token_num * d_v * h);            // (token_num, d_v * h) matrix
    attention = malloc(sizeof(float32_t) * token_num * d_v * h);    // (token_num, d_v * h) matrix
    mh_attention = malloc(sizeof(float32_t) * token_num * d_model); // (token_num, d_model) matrix
    sub2_output = malloc(sizeof(float32_t) * token_num * d_model);  // (token_num, d_model) matrix

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

    /************************ SUBLAYER 1 ************************/

    /* Generate Q, K, V */
    mTm_simd(input_embedded, W1_Q, Q, token_num, d_k * h, d_model);
    mTm_simd(input_embedded, W1_K, K, token_num, d_k * h, d_model);
    mTm_simd(input_embedded, W1_V, V, token_num, d_v * h, d_model);

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
    mmT_simd(attention, W1_O, mh_attention, token_num, d_model, d_v * h);

    /* Add & Normalize */
    sub1_output = mh_attention;
    addi_inplace_simd(input_embedded, sub1_output, token_num * d_model);
    normalize_inplace_simd(sub1_output, token_num * d_model);

    
    /************************ SUBLAYER 2 ************************/

    /* Generate Q, K, V */
    mTm_simd(input_encoder, W2_Q, Q, token_num, d_k * h, d_model);
    mTm_simd(input_encoder, W2_K, K, token_num, d_k * h, d_model);
    mTm_simd(sub1_output, W2_V, V, token_num, d_v * h, d_model);

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
    mmT_simd(attention, W2_O, sub2_output, token_num, d_model, d_v * h);

    /* Add & Normalize */
    addi_inplace_simd(sub1_output, sub2_output, token_num * d_model);
    normalize_inplace_simd(sub2_output, token_num * d_model);


    /************************ SUBLAYER 3 ************************/

    /* FFN */
    ff_hidden = malloc(sizeof(float32_t) * d_ff * token_num);   // (d_ff, token_num) matrix

    mm_T_simd(sub2_output, FC1, ff_hidden, token_num, d_ff, d_model);

    relu_inplace_simd(ff_hidden, d_ff * token_num);

    mTm_simd(FC2, ff_hidden, output, d_model, token_num, d_ff);

    /* Add & Normalize */
    addi_inplace_simd(sub2_output, output, token_num * d_model);
    normalize_inplace_simd(output, token_num * d_model);

}