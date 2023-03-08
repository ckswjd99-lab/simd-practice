#include <arm_neon.h>
#include <stdlib.h>
#include "operation.h"

#define EXPECTED_CACHE 64

void decoder_lbl(
    float32_t *input_embedded, float32_t *input_encoder,
    float32_t *W1_Q, float32_t *W1_K, float32_t *W1_V, float32_t *W1_O,
    float32_t *W2_Q, float32_t *W2_K, float32_t *W2_V, float32_t *W2_O,
    float32_t *FC1, float32_t *bias1, float32_t *FC2, float32_t *bias2,
    float32_t *output, 
    uint32_t token_num, uint32_t d_model, uint32_t d_k, uint32_t d_v, uint32_t h, uint32_t d_ff
);

