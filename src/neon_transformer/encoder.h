#include <arm_neon.h>
#include <stdlib.h>
#include "operation.h"

#define EXPECTED_CACHE 64

void encoder_lbl(
    float32_t *input, 
    float32_t *W_Q, float32_t *W_K, float32_t *W_V, float32_t *W_O,
    float32_t *FC1, float32_t *bias1, float32_t *FC2, float32_t *bias2,
    float32_t *output, 
    uint32_t token_num, uint32_t d_model, uint32_t d_k, uint32_t d_v, uint32_t h, uint32_t d_ff
);

