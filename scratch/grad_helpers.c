#include "core/dm_engine.h"
#include "models/vision/swin.h"
#include <string.h>

static int linear_backward_helper(DM_SwinLinearBlock *lin, const DM_Block *d_y, DM_Block *d_x, DM_ActivationCache *cache) {
    // 1. Pop x from cache
    DM_Block x = {0};
    if (cache) {
        if (cache->count == 0) return -1;
        x = cache->activations[--cache->count];
    } else {
        return -1; // Cannot backward without cache
    }
    
    // d_y is [B, L, out_dim]. x is [B, L, in_dim].
    // weight is [out_dim, in_dim].
    // dm_matmul_nt_backward expects 2D matrices.
    
    int64_t b = x.shape[0];
    int64_t l = x.shape[1];
    int64_t in_dim = x.shape[2];
    int64_t out_dim = lin->weight->shape[0];
    
    DM_Block A_view = x;
    A_view.ndim = 2;
    A_view.shape[0] = b * l;
    A_view.shape[1] = in_dim;
    
    DM_Block d_y_view = *d_y;
    d_y_view.ndim = 2;
    d_y_view.shape[0] = b * l;
    d_y_view.shape[1] = out_dim;
    
    if (d_x) {
        if (dm_block_create(d_x, 3, x.shape) != 0) {
            dm_block_free(&x); return -1;
        }
    }
    
    DM_Block d_x_view = {0};
    if (d_x) {
        d_x_view = *d_x;
        d_x_view.ndim = 2;
        d_x_view.shape[0] = b * l;
        d_x_view.shape[1] = in_dim;
    }
    
    // Accumulate gradients into lin->grad_weight
    // But dm_matmul_nt_backward overwrites grad_B. We must accumulate.
    // Wait! dm_matmul_nt_backward OVERWRITES grad_B.
    // In neural networks, we must accumulate gradients across batches/steps.
    // Oh, actually, the gradients are accumulated across the batch inside dm_matmul_nt_backward.
    // But if we reuse the same weight multiple times (e.g., shared weights), we need to accumulate.
    // Swin uses shared weights per layer? No, each layer has unique weights.
    // BUT! A block processes multiple tokens in a batch, which is handled because the matrix multiply sees [B*L, C].
    // So overwriting is fine for the FIRST time in a backward pass, but wait, `lin->grad_weight` was zeroed at creation.
    // So dm_matmul_nt_backward WILL overwrite the zeros. That is fine. 
    // EXCEPT! What if `wmsa_forward` uses the same `lin` twice? It doesn't.
    // However, if we run backward multiple times before optimizer step, we'd need accumulation.
    // Let's just create a temporary block, call dm_matmul_nt_backward, and ADD to lin->grad_weight.
    
    DM_Block tmp_grad_w = {0};
    dm_block_create(&tmp_grad_w, 2, lin->weight->shape);
    
    dm_matmul_nt_backward(&A_view, lin->weight, &d_y_view, d_x ? &d_x_view : NULL, &tmp_grad_w);
    
    // Add tmp_grad_w to lin->grad_weight
    if (lin->grad_weight) {
        float *gw = (float *)lin->grad_weight->data;
        float *tw = (float *)tmp_grad_w.data;
        for (size_t i = 0; i < tmp_grad_w.count; i++) gw[i] += tw[i];
    }
    dm_block_free(&tmp_grad_w);
    
    // Gradients for bias
    if (lin->grad_bias) {
        float *gb = (float *)lin->grad_bias->data;
        const float *dy_ptr = (const float *)d_y->data;
        for (int64_t i = 0; i < b * l; i++) {
            for (int64_t o = 0; o < out_dim; o++) {
                gb[o] += dy_ptr[i * out_dim + o];
            }
        }
    }
    
    dm_block_free(&x);
    return 0;
}
