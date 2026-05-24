#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "models/mlp_train.h"
#include "core/dm_engine.h"

static float random_weight() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

int dm_mlp_train_cli(int argc, char **argv) {
    (void)argc;
    (void)argv;
    
    printf("--- Phase 2: XOR MLP Training ---\n");
    
    DM_TrainingContext ctx;
    if (dm_training_context_init(&ctx, 4) != 0) {
        fprintf(stderr, "Failed to init training context.\n");
        return -1;
    }
    
    // W1: [16, 2], b1: [16]
    // W2: [1, 16], b2: [1]
    for (int i = 0; i < 4; i++) {
        ctx.params[i].param = (DM_Parameter *)calloc(1, sizeof(DM_Parameter));
        ctx.params[i].grad = (DM_Gradient *)calloc(1, sizeof(DM_Gradient));
    }
    
    DM_Parameter *W1 = ctx.params[0].param;
    DM_Gradient *gW1 = ctx.params[0].grad;
    DM_Parameter *b1 = ctx.params[1].param;
    DM_Gradient *gb1 = ctx.params[1].grad;
    DM_Parameter *W2 = ctx.params[2].param;
    DM_Gradient *gW2 = ctx.params[2].grad;
    DM_Parameter *b2 = ctx.params[3].param;
    DM_Gradient *gb2 = ctx.params[3].grad;
    
    dm_block_create(&W1->weight, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){16, 2});
    dm_block_create(&b1->weight, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 1, (int64_t[]){16});
    dm_block_create(&W2->weight, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){1, 16});
    dm_block_create(&b2->weight, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 1, (int64_t[]){1});
    
    dm_block_create(&gW1->grad, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){16, 2});
    dm_block_create(&gb1->grad, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 1, (int64_t[]){16});
    dm_block_create(&gW2->grad, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){1, 16});
    dm_block_create(&gb2->grad, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 1, (int64_t[]){1});
    
    for (size_t i = 0; i < W1->weight.count; i++) ((float*)W1->weight.data)[i] = random_weight() * 0.5f;
    for (size_t i = 0; i < b1->weight.count; i++) ((float*)b1->weight.data)[i] = 0.0f;
    for (size_t i = 0; i < W2->weight.count; i++) ((float*)W2->weight.data)[i] = random_weight() * 0.5f;
    for (size_t i = 0; i < b2->weight.count; i++) ((float*)b2->weight.data)[i] = 0.0f;
    
    ctx.learning_rate = 0.05f; // Fast learning rate for XOR
    
    // Dataset
    DM_Block X, Y_true;
    dm_block_create(&X, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){4, 2});
    dm_block_create(&Y_true, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){4, 1});
    
    float x_data[8] = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    float y_data[4] = {0.0f, 1.0f, 1.0f, 0.0f};
    memcpy(X.data, x_data, 8 * sizeof(float));
    memcpy(Y_true.data, y_data, 4 * sizeof(float));
    
    DM_ActivationCache cache;
    dm_activation_cache_init(&cache, 10);
    
    DM_Block Z1, A1, Z2, Y_pred;
    dm_block_create(&Z1, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){4, 16});
    dm_block_create(&A1, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){4, 16});
    dm_block_create(&Z2, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){4, 1});
    dm_block_create(&Y_pred, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){4, 1});
    
    DM_Block dZ2, dA1, dZ1;
    dm_block_create(&dZ2, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){4, 1});
    dm_block_create(&dA1, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){4, 16});
    dm_block_create(&dZ1, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){4, 16});

    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward Pass
        // 1. Z1 = X @ W1^T
        dm_matmul_nt(&X, &W1->weight, &Z1);
        // Add bias b1
        float *z1p = (float *)Z1.data;
        float *b1p = (float *)b1->weight.data;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 16; j++) {
                z1p[i * 16 + j] += b1p[j];
            }
        }
        
        // Push Z1 to cache before activation
        dm_activation_cache_push(&cache, &Z1);
        
        // 2. A1 = GELU(Z1)
        memcpy(A1.data, Z1.data, Z1.bytes);
        dm_gelu_inplace((float*)A1.data, (int)A1.count);
        
        // Push A1 to cache
        dm_activation_cache_push(&cache, &A1);
        
        // 3. Z2 = A1 @ W2^T
        dm_matmul_nt(&A1, &W2->weight, &Z2);
        // Add bias b2
        float *z2p = (float *)Z2.data;
        float *b2p = (float *)b2->weight.data;
        for (int i = 0; i < 4; i++) {
            z2p[i] += b2p[0];
        }
        
        // 4. Y_pred = Sigmoid(Z2)
        memcpy(Y_pred.data, Z2.data, Z2.bytes);
        dm_sigmoid_inplace(&Y_pred);
        
        // Compute MSE Loss
        float loss = 0.0f;
        float *ypredp = (float *)Y_pred.data;
        float *ytruep = (float *)Y_true.data;
        for (int i = 0; i < 4; i++) {
            float diff = ypredp[i] - ytruep[i];
            loss += diff * diff;
            // dL/dZ2 for Sigmoid with MSE is (y_pred - y_true) * y_pred * (1 - y_pred)
            // Wait, standard backprop: dL/dY_pred = 2 * (y_pred - y_true) / N
            // dY_pred/dZ2 = y_pred * (1 - y_pred)
            float dL_dY = 2.0f * diff / 4.0f;
            ((float*)dZ2.data)[i] = dL_dY * ypredp[i] * (1.0f - ypredp[i]);
        }
        
        if (epoch % 100 == 0 || epoch == epochs - 1) {
            printf("Epoch %d, Loss: %f\n", epoch, loss);
        }
        
        // Backward Pass
        // 1. dL/dW2 and dL/db2, dL/dA1
        // dZ2 is [4, 1]. A1 is [4, 16].
        // dm_matmul_nt_backward computes grad_A (dA1) and grad_B (dW2)
        // dA1 = dZ2 @ W2, dW2 = dZ2^T @ A1
        memset(gW2->grad.data, 0, gW2->grad.bytes);
        dm_matmul_nt_backward(&A1, &W2->weight, &dZ2, &dA1, &gW2->grad);
        
        // db2 = sum(dZ2) over batch
        float db2_val = 0.0f;
        for (int i = 0; i < 4; i++) db2_val += ((float*)dZ2.data)[i];
        ((float*)gb2->grad.data)[0] = db2_val;
        
        // 2. dL/dZ1 = dL/dA1 * dGELU(Z1)
        // We pop Z1 from cache
        DM_Block *cached_Z1 = &cache.activations[0];
        dm_gelu_backward(cached_Z1, &dA1, &dZ1);
        
        // 3. dL/dW1 and dL/db1
        memset(gW1->grad.data, 0, gW1->grad.bytes);
        dm_matmul_nt_backward(&X, &W1->weight, &dZ1, NULL, &gW1->grad);
        
        memset(gb1->grad.data, 0, gb1->grad.bytes);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 16; j++) {
                ((float*)gb1->grad.data)[j] += ((float*)dZ1.data)[i * 16 + j];
            }
        }
        
        // Optimizer Step (Adam)
        for (int i = 0; i < 4; i++) {
            DM_TrainableParam *tp = &ctx.params[i];
            DM_OptimizerState *os = &ctx.opt_states[i];
            if (os->m.count == 0) {
                dm_block_create(&os->m, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, tp->param->weight.ndim, tp->param->weight.shape);
                dm_block_create(&os->v, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, tp->param->weight.ndim, tp->param->weight.shape);
                memset(os->m.data, 0, os->m.bytes);
                memset(os->v.data, 0, os->v.bytes);
            }
            dm_adam_step((float*)tp->param->weight.data, (float*)tp->grad->grad.data, 
                         (float*)os->m.data, (float*)os->v.data,
                         (int)tp->param->weight.count, ctx.learning_rate,
                         ctx.beta1, ctx.beta2, ctx.eps, ctx.weight_decay, epoch + 1);
        }
        
        // Clear cache for next epoch
        for (size_t i = 0; i < cache.count; i++) dm_block_free(&cache.activations[i]);
        cache.count = 0;
    }
    
    printf("Final Predictions:\n");
    for (int i = 0; i < 4; i++) {
        printf("X=[%.0f, %.0f] -> Y_pred=%.4f (True=%.0f)\n", 
               ((float*)X.data)[i*2], ((float*)X.data)[i*2+1], 
               ((float*)Y_pred.data)[i], ((float*)Y_true.data)[i]);
    }
    
    dm_activation_cache_free(&cache);
    dm_block_free(&X);
    dm_block_free(&Y_true);
    dm_block_free(&Z1);
    dm_block_free(&A1);
    dm_block_free(&Z2);
    dm_block_free(&Y_pred);
    dm_block_free(&dZ2);
    dm_block_free(&dA1);
    dm_block_free(&dZ1);
    dm_training_context_free(&ctx);
    for (int i = 0; i < 4; i++) {
        free(ctx.params[i].param);
        free(ctx.params[i].grad);
    }
    
    return 0;
}
