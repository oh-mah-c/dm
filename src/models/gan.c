#include "models/gan.h"
#include <stdlib.h>
#include <math.h>

static float rand_normal(float mean, float stddev) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    if (u1 <= 1e-7f) u1 = 1e-7f;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
    return z0 * stddev + mean;
}

static void alloc_and_init(float **w, float **g, float **v, int size) {
    *w = malloc(sizeof(float) * size);
    *g = calloc(size, sizeof(float));
    *v = calloc(size, sizeof(float));
    for (int i = 0; i < size; i++) (*w)[i] = rand_normal(0.0f, 0.01f);
}

void dm_gan_init(DM_GAN *gan, int input_dim, int g_hidden, int noise_dim, int d_hidden, int maxout_k, float drop_prob, float lr, float momentum, int nesterov) {
    gan->input_dim = input_dim;
    gan->g_hidden_dim = g_hidden;
    gan->noise_dim = noise_dim;
    gan->d_hidden_dim = d_hidden;
    gan->maxout_k = maxout_k;
    gan->drop_prob = drop_prob;
    gan->lr = lr;
    gan->momentum = momentum;
    gan->nesterov = nesterov;
    
    alloc_and_init(&gan->g_w1, &gan->g_gw1, &gan->g_vw1, g_hidden * noise_dim);
    alloc_and_init(&gan->g_b1, &gan->g_gb1, &gan->g_vb1, g_hidden);
    alloc_and_init(&gan->g_w2, &gan->g_gw2, &gan->g_vw2, input_dim * g_hidden);
    alloc_and_init(&gan->g_b2, &gan->g_gb2, &gan->g_vb2, input_dim);
    
    alloc_and_init(&gan->d_w1, &gan->d_gw1, &gan->d_vw1, d_hidden * maxout_k * input_dim);
    alloc_and_init(&gan->d_b1, &gan->d_gb1, &gan->d_vw1, d_hidden * maxout_k);
    alloc_and_init(&gan->d_w2, &gan->d_gw2, &gan->d_vw2, 1 * d_hidden);
    alloc_and_init(&gan->d_b2, &gan->d_gb2, &gan->d_vb2, 1);
}

void dm_gan_free(DM_GAN *gan) {
    free(gan->g_w1); free(gan->g_b1); free(gan->g_w2); free(gan->g_b2);
    free(gan->g_gw1); free(gan->g_gb1); free(gan->g_gw2); free(gan->g_gb2);
    free(gan->g_vw1); free(gan->g_vb1); free(gan->g_vw2); free(gan->g_vb2);
    
    free(gan->d_w1); free(gan->d_b1); free(gan->d_w2); free(gan->d_b2);
    free(gan->d_gw1); free(gan->d_gb1); free(gan->d_gw2); free(gan->d_gb2);
    free(gan->d_vw1); free(gan->d_vb1); free(gan->d_vw2); free(gan->d_vb2);
}

void dm_gan_generate(DM_GAN *gan, const DM_Block *z, DM_Block *out) {
    int batch_size = DM_NCHW_N(z);
    DM_Block g_h1;
    dm_block_create(&g_h1, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->g_hidden_dim, 1, 1});
    
    dm_linear(z, &g_h1, gan->g_w1, gan->g_b1, gan->g_hidden_dim);
    dm_relu(&g_h1);
    
    dm_linear(&g_h1, out, gan->g_w2, gan->g_b2, gan->input_dim);
    dm_sigmoid_inplace(out);
    
    dm_block_free(&g_h1);
}

static float forward_discriminator(DM_GAN *gan, const DM_Block *x, int *argmax, int *mask, 
                                   DM_Block *d_h1_pre, DM_Block *d_h1_max, DM_Block *d_h1_drop, DM_Block *d_out) {
    int batch_size = DM_NCHW_N(x);
    dm_linear(x, d_h1_pre, gan->d_w1, gan->d_b1, gan->d_hidden_dim * gan->maxout_k);
    
    dm_maxout(d_h1_pre, d_h1_max, gan->maxout_k, argmax);
    
    if (mask) {
        dm_dropout(d_h1_max, d_h1_drop, gan->drop_prob, mask);
    } else {
        // inference mode
        size_t count = (d_h1_max)->count;
        for (size_t i = 0; i < count; i++) ((float*)((float*)d_h1_drop->data))[i] = ((float*)((float*)d_h1_max->data))[i];
    }
    
    dm_linear(d_h1_drop, d_out, gan->d_w2, gan->d_b2, 1);
    dm_sigmoid_inplace(d_out);
    
    float loss = 0.0f;
    return loss; // calculated by caller
}

float dm_gan_train_d_step(DM_GAN *gan, const DM_Block *real_x, const DM_Block *z) {
    int batch_size = DM_NCHW_N(real_x);
    float loss = 0.0f;
    
    // G(z)
    DM_Block fake_x, g_h1;
    dm_block_create(&g_h1, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->g_hidden_dim, 1, 1});
    dm_block_create(&fake_x, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->input_dim, 1, 1});
    dm_linear(z, &g_h1, gan->g_w1, gan->g_b1, gan->g_hidden_dim);
    dm_relu(&g_h1);
    dm_linear(&g_h1, &fake_x, gan->g_w2, gan->g_b2, gan->input_dim);
    dm_sigmoid_inplace(&fake_x);
    
    // Tensors for D
    DM_Block d_h1_pre, d_h1_max, d_h1_drop, d_out, d_grad_out, d_grad_drop, d_grad_max, d_grad_pre, d_grad_x;
    dm_block_create(&d_h1_pre, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim * gan->maxout_k, 1, 1});
    dm_block_create(&d_h1_max, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim, 1, 1});
    dm_block_create(&d_h1_drop, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim, 1, 1});
    dm_block_create(&d_out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, 1, 1, 1});
    
    dm_block_create(&d_grad_out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, 1, 1, 1});
    dm_block_create(&d_grad_drop, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim, 1, 1});
    dm_block_create(&d_grad_max, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim, 1, 1});
    dm_block_create(&d_grad_pre, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim * gan->maxout_k, 1, 1});
    dm_block_create(&d_grad_x, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->input_dim, 1, 1});
    
    int *argmax = malloc(sizeof(int) * batch_size * gan->d_hidden_dim);
    int *mask = malloc(sizeof(int) * batch_size * gan->d_hidden_dim);
    
    // D(real_x)
    forward_discriminator(gan, real_x, argmax, mask, &d_h1_pre, &d_h1_max, &d_h1_drop, &d_out);
    for (int i = 0; i < batch_size; i++) {
        float p = ((float*)((float*)d_out.data))[i];
        if (p < 1e-7f) p = 1e-7f;
        loss -= logf(p);
        ((float*)((float*)d_grad_out.data))[i] = (p - 1.0f) / batch_size;
    }
    dm_linear_backward(&d_h1_drop, &d_grad_out, &d_grad_drop, gan->d_gw2, gan->d_gb2, gan->d_w2, 1);
    dm_dropout_backward(&d_grad_drop, &d_grad_max, gan->drop_prob, mask);
    dm_maxout_backward(&d_grad_max, &d_grad_pre, gan->maxout_k, argmax);
    dm_linear_backward(real_x, &d_grad_pre, NULL, gan->d_gw1, gan->d_gb1, gan->d_w1, gan->d_hidden_dim * gan->maxout_k);
    
    // D(fake_x)
    forward_discriminator(gan, &fake_x, argmax, mask, &d_h1_pre, &d_h1_max, &d_h1_drop, &d_out);
    for (int i = 0; i < batch_size; i++) {
        float p = ((float*)((float*)d_out.data))[i];
        if (p > 1.0f - 1e-7f) p = 1.0f - 1e-7f;
        loss -= logf(1.0f - p);
        ((float*)((float*)d_grad_out.data))[i] = (p - 0.0f) / batch_size;
    }
    dm_linear_backward(&d_h1_drop, &d_grad_out, &d_grad_drop, gan->d_gw2, gan->d_gb2, gan->d_w2, 1);
    dm_dropout_backward(&d_grad_drop, &d_grad_max, gan->drop_prob, mask);
    dm_maxout_backward(&d_grad_max, &d_grad_pre, gan->maxout_k, argmax);
    dm_linear_backward(&fake_x, &d_grad_pre, NULL, gan->d_gw1, gan->d_gb1, gan->d_w1, gan->d_hidden_dim * gan->maxout_k);
    
    dm_sgd_momentum_step(gan->d_w1, gan->d_gw1, gan->d_vw1, gan->d_hidden_dim * gan->maxout_k * gan->input_dim, gan->lr, gan->momentum, 0.0f, gan->nesterov);
    dm_sgd_momentum_step(gan->d_b1, gan->d_gb1, gan->d_vb1, gan->d_hidden_dim * gan->maxout_k, gan->lr, gan->momentum, 0.0f, gan->nesterov);
    dm_sgd_momentum_step(gan->d_w2, gan->d_gw2, gan->d_vw2, 1 * gan->d_hidden_dim, gan->lr, gan->momentum, 0.0f, gan->nesterov);
    dm_sgd_momentum_step(gan->d_b2, gan->d_gb2, gan->d_vb2, 1, gan->lr, gan->momentum, 0.0f, gan->nesterov);
    
    free(argmax); free(mask);
    dm_block_free(&fake_x); dm_block_free(&g_h1);
    dm_block_free(&d_h1_pre); dm_block_free(&d_h1_max); dm_block_free(&d_h1_drop); dm_block_free(&d_out);
    dm_block_free(&d_grad_out); dm_block_free(&d_grad_drop); dm_block_free(&d_grad_max); dm_block_free(&d_grad_pre); dm_block_free(&d_grad_x);
    
    return loss / batch_size;
}

float dm_gan_train_g_step(DM_GAN *gan, const DM_Block *z) {
    int batch_size = DM_NCHW_N(z);
    float loss = 0.0f;
    
    // G(z)
    DM_Block fake_x, g_h1;
    dm_block_create(&g_h1, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->g_hidden_dim, 1, 1});
    dm_block_create(&fake_x, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->input_dim, 1, 1});
    dm_linear(z, &g_h1, gan->g_w1, gan->g_b1, gan->g_hidden_dim);
    dm_relu(&g_h1);
    dm_linear(&g_h1, &fake_x, gan->g_w2, gan->g_b2, gan->input_dim);
    dm_sigmoid_inplace(&fake_x);
    
    DM_Block d_h1_pre, d_h1_max, d_h1_drop, d_out, d_grad_out, d_grad_drop, d_grad_max, d_grad_pre, d_grad_x;
    dm_block_create(&d_h1_pre, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim * gan->maxout_k, 1, 1});
    dm_block_create(&d_h1_max, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim, 1, 1});
    dm_block_create(&d_h1_drop, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim, 1, 1});
    dm_block_create(&d_out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, 1, 1, 1});
    
    dm_block_create(&d_grad_out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, 1, 1, 1});
    dm_block_create(&d_grad_drop, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim, 1, 1});
    dm_block_create(&d_grad_max, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim, 1, 1});
    dm_block_create(&d_grad_pre, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->d_hidden_dim * gan->maxout_k, 1, 1});
    dm_block_create(&d_grad_x, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->input_dim, 1, 1});
    
    int *argmax = malloc(sizeof(int) * batch_size * gan->d_hidden_dim);
    int *mask = malloc(sizeof(int) * batch_size * gan->d_hidden_dim);
    
    // D(fake_x)
    forward_discriminator(gan, &fake_x, argmax, mask, &d_h1_pre, &d_h1_max, &d_h1_drop, &d_out);
    for (int i = 0; i < batch_size; i++) {
        float p = ((float*)((float*)d_out.data))[i];
        if (p < 1e-7f) p = 1e-7f;
        loss -= logf(p);
        ((float*)((float*)d_grad_out.data))[i] = (p - 1.0f) / batch_size;
    }
    
    // Backprop through D (no parameter updates)
    float *dummy_gw2 = calloc(gan->d_hidden_dim, sizeof(float));
    float *dummy_gb2 = calloc(1, sizeof(float));
    dm_linear_backward(&d_h1_drop, &d_grad_out, &d_grad_drop, dummy_gw2, dummy_gb2, gan->d_w2, 1);
    free(dummy_gw2); free(dummy_gb2);
    
    dm_dropout_backward(&d_grad_drop, &d_grad_max, gan->drop_prob, mask);
    dm_maxout_backward(&d_grad_max, &d_grad_pre, gan->maxout_k, argmax);
    
    float *dummy_gw1 = calloc(gan->d_hidden_dim * gan->maxout_k * gan->input_dim, sizeof(float));
    float *dummy_gb1 = calloc(gan->d_hidden_dim * gan->maxout_k, sizeof(float));
    dm_linear_backward(&fake_x, &d_grad_pre, &d_grad_x, dummy_gw1, dummy_gb1, gan->d_w1, gan->d_hidden_dim * gan->maxout_k);
    free(dummy_gw1); free(dummy_gb1);
    
    // Backprop through G
    DM_Block g_grad_h1, g_grad_pre_relu;
    dm_block_create(&g_grad_h1, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->g_hidden_dim, 1, 1});
    dm_block_create(&g_grad_pre_relu, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch_size, gan->g_hidden_dim, 1, 1});
    
    for (int i = 0; i < batch_size * gan->input_dim; i++) {
        float o = ((float*)((float*)fake_x.data))[i];
        ((float*)((float*)d_grad_x.data))[i] = ((float*)((float*)d_grad_x.data))[i] * o * (1.0f - o);
    }
    dm_linear_backward(&g_h1, &d_grad_x, &g_grad_h1, gan->g_gw2, gan->g_gb2, gan->g_w2, gan->input_dim);
    dm_relu_backward(&g_h1, &g_grad_h1, &g_grad_pre_relu);
    dm_linear_backward(z, &g_grad_pre_relu, NULL, gan->g_gw1, gan->g_gb1, gan->g_w1, gan->g_hidden_dim);
    
    dm_sgd_momentum_step(gan->g_w1, gan->g_gw1, gan->g_vw1, gan->g_hidden_dim * gan->noise_dim, gan->lr, gan->momentum, 0.0f, gan->nesterov);
    dm_sgd_momentum_step(gan->g_b1, gan->g_gb1, gan->g_vb1, gan->g_hidden_dim, gan->lr, gan->momentum, 0.0f, gan->nesterov);
    dm_sgd_momentum_step(gan->g_w2, gan->g_gw2, gan->g_vw2, gan->input_dim * gan->g_hidden_dim, gan->lr, gan->momentum, 0.0f, gan->nesterov);
    dm_sgd_momentum_step(gan->g_b2, gan->g_gb2, gan->g_vb2, gan->input_dim, gan->lr, gan->momentum, 0.0f, gan->nesterov);
    
    free(argmax); free(mask);
    dm_block_free(&fake_x); dm_block_free(&g_h1);
    dm_block_free(&d_h1_pre); dm_block_free(&d_h1_max); dm_block_free(&d_h1_drop); dm_block_free(&d_out);
    dm_block_free(&d_grad_out); dm_block_free(&d_grad_drop); dm_block_free(&d_grad_max); dm_block_free(&d_grad_pre); dm_block_free(&d_grad_x);
    dm_block_free(&g_grad_h1); dm_block_free(&g_grad_pre_relu);
    
    return loss / batch_size;
}
