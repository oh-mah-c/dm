#include "models/vae.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float rand_normal(float mean, float stddev) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    if (u1 <= 1e-7f) u1 = 1e-7f;
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return z0 * stddev + mean;
}

static void alloc_and_init(float **w, float **g, float **s, float **v, int size) {
    *w = malloc(sizeof(float) * size);
    *g = calloc(size, sizeof(float));
    *s = calloc(size, sizeof(float));
    *v = calloc(size, sizeof(float));
    for (int i = 0; i < size; i++) (*w)[i] = rand_normal(0.0f, 0.01f);
}

void dm_vae_init(DM_VAE *vae, int input_dim, int hidden_dim, int latent_dim, float lr) {
    vae->input_dim = input_dim;
    vae->hidden_dim = hidden_dim;
    vae->latent_dim = latent_dim;
    vae->lr = lr;
    vae->t = 0;
    
    // Encoder: input -> hidden
    alloc_and_init(&vae->enc_w1, &vae->g_enc_w1, &vae->s_enc_w1, &vae->v_enc_w1, hidden_dim * input_dim);
    alloc_and_init(&vae->enc_b1, &vae->g_enc_b1, &vae->s_enc_b1, &vae->v_enc_b1, hidden_dim);
    
    // Encoder: hidden -> 2*latent
    alloc_and_init(&vae->enc_w2, &vae->g_enc_w2, &vae->s_enc_w2, &vae->v_enc_w2, 2 * latent_dim * hidden_dim);
    alloc_and_init(&vae->enc_b2, &vae->g_enc_b2, &vae->s_enc_b2, &vae->v_enc_b2, 2 * latent_dim);
    
    // Decoder: latent -> hidden
    alloc_and_init(&vae->dec_w1, &vae->g_dec_w1, &vae->s_dec_w1, &vae->v_dec_w1, hidden_dim * latent_dim);
    alloc_and_init(&vae->dec_b1, &vae->g_dec_b1, &vae->s_dec_b1, &vae->v_dec_b1, hidden_dim);
    
    // Decoder: hidden -> input
    alloc_and_init(&vae->dec_w2, &vae->g_dec_w2, &vae->s_dec_w2, &vae->v_dec_w2, input_dim * hidden_dim);
    alloc_and_init(&vae->dec_b2, &vae->g_dec_b2, &vae->s_dec_b2, &vae->v_dec_b2, input_dim);
}

void dm_vae_free(DM_VAE *vae) {
    free(vae->enc_w1); free(vae->enc_b1); free(vae->enc_w2); free(vae->enc_b2);
    free(vae->dec_w1); free(vae->dec_b1); free(vae->dec_w2); free(vae->dec_b2);
    free(vae->g_enc_w1); free(vae->g_enc_b1); free(vae->g_enc_w2); free(vae->g_enc_b2);
    free(vae->g_dec_w1); free(vae->g_dec_b1); free(vae->g_dec_w2); free(vae->g_dec_b2);
    free(vae->s_enc_w1); free(vae->s_enc_b1); free(vae->s_enc_w2); free(vae->s_enc_b2);
    free(vae->s_dec_w1); free(vae->s_dec_b1); free(vae->s_dec_w2); free(vae->s_dec_b2);
    free(vae->v_enc_w1); free(vae->v_enc_b1); free(vae->v_enc_w2); free(vae->v_enc_b2);
    free(vae->v_dec_w1); free(vae->v_dec_b1); free(vae->v_dec_w2); free(vae->v_dec_b2);
}

void dm_vae_encode(DM_VAE *vae, const DM_Block *x, DM_Block *mean, DM_Block *logvar) {
    DM_Block h1;
    dm_linear(x, &h1, vae->enc_w1, vae->enc_b1, vae->hidden_dim);
    dm_tanh_inplace(&h1);
    
    DM_Block h2;
    dm_linear(&h1, &h2, vae->enc_w2, vae->enc_b2, 2 * vae->latent_dim);
    
    dm_block_create(mean, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(x), vae->latent_dim, 1, 1});
    dm_block_create(logvar, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(x), vae->latent_dim, 1, 1});
    
    for (int n = 0; n < DM_NCHW_N(x); n++) {
        for (int i = 0; i < vae->latent_dim; i++) {
            dm_tensor_set(mean, n, i, 0, 0, dm_tensor_get(&h2, n, i, 0, 0));
            dm_tensor_set(logvar, n, i, 0, 0, dm_tensor_get(&h2, n, vae->latent_dim + i, 0, 0));
        }
    }
    
    dm_block_free(&h1);
    dm_block_free(&h2);
}

void dm_vae_decode(DM_VAE *vae, const DM_Block *z, DM_Block *out) {
    DM_Block h1;
    dm_linear(z, &h1, vae->dec_w1, vae->dec_b1, vae->hidden_dim);
    dm_tanh_inplace(&h1);
    
    dm_linear(&h1, out, vae->dec_w2, vae->dec_b2, vae->input_dim);
    dm_sigmoid_inplace(out);
    
    dm_block_free(&h1);
}

float dm_vae_train_step(DM_VAE *vae, const DM_Block *x) {
    int batch = DM_NCHW_N(x);
    
    // --- FORWARD PASS ---
    DM_Block enc_h1;
    dm_linear(x, &enc_h1, vae->enc_w1, vae->enc_b1, vae->hidden_dim);
    dm_tanh_inplace(&enc_h1);
    
    DM_Block enc_h2;
    dm_linear(&enc_h1, &enc_h2, vae->enc_w2, vae->enc_b2, 2 * vae->latent_dim);
    
    DM_Block mean, logvar, z, eps;
    dm_block_create(&mean, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch, vae->latent_dim, 1, 1});
    dm_block_create(&logvar, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch, vae->latent_dim, 1, 1});
    dm_block_create(&z, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch, vae->latent_dim, 1, 1});
    dm_block_create(&eps, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch, vae->latent_dim, 1, 1});
    
    float kl_loss = 0.0f;
    for (int n = 0; n < batch; n++) {
        for (int i = 0; i < vae->latent_dim; i++) {
            float m = dm_tensor_get(&enc_h2, n, i, 0, 0);
            float lv = dm_tensor_get(&enc_h2, n, vae->latent_dim + i, 0, 0);
            dm_tensor_set(&mean, n, i, 0, 0, m);
            dm_tensor_set(&logvar, n, i, 0, 0, lv);
            
            float e = rand_normal(0.0f, 1.0f);
            dm_tensor_set(&eps, n, i, 0, 0, e);
            dm_tensor_set(&z, n, i, 0, 0, m + expf(0.5f * lv) * e);
            
            kl_loss += -0.5f * (1.0f + lv - m * m - expf(lv));
        }
    }
    kl_loss /= batch;
    
    DM_Block dec_h1;
    dm_linear(&z, &dec_h1, vae->dec_w1, vae->dec_b1, vae->hidden_dim);
    dm_tanh_inplace(&dec_h1);
    
    DM_Block out;
    dm_linear(&dec_h1, &out, vae->dec_w2, vae->dec_b2, vae->input_dim);
    dm_sigmoid_inplace(&out);
    
    float recon_loss = 0.0f;
    DM_Block g_out;
    dm_block_create(&g_out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch, vae->input_dim, 1, 1});
    
    for (int n = 0; n < batch; n++) {
        for (int i = 0; i < vae->input_dim; i++) {
            float target = dm_tensor_get(x, n, i, 0, 0);
            float pred = dm_tensor_get(&out, n, i, 0, 0);
            if (pred <= 1e-7f) pred = 1e-7f;
            if (pred >= 1.0f - 1e-7f) pred = 1.0f - 1e-7f;
            
            recon_loss -= target * logf(pred) + (1.0f - target) * logf(1.0f - pred);
            
            // dL/d(out_pre_sigmoid) for BCE + Sigmoid = pred - target
            dm_tensor_set(&g_out, n, i, 0, 0, (pred - target) / batch);
        }
    }
    recon_loss /= batch;
    
    // --- BACKWARD PASS ---
    DM_Block g_dec_h1;
    dm_linear_backward(&dec_h1, &g_out, &g_dec_h1, vae->g_dec_w2, vae->g_dec_b2, vae->dec_w2, vae->input_dim);
    
    DM_Block g_dec_h1_pre;
    dm_tanh_backward(&dec_h1, &g_dec_h1, &g_dec_h1_pre);
    
    DM_Block g_z;
    dm_linear_backward(&z, &g_dec_h1_pre, &g_z, vae->g_dec_w1, vae->g_dec_b1, vae->dec_w1, vae->hidden_dim);
    
    DM_Block g_enc_h2;
    dm_block_create(&g_enc_h2, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){batch, 2 * vae->latent_dim, 1, 1});
    for (int n = 0; n < batch; n++) {
        for (int i = 0; i < vae->latent_dim; i++) {
            float m = dm_tensor_get(&mean, n, i, 0, 0);
            float lv = dm_tensor_get(&logvar, n, i, 0, 0);
            float e = dm_tensor_get(&eps, n, i, 0, 0);
            float gz = dm_tensor_get(&g_z, n, i, 0, 0);
            
            // dL/d(mean) = dL/dz * 1 + dKL/d(mean)
            float gm = gz + m / batch;
            // dL/d(logvar) = dL/dz * 0.5 * exp(0.5*logvar) * eps + dKL/d(logvar)
            float glv = gz * 0.5f * expf(0.5f * lv) * e + 0.5f * (expf(lv) - 1.0f) / batch;
            
            dm_tensor_set(&g_enc_h2, n, i, 0, 0, gm);
            dm_tensor_set(&g_enc_h2, n, vae->latent_dim + i, 0, 0, glv);
        }
    }
    
    DM_Block g_enc_h1;
    dm_linear_backward(&enc_h1, &g_enc_h2, &g_enc_h1, vae->g_enc_w2, vae->g_enc_b2, vae->enc_w2, 2 * vae->latent_dim);
    
    DM_Block g_enc_h1_pre;
    dm_tanh_backward(&enc_h1, &g_enc_h1, &g_enc_h1_pre);
    
    dm_linear_backward(x, &g_enc_h1_pre, NULL, vae->g_enc_w1, vae->g_enc_b1, vae->enc_w1, vae->hidden_dim);
    
    // --- OPTIMIZER STEP (Adam) ---
    float eps_adam = 1e-8f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    // We need a timestep `t` for Adam bias correction. We can maintain it in the VAE struct.
    vae->t += 1;
    
    dm_adam_step(vae->enc_w1, vae->g_enc_w1, vae->s_enc_w1, vae->v_enc_w1, vae->hidden_dim * vae->input_dim, vae->lr, beta1, beta2, eps_adam, 0.0f, vae->t);
    dm_adam_step(vae->enc_b1, vae->g_enc_b1, vae->s_enc_b1, vae->v_enc_b1, vae->hidden_dim, vae->lr, beta1, beta2, eps_adam, 0.0f, vae->t);
    dm_adam_step(vae->enc_w2, vae->g_enc_w2, vae->s_enc_w2, vae->v_enc_w2, 2 * vae->latent_dim * vae->hidden_dim, vae->lr, beta1, beta2, eps_adam, 0.0f, vae->t);
    dm_adam_step(vae->enc_b2, vae->g_enc_b2, vae->s_enc_b2, vae->v_enc_b2, 2 * vae->latent_dim, vae->lr, beta1, beta2, eps_adam, 0.0f, vae->t);
    dm_adam_step(vae->dec_w1, vae->g_dec_w1, vae->s_dec_w1, vae->v_dec_w1, vae->hidden_dim * vae->latent_dim, vae->lr, beta1, beta2, eps_adam, 0.0f, vae->t);
    dm_adam_step(vae->dec_b1, vae->g_dec_b1, vae->s_dec_b1, vae->v_dec_b1, vae->hidden_dim, vae->lr, beta1, beta2, eps_adam, 0.0f, vae->t);
    dm_adam_step(vae->dec_w2, vae->g_dec_w2, vae->s_dec_w2, vae->v_dec_w2, vae->input_dim * vae->hidden_dim, vae->lr, beta1, beta2, eps_adam, 0.0f, vae->t);
    dm_adam_step(vae->dec_b2, vae->g_dec_b2, vae->s_dec_b2, vae->v_dec_b2, vae->input_dim, vae->lr, beta1, beta2, eps_adam, 0.0f, vae->t);
    
    // Free tensors
    dm_block_free(&enc_h1);
    dm_block_free(&enc_h2);
    dm_block_free(&mean);
    dm_block_free(&logvar);
    dm_block_free(&z);
    dm_block_free(&eps);
    dm_block_free(&dec_h1);
    dm_block_free(&out);
    dm_block_free(&g_out);
    dm_block_free(&g_dec_h1);
    dm_block_free(&g_dec_h1_pre);
    dm_block_free(&g_z);
    dm_block_free(&g_enc_h2);
    dm_block_free(&g_enc_h1);
    dm_block_free(&g_enc_h1_pre);
    
    return recon_loss + kl_loss;
}
