#ifndef DM_VAE_H
#define DM_VAE_H

#include "models/tensor.h"

typedef struct {
    int input_dim;
    int hidden_dim;
    int latent_dim;
    
    // Encoder
    float *enc_w1, *enc_b1;
    float *enc_w2, *enc_b2;
    // Decoder
    float *dec_w1, *dec_b1;
    float *dec_w2, *dec_b2;
    
    // Gradients
    float *g_enc_w1, *g_enc_b1;
    float *g_enc_w2, *g_enc_b2;
    float *g_dec_w1, *g_dec_b1;
    float *g_dec_w2, *g_dec_b2;
    
    // Adagrad / Adam accumulated gradients / moments
    float *s_enc_w1, *s_enc_b1;
    float *s_enc_w2, *s_enc_b2;
    float *s_dec_w1, *s_dec_b1;
    float *s_dec_w2, *s_dec_b2;
    
    // Adam v vectors
    float *v_enc_w1, *v_enc_b1;
    float *v_enc_w2, *v_enc_b2;
    float *v_dec_w1, *v_dec_b1;
    float *v_dec_w2, *v_dec_b2;
    
    float lr;
    int t;
} DM_VAE;

void dm_vae_init(DM_VAE *vae, int input_dim, int hidden_dim, int latent_dim, float lr);
void dm_vae_free(DM_VAE *vae);

/* Forward pass and loss computation. Returns the total loss (BCE + KL) for the batch */
float dm_vae_train_step(DM_VAE *vae, const DM_Tensor *x);

/* Inference: encode */
void dm_vae_encode(DM_VAE *vae, const DM_Tensor *x, DM_Tensor *mean, DM_Tensor *logvar);

/* Inference: decode */
void dm_vae_decode(DM_VAE *vae, const DM_Tensor *z, DM_Tensor *out);

#endif /* DM_VAE_H */
