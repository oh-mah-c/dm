#ifndef DM_GAN_H
#define DM_GAN_H

#include "core/dm_engine.h"

typedef struct {
    int input_dim;
    int g_hidden_dim;
    int noise_dim;
    int d_hidden_dim;
    int maxout_k;
    float drop_prob;
    
    // Generator weights
    float *g_w1, *g_b1;
    float *g_w2, *g_b2;
    // Generator gradients
    float *g_gw1, *g_gb1;
    float *g_gw2, *g_gb2;
    // Generator velocities
    float *g_vw1, *g_vb1;
    float *g_vw2, *g_vb2;
    
    // Discriminator weights
    float *d_w1, *d_b1;
    float *d_w2, *d_b2;
    // Discriminator gradients
    float *d_gw1, *d_gb1;
    float *d_gw2, *d_gb2;
    // Discriminator velocities
    float *d_vw1, *d_vb1;
    float *d_vw2, *d_vb2;
    
    float lr;
    float momentum;
    int nesterov;
} DM_GAN;

void dm_gan_init(DM_GAN *gan, int input_dim, int g_hidden, int noise_dim, int d_hidden, int maxout_k, float drop_prob, float lr, float momentum, int nesterov);
void dm_gan_free(DM_GAN *gan);

/* Generate fake samples given noise z */
void dm_gan_generate(DM_GAN *gan, const DM_Tensor *z, DM_Tensor *out);

/* Returns D loss */
float dm_gan_train_d_step(DM_GAN *gan, const DM_Tensor *real_x, const DM_Tensor *z);

/* Returns G loss */
float dm_gan_train_g_step(DM_GAN *gan, const DM_Tensor *z);

#endif /* DM_GAN_H */
