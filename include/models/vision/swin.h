#ifndef DM_SWIN_H
#define DM_SWIN_H

#include "core/dm_engine.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DM_SWIN_MAX_STAGES 4

typedef enum {
    DM_SWIN_TINY = 0,
    DM_SWIN_SMALL = 1,
    DM_SWIN_BASE = 2,
    DM_SWIN_LARGE = 3
} DM_SwinVariant;

typedef struct {
    DM_SwinVariant variant;
    int img_size;
    int patch_size;
    int in_chans;
    int num_classes;
    int embed_dim;
    int depths[DM_SWIN_MAX_STAGES];
    int num_heads[DM_SWIN_MAX_STAGES];
    int window_size;
    float mlp_ratio;
    int qkv_bias;
    int ape;
    int patch_norm;
} DM_SwinConfig;

typedef struct {
    char *name;
    DM_Block tensor;
    DM_Block grad;
    int used;
} DM_SwinNamedTensor;

typedef struct {
    DM_SwinNamedTensor *items;
    int count;
    int capacity;
} DM_SwinTensorStore;

typedef struct {
    DM_Block *weight;
    DM_Block *bias;
    DM_Block *grad_weight;
    DM_Block *grad_bias;
} DM_SwinLayerNormBlock;

typedef struct {
    DM_Block *weight;
    DM_Block *bias;
    DM_Block *grad_weight;
    DM_Block *grad_bias;
} DM_SwinLinearBlock;

typedef struct {
    DM_SwinLinearBlock fc1;
    DM_SwinLinearBlock fc2;
    int dim;
    int hidden_dim;
} DM_SwinMlpBlock;

typedef struct {
    DM_SwinLinearBlock qkv;
    DM_SwinLinearBlock proj;
    DM_Block *relative_position_bias_table;
    DM_Block *grad_relative_position_bias_table;
    int *relative_position_index;
    int dim;
    int window_size;
    int num_heads;
    int head_dim;
    float scale;
} DM_SwinWindowAttentionBlock;

typedef struct {
    DM_SwinLayerNormBlock norm1;
    DM_SwinWindowAttentionBlock attn;
    DM_SwinLayerNormBlock norm2;
    DM_SwinMlpBlock mlp;
    DM_Block attn_mask;
    int has_attn_mask;
    int dim;
    int input_h;
    int input_w;
    int num_heads;
    int window_size;
    int shift_size;
} DM_SwinBlock;

typedef struct {
    DM_SwinLayerNormBlock norm;
    DM_SwinLinearBlock reduction;
    int input_h;
    int input_w;
    int dim;
} DM_SwinPatchMergingBlock;

typedef struct {
    DM_SwinBlock *blocks;
    int depth;
    int dim;
    int input_h;
    int input_w;
    int has_downsample;
    DM_SwinPatchMergingBlock downsample;
} DM_SwinStageBlock;

typedef struct {
    DM_Block *proj_weight;
    DM_Block *proj_bias;
    DM_Block *grad_proj_weight;
    DM_Block *grad_proj_bias;
    DM_SwinLayerNormBlock norm;
    int has_norm;
    int img_size;
    int patch_size;
    int in_chans;
    int embed_dim;
    int patches_h;
    int patches_w;
} DM_SwinPatchEmbedBlock;

typedef struct {
    DM_SwinConfig cfg;
    DM_SwinTensorStore tensors;
    DM_SwinPatchEmbedBlock patch_embed;
    DM_Block *absolute_pos_embed;
    DM_Block *grad_absolute_pos_embed;
    DM_SwinStageBlock stages[DM_SWIN_MAX_STAGES];
    DM_SwinLayerNormBlock norm;
    DM_SwinLinearBlock head;
    DM_TrainingContext *ctx;
    int initialized;
} DM_SwinModel;

void dm_swin_config_init(DM_SwinConfig *cfg, DM_SwinVariant variant, int num_classes, int img_size);

int dm_swin_model_init(DM_SwinModel *model, const DM_SwinConfig *cfg);
int dm_swin_model_load_weights(DM_SwinModel *model, const char *metadata_path);
int dm_swin_model_forward(DM_SwinModel *model, const DM_Block *input_nhwc, DM_ActivationCache *cache, DM_Block *logits);
int dm_swin_model_backward(DM_SwinModel *model, const DM_Block *d_logits, DM_Block *d_input_nhwc, DM_ActivationCache *cache);
void dm_swin_model_free(DM_SwinModel *model);

int dm_swin_window_partition(const DM_Block *x_bhwc, int window_size, DM_Block *windows);
int dm_swin_window_reverse(const DM_Block *windows, int window_size, int h, int w, DM_Block *x_bhwc);
int dm_swin_build_relative_position_index(int window_size, int *out_index);
int dm_swin_build_attention_mask(int h, int w, int window_size, int shift_size, DM_Block *mask);
int dm_swin_window_attention_forward_fixture(const DM_Block *x,
                                             const DM_Block *qkv_weight,
                                             const DM_Block *qkv_bias,
                                             const DM_Block *proj_weight,
                                             const DM_Block *proj_bias,
                                             const DM_Block *relative_position_bias_table,
                                             int num_heads,
                                             const DM_Block *mask,
                                             DM_Block *y);

int dm_swin_cli(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif /* DM_SWIN_H */
