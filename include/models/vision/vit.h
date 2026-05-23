#ifndef DM_VIT_H
#define DM_VIT_H

/*
 * vit.h — Vision Transformer (ViT) public API
 *
 * Paper: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers
 *        for Image Recognition at Scale," ICLR 2021. arXiv:2010.11929v2
 *
 * Variants from Table 1 of the paper:
 *   ViT-S/16  (not in original table; widely used 6-layer small variant)
 *   ViT-B/16  — 12 layers, D=768,  MLP=3072, heads=12, params≈86M
 *   ViT-L/16  — 24 layers, D=1024, MLP=4096, heads=16, params≈307M
 *   ViT-H/14  — 32 layers, D=1280, MLP=5120, heads=16, params≈632M
 *
 * Layout used throughout: sequences are stored as row-major float arrays
 * [seq_len × d_model].  The NCHW DM_Tensor is only used for input/output.
 */

#include "core/dm_engine.h"

/* ── Variant selector ────────────────────────────────────────────────────── */
typedef enum {
    VIT_TINY  = 0,   /* 5L  D=192  MLP=768  heads=3   patch=16  (custom)   */
    VIT_SMALL = 1,   /* 6L  D=384  MLP=1536 heads=6   patch=16  (common)   */
    VIT_BASE  = 2,   /* 12L D=768  MLP=3072 heads=12  patch=16  (paper B)  */
    VIT_LARGE = 3,   /* 24L D=1024 MLP=4096 heads=16  patch=16  (paper L)  */
    VIT_HUGE  = 4,   /* 32L D=1280 MLP=5120 heads=16  patch=14  (paper H)  */
} ViTVariant;

/* ── Config ──────────────────────────────────────────────────────────────── */
typedef struct {
    ViTVariant variant;
    int        img_size;   /* H = W, default 224                            */
    int        patch_size; /* P,   e.g. 16 or 14                            */
    int        num_layers; /* L transformer encoder layers                  */
    int        d_model;    /* D = hidden/embedding dim                      */
    int        mlp_dim;    /* 4×D by default                                */
    int        num_heads;  /* h heads for MHSA                              */
    int        num_classes;/* K downstream classification head              */
} ViTConfig;

/* ── Forward passes ──────────────────────────────────────────────────────── */

/* Initialise config from variant (fills in paper defaults). */
void dm_vit_config_init(ViTConfig *cfg, ViTVariant v,
                        int num_classes, int img_size);

/*
 * Full forward pass.  Weights are provided as a single contiguous float
 * array (random / loaded from file) whose layout is documented in vit.c.
 * If weights == NULL, random Gaussian weights are generated internally
 * using the supplied seed.
 *
 * Input  : NCHW tensor  shape (1, 3, img_size, img_size)
 * Output : logits tensor shape (1, num_classes, 1, 1)
 * Returns: 0 on success, -1 on error.
 */
int dm_vit_forward(const DM_Tensor *input, DM_Tensor *logits,
                   const ViTConfig *cfg, const float *weights,
                   unsigned int seed);

/* Convenience wrapper that allocates cfg from variant. */
int dm_vit_forward_variant(const DM_Tensor *input, DM_Tensor *logits,
                            ViTVariant v, int num_classes,
                            const float *weights, unsigned int seed);

/* Count total number of float parameters for the given config. */
size_t dm_vit_param_count(const ViTConfig *cfg);

/* CLI entry point: registered in main.c as "vit" */
int dm_vit_cli(int argc, char **argv);

#endif /* DM_VIT_H */
