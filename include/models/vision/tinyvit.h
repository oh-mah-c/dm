/*
 * tinyvit.h — TinyViT: Fast Pretraining Distillation for Small Vision Transformers
 *
 * Implements the architecture and distillation framework from:
 *   Wu, Zhang, Peng et al., arXiv:2207.10666v1, 2022
 *
 * Architecture (Table 1, Supplementary, Section 3.2):
 *   ┌─────────────┬──────────────────────────────────────────────────────┐
 *   │ Patch Embed │ 2× Conv3×3 (stride 2, pad 1)  → 56×56×D1            │
 *   │ Stage 1     │ N1=2 MBConv blocks             → 56×56×D1            │
 *   │ Downsample  │ MBConv (stride 2)              → 28×28×D2            │
 *   │ Stage 2     │ N2=2 Transformer(window W2×W2) → 28×28×D2            │
 *   │ Downsample  │ MBConv (stride 2)              → 14×14×D3            │
 *   │ Stage 3     │ N3=6 Transformer(window W3×W3) → 14×14×D3            │
 *   │ Downsample  │ MBConv (stride 2)              → 7×7×D4              │
 *   │ Stage 4     │ N4=2 Transformer(window W4×W4) → 7×7×D4              │
 *   │ Classifier  │ AvgPool + LayerNorm + Linear   → num_classes         │
 *   └─────────────┴──────────────────────────────────────────────────────┘
 *
 * Shared factors (all variants):
 *   depths  = {2, 2, 6, 2}   (γ_N1…γ_N4)
 *   windows = {7, 14, 7}     (γ_W2, γ_W3, γ_W4 — stages 2, 3, 4)
 *   γ_R  = 4  (MBConv expansion ratio)
 *   γ_M  = 4  (MLP hidden ratio)
 *   γ_E  = 32 (attention head dimension)
 *
 * Model variants — embed dims {D1, D2, D3, D4}:
 *   TinyViT-5M:  {64,  128, 160, 320}  ~  5 M params
 *   TinyViT-11M: {64,  128, 256, 448}  ~ 11 M params
 *   TinyViT-21M: {96,  192, 384, 576}  ~ 21 M params
 *
 * All in pure C99.  Training: TF/Keras Python backend (subprocess), same
 * pattern as mobilenet_tiny.c.  Distillation: pure-C sparse-label I/O.
 */

#ifndef DM_TINYVIT_H
#define DM_TINYVIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Model configuration                                                  */
/* ------------------------------------------------------------------ */

typedef enum {
    TINYVIT_5M  = 0,
    TINYVIT_11M = 1,
    TINYVIT_21M = 2
} TinyViTVariant;

typedef struct {
    TinyViTVariant variant;
    int embed_dims[4];   /* {D1, D2, D3, D4} */
    int depths[4];       /* {N1, N2, N3, N4} — always {2,2,6,2} per paper */
    int window_sizes[3]; /* {W2, W3, W4}     — always {7,14,7} per paper  */
    int mbconv_expand;   /* γ_R = 4 */
    int mlp_ratio;       /* γ_M = 4 */
    int head_dim;        /* γ_E = 32 */
    int num_classes;
    int img_size;        /* default 224 */
} TinyViTConfig;

/* Fill cfg with the canonical paper settings for a given variant. */
void dm_tinyvit_config_init(TinyViTConfig *cfg, TinyViTVariant v,
                             int num_classes, int img_size);

/* ------------------------------------------------------------------ */
/* Sparse soft-label distillation (Section 3.1, Eq. 1–2)              */
/* ------------------------------------------------------------------ */
/*
 * The fast pretraining distillation framework stores only the top-K
 * teacher logits per image to avoid re-running the large teacher model
 * during student training.
 *
 * Label recovery (Eq. 2):
 *   ŷ_c = ŷ_{I(k)}                           if c ∈ {I(1)…I(K)}
 *        = (1 − Σ_{k=1}^K ŷ_{I(k)}) / (C−K)  otherwise
 *
 * Data augmentation is encoded as a single PCG seed d_0 (Section 3.1).
 */

typedef struct {
    uint32_t *indices;  /* top-K class indices [K], sorted descending by value */
    float    *values;   /* top-K logit values  [K] (after teacher softmax)     */
    int       K;        /* sparsity factor (e.g. K=100 for IN-21k, K=10 for IN-1k) */
    int       C;        /* total number of classes                              */
    uint32_t  aug_seed; /* PCG d_0 — encodes data-augmentation parameters      */
} TinyViTSparseLabel;

/* ------------------------------------------------------------------ */
/* Forward inference (pure C, no TF dependency)                        */
/* ------------------------------------------------------------------ */

/* Total number of float weights required for the given config.
 * Call this first to allocate/load weights before dm_tinyvit_cfg_forward. */
size_t dm_tinyvit_cfg_count(const TinyViTConfig *cfg);

/*
 * Run a single forward pass.
 *
 * input_nhwc : float[batch × img_size × img_size × 3]   (NHWC, [0,1])
 * weights    : float[dm_tinyvit_cfg_count(cfg)]       (see weight layout doc)
 * logits_out : float[batch × cfg->num_classes]           (pre-softmax)
 *
 * Returns 0 on success, negative on allocation failure.
 */
int dm_tinyvit_cfg_forward(const TinyViTConfig *cfg,
                        const float         *weights,
                        const float         *input_nhwc,
                        int                  batch,
                        float               *logits_out);

/* ------------------------------------------------------------------ */
/* Weight file I/O                                                      */
/* ------------------------------------------------------------------ */

/*
 * Binary weight file layout:
 *   [0]  uint32  magic   = 0x54564954  ("TVIT")
 *   [4]  uint32  version = 1
 *   [8]  uint32  variant (TinyViTVariant enum)
 *   [12] uint32  num_classes
 *   [16] uint32  img_size
 *   [20] uint64  weight_count
 *   [28] float32 weights[weight_count]
 */
#define DM_TINYVIT_WEIGHT_MAGIC  0x54564954u   /* "TVIT" */
#define DM_TINYVIT_WEIGHT_VER    1u

int dm_tinyvit_cfg_save(const char *path,
                             const TinyViTConfig *cfg,
                             const float *weights);

/* Allocates *weights_out; caller must free(). Fills *cfg from file header. */
int dm_tinyvit_cfg_load(const char *path,
                             TinyViTConfig *cfg,
                             float **weights_out);

/* ------------------------------------------------------------------ */
/* Sparse label file I/O                                               */
/* ------------------------------------------------------------------ */

/*
 * Binary sparse-label file layout (one file per dataset split / epoch):
 *   [0]   uint32  magic       = 0x4C505354  ("TSPL")
 *   [4]   uint32  version     = 1
 *   [8]   uint32  num_images
 *   [12]  uint32  num_classes
 *   [16]  uint32  topK
 *   Per image (repeated num_images times):
 *     uint32  aug_seed
 *     uint32  indices[K]
 *     float32 values[K]
 */
#define DM_TINYVIT_LABEL_MAGIC  0x4C505354u   /* "TSPL" */
#define DM_TINYVIT_LABEL_VER    1u

int dm_tinyvit_save_sparse_labels(const char *path,
                                   int num_images,
                                   int num_classes,
                                   int topK,
                                   const TinyViTSparseLabel *labels);

/* Allocates labels array and each label's indices/values; caller frees via
 * dm_tinyvit_free_sparse_labels(). */
int dm_tinyvit_load_sparse_labels(const char *path,
                                   int *num_images,
                                   int *num_classes,
                                   int *topK,
                                   TinyViTSparseLabel **labels_out);

void dm_tinyvit_free_sparse_labels(TinyViTSparseLabel *labels, int n);

/* ------------------------------------------------------------------ */
/* Distillation loss                                                    */
/* ------------------------------------------------------------------ */

/*
 * Cross-entropy loss against sparse soft labels (Eq. 1):
 *   L = CE(ŷ_teacher, softmax(student_logits / temperature))
 *
 * student_logits: float[label->C]  (raw pre-softmax logits)
 * Returns the scalar loss value.
 */
float dm_tinyvit_cfg_loss(const float             *student_logits,
                               const TinyViTSparseLabel *label,
                               float                    temperature);

/* ------------------------------------------------------------------ */
/* CLI                                                                  */
/* ------------------------------------------------------------------ */

/*
 * Subcommands:
 *   dm tinyvit infer   -i image.ppm [--model weights.bin] [--variant 21m]
 *                      [--classes N] [--top K]
 *   dm tinyvit train   --manifest train.txt -o weights.bin
 *                      [--variant 21m] [--classes N] [--epochs N] [--lr F]
 *   dm tinyvit distill --labels labels.bin --manifest train.txt -o weights.bin
 *                      [--variant 21m] [--classes N] [--epochs N] [--K N]
 *   dm tinyvit gen-labels --teacher <saved_model_dir> --manifest imgs.txt
 *                          -o labels.bin [--K N] [--classes N]
 *   dm tinyvit bench   [--variant 21m] [--batch N]
 */
int dm_tinyvit_cli(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif /* DM_TINYVIT_H */
