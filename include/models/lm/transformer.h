/*
 * transformer.h — "Attention Is All You Need"
 *
 * Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
 * 31st Conference on Neural Information Processing Systems (NeurIPS 2017)
 * arXiv:1706.03762
 *
 * This header exposes every building block individually so they can be
 * composed into other architectures (BERT, GPT, ViT, …).  The full
 * encoder-decoder Transformer is also available as a single unit.
 *
 * Paper sections implemented:
 *   §3.1  Encoder and Decoder Stacks
 *   §3.2  Attention
 *         §3.2.1  Scaled Dot-Product Attention  (Eq. 1)
 *         §3.2.2  Multi-Head Attention           (Eq. 2)
 *   §3.3  Position-wise Feed-Forward Networks   (Eq. 2)
 *   §3.4  Embeddings and Softmax (shared weight matrix, ×√d_model scale)
 *   §3.5  Positional Encoding (sinusoidal, sin/cos frequencies)
 *   §5.3  Optimizer — Adam β1=0.9, β2=0.98, ε=1e-9, warmup schedule (Eq. 3)
 *   §5.4  Regularization — residual dropout P_drop=0.1, label smoothing ε_ls=0.1
 *
 * All forward-pass code is pure C99 with no external dependencies.
 * Training is delegated to a TF/Keras Python subprocess (same pattern as
 * mobilenet_tiny.c and tinyvit.c).
 *
 * ──────────────────────────────────────────────────────────────────────────
 * Weight layout (flat float array, same order for counting and forward):
 *
 *   [A]  Token embedding   [vocab_size × d_model]   (SHARED — used for
 *                          source embed, target embed, and output projection)
 *        Multiply by √d_model when applied (§3.4).
 *
 *   [B]  Per encoder layer  ×  N  (layers 0 … N-1):
 *          Self-MHA:  W_Q[d_model×d_model], b_Q[d_model]
 *                     W_K[d_model×d_model], b_K[d_model]
 *                     W_V[d_model×d_model], b_V[d_model]
 *                     W_O[d_model×d_model], b_O[d_model]
 *          LN_1:      gamma[d_model], beta[d_model]
 *          FFN:       W_1[d_model×d_ff], b_1[d_ff]
 *                     W_2[d_ff×d_model],  b_2[d_model]
 *          LN_2:      gamma[d_model], beta[d_model]
 *
 *   [C]  Per decoder layer  ×  N  (layers 0 … N-1):
 *          Self-MHA (masked):  same layout as encoder self-MHA
 *          LN_1:      gamma[d_model], beta[d_model]
 *          Cross-MHA: same layout as encoder self-MHA
 *          LN_2:      gamma[d_model], beta[d_model]
 *          FFN:       same layout as encoder FFN
 *          LN_3:      gamma[d_model], beta[d_model]
 *
 *   [D]  Final encoder LayerNorm: gamma[d_model], beta[d_model]
 *        (applied after the last encoder layer — "encoder output LN")
 *
 * Notes:
 *   • d_k = d_v = d_model / h  (enforced by config)
 *   • No decoder final LN in the original paper; the last Add&Norm in the
 *     last decoder layer already normalises the output before the linear
 *     projection.
 *   • Output projection = token_embedding^T  (shared weight, §3.4).
 */

#ifndef DM_TRANSFORMER_H
#define DM_TRANSFORMER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * TransformerVariant — predefined hyperparameter sets from Table 3.
 *
 *  BASE: N=6  d_model=512   d_ff=2048  h=8   d_k=d_v=64  P_drop=0.1
 *             params ≈ 65 M
 *  BIG:  N=6  d_model=1024  d_ff=4096  h=16  d_k=d_v=64  P_drop=0.3
 *             params ≈ 213 M
 */
typedef enum {
    TRANSFORMER_BASE = 0,   /* base model (Table 3, row "base") */
    TRANSFORMER_BIG  = 1    /* big  model (Table 3, row "big")  */
} TransformerVariant;

typedef struct {
    TransformerVariant variant;

    /* Architecture (§3) */
    int  num_layers;    /* N    — encoder and decoder depth (paper: 6)       */
    int  d_model;       /* d_model — embedding dimension (paper: 512 / 1024) */
    int  d_ff;          /* d_ff    — FFN inner dimension  (paper: 2048/ 4096) */
    int  num_heads;     /* h       — attention heads      (paper: 8 / 16)    */
    int  d_k;           /* d_k = d_v = d_model / h        (paper: 64)        */
    int  vocab_size;    /* shared source + target vocabulary size             */
    int  max_seq_len;   /* maximum sequence length for positional encoding    */

    /* Regularization (§5.4) */
    float dropout;      /* P_drop residual dropout        (paper: 0.1 / 0.3) */
    float label_smooth; /* ε_ls label smoothing           (paper: 0.1)       */

    /* Optimizer (§5.3) */
    int   warmup_steps; /* warmup_steps for lr schedule   (paper: 4000)      */
} TransformerConfig;

/**
 * Fill cfg with canonical paper settings for the given variant.
 * vocab_size and max_seq_len must be provided by the caller.
 */
void dm_transformer_config_init(TransformerConfig *cfg,
                                 TransformerVariant  variant,
                                 int                 vocab_size,
                                 int                 max_seq_len);

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight counting and I/O
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Return the total number of float32 weights needed for cfg.
 * Allocate with:  float *w = malloc(dm_transformer_weight_count(cfg) * sizeof(float));
 */
size_t dm_transformer_weight_count(const TransformerConfig *cfg);

/**
 * Binary weight file format:
 *   [0]   uint32  magic        = 0x54524E53  ("TRNS")
 *   [4]   uint32  version      = 1
 *   [8]   uint32  variant      (TransformerVariant enum)
 *   [12]  uint32  vocab_size
 *   [16]  uint32  max_seq_len
 *   [20]  uint64  weight_count
 *   [28]  float32 weights[weight_count]
 */
#define DM_TRANSFORMER_WEIGHT_MAGIC  0x54524E53u   /* "TRNS" */
#define DM_TRANSFORMER_WEIGHT_VER    1u

int dm_transformer_save(const char              *path,
                         const TransformerConfig *cfg,
                         const float             *weights);

/** Allocates *weights_out (caller must free).  Fills *cfg from file header. */
int dm_transformer_load(const char        *path,
                         TransformerConfig *cfg,
                         float            **weights_out);

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.5  Positional Encoding (sinusoidal)
 *
 *   PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Write the positional encoding table into pe_out[max_len × d_model].
 *
 * The caller typically pre-computes this once and adds it to embeddings:
 *   for (int t = 0; t < seq_len; t++)
 *     for (int d = 0; d < d_model; d++)
 *       x[t*d_model + d] += pe[t*d_model + d];
 */
void dm_transformer_positional_encoding(int    max_len,
                                         int    d_model,
                                         float *pe_out);

/* ═══════════════════════════════════════════════════════════════════════════
 * §1  Layer Normalization  (Ba et al. 2016)
 *
 *   LN(x) = (x - μ) / √(σ² + ε)  * γ + β
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * In-place layer normalisation over the last dimension.
 *
 *   x     : float[n_rows × d]   (modified in-place)
 *   gamma : float[d]            (learned scale; pass NULL for all-ones)
 *   beta  : float[d]            (learned bias;  pass NULL for all-zeros)
 *   n_rows: number of independent rows (tokens, batch elements, …)
 *   d     : dimension to normalize over
 *   eps   : numerical stability constant (typically 1e-6)
 */
void dm_transformer_layer_norm(float       *x,
                                const float *gamma,
                                const float *beta,
                                int          n_rows,
                                int          d,
                                float        eps);

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.2.1  Scaled Dot-Product Attention  (Eq. 1)
 *
 *   Attention(Q, K, V) = softmax(QK^T / √d_k) V
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Compute scaled dot-product attention for a single head.
 *
 *   Q      : float[seq_q × d_k]
 *   K      : float[seq_k × d_k]
 *   V      : float[seq_k × d_v]
 *   mask   : float[seq_q × seq_k]  or NULL (no masking)
 *            Set mask[i][j] = -1e9 to prevent position i from attending to j.
 *            For causal (decoder self-attention) pass the prebuilt causal mask.
 *   out    : float[seq_q × d_v]    (output, must be pre-allocated)
 *   scratch: float[seq_q × seq_k]  (temporary scores buffer, pre-allocated)
 *
 * Returns 0 on success, -1 if allocation fails.
 */
int dm_transformer_sdp_attention(const float *Q,
                                  const float *K,
                                  const float *V,
                                  const float *mask,
                                  int          seq_q,
                                  int          seq_k,
                                  int          d_k,
                                  int          d_v,
                                  float       *out,
                                  float       *scratch);

/**
 * Build a causal (upper-triangular) mask for decoder self-attention.
 *
 *   mask_out: float[seq × seq]
 *             mask_out[i][j] = 0 if j <= i, else -1e9
 *
 * Position i can attend to all positions j ≤ i ("no future leakage").
 */
void dm_transformer_causal_mask(int seq, float *mask_out);

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.2.2  Multi-Head Attention  (Eq. 2)
 *
 *   MultiHead(Q, K, V) = Concat(head_1,...,head_h) W^O
 *   where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
 *
 *   W_i^Q, W_i^K ∈ ℝ^(d_model × d_k)
 *   W_i^V        ∈ ℝ^(d_model × d_v)
 *   W^O          ∈ ℝ^(h·d_v × d_model)
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Multi-head attention.  All four weight matrices and biases are expected as
 * contiguous float arrays (row-major):
 *
 *   W_Q : float[d_model × d_model]    (stacks all h projections: W_1^Q‖…‖W_h^Q)
 *   b_Q : float[d_model]
 *   W_K : float[d_model × d_model]
 *   b_K : float[d_model]
 *   W_V : float[d_model × d_model]
 *   b_V : float[d_model]
 *   W_O : float[d_model × d_model]
 *   b_O : float[d_model]
 *
 *   Q_in : float[seq_q × d_model]     query input
 *   K_in : float[seq_k × d_model]     key   input  (= Q_in for self-attention)
 *   V_in : float[seq_k × d_model]     value input  (= Q_in for self-attention)
 *   mask : float[seq_q × seq_k]       attention mask or NULL
 *   out  : float[seq_q × d_model]     output (pre-allocated)
 *
 *   h       : number of heads
 *   d_model : model dimension
 *   seq_q, seq_k: query / key-value sequence lengths
 *
 * Returns 0 on success, -1 on allocation failure.
 */
int dm_transformer_mha(const float *W_Q, const float *b_Q,
                        const float *W_K, const float *b_K,
                        const float *W_V, const float *b_V,
                        const float *W_O, const float *b_O,
                        const float *Q_in,
                        const float *K_in,
                        const float *V_in,
                        const float *mask,
                        int          seq_q,
                        int          seq_k,
                        int          d_model,
                        int          h,
                        float       *out);

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.3  Position-wise Feed-Forward Network  (Eq. 2)
 *
 *   FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Position-wise FFN applied to every row of x independently.
 *
 *   x   : float[seq × d_model]  (input; NOT modified)
 *   W_1 : float[d_model × d_ff]
 *   b_1 : float[d_ff]
 *   W_2 : float[d_ff × d_model]
 *   b_2 : float[d_model]
 *   out : float[seq × d_model]  (pre-allocated output)
 *
 * Returns 0 on success, -1 on allocation failure.
 */
int dm_transformer_ffn(const float *x,
                        const float *W_1, const float *b_1,
                        const float *W_2, const float *b_2,
                        int          seq,
                        int          d_model,
                        int          d_ff,
                        float       *out);

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.1  Encoder Layer
 *
 *   sublayer_1 = LayerNorm(x   + MHA_self(x, x, x))
 *   sublayer_2 = LayerNorm(sub1 + FFN(sub1))
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Single encoder layer (forward pass, no dropout at inference).
 *
 * Weights are read from a flat array at the position pointed to by the
 * weight layout documented at the top of this file (section [B]).
 *
 *   x_in   : float[seq × d_model]  (input)
 *   weights: pointer to the start of this layer's weight block
 *             (i.e. &w[offset_of_encoder_layer_i])
 *   x_out  : float[seq × d_model]  (output, pre-allocated)
 *
 * Returns 0 on success, -1 on allocation failure.
 */
int dm_transformer_encoder_layer(const float             *x_in,
                                  const TransformerConfig *cfg,
                                  const float             *layer_weights,
                                  int                      seq,
                                  float                   *x_out);

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.1  Decoder Layer
 *
 *   sublayer_1 = LayerNorm(x    + MaskedMHA_self(x, x, x))
 *   sublayer_2 = LayerNorm(sub1 + CrossMHA(sub1, enc_out, enc_out))
 *   sublayer_3 = LayerNorm(sub2 + FFN(sub2))
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Single decoder layer (forward pass, no dropout at inference).
 *
 *   x_in    : float[tgt_seq × d_model]   (target sequence input)
 *   enc_out : float[src_seq × d_model]   (encoder output; keys/values for
 *                                         cross-attention)
 *   layer_weights: pointer to this decoder layer's weight block (section [C])
 *   tgt_seq, src_seq: sequence lengths
 *   x_out   : float[tgt_seq × d_model]  (output, pre-allocated)
 *
 * Returns 0 on success, -1 on allocation failure.
 */
int dm_transformer_decoder_layer(const float             *x_in,
                                  const float             *enc_out,
                                  const TransformerConfig *cfg,
                                  const float             *layer_weights,
                                  int                      tgt_seq,
                                  int                      src_seq,
                                  float                   *x_out);

/* ═══════════════════════════════════════════════════════════════════════════
 * Full Encoder Stack
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Run the full encoder: embedding + positional encoding + N encoder layers
 * + final LayerNorm.
 *
 *   src_tokens : int[src_seq]       (token indices, 0-based)
 *   weights    : float[dm_transformer_weight_count(cfg)]
 *   enc_out    : float[src_seq × d_model]  (pre-allocated output)
 *
 * Returns 0 on success, -1 on allocation failure.
 */
int dm_transformer_encode(const TransformerConfig *cfg,
                           const float             *weights,
                           const int               *src_tokens,
                           int                      src_seq,
                           float                   *enc_out);

/* ═══════════════════════════════════════════════════════════════════════════
 * Full Decoder Stack (requires encoder output)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Run the full decoder: embedding + positional encoding + N decoder layers
 * + final linear projection (= token_embedding^T) → logits.
 *
 *   tgt_tokens : int[tgt_seq]
 *   enc_out    : float[src_seq × d_model]  (from dm_transformer_encode)
 *   logits_out : float[tgt_seq × vocab_size]  (pre-allocated)
 *
 * Returns 0 on success, -1 on allocation failure.
 */
int dm_transformer_decode(const TransformerConfig *cfg,
                           const float             *weights,
                           const int               *tgt_tokens,
                           int                      tgt_seq,
                           const float             *enc_out,
                           int                      src_seq,
                           float                   *logits_out);

/* ═══════════════════════════════════════════════════════════════════════════
 * Full Forward Pass  (encode + decode in one call)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Full Transformer forward pass (single item, no batch dimension).
 *
 *   src_tokens : int[src_seq]
 *   tgt_tokens : int[tgt_seq]
 *   logits_out : float[tgt_seq × vocab_size]  (pre-allocated)
 *
 * Returns 0 on success, -1 on allocation failure.
 */
int dm_transformer_forward(const TransformerConfig *cfg,
                            const float             *weights,
                            const int               *src_tokens,
                            int                      src_seq,
                            const int               *tgt_tokens,
                            int                      tgt_seq,
                            float                   *logits_out);

/* ═══════════════════════════════════════════════════════════════════════════
 * §5.3  Learning Rate Schedule  (Eq. 3)
 *
 *   lrate = d_model^{-0.5} · min(step^{-0.5}, step · warmup_steps^{-1.5})
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Compute the paper's warmup learning rate for a given training step.
 * step is 1-indexed (step=1 is the first training step).
 */
float dm_transformer_lr_schedule(int d_model, int step, int warmup_steps);

/* ═══════════════════════════════════════════════════════════════════════════
 * CLI
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Subcommands:
 *
 *   dm transformer train  --src train.src.txt --tgt train.tgt.txt
 *                         -o weights.bin [--variant base|big]
 *                         [--vocab-size N] [--max-len N]
 *                         [--epochs N] [--batch N] [--lr F]
 *                         [--warmup N]
 *
 *   dm transformer infer  --src "Hello world" -m weights.bin
 *                         [--max-tokens N] [--beam N]
 *
 *   dm transformer encode --src tokens.txt -m weights.bin -o enc.bin
 *
 *   dm transformer bench  [--variant base|big] [--src-len N] [--tgt-len N]
 *
 *   dm transformer info   -m weights.bin
 */
int dm_transformer_cli(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif /* DM_TRANSFORMER_H */
