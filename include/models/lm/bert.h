/*
 * bert.h — BERT: Pre-training of Deep Bidirectional Transformers for
 *          Language Understanding
 *
 * Devlin, Chang, Lee, Toutanova — Google AI Language
 * NAACL-HLT 2019, pages 4171–4186  (N19-1423)
 *
 * ─── Architecture (§3, Figure 1) ─────────────────────────────────────────
 *
 *   Input (Figure 2):
 *     x[i] = TokenEmb(tok[i]) + SegmentEmb(seg[i]) + PositionEmb(i)
 *     x    = LayerNorm(x)   [ε = 1e-12]
 *
 *   L identical Transformer encoder layers (bidirectional — no causal mask):
 *     Sub-layer 1:  x = LayerNorm(x + MultiHeadSelfAttn(x))
 *     Sub-layer 2:  x = LayerNorm(x + FFN_GELU(x))
 *
 *   Pooler (for classification tasks):
 *     cls_out = tanh( x[0] · W_pool + b_pool )
 *
 * ─── Key differences from original Transformer (Vaswani et al. 2017) ─────
 *   1. Encoder-only    — no decoder, no causal masking → fully bidirectional
 *   2. GELU activation — FFN uses GELU, not ReLU
 *   3. Learned PE      — position embeddings are learned, not sinusoidal
 *   4. Segment embeddings — sentence A (0) vs sentence B (1)
 *   5. Pooler          — Dense(H→H, tanh) on the [CLS] token
 *   6. LN epsilon      — 1e-12 (vs 1e-6 in original Transformer)
 *
 * ─── Model sizes (Table 6) ───────────────────────────────────────────────
 *   BERT_BASE:   L=12, H=768,  A=12, d_ff=3072 → ≈110 M params
 *   BERT_LARGE:  L=24, H=1024, A=16, d_ff=4096 → ≈340 M params
 *
 * ─── Pre-training tasks (§3.1) ───────────────────────────────────────────
 *   Task 1 — Masked Language Model (MLM):
 *     Mask 15% of WordPiece tokens at random.
 *     Of those 15%: 80% → [MASK], 10% → random token, 10% → unchanged.
 *     Predict the original token (cross-entropy over vocabulary).
 *
 *   Task 2 — Next Sentence Prediction (NSP):
 *     50% of time: sentence B is the actual next sentence  (label = IsNext).
 *     50% of time: sentence B is a random sentence         (label = NotNext).
 *     Binary classification on the pooled [CLS] output.
 *
 * ─── Training hyper-parameters (Appendix A.2) ────────────────────────────
 *   Adam: lr=1e-4, β₁=0.9, β₂=0.999, ε=1e-8, L2 weight_decay=0.01
 *   LR warmup: linear over first 10,000 steps, then linear decay to 0
 *   Batch: 256 sequences × 512 tokens = 128,000 tokens/batch
 *   Steps: 1,000,000  (≈40 epochs over the 3.3 B-word corpus)
 *   Dropout: 0.1 on all layers
 *
 * ─── Fine-tuning hyper-parameters (Appendix A.3) ─────────────────────────
 *   Batch size: 16 or 32
 *   Learning rate (Adam): 5e-5 | 3e-5 | 2e-5
 *   Epochs: 2, 3, or 4
 *   Dropout: 0.1 (unchanged from pre-training)
 *
 * ─── Special token conventions (§3) ──────────────────────────────────────
 *   [CLS] — always the first token; its hidden state → classification head
 *   [SEP] — separates sentence A from sentence B (and ends sentence B)
 *   [MASK] — replaces masked tokens during MLM pre-training
 *
 *   WordPiece token IDs:  [CLS]=101, [SEP]=102, [MASK]=103  (30,522 vocab)
 *   Segment IDs: 0 = sentence A, 1 = sentence B
 *
 * ─── Flat weight-buffer layout ────────────────────────────────────────────
 *   [A] Token embedding        float[vocab_size × H]
 *   [B] Segment embedding      float[2 × H]
 *   [C] Position embedding     float[max_seq_len × H]  (learned)
 *   [D] Embedding LayerNorm    gamma[H], beta[H]
 *
 *   Per encoder layer i = 0 … L-1 (section [E]):
 *     Attention:
 *       W_Q[H × H], b_Q[H]    — query projection
 *       W_K[H × H], b_K[H]    — key   projection
 *       W_V[H × H], b_V[H]    — value projection
 *       W_O[H × H], b_O[H]    — output projection
 *     LN_1: gamma[H], beta[H]  — after attention residual
 *     FFN:
 *       W_1[H × d_ff], b_1[d_ff]    — intermediate.dense
 *       W_2[d_ff × H], b_2[H]       — output.dense
 *     LN_2: gamma[H], beta[H]  — after FFN residual
 *
 *   [F] Pooler: W_pool[H × H], b_pool[H]
 *
 * Notes:
 *   • d_ff = 4 × H  (paper: "feed-forward/filter size = 4H")
 *   • d_k  = d_v = H / A  (BERT_BASE: 64, BERT_LARGE: 64)
 *   • W_Q/K/V/O are [H × H] — each stacks all A heads together
 *   • Weight matrices stored as [out_dim × in_dim] (row = output neuron)
 *     → TF Dense kernel needs to be transposed before storing
 */

#ifndef DM_BERT_H
#define DM_BERT_H

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
 * BERT_BASE  — L=12 H=768  A=12  d_ff=3072  ~110 M params
 * BERT_LARGE — L=24 H=1024 A=16  d_ff=4096  ~340 M params
 */
typedef enum {
    BERT_BASE  = 0,
    BERT_LARGE = 1
} BertVariant;

typedef struct {
    BertVariant variant;

    /* Architecture (§3, Table 6) */
    int  num_layers;      /* L — Transformer encoder depth: 12 (BASE) / 24 (LARGE) */
    int  hidden_size;     /* H — embedding / hidden dimension: 768 / 1024          */
    int  num_heads;       /* A — self-attention heads: 12 / 16                     */
    int  intermediate;    /* d_ff = 4H — FFN inner dimension: 3072 / 4096          */
    int  vocab_size;      /* WordPiece vocabulary size (default 30522)              */
    int  max_seq_len;     /* maximum sequence length (default 512)                 */
    int  num_seg_types;   /* segment embedding types (always 2: A and B)           */

    /* Regularisation (Appendix A.2) */
    float dropout;        /* residual dropout probability — 0.1                   */
} BertConfig;

/**
 * Populate *cfg with the canonical paper settings for the given variant.
 * vocab_size:  pass 30522 for the original WordPiece vocabulary, or any
 *              custom value for a different tokenizer.
 * max_seq_len: pass 512 (paper default) or 0 to use 512.
 */
void dm_bert_config_init(BertConfig *cfg,
                          BertVariant  variant,
                          int          vocab_size,
                          int          max_seq_len);

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight counting and file I/O
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Return the total number of float32 weights for the core BERT model
 * (sections A–F in the layout above).  Does NOT include MLM or NSP heads.
 *
 * Allocate with:
 *   float *w = malloc(dm_bert_weight_count(&cfg) * sizeof(float));
 */
size_t dm_bert_weight_count(const BertConfig *cfg);

/**
 * Binary weight file layout:
 *   [0]  uint32  magic       = 0x42455254  ("BERT")
 *   [4]  uint32  version     = 1
 *   [8]  uint32  variant     (BertVariant enum value)
 *   [12] uint32  vocab_size
 *   [16] uint32  max_seq_len
 *   [20] uint64  weight_count
 *   [28] float32 weights[weight_count]
 */
#define DM_BERT_WEIGHT_MAGIC  0x42455254u   /* "BERT" */
#define DM_BERT_WEIGHT_VER    1u

int dm_bert_save(const char       *path,
                  const BertConfig *cfg,
                  const float      *weights);

/**
 * Load weights from file.  Allocates *weights_out (caller must free).
 * Fills *cfg from file header.
 */
int dm_bert_load(const char  *path,
                  BertConfig  *cfg,
                  float      **weights_out);

/* ═══════════════════════════════════════════════════════════════════════════
 * Forward pass
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Full BERT forward pass.
 *
 *   token_ids   : int[seq]  — token IDs (0-based); [CLS]=101, [SEP]=102
 *   segment_ids : int[seq]  — 0 for sentence A tokens, 1 for sentence B tokens
 *   seq         : sequence length (≤ cfg->max_seq_len)
 *
 *   hidden_out  : float[seq × H]  — final (L-th) encoder layer hidden states
 *                 (pre-allocated; the full contextual representation of each token)
 *   cls_out     : float[H]  — pooled [CLS] representation after the pooler
 *                 ( tanh(x[0] · W_pool + b_pool) )
 *                 Pass NULL if you only need token-level outputs.
 *
 * Returns 0 on success, -1 on allocation failure.
 */
int dm_bert_forward(const BertConfig *cfg,
                     const float      *weights,
                     const int        *token_ids,
                     const int        *segment_ids,
                     int               seq,
                     float            *hidden_out,
                     float            *cls_out);

/**
 * Same as dm_bert_forward, with an optional attention mask.
 *
 *   attention_mask : int[seq] where 1 keeps a token visible as an attention key
 *                    and 0 hides it (padding). Pass NULL for all-visible.
 */
int dm_bert_forward_masked(const BertConfig *cfg,
                            const float      *weights,
                            const int        *token_ids,
                            const int        *segment_ids,
                            const int        *attention_mask,
                            int               seq,
                            float            *hidden_out,
                            float            *cls_out);

/**
 * BERT forward pass returning ALL encoder layer outputs.
 *
 *   all_hidden_out : float[L × seq × H]  — hidden states for every layer
 *                    Layer i is at offset: i * seq * H
 *   cls_out        : float[H]  — pooled [CLS] (same as dm_bert_forward)
 *
 * Useful for the feature-based approach (§5.3) where different layers are
 * concatenated as input to downstream models (e.g., NER BiLSTM).
 *
 * Returns 0 on success, -1 on allocation failure.
 */
int dm_bert_forward_all_layers(const BertConfig *cfg,
                                 const float      *weights,
                                 const int        *token_ids,
                                 const int        *segment_ids,
                                 int               seq,
                                 float            *all_hidden_out,
                                 float            *cls_out);

int dm_bert_forward_all_layers_masked(const BertConfig *cfg,
                                       const float      *weights,
                                       const int        *token_ids,
                                       const int        *segment_ids,
                                       const int        *attention_mask,
                                       int               seq,
                                       float            *all_hidden_out,
                                       float            *cls_out);

/* ═══════════════════════════════════════════════════════════════════════════
 * CLI
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Subcommands:
 *
 *   dm bert encode    --text "token_ids..."  -m <weights.bin>
 *                     [--all-layers]         (run the encoder)
 *
 *   dm bert bench     [--variant base|large] [--seq-len N]
 *
 *   dm bert info      -m <weights.bin>
 */
int dm_bert_cli(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif /* DM_BERT_H */
