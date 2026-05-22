/*
 * dm_gpu.h — GPU compute acceleration for DM tokenizer training
 *
 * Backend: Vulkan 1.1 compute (NVIDIA, AMD, Intel, Apple via MoltenVK).
 * Loaded at runtime — no link-time dependency on vulkan-1.lib.
 * Falls back to CPU silently when a GPU is unavailable.
 *
 * Supported operations:
 *   dm_gpu_bpe_pair_count   — parallel pair-frequency histogram (BPE training)
 *   dm_gpu_sinkhorn         — Sinkhorn balanced OT iterations (VOLT)
 *   dm_gpu_unigram_em_step  — forward-backward EM E-step (Unigram training)
 *
 * Usage pattern (every tokenizer):
 *   DmGpuCtx *gpu = use_gpu ? dm_gpu_create(gpu_device) : NULL;
 *   if (use_gpu && !gpu) fprintf(stderr, "warn: GPU unavailable, using CPU\n");
 *   // ... pass gpu to hot-path functions; they check for NULL themselves ...
 *   dm_gpu_destroy(gpu);
 */

#ifndef DM_GPU_H
#define DM_GPU_H

#include <stddef.h>
#include <stdint.h>

/* Opaque GPU context — one per training run, not thread-safe. */
typedef struct DmGpuCtx DmGpuCtx;

/* Return codes (negative = error). */
#define DM_GPU_OK                   0
#define DM_GPU_ERR_NO_DEVICE       -1   /* no Vulkan-capable GPU found        */
#define DM_GPU_ERR_INIT            -2   /* Vulkan init failed                 */
#define DM_GPU_ERR_SHADER          -3   /* .spv file not found or invalid     */
#define DM_GPU_ERR_OOM             -4   /* GPU out of memory                  */
#define DM_GPU_ERR_VOCAB_TOO_LARGE -5   /* vocab > DM_GPU_BPE_MAX_VOCAB       */
#define DM_GPU_ERR_NO_ATOMIC_FLOAT -6   /* VK_EXT_shader_atomic_float absent  */

/* -------------------------------------------------------------------------
 * Lifecycle
 * ---------------------------------------------------------------------- */

/*
 * Create a Vulkan compute context.
 *   device_index  0 = first discrete GPU, -1 = let the driver pick.
 *   shader_dir    path to the directory containing compiled .spv files,
 *                 or NULL to search: shaders/ → bin/shaders/ → exe-dir/shaders/
 * Returns NULL on failure; caller owns and must call dm_gpu_destroy().
 */
DmGpuCtx *dm_gpu_create(int device_index, const char *shader_dir);

/* Release all Vulkan resources.  Safe to call with NULL. */
void dm_gpu_destroy(DmGpuCtx *ctx);

/* Returns 1 if ctx is non-NULL and all required shaders loaded, else 0. */
int dm_gpu_ready(const DmGpuCtx *ctx);

/* Write device name into buf[buf_len].  Returns DM_GPU_OK or negative. */
int dm_gpu_device_name(const DmGpuCtx *ctx, char *buf, size_t buf_len);

/* -------------------------------------------------------------------------
 * BPE pair-frequency histogram
 *
 * Used by: bpe_subword, sentencepiece_lite (BPE mode),
 *          grapheme_pair_encoding, tokenizer_lab.
 *
 * All symbol strings must be pre-interned to compact uint32_t IDs [0, V).
 * The helper dm_gpu_intern_sym() below handles that for you.
 * ---------------------------------------------------------------------- */

/* Maximum vocabulary size for the dense pair-count matrix.
 * At 4096: matrix = 4096 * 4096 * 4 B = 64 MB (fits on any modern GPU). */
#define DM_GPU_BPE_MAX_VOCAB 4096

/*
 * Flat corpus description for BPE pair counting.
 * All words are concatenated into sym_ids[]; word w spans
 * [word_starts[w], word_starts[w] + word_lens[w]).
 */
typedef struct {
    const uint32_t *sym_ids;      /* flat symbol-ID array, length total_syms  */
    size_t          total_syms;
    const uint32_t *word_starts;  /* word_starts[w] = first index in sym_ids  */
    const uint32_t *word_lens;    /* word_lens[w]   = symbol count of word w  */
    const uint32_t *word_freqs;   /* word_freqs[w]  = corpus frequency        */
    size_t          n_words;
    uint32_t        vocab_size;   /* distinct symbol IDs, must be ≤ MAX_VOCAB */
} DmGpuBpeInput;

/*
 * Count adjacent-pair frequencies on the GPU (or CPU fallback).
 *
 * pair_counts[left * vocab_size + right] += word_freqs[w]
 * for every adjacent (left, right) pair in word w.
 *
 * Caller allocates pair_counts[vocab_size * vocab_size] and zeroes it.
 * Returns DM_GPU_OK (GPU or CPU path) or DM_GPU_ERR_VOCAB_TOO_LARGE.
 */
int dm_gpu_bpe_pair_count(DmGpuCtx            *ctx,
                           const DmGpuBpeInput *in,
                           uint32_t            *pair_counts);

/* -------------------------------------------------------------------------
 * Sinkhorn balanced OT  (VOLT)
 *
 * Sparse kernel K in CSR format: n_tok × n_char matrix.
 * Also requires K-transpose in CSR: n_char × n_tok.
 * Build both on CPU with dm_gpu_build_csr() below.
 * ---------------------------------------------------------------------- */

typedef struct {
    const uint32_t *row_ptr;   /* length n_rows + 1                          */
    const uint32_t *col_idx;   /* length nnz                                 */
    const float    *vals;      /* length nnz                                 */
    uint32_t        n_rows;
    uint32_t        n_cols;
    uint32_t        nnz;
} DmGpuCSR;

/*
 * Run Sinkhorn iterations on GPU (or CPU fallback).
 *
 * K and Kt must be the same matrix and its transpose in CSR form.
 * p_tok[n_tok] and p_char[n_char] are the marginal distributions.
 * row_sums_out[n_tok] receives sum_j u[i]*K[i,j]*v[j] for each row.
 *
 * Returns DM_GPU_OK on success.
 */
int dm_gpu_sinkhorn(DmGpuCtx     *ctx,
                    const DmGpuCSR *K,
                    const DmGpuCSR *Kt,
                    const float  *p_tok,
                    const float  *p_char,
                    int           max_iter,
                    float         tol,
                    float        *row_sums_out);

/* -------------------------------------------------------------------------
 * Unigram EM E-step
 *
 * Runs forward-backward DP per word in parallel across words.
 * ---------------------------------------------------------------------- */

typedef struct {
    const float   *log_probs;    /* log(prob) per piece, length n_pieces      */
    uint32_t       n_pieces;
    uint32_t       max_piece_len; /* max codepoint-length of any piece        */
} DmGpuUnigramModel;

/*
 * Corpus as codepoint-ID sequences (uint16_t).
 * Piece coverage (which pieces start at each codepoint position) is
 * pre-computed on the CPU as a CSR table.
 */
typedef struct {
    const uint16_t *cp_ids;       /* flat codepoint-ID array, length total_cps */
    size_t          total_cps;
    const uint32_t *word_starts;  /* start index in cp_ids for word w          */
    const uint32_t *word_lens;    /* codepoint length of word w                */
    const uint32_t *word_freqs;   /* corpus frequency of word w               */
    size_t          n_words;
    /* CSR coverage: piece_row_ptr[pos], piece_col[...] = piece IDs valid at pos */
    const uint32_t *piece_row_ptr; /* length total_cps + 1                    */
    const uint32_t *piece_col;     /* length total_covered_arcs               */
    uint32_t        total_covered_arcs;
} DmGpuUnigramCorpus;

/*
 * Run one EM E-step on GPU (or CPU fallback if float atomics unavailable).
 *
 * new_counts[n_pieces] and *expected_total are accumulated (added to, not set).
 * Returns DM_GPU_OK or DM_GPU_ERR_NO_ATOMIC_FLOAT (CPU fallback used).
 */
int dm_gpu_unigram_em_step(DmGpuCtx                 *ctx,
                            const DmGpuUnigramModel  *model,
                            const DmGpuUnigramCorpus *corpus,
                            float                    *new_counts,
                            float                    *expected_total);

/* -------------------------------------------------------------------------
 * CPU-side helpers (no GPU required)
 * ---------------------------------------------------------------------- */

/*
 * Build a CSR matrix from COO (row, col, val) triples.
 * Caller must free *row_ptr_out, *col_idx_out, *vals_out with free().
 */
void dm_gpu_build_csr(const uint32_t *coo_row,
                      const uint32_t *coo_col,
                      const float    *coo_val,
                      uint32_t        nnz,
                      uint32_t        n_rows,
                      uint32_t        n_cols,
                      uint32_t      **row_ptr_out,
                      uint32_t      **col_idx_out,
                      float         **vals_out);

/*
 * Build the transpose of a CSR matrix in CSR form.
 * Caller must free the three output arrays with free().
 */
void dm_gpu_csr_transpose(const DmGpuCSR *K,
                           uint32_t      **kt_row_ptr_out,
                           uint32_t      **kt_col_idx_out,
                           float         **kt_vals_out);

/*
 * CPU fallback for BPE pair counting (called automatically when ctx is NULL
 * or vocab_size > DM_GPU_BPE_MAX_VOCAB).
 */
void dm_gpu_bpe_pair_count_cpu(const DmGpuBpeInput *in, uint32_t *pair_counts);

/*
 * CPU fallback for Sinkhorn (called automatically when ctx is NULL).
 */
void dm_gpu_sinkhorn_cpu(const DmGpuCSR *K,
                         const DmGpuCSR *Kt,
                         const float    *p_tok,
                         const float    *p_char,
                         int             max_iter,
                         float           tol,
                         float          *row_sums_out);

#endif /* DM_GPU_H */
