/*
 * gpu_unigram_em.c — GPU Unigram EM E-step via Vulkan compute.
 *
 * Shader: unigram_word.spv
 *   One workgroup per word. Within a workgroup, one thread runs the full
 *   forward-backward DP for that word (sequential but parallel across words).
 *   After backward pass, atomicAdd (float, requires VK_EXT_shader_atomic_float)
 *   scatters expected counts into the shared new_counts[] buffer.
 *
 * Bindings for unigram_word.spv  (9 storage buffers):
 *   0  log_probs     float[n_pieces]
 *   1  cp_ids        uint16[] (padded to uint32[] — shader reads as uint, extracts half)
 *   2  word_starts   uint[]
 *   3  word_lens     uint[]
 *   4  word_freqs    uint[]
 *   5  piece_row_ptr uint[]
 *   6  piece_col     uint[]
 *   7  new_counts    float[n_pieces]   (atomicAdd)
 *   8  exp_total     float[1]          (atomicAdd)
 *
 * Push constants (16 bytes):
 *   uint n_words, n_pieces, max_word_len, pad
 */

#include "dm_gpu_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    uint32_t n_words;
    uint32_t n_pieces;
    uint32_t max_word_len;
    uint32_t _pad;
} UnigramPc;

int _gpu_unigram_em_step(DmGpuCtx *ctx,
                          const DmGpuUnigramModel  *model,
                          const DmGpuUnigramCorpus *corpus,
                          float *new_counts, float *expected_total) {

    uint32_t n_words  = (uint32_t)corpus->n_words;
    uint32_t n_pieces = model->n_pieces;
    uint32_t max_wlen = model->max_piece_len; /* max word length in codepoints */

    /* cp_ids is uint16_t; we pack two into each uint32_t for GPU upload */
    size_t cp_u32_count = (corpus->total_cps + 1) / 2;

    GpuBuf b_lp, b_cp, b_ws, b_wl, b_wf, b_prp, b_pc, b_nc, b_et;
    memset(&b_lp, 0, sizeof(GpuBuf)); memset(&b_cp,  0, sizeof(GpuBuf));
    memset(&b_ws, 0, sizeof(GpuBuf)); memset(&b_wl,  0, sizeof(GpuBuf));
    memset(&b_wf, 0, sizeof(GpuBuf)); memset(&b_prp, 0, sizeof(GpuBuf));
    memset(&b_pc, 0, sizeof(GpuBuf)); memset(&b_nc,  0, sizeof(GpuBuf));
    memset(&b_et, 0, sizeof(GpuBuf));

    int rc = DM_GPU_ERR_OOM;

#define ALLOC(buf, sz) \
    if (gpu_buf_alloc(ctx, (sz), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &(buf)) != 0) goto fail

    ALLOC(b_lp,  n_pieces                         * 4);
    ALLOC(b_cp,  cp_u32_count                     * 4);  /* uint16 packed */
    ALLOC(b_ws,  n_words                          * 4);
    ALLOC(b_wl,  n_words                          * 4);
    ALLOC(b_wf,  n_words                          * 4);
    ALLOC(b_prp, (corpus->total_cps + 1)          * 4);
    ALLOC(b_pc,  corpus->total_covered_arcs        * 4);
    ALLOC(b_nc,  n_pieces                          * 4);
    ALLOC(b_et,  1                                 * 4);
#undef ALLOC

    gpu_buf_upload(&b_lp,  model->log_probs,          n_pieces * 4);
    /* Pack uint16 → uint32: two cp_ids per element (little-endian) */
    {
        uint32_t *packed = (uint32_t *)b_cp.mapped;
        const uint16_t *src = corpus->cp_ids;
        for (size_t k = 0; k < corpus->total_cps; k += 2) {
            uint32_t lo = src[k];
            uint32_t hi = (k + 1 < corpus->total_cps) ? src[k + 1] : 0;
            packed[k / 2] = lo | (hi << 16);
        }
    }
    gpu_buf_upload(&b_ws,  corpus->word_starts,        n_words * 4);
    gpu_buf_upload(&b_wl,  corpus->word_lens,           n_words * 4);
    gpu_buf_upload(&b_wf,  corpus->word_freqs,          n_words * 4);
    gpu_buf_upload(&b_prp, corpus->piece_row_ptr, (corpus->total_cps + 1) * 4);
    gpu_buf_upload(&b_pc,  corpus->piece_col, corpus->total_covered_arcs * 4);

    /* Zero accumulators (atomic targets) */
    memset(b_nc.mapped, 0, n_pieces * 4);
    memset(b_et.mapped, 0, 4);

    if (gpu_cmd_begin(ctx) != 0) { rc = DM_GPU_ERR_INIT; goto fail; }

    GpuBuf *bufs[] = { &b_lp, &b_cp, &b_ws, &b_wl, &b_wf, &b_prp, &b_pc, &b_nc, &b_et };
    VkDescriptorSet ds = gpu_bind_buffers(ctx, PIPE_UNIGRAM_WORD, bufs, 9);
    if (ds == VK_NULL_HANDLE) { rc = DM_GPU_ERR_INIT; goto fail; }

    UnigramPc pc = { n_words, n_pieces, max_wlen, 0 };
    /* One thread per word; workgroup size = 1 (defined in shader) */
    gpu_dispatch(ctx, PIPE_UNIGRAM_WORD, ds, &pc, sizeof(pc), n_words);

    if (gpu_cmd_end_submit(ctx) != 0) { rc = DM_GPU_ERR_INIT; goto fail; }

    /* Accumulate into caller's arrays */
    const float *gpu_nc = (const float *)b_nc.mapped;
    const float *gpu_et = (const float *)b_et.mapped;
    for (uint32_t i = 0; i < n_pieces; i++) new_counts[i] += gpu_nc[i];
    *expected_total += gpu_et[0];

    ctx->vk.vkFreeDescriptorSets(ctx->dev, ctx->dpool, 1, &ds);
    rc = DM_GPU_OK;

fail:
    gpu_buf_free(ctx, &b_lp); gpu_buf_free(ctx, &b_cp);
    gpu_buf_free(ctx, &b_ws); gpu_buf_free(ctx, &b_wl); gpu_buf_free(ctx, &b_wf);
    gpu_buf_free(ctx, &b_prp); gpu_buf_free(ctx, &b_pc);
    gpu_buf_free(ctx, &b_nc); gpu_buf_free(ctx, &b_et);
    return rc;
}
