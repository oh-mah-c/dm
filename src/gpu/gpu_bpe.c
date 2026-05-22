/*
 * gpu_bpe.c — GPU BPE pair-frequency histogram via Vulkan compute.
 *
 * Shader: bpe_pair_count.spv
 *   Each GPU thread handles one word, atomically increments
 *   pair_counts[left_id * vocab_size + right_id] for each adjacent pair.
 *
 * Vocab limit: DM_GPU_BPE_MAX_VOCAB (4096) — keeps the pair matrix ≤ 64 MB.
 * Falls back to CPU automatically when vocab exceeds limit.
 */

#include "dm_gpu_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Push constants for bpe_pair_count.spv */
typedef struct { uint32_t n_words; uint32_t vocab_size; } BpePc;

int _gpu_bpe_pair_count(DmGpuCtx *ctx, const DmGpuBpeInput *in,
                         uint32_t *pair_counts) {
    uint32_t V        = in->vocab_size;
    size_t   n_words  = in->n_words;
    size_t   tot_syms = in->total_syms;

    /* Allocate GPU buffers */
    GpuBuf b_sym_ids, b_starts, b_lens, b_freqs, b_counts;

    int rc;
    if ((rc = gpu_buf_alloc(ctx, tot_syms  * sizeof(uint32_t),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_sym_ids)) != 0) goto fail;
    if ((rc = gpu_buf_alloc(ctx, n_words   * sizeof(uint32_t),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_starts))  != 0) goto fail;
    if ((rc = gpu_buf_alloc(ctx, n_words   * sizeof(uint32_t),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_lens))    != 0) goto fail;
    if ((rc = gpu_buf_alloc(ctx, n_words   * sizeof(uint32_t),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &b_freqs))   != 0) goto fail;
    if ((rc = gpu_buf_alloc(ctx, (size_t)V * V * sizeof(uint32_t),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,  &b_counts))   != 0) goto fail;

    /* Upload inputs */
    gpu_buf_upload(&b_sym_ids, in->sym_ids,    tot_syms * sizeof(uint32_t));
    gpu_buf_upload(&b_starts,  in->word_starts, n_words * sizeof(uint32_t));
    gpu_buf_upload(&b_lens,    in->word_lens,   n_words * sizeof(uint32_t));
    gpu_buf_upload(&b_freqs,   in->word_freqs,  n_words * sizeof(uint32_t));
    /* Zero pair count buffer */
    memset(b_counts.mapped, 0, (size_t)V * V * sizeof(uint32_t));

    /* Record commands */
    if (gpu_cmd_begin(ctx) != 0) { rc = DM_GPU_ERR_INIT; goto fail; }

    GpuBuf *bufs[] = { &b_sym_ids, &b_starts, &b_lens, &b_freqs, &b_counts };
    VkDescriptorSet ds = gpu_bind_buffers(ctx, PIPE_BPE_PAIR_COUNT, bufs, 5);
    if (ds == VK_NULL_HANDLE) { rc = DM_GPU_ERR_INIT; goto fail; }

    BpePc pc = { (uint32_t)n_words, V };
    uint32_t groups = ((uint32_t)n_words + 255u) / 256u;
    gpu_dispatch(ctx, PIPE_BPE_PAIR_COUNT, ds, &pc, sizeof(pc), groups);

    if (gpu_cmd_end_submit(ctx) != 0) { rc = DM_GPU_ERR_INIT; goto fail; }

    /* Download pair counts and add into caller's array */
    const uint32_t *gpu_counts = (const uint32_t *)b_counts.mapped;
    size_t total = (size_t)V * V;
    for (size_t k = 0; k < total; k++) pair_counts[k] += gpu_counts[k];

    ctx->vk.vkFreeDescriptorSets(ctx->dev, ctx->dpool, 1, &ds);
    rc = DM_GPU_OK;

fail:
    gpu_buf_free(ctx, &b_sym_ids);
    gpu_buf_free(ctx, &b_starts);
    gpu_buf_free(ctx, &b_lens);
    gpu_buf_free(ctx, &b_freqs);
    gpu_buf_free(ctx, &b_counts);
    return rc;
}
