/*
 * gpu_sinkhorn.c — GPU Sinkhorn balanced OT iterations for VOLT.
 *
 * Two shaders (both reuse sinkhorn_spmv.spv with different push constants):
 *   sinkhorn_spmv.spv : out[i] = p[i] / (K[i,:] @ v)   (K CSR, one thread/row)
 *   sinkhorn_rowsum.spv : out[i] = u[i] * (K[i,:] @ v)  (final row-sums)
 *
 * Algorithm:
 *   Init u = 1, v = 1
 *   Repeat max_iter times:
 *     u[i] = p_tok[i]  / (K   @ v)[i]
 *     v[j] = p_char[j] / (K^T @ u)[j]
 *     check convergence (||u_new - u_old|| < tol)
 *   row_sums[i] = u[i] * (K @ v)[i]
 */

#include "dm_gpu_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Push constants for sinkhorn_spmv.spv
 *   n_rows  : rows in this matrix (n_tok for K@v, n_char for K^T@u)
 *   mode    : 0 = out[i] = p[i]/(K@v)[i],  1 = out[i] = u[i]*(K@v)[i]
 */
typedef struct { uint32_t n_rows; uint32_t mode; } SpmvPc;

/*
 * sinkhorn_spmv.spv bindings (6 storage buffers):
 *   0 row_ptr  uint[]
 *   1 col_idx  uint[]
 *   2 vals     float[]
 *   3 vec_in   float[]   (v or u)
 *   4 p        float[]   (p_tok or p_char; ignored in mode 1)
 *   5 vec_out  float[]   (u or v; in mode 1 = u for multiplication)
 *
 * For mode 1 (row-sums), binding 4 is re-used as u (the scalar multiplier).
 */

int _gpu_sinkhorn(DmGpuCtx *ctx,
                   const DmGpuCSR *K, const DmGpuCSR *Kt,
                   const float *p_tok, const float *p_char,
                   int max_iter, float tol, float *row_sums_out) {

    uint32_t n_tok  = K->n_rows;
    uint32_t n_char = K->n_cols;
    uint32_t nnz    = K->nnz;

    /* --- Allocate all GPU buffers up front --- */
    GpuBuf b_K_rp, b_K_ci, b_K_v;    /* K CSR */
    GpuBuf b_Kt_rp, b_Kt_ci, b_Kt_v; /* K^T CSR */
    GpuBuf b_p_tok, b_p_char;         /* marginals */
    GpuBuf b_u, b_v;                  /* Sinkhorn vectors */
    GpuBuf b_kv;                      /* K @ v intermediate */
    GpuBuf b_row_sums;                /* output */

    /* Zero-init all so gpu_buf_free is safe on partial failure */
    memset(&b_K_rp,    0, sizeof(GpuBuf));
    memset(&b_K_ci,    0, sizeof(GpuBuf));
    memset(&b_K_v,     0, sizeof(GpuBuf));
    memset(&b_Kt_rp,   0, sizeof(GpuBuf));
    memset(&b_Kt_ci,   0, sizeof(GpuBuf));
    memset(&b_Kt_v,    0, sizeof(GpuBuf));
    memset(&b_p_tok,   0, sizeof(GpuBuf));
    memset(&b_p_char,  0, sizeof(GpuBuf));
    memset(&b_u,       0, sizeof(GpuBuf));
    memset(&b_v,       0, sizeof(GpuBuf));
    memset(&b_kv,      0, sizeof(GpuBuf));
    memset(&b_row_sums,0, sizeof(GpuBuf));

    int rc = DM_GPU_ERR_OOM;

#define ALLOC(buf, sz, usage) \
    if (gpu_buf_alloc(ctx, (sz), (usage), &(buf)) != 0) goto fail

    ALLOC(b_K_rp,    (n_tok + 1)  * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ALLOC(b_K_ci,    nnz          * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ALLOC(b_K_v,     nnz          * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ALLOC(b_Kt_rp,   (n_char + 1) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ALLOC(b_Kt_ci,   nnz          * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ALLOC(b_Kt_v,    nnz          * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ALLOC(b_p_tok,   n_tok        * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ALLOC(b_p_char,  n_char       * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ALLOC(b_u,       n_tok        * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ALLOC(b_v,       n_char       * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ALLOC(b_kv,      n_tok        * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    ALLOC(b_row_sums,n_tok        * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
#undef ALLOC

    /* Upload K and K^T */
    gpu_buf_upload(&b_K_rp,  K->row_ptr, (n_tok + 1)  * 4);
    gpu_buf_upload(&b_K_ci,  K->col_idx, nnz * 4);
    gpu_buf_upload(&b_K_v,   K->vals,    nnz * 4);
    gpu_buf_upload(&b_Kt_rp, Kt->row_ptr, (n_char + 1) * 4);
    gpu_buf_upload(&b_Kt_ci, Kt->col_idx, nnz * 4);
    gpu_buf_upload(&b_Kt_v,  Kt->vals,    nnz * 4);
    gpu_buf_upload(&b_p_tok,  p_tok,  n_tok  * 4);
    gpu_buf_upload(&b_p_char, p_char, n_char * 4);

    /* Init u = 1, v = 1 */
    {
        float *pu = (float *)b_u.mapped;
        float *pv = (float *)b_v.mapped;
        for (uint32_t i = 0; i < n_tok;  i++) pu[i] = 1.0f;
        for (uint32_t j = 0; j < n_char; j++) pv[j] = 1.0f;
    }

    uint32_t grp_tok  = (n_tok  + 63u) / 64u;
    uint32_t grp_char = (n_char + 63u) / 64u;

    /* Sinkhorn iteration loop */
    for (int iter = 0; iter < max_iter; iter++) {

        if (gpu_cmd_begin(ctx) != 0) { rc = DM_GPU_ERR_INIT; goto fail; }

        /* u = p_tok / (K @ v) */
        {
            GpuBuf *bufs[] = { &b_K_rp, &b_K_ci, &b_K_v, &b_v, &b_p_tok, &b_u };
            VkDescriptorSet ds = gpu_bind_buffers(ctx, PIPE_SINKHORN_SPMV, bufs, 6);
            SpmvPc pc = { n_tok, 0 };
            gpu_dispatch(ctx, PIPE_SINKHORN_SPMV, ds, &pc, sizeof(pc), grp_tok);
            ctx->vk.vkFreeDescriptorSets(ctx->dev, ctx->dpool, 1, &ds);
        }

        gpu_barrier(ctx);

        /* v = p_char / (K^T @ u) */
        {
            GpuBuf *bufs[] = { &b_Kt_rp, &b_Kt_ci, &b_Kt_v, &b_u, &b_p_char, &b_v };
            VkDescriptorSet ds = gpu_bind_buffers(ctx, PIPE_SINKHORN_SPMV, bufs, 6);
            SpmvPc pc = { n_char, 0 };
            gpu_dispatch(ctx, PIPE_SINKHORN_SPMV, ds, &pc, sizeof(pc), grp_char);
            ctx->vk.vkFreeDescriptorSets(ctx->dev, ctx->dpool, 1, &ds);
        }

        if (gpu_cmd_end_submit(ctx) != 0) { rc = DM_GPU_ERR_INIT; goto fail; }

        /* Convergence check: read back u, recompute K@v, compare */
        if (tol > 0.0f && (iter % 20 == 19 || iter == max_iter - 1)) {
            const float *u_ptr = (const float *)b_u.mapped;
            const float *v_ptr = (const float *)b_v.mapped;

            /* Quick convergence proxy: check max |u[i] - p_tok[i]/(K@v)[i]| */
            float max_d = 0.0f;
            for (uint32_t i = 0; i < n_tok; i++) {
                float dot = 0.0f;
                for (uint32_t k = K->row_ptr[i]; k < K->row_ptr[i + 1]; k++)
                    dot += K->vals[k] * v_ptr[K->col_idx[k]];
                float expected = (dot > 1e-30f) ? p_tok[i] / dot : 0.0f;
                float d = fabsf(u_ptr[i] - expected);
                if (d > max_d) max_d = d;
            }
            if (max_d < tol) break;
        }
    }

    /* Final row_sums[i] = u[i] * (K @ v)[i] */
    if (gpu_cmd_begin(ctx) != 0) { rc = DM_GPU_ERR_INIT; goto fail; }
    {
        /* binding 4 = u (used as scalar multiplier in mode 1) */
        GpuBuf *bufs[] = { &b_K_rp, &b_K_ci, &b_K_v, &b_v, &b_u, &b_row_sums };
        VkDescriptorSet ds = gpu_bind_buffers(ctx, PIPE_SINKHORN_ROWSUM, bufs, 6);
        SpmvPc pc = { n_tok, 1 };
        gpu_dispatch(ctx, PIPE_SINKHORN_ROWSUM, ds, &pc, sizeof(pc), grp_tok);
        ctx->vk.vkFreeDescriptorSets(ctx->dev, ctx->dpool, 1, &ds);
    }
    if (gpu_cmd_end_submit(ctx) != 0) { rc = DM_GPU_ERR_INIT; goto fail; }

    /* Copy row_sums back to caller */
    gpu_buf_download(&b_row_sums, row_sums_out, n_tok * sizeof(float));
    rc = DM_GPU_OK;

fail:
    gpu_buf_free(ctx, &b_K_rp);   gpu_buf_free(ctx, &b_K_ci);   gpu_buf_free(ctx, &b_K_v);
    gpu_buf_free(ctx, &b_Kt_rp);  gpu_buf_free(ctx, &b_Kt_ci);  gpu_buf_free(ctx, &b_Kt_v);
    gpu_buf_free(ctx, &b_p_tok);  gpu_buf_free(ctx, &b_p_char);
    gpu_buf_free(ctx, &b_u);      gpu_buf_free(ctx, &b_v);
    gpu_buf_free(ctx, &b_kv);     gpu_buf_free(ctx, &b_row_sums);
    return rc;
}
