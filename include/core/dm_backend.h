/**
 * include/core/dm_backend.h — DM portable backend dispatch layer
 *
 * Defines the DM_Backend enum, DM_BackendInfo struct, and the internal
 * dispatch API used by dm_engine.c and the rest of the framework.
 *
 * Tier 0: CPU baseline   — pure-C, always available (dm_cpu_kernels.c)
 * Tier 1: Vulkan compute — portable compute shader (dm_gpu.c Vulkan path)
 * Tier 2: TensorFlow     — TFE/XLA/cuDNN/oneDNN (dm_engine.c)
 * Tier 3: CUDA / ROCm    — vendor-optimised stubs (future work)
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef DM_BACKEND_H
#define DM_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── Backend tier enum ──────────────────────────────────────────────────── */

typedef enum {
    DM_BACKEND_CPU             = 0,   /* pure-C, always available             */
    DM_BACKEND_VULKAN_COMPUTE  = 1,   /* Vulkan compute shader (portable)     */
    DM_BACKEND_VULKAN_COOP_MAT = 2,   /* Vulkan cooperative matrix (optional) */
    DM_BACKEND_TENSORFLOW      = 3,   /* TFE — XLA/cuDNN/oneDNN               */
    DM_BACKEND_CUDA            = 4,   /* CUDA/Tensor Core (future)            */
    DM_BACKEND_ROCM            = 5,   /* ROCm/HIP (future)                    */
    DM_BACKEND_EXTERNAL        = 6,   /* External handle / custom backend     */
    DM_BACKEND_AUTO            = 255  /* runtime picks best available         */
} DM_Backend;

/* ── Runtime capability info ─────────────────────────────────────────────── */

typedef struct {
    DM_Backend active;              /* currently selected backend             */
    int        tf_available;        /* 1 if TFE context init succeeded        */
    int        vulkan_available;    /* 1 if Vulkan context init succeeded     */
    int        coop_mat_available;  /* 1 if VK_KHR_cooperative_matrix avail  */
    int        cuda_available;      /* 1 if CUDA detected (future)            */
    int        rocm_available;      /* 1 if ROCm detected (future)            */
} DM_BackendInfo;

/* ── Public lifecycle API ────────────────────────────────────────────────── */

/**
 * Detect available backends and select the best one.
 * Respects DM_BACKEND env var: cpu | vulkan | tensorflow | auto
 * Thread-safe: uses a static once-flag; safe to call multiple times.
 */
void dm_backend_init(void);

/** Return the currently active backend. */
DM_Backend dm_backend_get(void);

/**
 * Override the active backend.
 * Pass DM_BACKEND_AUTO to re-run auto-detection.
 */
void dm_backend_set(DM_Backend b);

/** Return a full capability snapshot. */
DM_BackendInfo dm_backend_query(void);

/** Human-readable name for a backend constant. */
const char *dm_backend_name(DM_Backend b);

/* ── Internal dispatch entry points ─────────────────────────────────────── */

/**
 * Dispatched matrix multiply: C = A × B  (transpose_B=0)
 *                             C = A × Bᵀ (transpose_B=1)
 *
 * Routes to:
 *   TF available   → dm_matmul_nn / dm_matmul_nt  (TFE path in dm_engine.c)
 *   Vulkan avail   → dm_gpu_matmul_fallback()      (CPU kernel via Vulkan ctx)
 *   Otherwise      → dm_cpu_matmul / dm_cpu_matmul_nt  (pure-C tier-0)
 */
void dm_matmul_dispatch(const float *A, const float *B, float *C,
                         int M, int K, int N, int transpose_B);

#ifdef __cplusplus
}
#endif

#endif /* DM_BACKEND_H */
