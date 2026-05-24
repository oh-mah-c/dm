/**
 * src/core/dm_backend.c — Portable backend detection and dispatch
 *
 * Implements runtime detection of available compute backends (TFE, Vulkan,
 * CPU) and routes dm_matmul_dispatch() to the best available path.
 *
 * Detection order (highest to lowest priority):
 *   3. TensorFlow Eager (TFE) — requires libtensorflow.so at runtime
 *   1. Vulkan compute       — requires Vulkan loader + GPU
 *   0. Pure-C CPU           — always available
 *
 * The DM_BACKEND environment variable can override auto-detection:
 *   DM_BACKEND=cpu         force CPU-only
 *   DM_BACKEND=vulkan      force Vulkan portable
 *   DM_BACKEND=tensorflow  force TFE
 *   DM_BACKEND=auto        auto-detect (default)
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/dm_backend.h"
#include "core/dm_cpu_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Forward declarations for internal symbols ──────────────────────────── */

/* From dm_engine.c — TFE-backed matmul (compiled only when TF headers avail) */
extern void dm_matmul_nn(const float *A, const float *B, float *C,
                          int M, int K, int N);
extern void dm_matmul_nt(const float *A, const float *B, float *C,
                          int M, int N, int K);

/* From dm_gpu.c — Vulkan context factory (runtime-loads Vulkan; returns NULL
 * when no Vulkan loader or GPU is present — safe to call always). */
extern void *dm_gpu_create(int device_index, const char *shader_dir);
extern void  dm_gpu_destroy(void *ctx);

/* ── Module state ─────────────────────────────────────────────────────────── */

static DM_BackendInfo g_info   = {DM_BACKEND_CPU, 0, 0, 0, 0, 0};
static int            g_inited = 0;

/* ── Internal: probe TFE availability without hard-linking libtensorflow ─── */

/*
 * We cannot call TFE_NewContext here directly because dm_backend.c must
 * compile even when TF headers are absent (the backend layer is always
 * compiled; dm_engine.c is only compiled when TF is available).
 *
 * Instead we rely on the fact that dm_engine.c is already linked into
 * libdm.so and exposes dm_matmul_nn / dm_matmul_nt.  If the symbols are
 * reachable at compile-time (they always are — we declared them extern
 * above), we trust that the TFE path is live.
 *
 * For a more robust runtime check we attempt to dlsym "TFE_NewContext" from
 * the process image.  If it resolves, libtensorflow was loaded.
 */
static int probe_tf(void)
{
#if defined(__GNUC__) || defined(__clang__)
    /* dlsym approach — works on Linux/macOS */
#  ifndef _WIN32
#    include <dlfcn.h>  /* NOLINT — include inside function is intentional */
    void *sym = dlsym(RTLD_DEFAULT, "TFE_NewContext");
    return sym != NULL ? 1 : 0;
#  else
    /* Windows: check with GetProcAddress on a well-known module name */
#    include <windows.h>
    HMODULE hm = GetModuleHandleA("tensorflow.dll");
    if (!hm) hm = GetModuleHandleA("libtensorflow.dll");
    if (!hm) return 0;
    return GetProcAddress(hm, "TFE_NewContext") != NULL ? 1 : 0;
#  endif
#else
    return 0; /* conservative: unknown toolchain */
#endif
}

/* ── Internal: probe Vulkan availability ─────────────────────────────────── */

static int probe_vulkan(void)
{
    /* dm_gpu_create() performs runtime Vulkan loading internally.
     * It returns NULL when the Vulkan loader or a compatible GPU is absent.
     * We create a context, check it, then immediately destroy it. */
    void *ctx = dm_gpu_create(0, NULL);
    if (!ctx) return 0;
    dm_gpu_destroy(ctx);
    return 1;
}

/* ── dm_backend_init ─────────────────────────────────────────────────────── */

void dm_backend_init(void)
{
    if (g_inited) return;
    g_inited = 1;

    /* Default: everything unavailable, CPU active */
    g_info.tf_available      = 0;
    g_info.vulkan_available  = 0;
    g_info.coop_mat_available= 0;
    g_info.cuda_available    = 0;
    g_info.rocm_available    = 0;
    g_info.active            = DM_BACKEND_CPU;

    /* Check env override first */
    const char *env = getenv("DM_BACKEND");
    if (env) {
        if (strcmp(env, "cpu") == 0) {
            g_info.active = DM_BACKEND_CPU;
            return;
        } else if (strcmp(env, "vulkan") == 0) {
            g_info.vulkan_available = probe_vulkan();
            g_info.active = g_info.vulkan_available
                            ? DM_BACKEND_VULKAN_COMPUTE
                            : DM_BACKEND_CPU;
            return;
        } else if (strcmp(env, "tensorflow") == 0) {
            g_info.tf_available = probe_tf();
            g_info.active = g_info.tf_available
                            ? DM_BACKEND_TENSORFLOW
                            : DM_BACKEND_CPU;
            return;
        }
        /* else "auto" or unknown — fall through to auto-detection */
    }

    /* Auto-detect from highest to lowest tier */

    /* Tier 2: TensorFlow */
    g_info.tf_available = probe_tf();
    if (g_info.tf_available) {
        g_info.active = DM_BACKEND_TENSORFLOW;
        /* Still probe Vulkan for reporting, but TF wins */
        g_info.vulkan_available = probe_vulkan();
        /* coop_mat requires VK 1.3 + extension — be conservative */
        g_info.coop_mat_available = 0;
        return;
    }

    /* Tier 1: Vulkan */
    g_info.vulkan_available = probe_vulkan();
    if (g_info.vulkan_available) {
        g_info.active = DM_BACKEND_VULKAN_COMPUTE;
        /* TODO: check VK_KHR_cooperative_matrix extension via
         * vkEnumerateDeviceExtensionProperties — currently conservative */
        g_info.coop_mat_available = 0;
        return;
    }

    /* Tier 0: CPU always available */
    g_info.active = DM_BACKEND_CPU;
}

/* ── dm_backend_get / set / query / name ─────────────────────────────────── */

DM_Backend dm_backend_get(void)
{
    if (!g_inited) dm_backend_init();
    return g_info.active;
}

void dm_backend_set(DM_Backend b)
{
    if (b == DM_BACKEND_AUTO) {
        g_inited = 0;       /* force re-detection */
        dm_backend_init();
        return;
    }
    if (!g_inited) dm_backend_init();
    g_info.active = b;
}

DM_BackendInfo dm_backend_query(void)
{
    if (!g_inited) dm_backend_init();
    return g_info;
}

const char *dm_backend_name(DM_Backend b)
{
    switch (b) {
        case DM_BACKEND_CPU:             return "cpu";
        case DM_BACKEND_VULKAN_COMPUTE:  return "vulkan";
        case DM_BACKEND_VULKAN_COOP_MAT: return "vulkan_coop_mat";
        case DM_BACKEND_TENSORFLOW:      return "tensorflow";
        case DM_BACKEND_CUDA:            return "cuda";
        case DM_BACKEND_ROCM:            return "rocm";
        case DM_BACKEND_EXTERNAL:        return "external";
        case DM_BACKEND_AUTO:            return "auto";
        default:                         return "unknown";
    }
}

/* ── dm_matmul_dispatch ──────────────────────────────────────────────────── */

/*
 * dm_gpu_matmul_fallback — Tier-1 Vulkan path.
 *
 * The real Vulkan cooperative-matrix shader is future work.
 * For now we delegate to the pure-C CPU kernels so the dispatch layer is
 * complete and correct even when TFE is absent.
 *
 * TODO: implement a real Vulkan cooperative matrix GEMM shader here.
 */
static void dm_gpu_matmul_fallback(const float *A, const float *B, float *C,
                                    int M, int K, int N, int transpose_B)
{
    if (transpose_B)
        dm_cpu_matmul_nt(A, B, C, M, N, K);
    else
        dm_cpu_matmul(A, B, C, M, K, N);
}

void dm_matmul_dispatch(const float *A, const float *B, float *C,
                         int M, int K, int N, int transpose_B)
{
    if (!g_inited) dm_backend_init();

    switch (g_info.active) {
        case DM_BACKEND_TENSORFLOW:
            /* Delegate to TFE path in dm_engine.c */
            if (transpose_B)
                dm_matmul_nt(A, B, C, M, N, K);
            else
                dm_matmul_nn(A, B, C, M, K, N);
            return;

        case DM_BACKEND_VULKAN_COMPUTE:
        case DM_BACKEND_VULKAN_COOP_MAT:
            dm_gpu_matmul_fallback(A, B, C, M, K, N, transpose_B);
            return;

        case DM_BACKEND_CPU:
        default:
            if (transpose_B)
                dm_cpu_matmul_nt(A, B, C, M, N, K);
            else
                dm_cpu_matmul(A, B, C, M, K, N);
            return;
    }
}
