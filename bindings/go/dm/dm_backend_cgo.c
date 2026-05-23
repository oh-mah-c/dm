/*
 * dm_backend_cgo.c — CGo shim for dm_backend_* functions.
 *
 * CGo language servers (gopls) sometimes fail to resolve enum-typed
 * parameters from headers found only via #cgo CFLAGS.  This file provides
 * plain-int wrappers that are always visible to the C compiler that builds
 * the CGo package, so `go build` and IDE analysis both work.
 *
 * This file is compiled as part of the CGo package (any .c file in the
 * package directory is compiled automatically by `cgo`).
 *
 * SPDX-License-Identifier: MIT
 */

/* Pull in the real declarations from the project header. */
#include "../../../include/dm.h"

void dm_cgo_backend_init(void) {
    dm_backend_init();
}

int dm_cgo_backend_get(void) {
    return (int)dm_backend_get();
}

void dm_cgo_backend_set(int b) {
    dm_backend_set((DM_Backend)b);
}

typedef struct {
    int active;
    int tf_available;
    int vulkan_available;
    int coop_mat_available;
    int cuda_available;
    int rocm_available;
} DmCgoBackendInfo;

DmCgoBackendInfo dm_cgo_backend_query(void) {
    DM_BackendInfo qi = dm_backend_query();
    DmCgoBackendInfo out;
    out.active            = (int)qi.active;
    out.tf_available      = qi.tf_available;
    out.vulkan_available  = qi.vulkan_available;
    out.coop_mat_available= qi.coop_mat_available;
    out.cuda_available    = qi.cuda_available;
    out.rocm_available    = qi.rocm_available;
    return out;
}

const char *dm_cgo_backend_name(int b) {
    return dm_backend_name((DM_Backend)b);
}
