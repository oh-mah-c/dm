#ifndef DM_LAYOUT_H
#define DM_LAYOUT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DM_LAYOUT_NONE,
    DM_LAYOUT_FLAT,
    DM_LAYOUT_ROW_MAJOR,
    DM_LAYOUT_COLUMN_MAJOR,
    DM_LAYOUT_NCHW,
    DM_LAYOUT_NHWC,
    DM_LAYOUT_CSR,
    DM_LAYOUT_COO,
    DM_LAYOUT_RAGGED,
    DM_LAYOUT_PACKED_TOKENS,
    DM_LAYOUT_GRAPH_CSR,
    DM_LAYOUT_VULKAN_BUFFER,
    DM_LAYOUT_TF_HANDLE,
    DM_LAYOUT_DATASET
} DM_Layout;

typedef enum {
    DM_LAYOUT_POLICY_STRICT,
    DM_LAYOUT_POLICY_AUTO_TRANSPOSE,
    DM_LAYOUT_POLICY_BACKEND_NATIVE
} DM_LayoutPolicy;

const char* dm_layout_name(DM_Layout layout);
const char* dm_layout_policy_name(DM_LayoutPolicy policy);

#ifdef __cplusplus
}
#endif

#endif // DM_LAYOUT_H
