#include "dm_layout.h"

const char* dm_layout_name(DM_Layout layout) {
    switch (layout) {
        case DM_LAYOUT_NONE:           return "NONE";
        case DM_LAYOUT_FLAT:           return "FLAT";
        case DM_LAYOUT_ROW_MAJOR:      return "ROW_MAJOR";
        case DM_LAYOUT_COLUMN_MAJOR:   return "COLUMN_MAJOR";
        case DM_LAYOUT_NCHW:           return "NCHW";
        case DM_LAYOUT_NHWC:           return "NHWC";
        case DM_LAYOUT_CSR:            return "CSR";
        case DM_LAYOUT_COO:            return "COO";
        case DM_LAYOUT_RAGGED:         return "RAGGED";
        case DM_LAYOUT_PACKED_TOKENS:  return "PACKED_TOKENS";
        case DM_LAYOUT_GRAPH_CSR:      return "GRAPH_CSR";
        case DM_LAYOUT_VULKAN_BUFFER:  return "VULKAN_BUFFER";
        case DM_LAYOUT_TF_HANDLE:      return "TF_HANDLE";
        case DM_LAYOUT_DATASET:        return "DATASET";
        default:                       return "UNKNOWN";
    }
}

const char* dm_layout_policy_name(DM_LayoutPolicy policy) {
    switch(policy) {
        case DM_LAYOUT_POLICY_STRICT: return "STRICT";
        case DM_LAYOUT_POLICY_AUTO_TRANSPOSE: return "AUTO_TRANSPOSE";
        case DM_LAYOUT_POLICY_BACKEND_NATIVE: return "BACKEND_NATIVE";
        default: return "UNKNOWN";
    }
}
