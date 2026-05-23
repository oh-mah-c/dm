#include "dm_meta.h"

const char* dm_kind_name(DM_BlockKind kind) {
    switch (kind) {
        case DM_KIND_DENSE:          return "DENSE";
        case DM_KIND_SPARSE_CSR:     return "SPARSE_CSR";
        case DM_KIND_SPARSE_COO:     return "SPARSE_COO";
        case DM_KIND_RAGGED:         return "RAGGED";
        case DM_KIND_SEQUENCE:       return "SEQUENCE";
        case DM_KIND_TOKEN_STREAM:   return "TOKEN_STREAM";
        case DM_KIND_GRAPH:          return "GRAPH";
        case DM_KIND_TABLE:          return "TABLE";
        case DM_KIND_IMAGE:          return "IMAGE";
        case DM_KIND_PATCHES:        return "PATCHES";
        case DM_KIND_DISTRIBUTION:   return "DISTRIBUTION";
        case DM_KIND_SYMBOLIC:       return "SYMBOLIC";
        case DM_KIND_EXTERNAL:       return "EXTERNAL";
        default:                     return "UNKNOWN";
    }
}
