#include "dm_dtype.h"

const char* dm_dtype_name(DM_DType dtype) {
    switch (dtype) {
        case DM_DTYPE_U8:     return "U8";
        case DM_DTYPE_I32:    return "I32";
        case DM_DTYPE_I64:    return "I64";
        case DM_DTYPE_F32:    return "F32";
        case DM_DTYPE_F64:    return "F64";
        case DM_DTYPE_BOOL:   return "BOOL";
        case DM_DTYPE_STRING: return "STRING";
        case DM_DTYPE_OBJECT: return "OBJECT";
        default:              return "UNKNOWN";
    }
}

int dm_dtype_size(DM_DType dtype) {
    switch (dtype) {
        case DM_DTYPE_U8:     return 1;
        case DM_DTYPE_I32:    return 4;
        case DM_DTYPE_I64:    return 8;
        case DM_DTYPE_F32:    return 4;
        case DM_DTYPE_F64:    return 8;
        case DM_DTYPE_BOOL:   return 1;
        case DM_DTYPE_STRING: return sizeof(void*);
        case DM_DTYPE_OBJECT: return sizeof(void*);
        default:              return 0;
    }
}
