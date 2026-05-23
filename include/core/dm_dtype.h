#ifndef DM_DTYPE_H
#define DM_DTYPE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DM_DTYPE_U8,
    DM_DTYPE_I32,
    DM_DTYPE_I64,
    DM_DTYPE_F32,
    DM_DTYPE_F64,
    DM_DTYPE_BOOL,
    DM_DTYPE_STRING,
    DM_DTYPE_OBJECT
} DM_DType;

const char* dm_dtype_name(DM_DType dtype);
int dm_dtype_size(DM_DType dtype);

#ifdef __cplusplus
}
#endif

#endif // DM_DTYPE_H
