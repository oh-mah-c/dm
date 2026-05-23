#ifndef DM_META_H
#define DM_META_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DM_KIND_DENSE,
    DM_KIND_SPARSE_CSR,
    DM_KIND_SPARSE_COO,
    DM_KIND_RAGGED,
    DM_KIND_SEQUENCE,
    DM_KIND_TOKEN_STREAM,
    DM_KIND_GRAPH,
    DM_KIND_TABLE,
    DM_KIND_IMAGE,
    DM_KIND_PATCHES,
    DM_KIND_DISTRIBUTION,
    DM_KIND_SYMBOLIC,
    DM_KIND_EXTERNAL
} DM_BlockKind;

const char* dm_kind_name(DM_BlockKind kind);

typedef struct {
    char key[64];
    char value[256];
} DM_MetaItem;

#ifdef __cplusplus
}
#endif

#endif // DM_META_H
