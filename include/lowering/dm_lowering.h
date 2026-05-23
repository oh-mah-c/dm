#ifndef DM_LOWERING_H
#define DM_LOWERING_H

#include "core/dm_block.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DM_LOWER_COPY,  /* Force copy of data to target backend */
    DM_LOWER_VIEW,  /* Create a view/reference if possible, otherwise copy */
    DM_LOWER_MOVE   /* Transfer ownership to target backend */
} DM_LowerPolicy;

int dm_lower_block(const DM_Block *src, DM_Block *dst, DM_Backend target_backend, DM_LowerPolicy policy);
int dm_raise_block(const DM_Block *src, DM_Block *dst, DM_Backend target_backend, DM_LowerPolicy policy);

#ifdef __cplusplus
}
#endif

#endif // DM_LOWERING_H
