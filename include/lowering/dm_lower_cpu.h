#ifndef DM_LOWER_CPU_H
#define DM_LOWER_CPU_H

#include "core/dm_block.h"

#ifdef __cplusplus
extern "C" {
#endif

int dm_block_to_cpu(const DM_Block *src, DM_Block *dst);
int dm_cpu_to_block(const DM_Block *src, DM_Block *dst);

#ifdef __cplusplus
}
#endif

#endif // DM_LOWER_CPU_H
