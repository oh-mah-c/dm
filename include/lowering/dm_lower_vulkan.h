#ifndef DM_LOWER_VULKAN_H
#define DM_LOWER_VULKAN_H

#include "core/dm_block.h"

#ifdef __cplusplus
extern "C" {
#endif

int dm_block_to_vulkan_buffer(const DM_Block *src, DM_Block *dst);
int dm_vulkan_buffer_to_block(const DM_Block *src, DM_Block *dst);

#ifdef __cplusplus
}
#endif

#endif // DM_LOWER_VULKAN_H
