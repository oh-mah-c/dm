#ifndef DM_LOWER_TF_H
#define DM_LOWER_TF_H

#include "core/dm_block.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void DM_TF_Tensor;

int dm_block_to_tf_tensor(const DM_Block *block, DM_TF_Tensor **out_tensor);
int dm_tf_tensor_to_block(const DM_TF_Tensor *tensor, DM_Block *out, DM_BlockKind kind, DM_Role role);
int dm_tf_tensor_copy_to_block(const DM_TF_Tensor *tensor, DM_Block *out);
void dm_tf_tensor_free(void *tensor);

#ifdef __cplusplus
}
#endif

#endif // DM_LOWER_TF_H
