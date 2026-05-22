#ifndef DM_MOBILENET_TINY_H
#define DM_MOBILENET_TINY_H

#include "models/tensor.h"

typedef enum {
    DM_UIB_FFN = 0,
    DM_UIB_IB = 1,
    DM_UIB_CONVNEXT = 2,
    DM_UIB_EXTRADW = 3
} DM_UIBKind;

int dm_uib_block(const DM_Tensor *in, DM_Tensor *out, DM_UIBKind kind,
                 int expanded_c, int out_c, int kernel1, int kernel2, int stride, unsigned int seed);
int dm_mobile_mqa_block(const DM_Tensor *in, DM_Tensor *out, int heads, int key_dim,
                        int spatial_reduction, unsigned int seed);
int dm_mobilenet_tiny_forward(const DM_Tensor *input, DM_Tensor *logits, int classes, unsigned int seed);
int dm_mobilenet_tiny_cli(int argc, char **argv);

#endif /* DM_MOBILENET_TINY_H */
