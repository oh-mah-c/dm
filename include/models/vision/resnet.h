#ifndef DM_RESNET_H
#define DM_RESNET_H

#include "core/dm_engine.h"

int dm_resnet_basic_block(const DM_Block *in, DM_Block *out, int out_c, int stride, unsigned int seed);
int dm_resnet18_forward(const DM_Block *input, DM_Block *logits, int classes, unsigned int seed);
int dm_resnet18_cli(int argc, char **argv);

#endif /* DM_RESNET_H */
