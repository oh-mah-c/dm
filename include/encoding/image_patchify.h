#ifndef DM_IMAGE_PATCHIFY_H
#define DM_IMAGE_PATCHIFY_H

#include "models/tensor.h"

int dm_image_load_ppm_rgb_f32(const char *path, DM_Tensor *out);
int dm_image_resize_nearest(const DM_Tensor *in, DM_Tensor *out, int h, int w);
int dm_image_patchify(const DM_Tensor *in, DM_Tensor *out, int patch_h, int patch_w);

#endif /* DM_IMAGE_PATCHIFY_H */
