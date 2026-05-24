#include "models/vision/swin.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    DM_SwinConfig cfg;
    DM_SwinModel model;
    DM_Block image = {0}, logits = {0};
    const char *weights = argc > 1 ? argv[1] : NULL;
    int rc;

    dm_swin_config_init(&cfg, DM_SWIN_TINY, 1000, 224);
    if (dm_swin_model_init(&model, &cfg) != 0) {
        fprintf(stderr, "swin init failed\n");
        return 1;
    }
    if (!weights || dm_swin_model_load_weights(&model, weights) != 0) {
        fprintf(stderr, "usage: %s weights/swin_tiny_patch4_window7_224/metadata.json\n", argv[0]);
        dm_swin_model_free(&model);
        return 1;
    }
    if (dm_block_create(&image, DM_KIND_IMAGE, DM_DTYPE_F32, DM_LAYOUT_NHWC, DM_BACKEND_CPU, 4,
                        (int64_t[]){1, 224, 224, 3}) != 0) {
        dm_swin_model_free(&model);
        return 1;
    }
    rc = dm_swin_model_forward(&model, &image, &logits);
    if (rc == 0) {
        printf("logits[0]=%g\n", ((float *)logits.data)[0]);
    }
    dm_block_free(&image);
    dm_block_free(&logits);
    dm_swin_model_free(&model);
    return rc == 0 ? 0 : 1;
}
