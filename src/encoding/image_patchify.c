#include "encoding/image_patchify.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ppm_next_token(FILE *fp, char *buf, size_t cap) {
    int c;
    size_t n = 0;
    do {
        c = fgetc(fp);
        if (c == '#') while (c != '\n' && c != EOF) c = fgetc(fp);
    } while (c != EOF && isspace(c));
    if (c == EOF) return -1;
    while (c != EOF && !isspace(c)) {
        if (n + 1 < cap) buf[n++] = (char)c;
        c = fgetc(fp);
    }
    buf[n] = '\0';
    return 0;
}

int dm_image_load_ppm_rgb_f32(const char *path, DM_Tensor *out) {
    FILE *fp;
    char tok[64];
    int w, h, maxv, y, x, c;
    fp = fopen(path, "rb");
    if (!fp) return -1;
    if (ppm_next_token(fp, tok, sizeof(tok)) != 0 || strcmp(tok, "P6") != 0) { fclose(fp); return -1; }
    if (ppm_next_token(fp, tok, sizeof(tok)) != 0) { fclose(fp); return -1; }
    w = atoi(tok);
    if (ppm_next_token(fp, tok, sizeof(tok)) != 0) { fclose(fp); return -1; }
    h = atoi(tok);
    if (ppm_next_token(fp, tok, sizeof(tok)) != 0) { fclose(fp); return -1; }
    maxv = atoi(tok);
    if (w <= 0 || h <= 0 || maxv <= 0 || dm_tensor_alloc(out, 1, 3, h, w) != 0) { fclose(fp); return -1; }
    for (y = 0; y < h; y++) for (x = 0; x < w; x++) for (c = 0; c < 3; c++) {
        int v = fgetc(fp);
        if (v == EOF) { fclose(fp); dm_tensor_free(out); return -1; }
        dm_tensor_set(out, 0, c, y, x, ((float)v / (float)maxv - 0.5f) * 2.0f);
    }
    fclose(fp);
    return 0;
}

int dm_image_resize_nearest(const DM_Tensor *in, DM_Tensor *out, int h, int w) {
    int n, c, y, x;
    if (!in || !out || h <= 0 || w <= 0) return -1;
    if (dm_tensor_alloc(out, in->n, in->c, h, w) != 0) return -1;
    for (n = 0; n < in->n; n++) for (c = 0; c < in->c; c++) for (y = 0; y < h; y++) for (x = 0; x < w; x++) {
        int sy = (int)((long long)y * in->h / h);
        int sx = (int)((long long)x * in->w / w);
        dm_tensor_set(out, n, c, y, x, dm_tensor_get(in, n, c, sy, sx));
    }
    return 0;
}

int dm_image_patchify(const DM_Tensor *in, DM_Tensor *out, int patch_h, int patch_w) {
    int oh, ow, n, c, py, px, y, x, oc;
    if (!in || !out || patch_h <= 0 || patch_w <= 0 || in->h % patch_h || in->w % patch_w) return -1;
    oh = in->h / patch_h;
    ow = in->w / patch_w;
    if (dm_tensor_alloc(out, in->n, in->c * patch_h * patch_w, oh, ow) != 0) return -1;
    for (n = 0; n < in->n; n++) for (c = 0; c < in->c; c++) for (py = 0; py < patch_h; py++) for (px = 0; px < patch_w; px++) {
        oc = (c * patch_h + py) * patch_w + px;
        for (y = 0; y < oh; y++) for (x = 0; x < ow; x++) {
            dm_tensor_set(out, n, oc, y, x, dm_tensor_get(in, n, c, y * patch_h + py, x * patch_w + px));
        }
    }
    return 0;
}
