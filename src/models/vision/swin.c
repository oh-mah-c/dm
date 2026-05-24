#include "models/vision/swin.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/dm_engine.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static int block_create(DM_Block *b, int ndim, const int64_t *shape) {
    return dm_block_create(b, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, ndim, shape);
}

static int is_f32(const DM_Block *b) {
    return b && b->dtype == DM_DTYPE_F32 && b->backend == DM_BACKEND_CPU && b->data;
}

static int init_attn(DM_SwinWindowAttentionBlock *attn, int dim, int window_size, int num_heads);

static void mark_dirty(DM_Block *b) {
    if (b) {
        b->dirty = 1;
        b->version++;
    }
}

static char *dm_swin_strdup(const char *s) {
    size_t n;
    char *out;
    if (!s) return NULL;
    n = strlen(s) + 1;
    out = (char *)malloc(n);
    if (out) memcpy(out, s, n);
    return out;
}

static uint32_t rng_next(uint32_t *s) {
    uint32_t x = *s ? *s : 2463534242u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static float rng_weight(uint32_t *s, float scale) {
    return (((rng_next(s) >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f) * scale;
}

static int store_reserve(DM_SwinTensorStore *store, int cap) {
    DM_SwinNamedTensor *next;
    if (store->capacity >= cap) return 0;
    next = (DM_SwinNamedTensor *)realloc(store->items, (size_t)cap * sizeof(*next));
    if (!next) return -1;
    memset(next + store->capacity, 0, (size_t)(cap - store->capacity) * sizeof(*next));
    store->items = next;
    store->capacity = cap;
    return 0;
}

static DM_Block *store_add(DM_SwinTensorStore *store, const char *name, int ndim, const int64_t *shape) {
    DM_SwinNamedTensor *it;
    if (store->count == store->capacity && store_reserve(store, store->capacity ? store->capacity * 2 : 128) != 0) return NULL;
    it = &store->items[store->count++];
    it->name = dm_swin_strdup(name);
    if (!it->name) return NULL;
    if (block_create(&it->tensor, ndim, shape) != 0) return NULL;
    if (block_create(&it->grad, ndim, shape) != 0) {
        dm_block_free(&it->tensor); return NULL;
    }
    memset(it->grad.data, 0, it->grad.bytes); // zero gradients initially
    return &it->tensor;
}

static DM_Block *store_find(DM_SwinTensorStore *store, const char *name, int mark_used) {
    for (int i = 0; i < store->count; i++) {
        if (store->items[i].name && strcmp(store->items[i].name, name) == 0) {
            if (mark_used) store->items[i].used = 1;
            return &store->items[i].tensor;
        }
    }
    return NULL;
}

static DM_Block *store_find_grad(DM_SwinTensorStore *store, const char *name) {
    for (int i = 0; i < store->count; i++) {
        if (store->items[i].name && strcmp(store->items[i].name, name) == 0) {
            return &store->items[i].grad;
        }
    }
    return NULL;
}

static void store_free(DM_SwinTensorStore *store) {
    if (!store) return;
    for (int i = 0; i < store->count; i++) {
        free(store->items[i].name);
        dm_block_free(&store->items[i].tensor);
        dm_block_free(&store->items[i].grad);
    }
    free(store->items);
    memset(store, 0, sizeof(*store));
}

static int shape_eq(const DM_Block *b, int ndim, const int64_t *shape) {
    if (!b || b->ndim != ndim) return 0;
    for (int i = 0; i < ndim; i++) {
        if (b->shape[i] != shape[i]) return 0;
    }
    return 1;
}

static DM_Block *bind_tensor(DM_SwinModel *model, const char *name, int ndim, const int64_t *shape) {
    DM_Block *b = store_find(&model->tensors, name, 1);
    if (!b) {
        fprintf(stderr, "swin: missing tensor %s\n", name);
        return NULL;
    }
    if (!shape_eq(b, ndim, shape)) {
        fprintf(stderr, "swin: tensor %s shape mismatch: got [", name);
        for (int i = 0; i < b->ndim; i++) fprintf(stderr, "%s%lld", i ? "," : "", (long long)b->shape[i]);
        fprintf(stderr, "] expected [");
        for (int i = 0; i < ndim; i++) fprintf(stderr, "%s%lld", i ? "," : "", (long long)shape[i]);
        fprintf(stderr, "]\n");
        return NULL;
    }
    return b;
}

static DM_Block *bind_grad(DM_SwinModel *model, const char *name) {
    DM_Block *b = store_find_grad(&model->tensors, name);
    if (!b) fprintf(stderr, "swin: missing grad %s\n", name);
    return b;
}

static void bind_ln(DM_SwinModel *model, DM_SwinLayerNormBlock *ln, const char *prefix, int dim, int *ok) {
    char name[512];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    ln->weight = bind_tensor(model, name, 1, (int64_t[]){dim});
    ln->grad_weight = bind_grad(model, name);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    ln->bias = bind_tensor(model, name, 1, (int64_t[]){dim});
    ln->grad_bias = bind_grad(model, name);
    if (!ln->weight || !ln->bias) *ok = 0;
}

static void bind_linear(DM_SwinModel *model, DM_SwinLinearBlock *lin, const char *prefix, int out_dim, int in_dim, int has_bias, int *ok) {
    char name[512];
    if (strlen(prefix) + 8 >= sizeof(name)) { *ok = 0; return; }
    strcpy(name, prefix);
    strcat(name, ".weight");
    lin->weight = bind_tensor(model, name, 2, (int64_t[]){out_dim, in_dim});
    lin->grad_weight = bind_grad(model, name);
    if (has_bias) {
        if (strlen(prefix) + 6 >= sizeof(name)) { *ok = 0; return; }
        strcpy(name, prefix);
        strcat(name, ".bias");
        lin->bias = bind_tensor(model, name, 1, (int64_t[]){out_dim});
        lin->grad_bias = bind_grad(model, name);
    } else {
        lin->bias = NULL;
        lin->grad_bias = NULL;
    }
    if (!lin->weight || (has_bias && !lin->bias)) *ok = 0;
}

static int linear_forward(const DM_SwinLinearBlock *lin, const DM_Block *x, DM_Block *y, DM_ActivationCache *cache) {
    int64_t b, l, in_dim, out_dim;
    float *yp;
    const float *bp;
    if (!lin || !is_f32(x) || !lin->weight || !is_f32(lin->weight)) return -1;
    if (x->ndim != 3 || lin->weight->ndim != 2) return -1;
    b = x->shape[0];
    l = x->shape[1];
    in_dim = x->shape[2];
    out_dim = lin->weight->shape[0];
    if (lin->weight->shape[1] != in_dim) return -1;
    if (lin->bias && (!is_f32(lin->bias) || lin->bias->ndim != 1 || lin->bias->shape[0] != out_dim)) return -1;
    if (block_create(y, 3, (int64_t[]){b, l, out_dim}) != 0) return -1;
    if (cache) dm_activation_cache_push(cache, x);

    DM_Block A_view = *x;
    A_view.ndim = 2;
    A_view.shape[0] = b * l;
    A_view.shape[1] = in_dim;

    DM_Block y_view = *y;
    y_view.ndim = 2;
    y_view.shape[0] = b * l;
    y_view.shape[1] = out_dim;

    dm_matmul_nt(&A_view, lin->weight, &y_view);

    if (lin->bias) {
        yp = (float *)y->data;
        bp = (const float *)lin->bias->data;
        for (int64_t i = 0; i < b * l; i++) {
            for (int64_t o = 0; o < out_dim; o++) {
                yp[i * out_dim + o] += bp[o];
            }
        }
    }
    mark_dirty(y);
    return 0;
}

static int linear_backward_helper(DM_SwinLinearBlock *lin, const DM_Block *d_y, DM_Block *d_x, DM_ActivationCache *cache) {
    DM_Block x = {0};
    if (!cache || cache->count == 0) return -1;
    x = cache->activations[--cache->count];
    
    int64_t b = x.shape[0], l = x.shape[1], in_dim = x.shape[2], out_dim = lin->weight->shape[0];
    DM_Block A_view = x; A_view.ndim = 2; A_view.shape[0] = b * l; A_view.shape[1] = in_dim;
    DM_Block d_y_view = *d_y; d_y_view.ndim = 2; d_y_view.shape[0] = b * l; d_y_view.shape[1] = out_dim;
    
    if (d_x && block_create(d_x, 3, x.shape) != 0) { dm_block_free(&x); return -1; }
    DM_Block d_x_view = {0};
    if (d_x) { d_x_view = *d_x; d_x_view.ndim = 2; d_x_view.shape[0] = b * l; d_x_view.shape[1] = in_dim; }
    
    DM_Block tmp_grad_w = {0};
    if (block_create(&tmp_grad_w, 2, lin->weight->shape) != 0) { dm_block_free(&x); return -1; }
    dm_matmul_nt_backward(&A_view, lin->weight, &d_y_view, d_x ? &d_x_view : NULL, &tmp_grad_w);
    
    if (lin->grad_weight) {
        float *gw = (float *)lin->grad_weight->data;
        float *tw = (float *)tmp_grad_w.data;
        for (size_t i = 0; i < tmp_grad_w.count; i++) gw[i] += tw[i];
    }
    dm_block_free(&tmp_grad_w);
    
    if (lin->grad_bias) {
        float *gb = (float *)lin->grad_bias->data;
        const float *dy_ptr = (const float *)d_y->data;
        for (int64_t i = 0; i < b * l; i++) {
            for (int64_t o = 0; o < out_dim; o++) gb[o] += dy_ptr[i * out_dim + o];
        }
    }
    dm_block_free(&x);
    return 0;
}

static int layer_norm_forward(const DM_SwinLayerNormBlock *ln, const DM_Block *x, DM_Block *y, DM_ActivationCache *cache) {
    int64_t dim;
    if (!ln || !is_f32(x) || !ln->weight || !ln->bias || !is_f32(ln->weight) || !is_f32(ln->bias)) return -1;
    if (x->ndim < 2) return -1;
    dim = x->shape[x->ndim - 1];
    if (ln->weight->ndim != 1 || ln->bias->ndim != 1 || ln->weight->shape[0] != dim || ln->bias->shape[0] != dim) return -1;
    if (block_create(y, x->ndim, x->shape) != 0) return -1;
    memcpy(y->data, x->data, x->bytes);
    if (cache) dm_activation_cache_push(cache, x);
    
    DM_Block y_view = *y;
    if (y_view.ndim == 2) {
        y_view.ndim = 3;
        y_view.shape[0] = 1;
        y_view.shape[1] = y->shape[0];
        y_view.shape[2] = y->shape[1];
    } else if (y_view.ndim == 4) {
        y_view.ndim = 3;
        y_view.shape[0] = y->shape[0];
        y_view.shape[1] = y->shape[1] * y->shape[2];
        y_view.shape[2] = y->shape[3];
    }
    
    return dm_layer_norm_seq(&y_view, ln->weight, ln->bias, 1e-5f);
}

static int layer_norm_backward_helper(DM_SwinLayerNormBlock *ln, const DM_Block *d_y, DM_Block *d_x, DM_ActivationCache *cache) {
    DM_Block x = {0};
    if (!cache || cache->count == 0) return -1;
    x = cache->activations[--cache->count];
    
    if (d_x && block_create(d_x, x.ndim, x.shape) != 0) { dm_block_free(&x); return -1; }
    
    DM_Block tmp_gw = {0}, tmp_gb = {0};
    if (block_create(&tmp_gw, 1, ln->weight->shape) != 0) { dm_block_free(&x); return -1; }
    if (block_create(&tmp_gb, 1, ln->bias->shape) != 0) { dm_block_free(&x); dm_block_free(&tmp_gw); return -1; }
    
    DM_Block d_y_view = *d_y, x_view = x;
    if (d_y_view.ndim == 2) {
        d_y_view.ndim = 3; d_y_view.shape[0] = 1; d_y_view.shape[1] = d_y->shape[0]; d_y_view.shape[2] = d_y->shape[1];
        x_view.ndim = 3; x_view.shape[0] = 1; x_view.shape[1] = x.shape[0]; x_view.shape[2] = x.shape[1];
    } else if (d_y_view.ndim == 4) {
        d_y_view.ndim = 3; d_y_view.shape[0] = d_y->shape[0]; d_y_view.shape[1] = d_y->shape[1]*d_y->shape[2]; d_y_view.shape[2] = d_y->shape[3];
        x_view.ndim = 3; x_view.shape[0] = x.shape[0]; x_view.shape[1] = x.shape[1]*x.shape[2]; x_view.shape[2] = x.shape[3];
    }
    DM_Block d_x_view = {0};
    if (d_x) {
        d_x_view = *d_x;
        d_x_view.ndim = 3; d_x_view.shape[0] = x_view.shape[0]; d_x_view.shape[1] = x_view.shape[1]; d_x_view.shape[2] = x_view.shape[2];
    }
    
    dm_layer_norm_seq_backward(&x_view, ln->weight, &d_y_view, 1e-5f, d_x ? &d_x_view : NULL, &tmp_gw, &tmp_gb);
    
    if (ln->grad_weight) {
        float *gw = (float *)ln->grad_weight->data, *tw = (float *)tmp_gw.data;
        for(size_t i=0; i<tmp_gw.count; i++) gw[i] += tw[i];
    }
    if (ln->grad_bias) {
        float *gb = (float *)ln->grad_bias->data, *tb = (float *)tmp_gb.data;
        for(size_t i=0; i<tmp_gb.count; i++) gb[i] += tb[i];
    }
    
    dm_block_free(&tmp_gw); dm_block_free(&tmp_gb); dm_block_free(&x);
    return 0;
}

static void gelu_inplace(DM_Block *x, DM_ActivationCache *cache) {
    if (cache) dm_activation_cache_push(cache, x);
    dm_gelu_inplace((float *)x->data, (int)x->count);
    mark_dirty(x);
}

static void softmax_rows(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + (size_t)r * cols;
        float m = row[0], sum = 0.0f;
        for (int c = 1; c < cols; c++) if (row[c] > m) m = row[c];
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - m);
            sum += row[c];
        }
        if (sum != 0.0f) {
            float inv = 1.0f / sum;
            for (int c = 0; c < cols; c++) row[c] *= inv;
        }
    }
}

static int add_inplace(DM_Block *x, const DM_Block *res) {
    float *a;
    const float *b;
    if (!is_f32(x) || !is_f32(res) || x->count != res->count) return -1;
    for (int i = 0; i < x->ndim; i++) if (x->shape[i] != res->shape[i]) return -1;
    a = (float *)x->data;
    b = (const float *)res->data;
    for (size_t i = 0; i < x->count; i++) a[i] += b[i];
    mark_dirty(x);
    return 0;
}

static int copy_block(const DM_Block *src, DM_Block *dst) {
    if (!src || !src->data) return -1;
    if (block_create(dst, src->ndim, src->shape) != 0) return -1;
    memcpy(dst->data, src->data, src->bytes);
    return 0;
}

static int nlc_to_bhwc(const DM_Block *x, int h, int w, DM_Block *y) {
    int64_t b, c;
    if (!is_f32(x) || x->ndim != 3) return -1;
    b = x->shape[0];
    c = x->shape[2];
    if (x->shape[1] != (int64_t)h * w) return -1;
    if (block_create(y, 4, (int64_t[]){b, h, w, c}) != 0) return -1;
    memcpy(y->data, x->data, x->bytes);
    return 0;
}

static int bhwc_to_nlc(const DM_Block *x, DM_Block *y) {
    if (!is_f32(x) || x->ndim != 4) return -1;
    if (block_create(y, 3, (int64_t[]){x->shape[0], x->shape[1] * x->shape[2], x->shape[3]}) != 0) return -1;
    memcpy(y->data, x->data, x->bytes);
    return 0;
}

static int roll_bhwc(const DM_Block *x, int shift_h, int shift_w, DM_Block *y) {
    int64_t b, h, w, c;
    const float *xp;
    float *yp;
    if (!is_f32(x) || x->ndim != 4) return -1;
    b = x->shape[0]; h = x->shape[1]; w = x->shape[2]; c = x->shape[3];
    if (block_create(y, 4, x->shape) != 0) return -1;
    xp = (const float *)x->data;
    yp = (float *)y->data;
    for (int64_t bi = 0; bi < b; bi++) {
        for (int64_t yy = 0; yy < h; yy++) {
            int64_t src_y = (yy - shift_h) % h;
            if (src_y < 0) src_y += h;
            for (int64_t xx = 0; xx < w; xx++) {
                int64_t src_x = (xx - shift_w) % w;
                if (src_x < 0) src_x += w;
                memcpy(yp + (((bi * h + yy) * w + xx) * c),
                       xp + (((bi * h + src_y) * w + src_x) * c),
                       (size_t)c * sizeof(float));
            }
        }
    }
    return 0;
}

int dm_swin_window_partition(const DM_Block *x, int window_size, DM_Block *windows) {
    int64_t b, h, w, c, nwh, nww;
    const float *xp;
    float *wp;
    if (!is_f32(x) || x->ndim != 4 || window_size <= 0) return -1;
    b = x->shape[0]; h = x->shape[1]; w = x->shape[2]; c = x->shape[3];
    if (h % window_size != 0 || w % window_size != 0) return -1;
    nwh = h / window_size;
    nww = w / window_size;
    if (block_create(windows, 4, (int64_t[]){b * nwh * nww, window_size, window_size, c}) != 0) return -1;
    xp = (const float *)x->data;
    wp = (float *)windows->data;
    for (int64_t bi = 0; bi < b; bi++)
    for (int64_t wh = 0; wh < nwh; wh++)
    for (int64_t ww = 0; ww < nww; ww++) {
        int64_t win = (bi * nwh + wh) * nww + ww;
        for (int64_t iy = 0; iy < window_size; iy++)
        for (int64_t ix = 0; ix < window_size; ix++) {
            memcpy(wp + (((win * window_size + iy) * window_size + ix) * c),
                   xp + (((bi * h + wh * window_size + iy) * w + ww * window_size + ix) * c),
                   (size_t)c * sizeof(float));
        }
    }
    return 0;
}

int dm_swin_window_reverse(const DM_Block *windows, int window_size, int h, int w, DM_Block *x_bhwc) {
    int64_t nw, c, nwh, nww, b;
    const float *wp;
    float *xp;
    if (!is_f32(windows) || windows->ndim != 4 || window_size <= 0 || h % window_size != 0 || w % window_size != 0) return -1;
    nw = windows->shape[0];
    c = windows->shape[3];
    nwh = h / window_size;
    nww = w / window_size;
    if (nw % (nwh * nww) != 0) return -1;
    b = nw / (nwh * nww);
    if (block_create(x_bhwc, 4, (int64_t[]){b, h, w, c}) != 0) return -1;
    wp = (const float *)windows->data;
    xp = (float *)x_bhwc->data;
    for (int64_t bi = 0; bi < b; bi++)
    for (int64_t wh = 0; wh < nwh; wh++)
    for (int64_t ww = 0; ww < nww; ww++) {
        int64_t win = (bi * nwh + wh) * nww + ww;
        for (int64_t iy = 0; iy < window_size; iy++)
        for (int64_t ix = 0; ix < window_size; ix++) {
            memcpy(xp + (((bi * h + wh * window_size + iy) * w + ww * window_size + ix) * c),
                   wp + (((win * window_size + iy) * window_size + ix) * c),
                   (size_t)c * sizeof(float));
        }
    }
    return 0;
}

int dm_swin_build_relative_position_index(int window_size, int *out_index) {
    int n = window_size * window_size;
    if (!out_index || window_size <= 0) return -1;
    for (int i = 0; i < n; i++) {
        int ih = i / window_size;
        int iw = i % window_size;
        for (int j = 0; j < n; j++) {
            int jh = j / window_size;
            int jw = j % window_size;
            int rh = ih - jh + window_size - 1;
            int rw = iw - jw + window_size - 1;
            out_index[i * n + j] = rh * (2 * window_size - 1) + rw;
        }
    }
    return 0;
}

int dm_swin_build_attention_mask(int h, int w, int window_size, int shift_size, DM_Block *mask) {
    DM_Block img = {0}, wins = {0};
    float *p;
    const float *mw;
    float *out;
    int h_starts[3], h_ends[3], w_starts[3], w_ends[3], cnt = 0;
    int n = window_size * window_size;
    if (shift_size <= 0) return -1;
    if (h % window_size != 0 || w % window_size != 0) return -1;
    if (block_create(&img, 4, (int64_t[]){1, h, w, 1}) != 0) return -1;
    p = (float *)img.data;
    h_starts[0] = 0; h_ends[0] = h - window_size;
    h_starts[1] = h - window_size; h_ends[1] = h - shift_size;
    h_starts[2] = h - shift_size; h_ends[2] = h;
    w_starts[0] = 0; w_ends[0] = w - window_size;
    w_starts[1] = w - window_size; w_ends[1] = w - shift_size;
    w_starts[2] = w - shift_size; w_ends[2] = w;
    for (int hi = 0; hi < 3; hi++)
    for (int wi = 0; wi < 3; wi++) {
        for (int yy = h_starts[hi]; yy < h_ends[hi]; yy++)
        for (int xx = w_starts[wi]; xx < w_ends[wi]; xx++)
            p[(size_t)yy * w + xx] = (float)cnt;
        cnt++;
    }
    if (dm_swin_window_partition(&img, window_size, &wins) != 0) { dm_block_free(&img); return -1; }
    if (block_create(mask, 3, (int64_t[]){wins.shape[0], n, n}) != 0) {
        dm_block_free(&img); dm_block_free(&wins); return -1;
    }
    mw = (const float *)wins.data;
    out = (float *)mask->data;
    for (int64_t ww = 0; ww < wins.shape[0]; ww++) {
        const float *vals = mw + ww * n;
        for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            out[(ww * n + i) * n + j] = vals[i] == vals[j] ? 0.0f : -100.0f;
    }
    dm_block_free(&img);
    dm_block_free(&wins);
    return 0;
}

static int flatten_windows(const DM_Block *wins4, DM_Block *wins3) {
    if (!is_f32(wins4) || wins4->ndim != 4) return -1;
    if (block_create(wins3, 3, (int64_t[]){wins4->shape[0], wins4->shape[1] * wins4->shape[2], wins4->shape[3]}) != 0) return -1;
    memcpy(wins3->data, wins4->data, wins4->bytes);
    return 0;
}

static int unflatten_windows(const DM_Block *wins3, int window_size, DM_Block *wins4) {
    if (!is_f32(wins3) || wins3->ndim != 3 || wins3->shape[1] != (int64_t)window_size * window_size) return -1;
    if (block_create(wins4, 4, (int64_t[]){wins3->shape[0], window_size, window_size, wins3->shape[2]}) != 0) return -1;
    memcpy(wins4->data, wins3->data, wins3->bytes);
    return 0;
}

static int mlp_forward(const DM_SwinMlpBlock *mlp, const DM_Block *x, DM_Block *y, DM_ActivationCache *cache) {
    DM_Block h = {0};
    if (linear_forward(&mlp->fc1, x, &h, cache) != 0) return -1;
    gelu_inplace(&h, cache);
    if (linear_forward(&mlp->fc2, &h, y, cache) != 0) {
        dm_block_free(&h);
        return -1;
    }
    dm_block_free(&h);
    return 0;
}

static int mlp_backward(DM_SwinMlpBlock *mlp, const DM_Block *d_y, DM_Block *d_x, DM_ActivationCache *cache) {
    DM_Block d_h2 = {0}, d_h1 = {0}, pre_gelu_x = {0};
    if (linear_backward_helper(&mlp->fc2, d_y, &d_h2, cache) != 0) return -1;
    if (!cache || cache->count == 0) { dm_block_free(&d_h2); return -1; }
    pre_gelu_x = cache->activations[--cache->count];
    if (block_create(&d_h1, pre_gelu_x.ndim, pre_gelu_x.shape) != 0) { dm_block_free(&d_h2); dm_block_free(&pre_gelu_x); return -1; }
    dm_gelu_backward(&pre_gelu_x, &d_h2, &d_h1);
    dm_block_free(&pre_gelu_x);
    dm_block_free(&d_h2);
    int rc = linear_backward_helper(&mlp->fc1, &d_h1, d_x, cache);
    dm_block_free(&d_h1);
    return rc;
}

static int window_attention_forward(const DM_SwinWindowAttentionBlock *attn, const DM_Block *x, const DM_Block *mask, DM_Block *y, DM_ActivationCache *cache) {
    DM_Block qkv = {0};
    int64_t bw, n, c, h, d;
    const float *qkvp, *rbp, *mp;
    float *score = NULL, *ctx = NULL;
    DM_Block ctx_block = {0};
    int has_mask;
    if (!attn || !is_f32(x) || x->ndim != 3 || !attn->relative_position_index || !attn->relative_position_bias_table) return -1;
    bw = x->shape[0]; n = x->shape[1]; c = x->shape[2]; h = attn->num_heads; d = attn->head_dim;
    if (c != attn->dim || n != (int64_t)attn->window_size * attn->window_size || c != h * d) return -1;
    has_mask = mask && is_f32(mask);
    if (has_mask && (mask->ndim != 3 || mask->shape[1] != n || mask->shape[2] != n || bw % mask->shape[0] != 0)) return -1;
    if (linear_forward(&attn->qkv, x, &qkv, cache) != 0) return -1;
    if (cache) dm_activation_cache_push(cache, &qkv);
    qkvp = (const float *)qkv.data;
    rbp = (const float *)attn->relative_position_bias_table->data;
    mp = has_mask ? (const float *)mask->data : NULL;
    
    DM_Block attn_score = {0};
    if (block_create(&attn_score, 4, (int64_t[]){bw, h, n, n}) != 0) {
        dm_block_free(&qkv); return -1;
    }
    score = (float *)attn_score.data;
    
    ctx = (float *)calloc((size_t)bw * n * c, sizeof(float));
    if (!ctx) {
        dm_block_free(&attn_score); dm_block_free(&qkv); return -1;
    }
    for (int64_t bi = 0; bi < bw; bi++) {
        int64_t mask_win = has_mask ? (bi % mask->shape[0]) : 0;
        for (int64_t head = 0; head < h; head++) {
            float *sp = score + (((bi * h + head) * n) * n);
            for (int64_t qi = 0; qi < n; qi++) {
                for (int64_t ki = 0; ki < n; ki++) {
                    float sum = 0.0f;
                    for (int64_t di = 0; di < d; di++) {
                        size_t qidx = (((size_t)bi * n + qi) * 3 * c) + head * d + di;
                        size_t kidx = (((size_t)bi * n + ki) * 3 * c) + c + head * d + di;
                        sum += qkvp[qidx] * qkvp[kidx];
                    }
                    sum *= attn->scale;
                    sum += rbp[(size_t)attn->relative_position_index[qi * n + ki] * h + head];
                    if (has_mask) sum += mp[((size_t)mask_win * n + qi) * n + ki];
                    sp[qi * n + ki] = sum;
                }
            }
            softmax_rows(sp, (int)n, (int)n);
            for (int64_t qi = 0; qi < n; qi++) {
                for (int64_t vi = 0; vi < n; vi++) {
                    float a = sp[qi * n + vi];
                    for (int64_t di = 0; di < d; di++) {
                        size_t vidx = (((size_t)bi * n + vi) * 3 * c) + 2 * c + head * d + di;
                        ctx[((size_t)bi * n + qi) * c + head * d + di] += a * qkvp[vidx];
                    }
                }
            }
        }
    }
    if (cache) dm_activation_cache_push(cache, &attn_score);
    if (block_create(&ctx_block, 3, (int64_t[]){bw, n, c}) != 0) {
        dm_block_free(&attn_score); free(ctx); dm_block_free(&qkv); return -1;
    }
    memcpy(ctx_block.data, ctx, (size_t)bw * n * c * sizeof(float));
    dm_block_free(&attn_score);
    free(ctx);
    dm_block_free(&qkv);
    if (linear_forward(&attn->proj, &ctx_block, y, cache) != 0) {
        dm_block_free(&ctx_block);
        return -1;
    }
    dm_block_free(&ctx_block);
    return 0;
}

static int window_attention_backward(DM_SwinWindowAttentionBlock *attn, const DM_Block *d_y, DM_Block *d_x, DM_ActivationCache *cache) {
    DM_Block d_ctx = {0}, attn_score = {0}, qkv = {0};
    if (linear_backward_helper(&attn->proj, d_y, &d_ctx, cache) != 0) return -1;
    if (!cache || cache->count < 2) { dm_block_free(&d_ctx); return -1; }
    attn_score = cache->activations[--cache->count];
    qkv = cache->activations[--cache->count];
    
    int64_t bw = qkv.shape[0], n = qkv.shape[1], c = qkv.shape[2] / 3;
    int h = attn->num_heads, d = attn->head_dim;
    DM_Block d_qkv = {0};
    if (block_create(&d_qkv, 3, qkv.shape) != 0) {
        dm_block_free(&d_ctx); dm_block_free(&attn_score); dm_block_free(&qkv); return -1;
    }
    memset(d_qkv.data, 0, d_qkv.bytes);
    
    float *dqkvp = (float *)d_qkv.data;
    const float *qkvp = (const float *)qkv.data;
    const float *dctx_p = (const float *)d_ctx.data;
    const float *score_p = (const float *)attn_score.data;
    float *d_rbp = attn->grad_relative_position_bias_table ? (float *)attn->grad_relative_position_bias_table->data : NULL;
    
    float *d_a_arr = (float *)malloc((size_t)n * sizeof(float));
    if (!d_a_arr) {
        dm_block_free(&d_qkv); dm_block_free(&d_ctx); dm_block_free(&attn_score); dm_block_free(&qkv); return -1;
    }
    
    for (int64_t bi = 0; bi < bw; bi++) {
        for (int64_t head = 0; head < h; head++) {
            const float *sp = score_p + (((size_t)bi * h + head) * n) * n;
            for (int64_t qi = 0; qi < n; qi++) {
                float sum_dy_y = 0.0f;
                for (int64_t ki = 0; ki < n; ki++) {
                    float a = sp[qi * n + ki], d_a = 0.0f;
                    for (int64_t di = 0; di < d; di++) {
                        size_t ctx_idx = ((size_t)bi * n + qi) * c + head * d + di;
                        size_t vidx = (((size_t)bi * n + ki) * 3 * c) + 2 * c + head * d + di;
                        float d_c = dctx_p[ctx_idx];
                        dqkvp[vidx] += d_c * a;
                        d_a += d_c * qkvp[vidx];
                    }
                    d_a_arr[ki] = d_a;
                    sum_dy_y += d_a * a;
                }
                for (int64_t ki = 0; ki < n; ki++) {
                    float a = sp[qi * n + ki];
                    float d_x_ki = a * (d_a_arr[ki] - sum_dy_y);
                    if (d_rbp) {
                        size_t rb_idx = (size_t)attn->relative_position_index[qi * n + ki] * h + head;
                        d_rbp[rb_idx] += d_x_ki;
                    }
                    float scaled_d_x = d_x_ki * attn->scale;
                    for (int64_t di = 0; di < d; di++) {
                        size_t qidx = (((size_t)bi * n + qi) * 3 * c) + head * d + di;
                        size_t kidx = (((size_t)bi * n + ki) * 3 * c) + c + head * d + di;
                        dqkvp[qidx] += scaled_d_x * qkvp[kidx];
                        dqkvp[kidx] += scaled_d_x * qkvp[qidx];
                    }
                }
            }
        }
    }
    free(d_a_arr);
    
    // qkv backward
    int rc = linear_backward_helper(&attn->qkv, &d_qkv, d_x, cache);
    
    dm_block_free(&d_qkv);
    dm_block_free(&d_ctx);
    dm_block_free(&attn_score);
    dm_block_free(&qkv);
    return rc;
}

int dm_swin_window_attention_forward_fixture(const DM_Block *x,
                                             const DM_Block *qkv_weight,
                                             const DM_Block *qkv_bias,
                                             const DM_Block *proj_weight,
                                             const DM_Block *proj_bias,
                                             const DM_Block *relative_position_bias_table,
                                             int num_heads,
                                             const DM_Block *mask,
                                             DM_Block *y) {
    DM_SwinWindowAttentionBlock attn;
    int rc;
    if (!x || x->ndim != 3 || !qkv_weight || !proj_weight || !relative_position_bias_table) return -1;
    memset(&attn, 0, sizeof(attn));
    if (init_attn(&attn, (int)x->shape[2], (int)lround(sqrt((double)x->shape[1])), num_heads) != 0) return -1;
    attn.qkv.weight = (DM_Block *)qkv_weight;
    attn.qkv.bias = (DM_Block *)qkv_bias;
    attn.proj.weight = (DM_Block *)proj_weight;
    attn.proj.bias = (DM_Block *)proj_bias;
    attn.relative_position_bias_table = (DM_Block *)relative_position_bias_table;
    rc = window_attention_forward(&attn, x, mask, y, NULL);
    free(attn.relative_position_index);
    return rc;
}

static int swin_block_forward(const DM_SwinBlock *blk, const DM_Block *x, DM_Block *y, DM_ActivationCache *cache) {
    DM_Block shortcut = {0}, normed = {0}, bhwc = {0}, shifted = {0};
    DM_Block wins4 = {0}, wins3 = {0}, attn3 = {0}, attn4 = {0}, merged = {0}, unshifted = {0};
    DM_Block x_nlc = {0}, shortcut2 = {0}, normed2 = {0}, mlp = {0};
    int rc = -1;
    if (!blk || !is_f32(x) || x->ndim != 3 || x->shape[1] != (int64_t)blk->input_h * blk->input_w || x->shape[2] != blk->dim) return -1;
    if (copy_block(x, &shortcut) != 0) goto done;
    if (layer_norm_forward(&blk->norm1, x, &normed, cache) != 0) goto done;
    if (nlc_to_bhwc(&normed, blk->input_h, blk->input_w, &bhwc) != 0) goto done;
    if (blk->shift_size > 0) {
        if (roll_bhwc(&bhwc, -blk->shift_size, -blk->shift_size, &shifted) != 0) goto done;
    } else if (copy_block(&bhwc, &shifted) != 0) {
        goto done;
    }
    if (dm_swin_window_partition(&shifted, blk->window_size, &wins4) != 0) goto done;
    if (flatten_windows(&wins4, &wins3) != 0) goto done;
    if (window_attention_forward(&blk->attn, &wins3, blk->has_attn_mask ? &blk->attn_mask : NULL, &attn3, cache) != 0) goto done;
    if (unflatten_windows(&attn3, blk->window_size, &attn4) != 0) goto done;
    if (dm_swin_window_reverse(&attn4, blk->window_size, blk->input_h, blk->input_w, &merged) != 0) goto done;
    if (blk->shift_size > 0) {
        if (roll_bhwc(&merged, blk->shift_size, blk->shift_size, &unshifted) != 0) goto done;
    } else if (copy_block(&merged, &unshifted) != 0) {
        goto done;
    }
    if (bhwc_to_nlc(&unshifted, &x_nlc) != 0) goto done;
    if (add_inplace(&x_nlc, &shortcut) != 0) goto done;
    if (copy_block(&x_nlc, &shortcut2) != 0) goto done;
    if (layer_norm_forward(&blk->norm2, &x_nlc, &normed2, cache) != 0) goto done;
    if (mlp_forward(&blk->mlp, &normed2, &mlp, cache) != 0) goto done;
    if (add_inplace(&mlp, &shortcut2) != 0) goto done;
    rc = copy_block(&mlp, y);
done:
    dm_block_free(&shortcut); dm_block_free(&normed); dm_block_free(&bhwc); dm_block_free(&shifted);
    dm_block_free(&wins4); dm_block_free(&wins3); dm_block_free(&attn3); dm_block_free(&attn4);
    dm_block_free(&merged); dm_block_free(&unshifted); dm_block_free(&x_nlc); dm_block_free(&shortcut2);
    dm_block_free(&normed2); dm_block_free(&mlp);
    return rc;
}

static int swin_block_backward(const DM_SwinBlock *blk, const DM_Block *d_y, DM_Block *d_x, DM_ActivationCache *cache) {
    DM_Block d_mlp = *d_y, d_normed2 = {0}, d_x_nlc_branch = {0};
    if (mlp_backward((DM_SwinMlpBlock *)&blk->mlp, &d_mlp, &d_normed2, cache) != 0) return -1;
    if (layer_norm_backward_helper((DM_SwinLayerNormBlock *)&blk->norm2, &d_normed2, &d_x_nlc_branch, cache) != 0) { dm_block_free(&d_normed2); return -1; }
    dm_block_free(&d_normed2);
    
    DM_Block d_x_nlc = {0};
    block_create(&d_x_nlc, 3, d_y->shape);
    float *dxn = (float *)d_x_nlc.data;
    const float *dy = (const float *)d_y->data, *dxnb = (const float *)d_x_nlc_branch.data;
    for (size_t i = 0; i < d_x_nlc.count; i++) dxn[i] = dxnb[i] + dy[i];
    dm_block_free(&d_x_nlc_branch);
    
    DM_Block d_unshifted = {0}, d_merged = {0}, d_attn4 = {0}, d_attn3 = {0}, d_wins3 = {0}, d_wins4 = {0}, d_shifted = {0}, d_bhwc = {0}, d_normed = {0}, d_x_branch = {0};
    nlc_to_bhwc(&d_x_nlc, blk->input_h, blk->input_w, &d_unshifted);
    
    if (blk->shift_size > 0) roll_bhwc(&d_unshifted, -blk->shift_size, -blk->shift_size, &d_merged);
    else copy_block(&d_unshifted, &d_merged);
    dm_block_free(&d_unshifted);
    
    dm_swin_window_partition(&d_merged, blk->window_size, &d_attn4);
    dm_block_free(&d_merged);
    
    flatten_windows(&d_attn4, &d_attn3);
    dm_block_free(&d_attn4);
    
    if (window_attention_backward((DM_SwinWindowAttentionBlock *)&blk->attn, &d_attn3, &d_wins3, cache) != 0) { dm_block_free(&d_attn3); dm_block_free(&d_x_nlc); return -1; }
    dm_block_free(&d_attn3);
    
    unflatten_windows(&d_wins3, blk->window_size, &d_wins4);
    dm_block_free(&d_wins3);
    
    dm_swin_window_reverse(&d_wins4, blk->window_size, blk->input_h, blk->input_w, &d_shifted);
    dm_block_free(&d_wins4);
    
    if (blk->shift_size > 0) roll_bhwc(&d_shifted, blk->shift_size, blk->shift_size, &d_bhwc);
    else copy_block(&d_shifted, &d_bhwc);
    dm_block_free(&d_shifted);
    
    bhwc_to_nlc(&d_bhwc, &d_normed);
    dm_block_free(&d_bhwc);
    
    if (layer_norm_backward_helper((DM_SwinLayerNormBlock *)&blk->norm1, &d_normed, &d_x_branch, cache) != 0) { dm_block_free(&d_normed); dm_block_free(&d_x_nlc); return -1; }
    dm_block_free(&d_normed);
    
    if (d_x) {
        block_create(d_x, 3, d_x_branch.shape);
        float *dxp = (float *)d_x->data;
        const float *dxb = (const float *)d_x_branch.data;
        for (size_t i = 0; i < d_x->count; i++) dxp[i] = dxb[i] + dxn[i];
    }
    dm_block_free(&d_x_branch);
    dm_block_free(&d_x_nlc);
    return 0;
}

static int patch_merging_forward(const DM_SwinPatchMergingBlock *pm, const DM_Block *x, DM_Block *y, DM_ActivationCache *cache) {
    DM_Block cat = {0}, normed = {0};
    const float *xp;
    float *cp;
    int b, h, w, c;
    int rc = -1;
    if (!pm || !is_f32(x) || x->ndim != 3) return -1;
    b = (int)x->shape[0]; h = pm->input_h; w = pm->input_w; c = pm->dim;
    if (x->shape[1] != (int64_t)h * w || x->shape[2] != c || h % 2 || w % 2) return -1;
    if (block_create(&cat, 3, (int64_t[]){b, (h / 2) * (w / 2), 4 * c}) != 0) return -1;
    xp = (const float *)x->data;
    cp = (float *)cat.data;
    for (int bi = 0; bi < b; bi++)
    for (int yy = 0; yy < h / 2; yy++)
    for (int xx = 0; xx < w / 2; xx++) {
        float *dst = cp + ((size_t)bi * (h / 2) * (w / 2) + yy * (w / 2) + xx) * 4 * c;
        const float *x0 = xp + ((size_t)bi * h * w + (2 * yy) * w + (2 * xx)) * c;
        const float *x1 = xp + ((size_t)bi * h * w + (2 * yy + 1) * w + (2 * xx)) * c;
        const float *x2 = xp + ((size_t)bi * h * w + (2 * yy) * w + (2 * xx + 1)) * c;
        const float *x3 = xp + ((size_t)bi * h * w + (2 * yy + 1) * w + (2 * xx + 1)) * c;
        memcpy(dst, x0, (size_t)c * sizeof(float));
        memcpy(dst + c, x1, (size_t)c * sizeof(float));
        memcpy(dst + 2 * c, x2, (size_t)c * sizeof(float));
        memcpy(dst + 3 * c, x3, (size_t)c * sizeof(float));
    }
    if (layer_norm_forward(&pm->norm, &cat, &normed, cache) != 0) goto done;
    rc = linear_forward(&pm->reduction, &normed, y, cache);
done:
    dm_block_free(&cat);
    dm_block_free(&normed);
    return rc;
}

static int patch_merging_backward(const DM_SwinPatchMergingBlock *pm, const DM_Block *d_y, DM_Block *d_x, DM_ActivationCache *cache) {
    DM_Block d_normed = {0}, d_cat = {0};
    if (linear_backward_helper(&pm->reduction, d_y, &d_normed, cache) != 0) return -1;
    if (layer_norm_backward_helper(&pm->norm, &d_normed, &d_cat, cache) != 0) { dm_block_free(&d_normed); return -1; }
    dm_block_free(&d_normed);
    
    if (d_x) {
        int b = d_cat.shape[0], h = pm->input_h, w = pm->input_w, c = pm->dim;
        block_create(d_x, 3, (int64_t[]){b, h * w, c});
        float *dxp = (float *)d_x->data;
        const float *dcp = (const float *)d_cat.data;
        
        for (int bi = 0; bi < b; bi++)
        for (int yy = 0; yy < h / 2; yy++)
        for (int xx = 0; xx < w / 2; xx++) {
            const float *src = dcp + ((size_t)bi * (h / 2) * (w / 2) + yy * (w / 2) + xx) * 4 * c;
            float *dx0 = dxp + ((size_t)bi * h * w + (2 * yy) * w + (2 * xx)) * c;
            float *dx1 = dxp + ((size_t)bi * h * w + (2 * yy + 1) * w + (2 * xx)) * c;
            float *dx2 = dxp + ((size_t)bi * h * w + (2 * yy) * w + (2 * xx + 1)) * c;
            float *dx3 = dxp + ((size_t)bi * h * w + (2 * yy + 1) * w + (2 * xx + 1)) * c;
            memcpy(dx0, src, (size_t)c * sizeof(float));
            memcpy(dx1, src + c, (size_t)c * sizeof(float));
            memcpy(dx2, src + 2 * c, (size_t)c * sizeof(float));
            memcpy(dx3, src + 3 * c, (size_t)c * sizeof(float));
        }
    }
    dm_block_free(&d_cat);
    return 0;
}

static int stage_forward(const DM_SwinStageBlock *stage, const DM_Block *x, DM_Block *y, DM_ActivationCache *cache) {
    DM_Block cur = {0}, next = {0};
    if (copy_block(x, &cur) != 0) return -1;
    for (int i = 0; i < stage->depth; i++) {
        if (swin_block_forward(&stage->blocks[i], &cur, &next, cache) != 0) {
            dm_block_free(&cur);
            return -1;
        }
        dm_block_free(&cur);
        cur = next;
        memset(&next, 0, sizeof(next));
    }
    if (stage->has_downsample) {
        if (patch_merging_forward(&stage->downsample, &cur, &next, cache) != 0) {
            dm_block_free(&cur);
            return -1;
        }
        dm_block_free(&cur);
        cur = next;
        memset(&next, 0, sizeof(next));
    }
    *y = cur;
    return 0;
}

static int stage_backward(const DM_SwinStageBlock *stage, const DM_Block *d_y, DM_Block *d_x, DM_ActivationCache *cache) {
    DM_Block d_cur = {0}, d_next = {0};
    if (copy_block(d_y, &d_cur) != 0) return -1;
    
    if (stage->has_downsample) {
        if (patch_merging_backward(&stage->downsample, &d_cur, &d_next, cache) != 0) { dm_block_free(&d_cur); return -1; }
        dm_block_free(&d_cur);
        d_cur = d_next;
        memset(&d_next, 0, sizeof(d_next));
    }
    
    for (int i = stage->depth - 1; i >= 0; i--) {
        if (swin_block_backward(&stage->blocks[i], &d_cur, &d_next, cache) != 0) { dm_block_free(&d_cur); return -1; }
        dm_block_free(&d_cur);
        d_cur = d_next;
        memset(&d_next, 0, sizeof(d_next));
    }
    
    if (d_x) *d_x = d_cur;
    else dm_block_free(&d_cur);
    return 0;
}

static int patch_embed_forward(const DM_SwinPatchEmbedBlock *pe, const DM_Block *input, DM_Block *tokens, DM_ActivationCache *cache) {
    DM_Block out = {0};
    const float *xp, *wp, *bp;
    float *op;
    int b, h, w, c, p, oh, ow, ed;
    if (!pe || !is_f32(input) || input->ndim != 4) return -1;
    b = (int)input->shape[0]; h = (int)input->shape[1]; w = (int)input->shape[2]; c = (int)input->shape[3];
    p = pe->patch_size; ed = pe->embed_dim; oh = h / p; ow = w / p;
    if (h != pe->img_size || w != pe->img_size || c != pe->in_chans || !pe->proj_weight || !pe->proj_bias) return -1;
    if (!shape_eq(pe->proj_weight, 4, (int64_t[]){ed, c, p, p}) || !shape_eq(pe->proj_bias, 1, (int64_t[]){ed})) return -1;
    if (block_create(&out, 3, (int64_t[]){b, oh * ow, ed}) != 0) return -1;
    if (cache) dm_activation_cache_push(cache, input);
    xp = (const float *)input->data;
    wp = (const float *)pe->proj_weight->data;
    bp = (const float *)pe->proj_bias->data;
    op = (float *)out.data;
    for (int bi = 0; bi < b; bi++)
    for (int yy = 0; yy < oh; yy++)
    for (int xx = 0; xx < ow; xx++) {
        float *dst = op + ((size_t)bi * oh * ow + yy * ow + xx) * ed;
        for (int eo = 0; eo < ed; eo++) {
            float sum = bp[eo];
            for (int ci = 0; ci < c; ci++)
            for (int py = 0; py < p; py++)
            for (int px = 0; px < p; px++) {
                float xv = xp[((size_t)bi * h * w + (yy * p + py) * w + (xx * p + px)) * c + ci];
                float wv = wp[(((size_t)eo * c + ci) * p + py) * p + px];
                sum += xv * wv;
            }
            dst[eo] = sum;
        }
    }
    if (pe->has_norm) {
        int rc = layer_norm_forward(&pe->norm, &out, tokens, cache);
        dm_block_free(&out);
        return rc;
    }
    *tokens = out;
    return 0;
}

static int patch_embed_backward(DM_SwinPatchEmbedBlock *pe, const DM_Block *d_tokens, DM_Block *d_input, DM_ActivationCache *cache) {
    DM_Block d_out = {0}, input = {0}, tmp_gw = {0}, tmp_gb = {0};
    if (layer_norm_backward_helper(&pe->norm, d_tokens, &d_out, cache) != 0) return -1;
    if (!cache || cache->count == 0) { dm_block_free(&d_out); return -1; }
    input = cache->activations[--cache->count];
    int b = input.shape[0], h = input.shape[1], w = input.shape[2], c = input.shape[3];
    int p = pe->patch_size, ed = pe->embed_dim, oh = h / p, ow = w / p;
    if (d_input) {
        block_create(d_input, 4, input.shape);
        memset(d_input->data, 0, d_input->bytes);
    }
    block_create(&tmp_gw, 4, pe->proj_weight->shape);
    block_create(&tmp_gb, 1, pe->proj_bias->shape);
    memset(tmp_gw.data, 0, tmp_gw.bytes); memset(tmp_gb.data, 0, tmp_gb.bytes);
    
    const float *xp = (const float *)input.data, *wp = (const float *)pe->proj_weight->data, *d_out_p = (const float *)d_out.data;
    float *d_inp = d_input ? (float *)d_input->data : NULL, *gw = (float *)tmp_gw.data, *gb = (float *)tmp_gb.data;
    
    for (int bi = 0; bi < b; bi++)
    for (int yy = 0; yy < oh; yy++)
    for (int xx = 0; xx < ow; xx++) {
        const float *src_d = d_out_p + ((size_t)bi * oh * ow + yy * ow + xx) * ed;
        for (int eo = 0; eo < ed; eo++) {
            float g = src_d[eo]; gb[eo] += g;
            for (int ci = 0; ci < c; ci++)
            for (int py = 0; py < p; py++)
            for (int px = 0; px < p; px++) {
                float xv = xp[((size_t)bi * h * w + (yy * p + py) * w + (xx * p + px)) * c + ci];
                float wv = wp[(((size_t)eo * c + ci) * p + py) * p + px];
                gw[(((size_t)eo * c + ci) * p + py) * p + px] += g * xv;
                if (d_inp) d_inp[((size_t)bi * h * w + (yy * p + py) * w + (xx * p + px)) * c + ci] += g * wv;
            }
        }
    }
    if (pe->grad_proj_weight) {
        float *pgw = (float *)pe->grad_proj_weight->data;
        for (size_t i = 0; i < tmp_gw.count; i++) pgw[i] += gw[i];
    }
    if (pe->grad_proj_bias) {
        float *pgb = (float *)pe->grad_proj_bias->data;
        for (size_t i = 0; i < tmp_gb.count; i++) pgb[i] += gb[i];
    }
    dm_block_free(&tmp_gw); dm_block_free(&tmp_gb); dm_block_free(&input); dm_block_free(&d_out);
    return 0;
}

static int final_pool_head(DM_SwinModel *model, const DM_Block *x, DM_Block *logits, DM_ActivationCache *cache) {
    DM_Block normed = {0}, pooled = {0};
    const float *xp;
    float *pp;
    int b, l, c;
    int rc = -1;
    if (layer_norm_forward(&model->norm, x, &normed, cache) != 0) return -1;
    b = (int)normed.shape[0]; l = (int)normed.shape[1]; c = (int)normed.shape[2];
    if (block_create(&pooled, 3, (int64_t[]){b, 1, c}) != 0) goto done;
    xp = (const float *)normed.data;
    pp = (float *)pooled.data;
    for (int bi = 0; bi < b; bi++)
    for (int ci = 0; ci < c; ci++) {
        double sum = 0.0;
        for (int ti = 0; ti < l; ti++) sum += xp[((size_t)bi * l + ti) * c + ci];
        pp[(size_t)bi * c + ci] = (float)(sum / l);
    }
    rc = linear_forward(&model->head, &pooled, logits, cache);
done:
    dm_block_free(&normed);
    dm_block_free(&pooled);
    return rc;
}

static int final_pool_head_backward(DM_SwinModel *model, const DM_Block *d_logits, DM_Block *d_x, DM_ActivationCache *cache) {
    DM_Block d_pooled = {0}, x = {0}, d_normed = {0};
    if (linear_backward_helper(&model->head, d_logits, &d_pooled, cache) != 0) return -1;
    if (!cache || cache->count == 0) { dm_block_free(&d_pooled); return -1; }
    x = cache->activations[--cache->count];
    int64_t b = x.shape[0], l = x.shape[1], c = x.shape[2];
    if (block_create(&d_normed, 3, x.shape) != 0) { dm_block_free(&d_pooled); dm_block_free(&x); return -1; }
    float *dp = (float *)d_pooled.data, *dn = (float *)d_normed.data;
    for (int bi = 0; bi < b; bi++) {
        for (int ci = 0; ci < c; ci++) {
            float val = dp[bi * c + ci] / l;
            for (int ti = 0; ti < l; ti++) dn[(bi * l + ti) * c + ci] = val;
        }
    }
    dm_block_free(&d_pooled);
    dm_activation_cache_push(cache, &x);
    dm_block_free(&x);
    int rc = layer_norm_backward_helper(&model->norm, &d_normed, d_x, cache);
    dm_block_free(&d_normed);
    return rc;
}

void dm_swin_config_init(DM_SwinConfig *cfg, DM_SwinVariant variant, int num_classes, int img_size) {
    static const int depths[4][4] = {
        {2, 2, 6, 2}, {2, 2, 18, 2}, {2, 2, 18, 2}, {2, 2, 18, 2}
    };
    static const int heads[4][4] = {
        {3, 6, 12, 24}, {3, 6, 12, 24}, {4, 8, 16, 32}, {6, 12, 24, 48}
    };
    static const int embed[4] = {96, 96, 128, 192};
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->variant = variant;
    cfg->img_size = img_size > 0 ? img_size : 224;
    cfg->patch_size = 4;
    cfg->in_chans = 3;
    cfg->num_classes = num_classes > 0 ? num_classes : 1000;
    cfg->embed_dim = embed[variant];
    cfg->window_size = 7;
    cfg->mlp_ratio = 4.0f;
    cfg->qkv_bias = 1;
    cfg->ape = 0;
    cfg->patch_norm = 1;
    for (int i = 0; i < 4; i++) {
        cfg->depths[i] = depths[variant][i];
        cfg->num_heads[i] = heads[variant][i];
    }
}

static int init_attn(DM_SwinWindowAttentionBlock *attn, int dim, int window_size, int num_heads) {
    attn->dim = dim;
    attn->window_size = window_size;
    attn->num_heads = num_heads;
    attn->head_dim = dim / num_heads;
    attn->scale = 1.0f / sqrtf((float)attn->head_dim);
    attn->relative_position_index = (int *)malloc((size_t)window_size * window_size * window_size * window_size * sizeof(int));
    if (!attn->relative_position_index) return -1;
    return dm_swin_build_relative_position_index(window_size, attn->relative_position_index);
}

int dm_swin_model_init(DM_SwinModel *model, const DM_SwinConfig *cfg) {
    int h, w, dim;
    if (!model || !cfg) return -1;
    memset(model, 0, sizeof(*model));
    model->cfg = *cfg;
    model->patch_embed.img_size = cfg->img_size;
    model->patch_embed.patch_size = cfg->patch_size;
    model->patch_embed.in_chans = cfg->in_chans;
    model->patch_embed.embed_dim = cfg->embed_dim;
    model->patch_embed.patches_h = cfg->img_size / cfg->patch_size;
    model->patch_embed.patches_w = cfg->img_size / cfg->patch_size;
    model->patch_embed.has_norm = cfg->patch_norm;
    h = model->patch_embed.patches_h;
    w = model->patch_embed.patches_w;
    dim = cfg->embed_dim;
    for (int s = 0; s < 4; s++) {
        DM_SwinStageBlock *stage = &model->stages[s];
        stage->depth = cfg->depths[s];
        stage->dim = dim;
        stage->input_h = h;
        stage->input_w = w;
        stage->blocks = (DM_SwinBlock *)calloc((size_t)stage->depth, sizeof(*stage->blocks));
        if (!stage->blocks) { dm_swin_model_free(model); return -1; }
        for (int i = 0; i < stage->depth; i++) {
            DM_SwinBlock *blk = &stage->blocks[i];
            int shift = (i % 2 == 0) ? 0 : cfg->window_size / 2;
            blk->dim = dim;
            blk->input_h = h;
            blk->input_w = w;
            blk->num_heads = cfg->num_heads[s];
            blk->window_size = cfg->window_size;
            blk->shift_size = shift;
            if (h <= blk->window_size || w <= blk->window_size) {
                blk->window_size = h < w ? h : w;
                blk->shift_size = 0;
            }
            blk->mlp.dim = dim;
            blk->mlp.hidden_dim = (int)(dim * cfg->mlp_ratio);
            if (init_attn(&blk->attn, dim, blk->window_size, cfg->num_heads[s]) != 0) {
                dm_swin_model_free(model); return -1;
            }
            if (blk->shift_size > 0) {
                if (dm_swin_build_attention_mask(h, w, blk->window_size, blk->shift_size, &blk->attn_mask) != 0) {
                    dm_swin_model_free(model); return -1;
                }
                blk->has_attn_mask = 1;
            }
        }
        if (s < 3) {
            stage->has_downsample = 1;
            stage->downsample.input_h = h;
            stage->downsample.input_w = w;
            stage->downsample.dim = dim;
            h /= 2;
            w /= 2;
            dim *= 2;
        }
    }
    model->initialized = 1;
    return 0;
}

static char *read_text_file(const char *path) {
    FILE *f;
    long n;
    char *buf;
    f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    buf[n] = '\0';
    fclose(f);
    return buf;
}

static int dirname_of(const char *path, char *out, size_t out_sz) {
    const char *slash;
    size_t n;
    if (!path || !out || out_sz == 0) return -1;
    slash = strrchr(path, '/');
    if (!slash) {
        snprintf(out, out_sz, ".");
        return 0;
    }
    n = (size_t)(slash - path);
    if (n >= out_sz) return -1;
    memcpy(out, path, n);
    out[n] = '\0';
    return 0;
}

static const char *skip_ws(const char *p) {
    while (p && *p && isspace((unsigned char)*p)) p++;
    return p;
}

static int json_string_after(const char *obj, const char *key, char *out, size_t out_sz) {
    char pat[96];
    const char *p, *q;
    size_t n;
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    p = strstr(obj, pat);
    if (!p) return -1;
    p = strchr(p + strlen(pat), ':');
    if (!p) return -1;
    p = skip_ws(p + 1);
    if (*p != '"') return -1;
    q = strchr(++p, '"');
    if (!q) return -1;
    n = (size_t)(q - p);
    if (n >= out_sz) return -1;
    memcpy(out, p, n);
    out[n] = '\0';
    return 0;
}

static int json_shape_after(const char *obj, int64_t *shape, int *ndim) {
    const char *p = strstr(obj, "\"shape\"");
    int n = 0;
    if (!p) return -1;
    p = strchr(p, '[');
    if (!p) return -1;
    p++;
    while (*p && *p != ']') {
        char *end = NULL;
        long long v;
        p = skip_ws(p);
        if (*p == ',') { p++; continue; }
        errno = 0;
        v = strtoll(p, &end, 10);
        if (errno || end == p || n >= DM_MAX_DIMS) return -1;
        shape[n++] = (int64_t)v;
        p = end;
    }
    if (*p != ']') return -1;
    *ndim = n;
    return 0;
}

static int load_float_file(const char *path, DM_Block *b) {
    FILE *f = fopen(path, "rb");
    size_t got;
    if (!f) return -1;
    got = fread(b->data, 1, b->bytes, f);
    fclose(f);
    return got == b->bytes ? 0 : -1;
}

static int parse_tensor_objects(DM_SwinModel *model, const char *json, const char *base_dir) {
    const char *p = strstr(json, "\"tensors\"");
    if (!p) return -1;
    p = strchr(p, '[');
    if (!p) return -1;
    while ((p = strchr(p, '{')) != NULL) {
        const char *end = strchr(p, '}');
        char name[512], file[512], full[PATH_MAX];
        int64_t shape[DM_MAX_DIMS];
        int ndim;
        DM_Block *b;
        if (!end) return -1;
        if (json_string_after(p, "name", name, sizeof(name)) != 0 ||
            json_string_after(p, "file", file, sizeof(file)) != 0 ||
            json_shape_after(p, shape, &ndim) != 0) {
            p = end + 1;
            continue;
        }
        b = store_add(&model->tensors, name, ndim, shape);
        if (!b) return -1;
        if ((size_t)snprintf(full, sizeof(full), "%s/%s", base_dir, file) >= sizeof(full)) {
            fprintf(stderr, "swin: tensor path too long for %s\n", name);
            return -1;
        }
        if (load_float_file(full, b) != 0) {
            fprintf(stderr, "swin: failed to load tensor file %s\n", full);
            return -1;
        }
        p = end + 1;
    }
    return model->tensors.count > 0 ? 0 : -1;
}

static int bind_all(DM_SwinModel *model) {
    int ok = 1;
    int dim = model->cfg.embed_dim;
    char name[512];
    model->patch_embed.proj_weight = bind_tensor(model, "patch_embed.proj.weight", 4, (int64_t[]){dim, model->cfg.in_chans, model->cfg.patch_size, model->cfg.patch_size});
    model->patch_embed.grad_proj_weight = bind_grad(model, "patch_embed.proj.weight");
    model->patch_embed.proj_bias = bind_tensor(model, "patch_embed.proj.bias", 1, (int64_t[]){dim});
    model->patch_embed.grad_proj_bias = bind_grad(model, "patch_embed.proj.bias");
    if (!model->patch_embed.proj_weight || !model->patch_embed.proj_bias) ok = 0;
    if (model->cfg.patch_norm) bind_ln(model, &model->patch_embed.norm, "patch_embed.norm", dim, &ok);
    if (model->cfg.ape) {
        model->absolute_pos_embed = bind_tensor(model, "absolute_pos_embed", 3, (int64_t[]){1, model->patch_embed.patches_h * model->patch_embed.patches_w, dim});
        model->grad_absolute_pos_embed = bind_grad(model, "absolute_pos_embed");
        if (!model->absolute_pos_embed) ok = 0;
    }
    for (int s = 0; s < 4; s++) {
        DM_SwinStageBlock *stage = &model->stages[s];
        for (int b = 0; b < stage->depth; b++) {
            DM_SwinBlock *blk = &stage->blocks[b];
            int hidden = (int)(blk->dim * model->cfg.mlp_ratio);
            snprintf(name, sizeof(name), "layers.%d.blocks.%d.norm1", s, b);
            bind_ln(model, &blk->norm1, name, blk->dim, &ok);
            snprintf(name, sizeof(name), "layers.%d.blocks.%d.attn.qkv", s, b);
            bind_linear(model, &blk->attn.qkv, name, 3 * blk->dim, blk->dim, model->cfg.qkv_bias, &ok);
            snprintf(name, sizeof(name), "layers.%d.blocks.%d.attn.proj", s, b);
            bind_linear(model, &blk->attn.proj, name, blk->dim, blk->dim, 1, &ok);
            snprintf(name, sizeof(name), "layers.%d.blocks.%d.attn.relative_position_bias_table", s, b);
            blk->attn.relative_position_bias_table = bind_tensor(model, name, 2, (int64_t[]){(2 * blk->window_size - 1) * (2 * blk->window_size - 1), blk->num_heads});
            blk->attn.grad_relative_position_bias_table = bind_grad(model, name);
            if (!blk->attn.relative_position_bias_table) ok = 0;
            snprintf(name, sizeof(name), "layers.%d.blocks.%d.norm2", s, b);
            bind_ln(model, &blk->norm2, name, blk->dim, &ok);
            snprintf(name, sizeof(name), "layers.%d.blocks.%d.mlp.fc1", s, b);
            bind_linear(model, &blk->mlp.fc1, name, hidden, blk->dim, 1, &ok);
            snprintf(name, sizeof(name), "layers.%d.blocks.%d.mlp.fc2", s, b);
            bind_linear(model, &blk->mlp.fc2, name, blk->dim, hidden, 1, &ok);
        }
        if (stage->has_downsample) {
            snprintf(name, sizeof(name), "layers.%d.downsample.norm", s);
            bind_ln(model, &stage->downsample.norm, name, 4 * stage->dim, &ok);
            snprintf(name, sizeof(name), "layers.%d.downsample.reduction", s);
            bind_linear(model, &stage->downsample.reduction, name, 2 * stage->dim, 4 * stage->dim, 0, &ok);
        }
    }
    bind_ln(model, &model->norm, "norm", model->cfg.embed_dim * 8, &ok);
    bind_linear(model, &model->head, "head", model->cfg.num_classes, model->cfg.embed_dim * 8, 1, &ok);
    for (int i = 0; i < model->tensors.count; i++) {
        if (!model->tensors.items[i].used) {
            fprintf(stderr, "swin: unused tensor in metadata: %s\n", model->tensors.items[i].name);
            ok = 0;
        }
    }
    return ok ? 0 : -1;
}

int dm_swin_model_load_weights(DM_SwinModel *model, const char *metadata_path) {
    char *json;
    char base[PATH_MAX];
    int rc;
    if (!model || !metadata_path) return -1;
    json = read_text_file(metadata_path);
    if (!json) return -1;
    if (dirname_of(metadata_path, base, sizeof(base)) != 0) { free(json); return -1; }
    store_free(&model->tensors);
    rc = parse_tensor_objects(model, json, base);
    free(json);
    if (rc != 0) return -1;
    return bind_all(model);
}

int dm_swin_model_forward(DM_SwinModel *model, const DM_Block *input_nhwc, DM_ActivationCache *cache, DM_Block *logits) {
    DM_Block cur = {0}, next = {0};
    int rc = -1;
    if (!model || !model->initialized || !is_f32(input_nhwc) || input_nhwc->ndim != 4) return -1;
    if (patch_embed_forward(&model->patch_embed, input_nhwc, &cur, cache) != 0) return -1;
    if (model->absolute_pos_embed) {
        if (add_inplace(&cur, model->absolute_pos_embed) != 0) goto done;
    }
    for (int s = 0; s < 4; s++) {
        if (stage_forward(&model->stages[s], &cur, &next, cache) != 0) goto done;
        dm_block_free(&cur);
        cur = next;
        memset(&next, 0, sizeof(next));
    }
    rc = final_pool_head(model, &cur, logits, cache);
done:
    dm_block_free(&cur);
    return rc;
}

int dm_swin_model_backward(DM_SwinModel *model, const DM_Block *d_logits, DM_Block *d_input_nhwc, DM_ActivationCache *cache) {
    DM_Block d_cur = {0}, d_next = {0};
    if (final_pool_head_backward(model, d_logits, &d_cur, cache) != 0) return -1;
    for (int s = 3; s >= 0; s--) {
        if (stage_backward(&model->stages[s], &d_cur, &d_next, cache) != 0) { dm_block_free(&d_cur); return -1; }
        dm_block_free(&d_cur);
        d_cur = d_next;
        memset(&d_next, 0, sizeof(d_next));
    }
    if (model->absolute_pos_embed && model->grad_absolute_pos_embed) {
        float *gpe = (float *)model->grad_absolute_pos_embed->data;
        const float *dc = (const float *)d_cur.data;
        for (size_t i = 0; i < model->grad_absolute_pos_embed->count; i++) gpe[i] += dc[i];
    }
    int rc = patch_embed_backward(&model->patch_embed, &d_cur, d_input_nhwc, cache);
    dm_block_free(&d_cur);
    return rc;
}

void dm_swin_model_free(DM_SwinModel *model) {
    if (!model) return;
    for (int s = 0; s < 4; s++) {
        DM_SwinStageBlock *stage = &model->stages[s];
        if (stage->blocks) {
            for (int b = 0; b < stage->depth; b++) {
                free(stage->blocks[b].attn.relative_position_index);
                dm_block_free(&stage->blocks[b].attn_mask);
            }
        }
        free(stage->blocks);
    }
    store_free(&model->tensors);
    memset(model, 0, sizeof(*model));
}

static int fill_synthetic_weights(DM_SwinModel *model) {
    uint32_t seed = 7;
    int ok = 1;
    int dim = model->cfg.embed_dim;
    if (store_reserve(&model->tensors, 512) != 0) return -1;
    model->patch_embed.proj_weight = store_add(&model->tensors, "patch_embed.proj.weight", 4, (int64_t[]){dim, model->cfg.in_chans, model->cfg.patch_size, model->cfg.patch_size});
    model->patch_embed.proj_bias = store_add(&model->tensors, "patch_embed.proj.bias", 1, (int64_t[]){dim});
    if (model->cfg.patch_norm) {
        model->patch_embed.norm.weight = store_add(&model->tensors, "patch_embed.norm.weight", 1, (int64_t[]){dim});
        model->patch_embed.norm.bias = store_add(&model->tensors, "patch_embed.norm.bias", 1, (int64_t[]){dim});
    }
    for (int s = 0; s < 4; s++) {
        DM_SwinStageBlock *stage = &model->stages[s];
        for (int b = 0; b < stage->depth; b++) {
            DM_SwinBlock *blk = &stage->blocks[b];
            int hidden = (int)(blk->dim * model->cfg.mlp_ratio);
            char pre[256];
            snprintf(pre, sizeof(pre), "synthetic.%d.%d", s, b);
            (void)pre;
            blk->norm1.weight = store_add(&model->tensors, "ln", 1, (int64_t[]){blk->dim});
            blk->norm1.bias = store_add(&model->tensors, "ln", 1, (int64_t[]){blk->dim});
            blk->attn.qkv.weight = store_add(&model->tensors, "qkvw", 2, (int64_t[]){3 * blk->dim, blk->dim});
            blk->attn.qkv.bias = store_add(&model->tensors, "qkvb", 1, (int64_t[]){3 * blk->dim});
            blk->attn.proj.weight = store_add(&model->tensors, "projw", 2, (int64_t[]){blk->dim, blk->dim});
            blk->attn.proj.bias = store_add(&model->tensors, "projb", 1, (int64_t[]){blk->dim});
            blk->attn.relative_position_bias_table = store_add(&model->tensors, "rpb", 2, (int64_t[]){(2 * blk->window_size - 1) * (2 * blk->window_size - 1), blk->num_heads});
            blk->norm2.weight = store_add(&model->tensors, "ln", 1, (int64_t[]){blk->dim});
            blk->norm2.bias = store_add(&model->tensors, "ln", 1, (int64_t[]){blk->dim});
            blk->mlp.fc1.weight = store_add(&model->tensors, "fc1w", 2, (int64_t[]){hidden, blk->dim});
            blk->mlp.fc1.bias = store_add(&model->tensors, "fc1b", 1, (int64_t[]){hidden});
            blk->mlp.fc2.weight = store_add(&model->tensors, "fc2w", 2, (int64_t[]){blk->dim, hidden});
            blk->mlp.fc2.bias = store_add(&model->tensors, "fc2b", 1, (int64_t[]){blk->dim});
        }
        if (stage->has_downsample) {
            stage->downsample.norm.weight = store_add(&model->tensors, "dnlnw", 1, (int64_t[]){4 * stage->dim});
            stage->downsample.norm.bias = store_add(&model->tensors, "dnlnb", 1, (int64_t[]){4 * stage->dim});
            stage->downsample.reduction.weight = store_add(&model->tensors, "dnw", 2, (int64_t[]){2 * stage->dim, 4 * stage->dim});
        }
    }
    model->norm.weight = store_add(&model->tensors, "normw", 1, (int64_t[]){model->cfg.embed_dim * 8});
    model->norm.bias = store_add(&model->tensors, "normb", 1, (int64_t[]){model->cfg.embed_dim * 8});
    model->head.weight = store_add(&model->tensors, "headw", 2, (int64_t[]){model->cfg.num_classes, model->cfg.embed_dim * 8});
    model->head.bias = store_add(&model->tensors, "headb", 1, (int64_t[]){model->cfg.num_classes});
    for (int i = 0; i < model->tensors.count; i++) {
        float *p = (float *)model->tensors.items[i].tensor.data;
        if (!p) { ok = 0; continue; }
        for (size_t j = 0; j < model->tensors.items[i].tensor.count; j++) p[j] = rng_weight(&seed, 0.02f);
        if (strstr(model->tensors.items[i].name, "normw") || strstr(model->tensors.items[i].name, "dnlnw") || strcmp(model->tensors.items[i].name, "ln") == 0) {
            for (size_t j = 0; j < model->tensors.items[i].tensor.count; j++) p[j] = 1.0f;
        }
    }
    return ok ? 0 : -1;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s swin bench [--variant tiny|small|base|large] [--size N] [--classes N] [--window N] [--weights metadata.json]\n",
        prog);
}

int dm_swin_cli(int argc, char **argv) {
    DM_SwinConfig cfg;
    DM_SwinModel model;
    DM_Block input = {0}, logits = {0};
    DM_SwinVariant variant = DM_SWIN_TINY;
    int size = 224, classes = 1000;
    int window = 7;
    const char *weights = NULL;
    int rc;
    if (argc < 3 || strcmp(argv[2], "bench") != 0) { usage(argv[0]); return 1; }
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "tiny") == 0) variant = DM_SWIN_TINY;
            else if (strcmp(v, "small") == 0) variant = DM_SWIN_SMALL;
            else if (strcmp(v, "base") == 0) variant = DM_SWIN_BASE;
            else if (strcmp(v, "large") == 0) variant = DM_SWIN_LARGE;
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--classes") == 0 && i + 1 < argc) {
            classes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--window") == 0 && i + 1 < argc) {
            window = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights = argv[++i];
        }
    }
    dm_swin_config_init(&cfg, variant, classes, size);
    cfg.window_size = window > 0 ? window : cfg.window_size;
    if (dm_swin_model_init(&model, &cfg) != 0) { fprintf(stderr, "swin: init failed\n"); return 1; }
    if (weights) {
        if (dm_swin_model_load_weights(&model, weights) != 0) {
            fprintf(stderr, "swin: failed to load weights\n");
            dm_swin_model_free(&model);
            return 1;
        }
    } else if (fill_synthetic_weights(&model) != 0) {
        fprintf(stderr, "swin: synthetic weights failed\n");
        dm_swin_model_free(&model);
        return 1;
    }
    if (block_create(&input, 4, (int64_t[]){1, size, size, 3}) != 0) {
        dm_swin_model_free(&model);
        return 1;
    }
    for (size_t i = 0; i < input.count; i++) ((float *)input.data)[i] = 1.0f;
    
    DM_ActivationCache cache;
    dm_activation_cache_init(&cache, 1000);
    
    rc = dm_swin_model_forward(&model, &input, &cache, &logits);
    if (rc == 0) {
        printf("Swin forward ok: logits shape [%lld,%lld,%lld], first=%g\n",
               (long long)logits.shape[0], (long long)logits.shape[1],
               (long long)logits.shape[2], ((float *)logits.data)[0]);
        printf("Swin activation cache size: %zu tensors populated\n", cache.count);
        
        DM_Block d_logits = {0}, d_input = {0};
        block_create(&d_logits, 3, logits.shape);
        for (size_t i = 0; i < d_logits.count; i++) ((float *)d_logits.data)[i] = 1e-4f;
        
        rc = dm_swin_model_backward(&model, &d_logits, &d_input, &cache);
        if (rc == 0) {
            printf("Swin backward ok: d_input shape [%lld,%lld,%lld,%lld], cache size left: %zu\n",
                   (long long)d_input.shape[0], (long long)d_input.shape[1],
                   (long long)d_input.shape[2], (long long)d_input.shape[3], cache.count);
        } else {
            fprintf(stderr, "swin: backward failed\n");
        }
        dm_block_free(&d_logits);
        dm_block_free(&d_input);
    } else {
        fprintf(stderr, "swin: forward failed\n");
    }
    
    dm_activation_cache_free(&cache);
    dm_block_free(&input);
    dm_block_free(&logits);
    dm_swin_model_free(&model);
    return rc == 0 ? 0 : 1;
}
