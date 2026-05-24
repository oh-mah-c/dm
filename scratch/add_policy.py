with open('src/core/dm_engine.c', 'r') as f: text = f.read()

cpu_transpose_code = """
/* CPU native transpose fallback: NCHW -> NHWC */
int dm_transpose_nchw_to_nhwc_cpu(const DM_Block *src, DM_Block *dst) {
    if (!dm_block_is_nchw4(src) || !dst) return -1;
    int n = DM_NCHW_N(src), c = DM_NCHW_C(src), h = DM_NCHW_H(src), w = DM_NCHW_W(src);
    if (dm_block_create(dst, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){n, h, w, c}) != 0) return -1;
    const float *sdata = (const float *)src->data;
    float *ddata = (float *)dst->data;
    for (int in = 0; in < n; in++) {
        for (int ic = 0; ic < c; ic++) {
            for (int ih = 0; ih < h; ih++) {
                for (int iw = 0; iw < w; iw++) {
                    ddata[in * h * w * c + ih * w * c + iw * c + ic] = sdata[in * c * h * w + ic * h * w + ih * w + iw];
                }
            }
        }
    }
    return 0;
}

/* CPU native transpose fallback: NHWC -> NCHW */
int dm_transpose_nhwc_to_nchw_cpu(const DM_Block *src, DM_Block *dst) {
    if (!src || src->ndim != 4 || !dst) return -1;
    int n = src->shape[0], h = src->shape[1], w = src->shape[2], c = src->shape[3];
    if (dm_block_create(dst, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){n, c, h, w}) != 0) return -1;
    const float *sdata = (const float *)src->data;
    float *ddata = (float *)dst->data;
    for (int in = 0; in < n; in++) {
        for (int ih = 0; ih < h; ih++) {
            for (int iw = 0; iw < w; iw++) {
                for (int ic = 0; ic < c; ic++) {
                    ddata[in * c * h * w + ic * h * w + ih * w + iw] = sdata[in * h * w * c + ih * w * c + iw * c + ic];
                }
            }
        }
    }
    return 0;
}
"""

text = text.replace('static TFE_TensorHandle *execute_tf_transpose', cpu_transpose_code + '\nstatic TFE_TensorHandle *execute_tf_transpose')

with open('src/core/dm_engine.c', 'w') as f: f.write(text)

with open('include/core/dm_engine.h', 'r') as f: text2 = f.read()
text2 = text2.replace('DM_LayoutPolicy dm_get_layout_policy(void);', 'DM_LayoutPolicy dm_get_layout_policy(void);\\n\\nint dm_transpose_nchw_to_nhwc_cpu(const DM_Block *src, DM_Block *dst);\\nint dm_transpose_nhwc_to_nchw_cpu(const DM_Block *src, DM_Block *dst);')
with open('include/core/dm_engine.h', 'w') as f: f.write(text2)
