import re

with open('src/core/dm_engine.c', 'r') as f:
    text = f.read()

# Fix DM_ERR_INCOMPATIBLE -> -1
text = text.replace('return DM_ERR_INCOMPATIBLE;', 'return -1;')

# Restore int32_array_to_tf and scalar_int32_to_tf
missing_helpers = """
static TFE_TensorHandle *scalar_int32_to_tf(int32_t v) {
    dm_tf_init();
    int64_t dims[1] = {1};
    int32_t *buf = (int32_t *)malloc(sizeof(int32_t));
    *buf = v;
    TF_Tensor *tft = TF_NewTensor(TF_INT32, dims, 1, buf, sizeof(int32_t), free_dealloc, NULL);
    TF_Status *s = TF_NewStatus();
    TFE_TensorHandle *h = TFE_NewTensorHandle(tft, s);
    TF_DeleteTensor(tft);
    TF_DeleteStatus(s);
    return h;
}

static TFE_TensorHandle *int32_array_to_tf(const int32_t *arr, int n) {
    dm_tf_init();
    int64_t dims[1] = {n};
    size_t sz = (size_t)n * sizeof(int32_t);
    int32_t *buf = (int32_t *)malloc(sz);
    memcpy(buf, arr, sz);
    TF_Tensor *tft = TF_NewTensor(TF_INT32, dims, 1, buf, sz, free_dealloc, NULL);
    TF_Status *s = TF_NewStatus();
    TFE_TensorHandle *h = TFE_NewTensorHandle(tft, s);
    TF_DeleteTensor(tft);
    TF_DeleteStatus(s);
    return h;
}

/* ── Op execution helpers"""

text = text.replace('/* ── Op execution helpers', missing_helpers)

# Fix idx4
text = text.replace('static size_t idx4(const DM_Tensor *t,', 'static size_t idx4(const DM_Block *t,')

# Revert dm_tensor_* to take DM_Tensor again to match header, 
# OR change the header to DM_Block.
# Since the header is what we want to keep as DM_Tensor (deprecated), we revert the dm_engine.c definitions:
text = text.replace('size_t dm_tensor_count(const DM_Block *t)', 'size_t dm_tensor_count(const DM_Tensor *t)')
text = text.replace('int dm_tensor_alloc(DM_Block *t, int n, int c, int h, int w)', 'int dm_tensor_alloc(DM_Tensor *t, int n, int c, int h, int w)')
text = text.replace('void dm_tensor_free(DM_Block *t)', 'void dm_tensor_free(DM_Tensor *t)')
text = text.replace('void dm_tensor_fill(DM_Block *t, float value)', 'void dm_tensor_fill(DM_Tensor *t, float value)')
text = text.replace('float dm_tensor_get(const DM_Block *t, int n, int c, int y, int x)', 'float dm_tensor_get(const DM_Tensor *t, int n, int c, int y, int x)')
text = text.replace('void dm_tensor_set(DM_Block *t, int n, int c, int y, int x, float v)', 'void dm_tensor_set(DM_Tensor *t, int n, int c, int y, int x, float v)')

# Also revert idx4 in dm_tensor_get/set to cast correctly if needed
# Wait, if dm_tensor_get uses t->n, the macro DM_NCHW_N doesn't work on DM_Tensor.
# Because in dm_engine.c, t->n was replaced with DM_NCHW_N(t). 
# But DM_NCHW_N assumes DM_Block.
# So inside dm_tensor_get, we should cast DM_Tensor to DM_Block before using DM_NCHW_N, or just use t->n.
# Let's just fix dm_tensor_*.c functions manually since there are only a few.

text = re.sub(r'size_t dm_tensor_count\(const DM_Tensor \*t\) \{.*?(?=int dm_tensor_alloc)', 
'''size_t dm_tensor_count(const DM_Tensor *t) {
    if (!t) return 0;
    return (size_t)t->n * t->c * t->h * t->w;
}

''', text, flags=re.DOTALL)

text = re.sub(r'int dm_tensor_alloc\(DM_Tensor \*t, int n, int c, int h, int w\) \{.*?(?=void dm_tensor_free)', 
'''int dm_tensor_alloc(DM_Tensor *t, int n, int c, int h, int w) {
    if (!t) return -1;
    t->n = n; t->c = c; t->h = h; t->w = w;
    size_t sz = (size_t)n * c * h * w * sizeof(float);
    if (sz == 0) { t->data = NULL; return 0; }
    t->data = (float*)malloc(sz);
    return t->data ? 0 : -1;
}

''', text, flags=re.DOTALL)

text = re.sub(r'void dm_tensor_free\(DM_Tensor \*t\) \{.*?(?=void dm_tensor_fill)', 
'''void dm_tensor_free(DM_Tensor *t) {
    if (!t) return;
    free(t->data);
    t->data = NULL;
}

''', text, flags=re.DOTALL)

text = re.sub(r'void dm_tensor_fill\(DM_Tensor \*t, float value\) \{.*?(?=static size_t idx4)', 
'''void dm_tensor_fill(DM_Tensor *t, float value) {
    if (!t || !t->data) return;
    size_t n = dm_tensor_count(t);
    for (size_t i = 0; i < n; i++) t->data[i] = value;
}

''', text, flags=re.DOTALL)

# In dm_engine.c, idx4 is currently:
# return (((size_t)n * (size_t)DM_NCHW_C(t) + (size_t)c) * (size_t)DM_NCHW_H(t) + (size_t)y) * (size_t)DM_NCHW_W(t) + (size_t)x;
# I need to fix it for DM_Block.
idx4_fixed = '''static size_t idx4(const DM_Block *t, int n, int c, int y, int x) {
    return (((size_t)n * (size_t)DM_NCHW_C(t) + (size_t)c) * (size_t)DM_NCHW_H(t) + (size_t)y) * (size_t)DM_NCHW_W(t) + (size_t)x;
}'''
text = re.sub(r'static size_t idx4\(const DM_Block \*t, int n, int c, int y, int x\) \{.*?\}', idx4_fixed, text, flags=re.DOTALL)

# Revert dm_tensor_get and set
text = re.sub(r'float dm_tensor_get\(const DM_Tensor \*t, int n, int c, int y, int x\) \{.*?(?=void dm_tensor_set)',
'''float dm_tensor_get(const DM_Tensor *t, int n, int c, int y, int x) {
    if (!t || !t->data) return 0.0f;
    size_t idx = (((size_t)n * t->c + c) * t->h + y) * t->w + x;
    return t->data[idx];
}

''', text, flags=re.DOTALL)

text = re.sub(r'void dm_tensor_set\(DM_Tensor \*t, int n, int c, int y, int x, float v\) \{.*?(?=int dm_conv2d_same)',
'''void dm_tensor_set(DM_Tensor *t, int n, int c, int y, int x, float v) {
    if (!t || !t->data) return;
    size_t idx = (((size_t)n * t->c + c) * t->h + y) * t->w + x;
    t->data[idx] = v;
}

''', text, flags=re.DOTALL)

with open('src/core/dm_engine.c', 'w') as f:
    f.write(text)

print("Engine legacy funcs fixed.")
