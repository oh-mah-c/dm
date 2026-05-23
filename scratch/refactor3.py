import re

with open('src/core/dm_engine.c', 'r') as f:
    text = f.read()

# Replace struct accesses
text = re.sub(r'([a-zA-Z0-9_]+)->n', r'DM_NCHW_N(\1)', text)
text = re.sub(r'([a-zA-Z0-9_]+)->c', r'DM_NCHW_C(\1)', text)
text = re.sub(r'([a-zA-Z0-9_]+)->h', r'DM_NCHW_H(\1)', text)
text = re.sub(r'([a-zA-Z0-9_]+)->w', r'DM_NCHW_W(\1)', text)

# Insert includes
if '#include "lowering/dm_lowering.h"' not in text:
    text = text.replace('#include "core/dm_engine.h"', '#include "core/dm_engine.h"\n#include "lowering/dm_lowering.h"\n#include "lowering/dm_lower_tf.h"\n')

helper_code = """
/* ── Handle helpers ─────────────────────────────────────────────────────── */

static TFE_TensorHandle *dm_to_tf(const DM_Block *t) {
    if (!t || !t->data) return NULL;
    dm_tf_init();
    DM_Block tf_b;
    // Note: CPU -> TF is a zero-copy wrap using TF_NewTensor with noop_dealloc.
    if (dm_lower_block(t, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return NULL;
    // dm_engine.c manually manages and frees TFE handles via TFE_DeleteTensorHandle.
    // We intentionally clear owns_handle so it doesn't try to free it if dm_block_free were called.
    tf_b.owns_handle = 0;
    return (TFE_TensorHandle *)tf_b.handle;
}

static void tf_to_dm(TFE_TensorHandle *h, DM_Block *out) {
    if (!h || !out || !out->data) return;
    DM_Block tf_src;
    memset(&tf_src, 0, sizeof(DM_Block));
    tf_src.backend = DM_BACKEND_TENSORFLOW;
    tf_src.kind = DM_KIND_EXTERNAL;
    tf_src.layout = DM_LAYOUT_TF_HANDLE;
    tf_src.handle = h;
    tf_src.owns_handle = 0; // The caller retains ownership of h
    
    // dm_raise_block triggers TFE_TensorHandleResolve and memcpy internally
    dm_raise_block(&tf_src, out, DM_BACKEND_CPU, DM_LOWER_COPY);
}

static TFE_TensorHandle *raw_to_tf(const float *data, const int64_t *dims, int ndim) {
    dm_tf_init();
    DM_Block b;
    if (dm_block_view(&b, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, ndim, dims, (void*)data) != 0) return NULL;
    
    DM_Block tf_b;
    if (dm_lower_block(&b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return NULL;
    tf_b.owns_handle = 0;
    return (TFE_TensorHandle *)tf_b.handle;
}

static void tf_to_raw(TFE_TensorHandle *h, float *out) {
    if (!h || !out) return;
    TF_Status *s = TF_NewStatus();
    TF_Tensor *tft = TFE_TensorHandleResolve(h, s);
    if (TF_GetCode(s) == TF_OK)
        memcpy(out, TF_TensorData(tft), TF_TensorByteSize(tft));
    TF_DeleteTensor(tft);
    TF_DeleteStatus(s);
}
"""

text = re.sub(r'/\* ── Handle helpers ─────────────────────────────────────────────────────── \*/.*?(?=/\* ── Op execution helpers ───────────────────────────────────────────────── \*/)', helper_code, text, flags=re.DOTALL)

with open('src/core/dm_engine.c', 'w') as f:
    f.write(text)

print("Engine refactored.")
