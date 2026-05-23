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

# We need to replace dm_to_tf with lower_view helper.
helper_code = """
/* ── DM_Block Lowering Helpers ──────────────────────────────────────────── */

static TFE_TensorHandle *block_to_tf(const DM_Block *t, DM_Block *tf_out) {
    if (!t) return NULL;
    if (dm_lower_block(t, tf_out, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return NULL;
    return (TFE_TensorHandle *)tf_out->handle;
}

static TFE_TensorHandle *raw_to_tf(const float *data, const int64_t *dims, int ndim) {
    DM_Block b;
    if (dm_block_view(&b, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, ndim, dims, (void*)data) != 0) return NULL;
    DM_Block tf_b;
    if (dm_lower_block(&b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return NULL;
    // We intentionally "leak" the tf_b wrapper lifecycle to the caller by returning just the handle.
    // The caller of raw_to_tf calls TFE_DeleteTensorHandle() explicitly.
    return (TFE_TensorHandle *)tf_b.handle;
}

static void tf_to_block(TFE_TensorHandle *h, DM_Block *out) {
    if (!h || !out || !out->data) return;
    
    // Create an external TF block wrapper for the handle
    DM_Block tf_src;
    memset(&tf_src, 0, sizeof(DM_Block));
    tf_src.backend = DM_BACKEND_TENSORFLOW;
    tf_src.kind = DM_KIND_EXTERNAL;
    tf_src.layout = DM_LAYOUT_TF_HANDLE;
    tf_src.handle = h;
    tf_src.owns_handle = 0; // We don't want dm_raise_block to free it, caller handles it.
    
    // Raise into CPU buffer
    dm_raise_block(&tf_src, out, DM_BACKEND_CPU, DM_LOWER_COPY);
}
"""

# Remove old dm_to_tf, tf_to_dm, raw_to_tf
text = re.sub(r'/\* ── Handle helpers ─────────────────────────────────────────────────────── \*/.*?(?=/\* ── Op execution helpers ───────────────────────────────────────────────── \*/)', helper_code, text, flags=re.DOTALL)

# Now find all usages of dm_to_tf(t) and replace them with:
# DM_Block tf_t; TFE_TensorHandle *h = block_to_tf(t, &tf_t);
# and add dm_block_free(&tf_t) at the end of the function!
# This is tricky using regex.

with open('src/core/dm_engine.c', 'w') as f:
    f.write(text)

print("Basic regex replaced.")
