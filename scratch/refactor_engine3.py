import re

filepath = 'src/core/dm_engine.c'
with open(filepath, 'r') as f:
    text = f.read()

# 1. Remove dm_to_tf, tf_to_dm, raw_to_tf, scalar_int32_to_tf, int32_array_to_tf, tf_to_raw
block_start = text.find('static TFE_TensorHandle *dm_to_tf')
block_end = text.find('/* ── Op execution helpers')
if block_start != -1 and block_end != -1:
    text = text[:block_start] + text[block_end:]

# 2. Add macros at the top of op execution helpers
macros = """
// --- DM_Block Lowering Macros ---
#define TF_VIEW(var, src_block) \\
    DM_Block var; \\
    if (dm_lower_block(src_block, &var, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) goto err_label

#define TF_VIEW_RAW(var, data, ndim, dims) \\
    DM_Block _blk_##var; \\
    dm_block_view(&_blk_##var, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, ndim, dims, (void*)(data)); \\
    DM_Block var; \\
    if (dm_lower_block(&_blk_##var, &var, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) goto err_label

#define TF_EXECUTE(res_var, expr) \\
    DM_Block res_var; \\
    if (expr != 0) goto err_label

#define TF_FETCH(tf_block, out_block) \\
    dm_lower_block(&tf_block, out_block, DM_BACKEND_CPU, DM_LOWER_COPY)

"""
text = text.replace('/* ── Op execution helpers', macros + '/* ── Op execution helpers')

# 3. Modify execute_* functions to return int and write to DM_Block *out
# Replace `TFE_TensorHandle *execute...` with `int execute...`
def repl_exec(m):
    return m.group(0).replace('static TFE_TensorHandle *', 'static int ')

text = re.sub(r'static TFE_TensorHandle \*execute_[^(]+\(', lambda m: repl_exec(m), text)
text = re.sub(r'static TFE_TensorHandle \*op[12]\(', lambda m: repl_exec(m), text)

# For op1, op2, execute_* we need to append `, DM_Block *out` to the signature.
# And inside, replace `return ret[0];` with:
# dm_block_create(out, DM_KIND_EXTERNAL, DM_DTYPE_F32, DM_LAYOUT_TF_HANDLE, DM_BACKEND_TENSORFLOW, 0, NULL);
# out->handle = ret[0]; out->owns_handle = 1; out->handle_destructor = dm_tf_tensor_free; return 0;
# And replace `TFE_TensorHandle *arg` with `const DM_Block *arg`, and `(TFE_TensorHandle*)arg->handle` inside.

# This is getting very complex to do reliably with regex.
# Let's use the Python script to do the C code transformation securely.
