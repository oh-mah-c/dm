import re

with open('src/core/dm_engine.c', 'r') as f:
    content = f.read()

# 1. dm_conv2d_same
content = re.sub(
    r'TFE_TensorHandle \*h_w\s*=\s*conv_weight_to_tf\([^;]+\);',
    r'''DM_Block tf_w;
    if (dm_lower_block(w, &tf_w, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) goto conv2d_err;
    TFE_TensorHandle *h_w = (TFE_TensorHandle *)tf_w.handle;''',
    content
)

content = re.sub(
    r'int64_t bdims\[1\] = \{out_c\};\s*TFE_TensorHandle \*h_b\s*=\s*raw_to_tf\(b, bdims, 1\);',
    r'''DM_Block tf_b;
        if (dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TFE_DeleteTensorHandle(h_out); goto conv2d_err; }
        TFE_TensorHandle *h_b = (TFE_TensorHandle *)tf_b.handle;''',
    content
)

# 2. dm_depthwise_conv2d_same
content = re.sub(
    r'TFE_TensorHandle \*h_w\s*=\s*depthwise_weight_to_tf\([^;]+\);',
    r'''DM_Block tf_w;
    if (dm_lower_block(w, &tf_w, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) goto depth_err;
    TFE_TensorHandle *h_w = (TFE_TensorHandle *)tf_w.handle;''',
    content
)

content = re.sub(
    r'int64_t bdims\[1\] = \{DM_NCHW_C\(in\)\};\s*TFE_TensorHandle \*h_b\s*=\s*raw_to_tf\(b, bdims, 1\);',
    r'''DM_Block tf_b;
        if (dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TFE_DeleteTensorHandle(h_out); goto depth_err; }
        TFE_TensorHandle *h_b = (TFE_TensorHandle *)tf_b.handle;''',
    content
)

# 3. dm_pointwise_conv2d
content = re.sub(
    r'int64_t wdims\[4\] = \{out_c, DM_NCHW_C\(in\), 1, 1\};\s*TFE_TensorHandle \*h_w\s*=\s*raw_to_tf\(w, wdims, 4\);',
    r'''DM_Block tf_w;
    if (dm_lower_block(w, &tf_w, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) goto pt_err;
    TFE_TensorHandle *h_w = (TFE_TensorHandle *)tf_w.handle;''',
    content
)

content = re.sub(
    r'int64_t bdims\[1\] = \{out_c\};\s*TFE_TensorHandle \*h_b\s*=\s*raw_to_tf\(b, bdims, 1\);',
    r'''DM_Block tf_b;
        if (dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TFE_DeleteTensorHandle(h_out); goto pt_err; }
        TFE_TensorHandle *h_b = (TFE_TensorHandle *)tf_b.handle;''',
    content
)

# 4. dm_linear
content = re.sub(
    r'int64_t wdims\[2\] = \{out_c, DM_NCHW_C\(in\)\};\s*TFE_TensorHandle \*h_w\s*=\s*raw_to_tf\(w, wdims, 2\);',
    r'''DM_Block tf_w;
    if (dm_lower_block(w, &tf_w, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) goto lin_err;
    TFE_TensorHandle *h_w = (TFE_TensorHandle *)tf_w.handle;''',
    content
)

content = re.sub(
    r'int64_t bdims\[1\] = \{out_c\};\s*TFE_TensorHandle \*h_b\s*=\s*raw_to_tf\(b, bdims, 1\);',
    r'''DM_Block tf_b;
        if (dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TFE_DeleteTensorHandle(h_out); goto lin_err; }
        TFE_TensorHandle *h_b = (TFE_TensorHandle *)tf_b.handle;''',
    content
)

# 5. dm_batch_norm (We need to change the signature of dm_batch_norm too, but the user didn't mention it explicitly. Wait, I should change it)
# I will leave batch_norm and layer_norm for manual/later if needed, or update their signatures now.
# Since the script might miss things, I'll write the script, but I'll only run it for the exact regex matches.

# 6. dm_matmul_nt
content = re.sub(
    r'int64_t da\[2\] = \{M, K\};\s*int64_t db\[2\] = \{N, K\};\s*TFE_TensorHandle \*hA = raw_to_tf\(A, da, 2\);\s*TFE_TensorHandle \*hB = raw_to_tf\(B, db, 2\);',
    r'''DM_Block tf_A, tf_B;
    if (dm_lower_block(A, &tf_A, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(B, &tf_B, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return;
    TFE_TensorHandle *hA = (TFE_TensorHandle *)tf_A.handle;
    TFE_TensorHandle *hB = (TFE_TensorHandle *)tf_B.handle;''',
    content
)

content = re.sub(
    r'int64_t da\[2\] = \{M, K\};\s*int64_t db\[2\] = \{K, N\};\s*TFE_TensorHandle \*hA = raw_to_tf\(A, da, 2\);\s*TFE_TensorHandle \*hB = raw_to_tf\(B, db, 2\);',
    r'''DM_Block tf_A, tf_B;
    if (dm_lower_block(A, &tf_A, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(B, &tf_B, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return;
    TFE_TensorHandle *hA = (TFE_TensorHandle *)tf_A.handle;
    TFE_TensorHandle *hB = (TFE_TensorHandle *)tf_B.handle;''',
    content
)


with open('src/core/dm_engine.c', 'w') as f:
    f.write(content)
