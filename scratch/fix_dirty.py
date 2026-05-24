import re

with open("src/core/dm_engine.c", "r") as f:
    text = f.read()

# Fix tf_to_dm
old_tf_to_dm = """static void tf_to_dm(TFE_TensorHandle *h, DM_Block *out) {
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
}"""

new_tf_to_dm = """static void tf_to_dm(TFE_TensorHandle *h, DM_Block *out) {
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
    out->dirty = 1;
    out->version++;
}"""

text = text.replace(old_tf_to_dm, new_tf_to_dm)
with open("src/core/dm_engine.c", "w") as f: f.write(text)

with open("src/models/vision/vit.c", "r") as f:
    text = f.read()

# I will just replace the manual loops with loops that also set dirty = 1!
# In mhsa_forward:
text = text.replace(
    "q_ptr[i*3*D + j] += qb_ptr[j];",
    "q_ptr[i*3*D + j] += qb_ptr[j];\n    qkv.dirty = 1; qkv.version++;"
)

text = text.replace(
    "x_ptr[(size_t)i*D + j] += pb_ptr[j];",
    "x_ptr[(size_t)i*D + j] += pb_ptr[j];\n    x->dirty = 1; x->version++;"
)

# In mlp_forward:
text = text.replace(
    "h_ptr[(size_t)i*M + j] += b1_ptr[j];",
    "h_ptr[(size_t)i*M + j] += b1_ptr[j];\n    hidden.dirty = 1; hidden.version++;"
)

text = text.replace(
    "dm_gelu_inplace(h_ptr, seq * M);",
    "dm_gelu_inplace(h_ptr, seq * M);\n    hidden.dirty = 1; hidden.version++;"
)

text = text.replace(
    "x_ptr[(size_t)i*D + j] += b2_ptr[j];",
    "x_ptr[(size_t)i*D + j] += b2_ptr[j];\n    x->dirty = 1; x->version++;"
)

# In dm_vit_forward:
text = text.replace(
    "z_ptr[(size_t)(i+1)*D + j] = pp_ptr[(size_t)i*D + j] + pb_ptr[j];",
    "z_ptr[(size_t)(i+1)*D + j] = pp_ptr[(size_t)i*D + j] + pb_ptr[j];\n            z.dirty = 1; z.version++;"
)

text = text.replace(
    "z_ptr[(size_t)i*D + j] += pe_ptr[(size_t)i*D + j];",
    "z_ptr[(size_t)i*D + j] += pe_ptr[(size_t)i*D + j];\n            z.dirty = 1; z.version++;"
)

text = text.replace(
    "for (size_t i = 0; i < (size_t)seq*D; i++) z_ptr[i] += ((float*)z_res.data)[i];",
    "for (size_t i = 0; i < (size_t)seq*D; i++) z_ptr[i] += ((float*)z_res.data)[i];\n        z.dirty = 1; z.version++;"
)

text = text.replace(
    "memcpy(z_cls.data, z.data, D * sizeof(float));",
    "memcpy(z_cls.data, z.data, D * sizeof(float));\n    z_cls.dirty = 1; z_cls.version++;"
)

with open("src/models/vision/vit.c", "w") as f: f.write(text)
