import re

with open("src/core/dm_engine.c", "r") as f: text = f.read()

# Replace dm_layer_norm_seq
old_ln = """int dm_layer_norm_seq(float *x, int seq_len, int d_model,
                      const float *gamma, const float *beta, float eps) {
    if (!x || !gamma || !beta || seq_len <= 0 || d_model <= 0) return -1;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    int64_t xdims[2]   = {seq_len, d_model};
    int64_t gdims[1]   = {d_model};

    TFE_TensorHandle *h_x  = raw_to_tf(x,     xdims, 2);
    TFE_TensorHandle *h_g  = raw_to_tf(gamma, gdims, 1);
    TFE_TensorHandle *h_b  = raw_to_tf(beta,  gdims, 1);"""

new_ln = """int dm_layer_norm_seq(DM_Block *x, const DM_Block *gamma, const DM_Block *beta, float eps) {
    if (!x || !gamma || !beta) return -1;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    
    DM_Block tf_x, tf_g, tf_b;
    if (dm_lower_block(x, &tf_x, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(gamma, &tf_g, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(beta, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return -1;

    TFE_TensorHandle *h_x = (TFE_TensorHandle*)tf_x.handle;
    TFE_TensorHandle *h_g = (TFE_TensorHandle*)tf_g.handle;
    TFE_TensorHandle *h_b = (TFE_TensorHandle*)tf_b.handle;"""

# Wait, there's another replace:
old_ln_end = """    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] LayerNorm(Seq) failed: %s\\n", TF_Message(s));

    if (h_out) { tf_to_raw(h_out, x); TFE_DeleteTensorHandle(h_out); }
    TFE_DeleteTensorHandle(h_x);
    TFE_DeleteTensorHandle(h_g);
    TFE_DeleteTensorHandle(h_b);
    TFE_DeleteTensorHandle(h_ax);
    TFE_DeleteTensorHandle(h_mu[0]);
    TFE_DeleteTensorHandle(h_x0);
    TFE_DeleteTensorHandle(h_sq);
    TFE_DeleteTensorHandle(h_var[0]);
    TFE_DeleteTensorHandle(h_eps);
    TFE_DeleteTensorHandle(h_veps);
    TFE_DeleteTensorHandle(h_inv);
    TFE_DeleteTensorHandle(h_norm);
    TFE_DeleteTensorHandle(h_scaled);
    TF_DeleteStatus(s);
    return 0;
}"""

new_ln_end = """    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] LayerNorm(Seq) failed: %s\\n", TF_Message(s));

    if (h_out) { tf_to_dm(h_out, x); TFE_DeleteTensorHandle(h_out); }
    // Handles h_x, h_g, h_b are cached inside DM_Blocks, do NOT delete them!
    TFE_DeleteTensorHandle(h_ax);
    TFE_DeleteTensorHandle(h_mu[0]);
    TFE_DeleteTensorHandle(h_x0);
    TFE_DeleteTensorHandle(h_sq);
    TFE_DeleteTensorHandle(h_var[0]);
    TFE_DeleteTensorHandle(h_eps);
    TFE_DeleteTensorHandle(h_veps);
    TFE_DeleteTensorHandle(h_inv);
    TFE_DeleteTensorHandle(h_norm);
    TFE_DeleteTensorHandle(h_scaled);
    TF_DeleteStatus(s);
    return 0;
}"""

text = text.replace(old_ln, new_ln).replace(old_ln_end, new_ln_end)

# Also need to add dm_softmax_last_dim
softmax_func = """
void dm_softmax_last_dim(DM_Block *t) {
    if (!t) return;
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "Softmax", s);
    
    DM_Block tf_in;
    if (dm_lower_block(t, &tf_in, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TF_DeleteStatus(s); return; }
    TFE_TensorHandle *h_in = (TFE_TensorHandle*)tf_in.handle;
    
    TFE_OpAddInput(op, h_in, s);
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    TFE_DeleteOp(op);
    if (TF_GetCode(s) != TF_OK) {
        fprintf(stderr, "[dm_engine] Softmax failed: %s\\n", TF_Message(s));
    }
    if (ret[0]) { tf_to_dm(ret[0], t); TFE_DeleteTensorHandle(ret[0]); }
    TF_DeleteStatus(s);
}
"""

text += softmax_func

with open("src/core/dm_engine.c", "w") as f: f.write(text)

