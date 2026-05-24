import re

with open('src/models/vision/swin.c', 'r') as f:
    code = f.read()

def repl_sig(m):
    return m.group(1) + ", DM_ActivationCache *cache) {"

code = re.sub(r'(static int linear_forward\([^)]+?)(?<!cache)\) \{', repl_sig, code)
code = re.sub(r'(static int layer_norm_forward\([^)]+?)(?<!cache)\) \{', repl_sig, code)
code = re.sub(r'(static int mlp_forward\([^)]+?)(?<!cache)\) \{', repl_sig, code)
code = re.sub(r'(static int window_attention_forward\([^)]+?)(?<!cache)\) \{', repl_sig, code)
code = re.sub(r'(static int swin_block_forward\([^)]+?)(?<!cache)\) \{', repl_sig, code)
code = re.sub(r'(static int patch_merging_forward\([^)]+?)(?<!cache)\) \{', repl_sig, code)
code = re.sub(r'(static int stage_forward\([^)]+?)(?<!cache)\) \{', repl_sig, code)
code = re.sub(r'(static int patch_embed_forward\([^)]+?)(?<!cache)\) \{', repl_sig, code)
code = re.sub(r'(int dm_swin_model_forward\(DM_SwinModel \*model, const DM_Block \*input_nhwc, DM_Block \*logits\) \{)',
              r'int dm_swin_model_forward(DM_SwinModel *model, const DM_Block *input_nhwc, DM_ActivationCache *cache, DM_Block *logits) {', code)

# Replace gelu
code = code.replace("static void gelu_inplace(DM_Block *x) {", "static void gelu_inplace(DM_Block *x, DM_ActivationCache *cache) {\n    if (cache) dm_activation_cache_push(cache, x);")

# Update calls
code = code.replace("linear_forward(&mlp->fc1, x, &h)", "linear_forward(&mlp->fc1, x, &h, cache)")
code = code.replace("linear_forward(&mlp->fc2, &h, y)", "linear_forward(&mlp->fc2, &h, y, cache)")
code = code.replace("gelu_inplace(&h)", "gelu_inplace(&h, cache)")

code = code.replace("linear_forward(&attn->qkv, x, &qkv)", "linear_forward(&attn->qkv, x, &qkv, cache)")
code = code.replace("linear_forward(&attn->proj, &ctx_block, y)", "linear_forward(&attn->proj, &ctx_block, y, cache)")

code = code.replace("window_attention_forward(&attn, x, mask, y)", "window_attention_forward(&attn, x, mask, y, NULL)") # In fixture
code = code.replace("window_attention_forward(&blk->attn, &wins3, blk->has_attn_mask ? &blk->attn_mask : NULL, &attn3)", "window_attention_forward(&blk->attn, &wins3, blk->has_attn_mask ? &blk->attn_mask : NULL, &attn3, cache)")

code = code.replace("layer_norm_forward(&blk->norm1, x, &normed)", "layer_norm_forward(&blk->norm1, x, &normed, cache)")
code = code.replace("layer_norm_forward(&blk->norm2, &x_nlc, &normed2)", "layer_norm_forward(&blk->norm2, &x_nlc, &normed2, cache)")
code = code.replace("mlp_forward(&blk->mlp, &normed2, &mlp)", "mlp_forward(&blk->mlp, &normed2, &mlp, cache)")

code = code.replace("layer_norm_forward(&pm->norm, &cat, &normed)", "layer_norm_forward(&pm->norm, &cat, &normed, cache)")
code = code.replace("linear_forward(&pm->reduction, &normed, y)", "linear_forward(&pm->reduction, &normed, y, cache)")

code = code.replace("swin_block_forward(&stage->blocks[i], &cur, &next)", "swin_block_forward(&stage->blocks[i], &cur, &next, cache)")
code = code.replace("patch_merging_forward(&stage->downsample, &cur, &next)", "patch_merging_forward(&stage->downsample, &cur, &next, cache)")

code = code.replace("layer_norm_forward(&pe->norm, &out, tokens)", "layer_norm_forward(&pe->norm, &out, tokens, cache)")

code = code.replace("patch_embed_forward(&model->patch_embed, input_nhwc, &cur)", "patch_embed_forward(&model->patch_embed, input_nhwc, &cur, cache)")
code = code.replace("stage_forward(&model->stages[s], &cur, &next)", "stage_forward(&model->stages[s], &cur, &next, cache)")
code = code.replace("layer_norm_forward(&model->norm, x, &normed)", "layer_norm_forward(&model->norm, x, &normed, cache)")
code = code.replace("linear_forward(&model->head, &pooled, logits)", "linear_forward(&model->head, &pooled, logits, cache)")

code = code.replace("dm_swin_model_forward(&model, &input, &logits)", "dm_swin_model_forward(&model, &input, NULL, &logits)")

# Add pushes inside linear and layer_norm
code = code.replace("if (block_create(y, 3, (int64_t[]){b, l, out_dim}) != 0) return -1;", "if (block_create(y, 3, (int64_t[]){b, l, out_dim}) != 0) return -1;\n    if (cache) dm_activation_cache_push(cache, x);")
code = code.replace("memcpy(y->data, x->data, x->bytes);", "memcpy(y->data, x->data, x->bytes);\n    if (cache) dm_activation_cache_push(cache, x);")

with open('src/models/vision/swin.c', 'w') as f:
    f.write(code)

