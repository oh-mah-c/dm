import re

with open("src/models/vision/resnet.c", "r") as f:
    text = f.read()

# Replace make_weights function completely, as it's no longer needed
text = re.sub(r'static float \*make_weights.*?return w;\n}\n', '', text, flags=re.DOTALL)

# Add DM_WeightCache *cache parameter to dm_resnet_basic_block
text = text.replace(
    'int dm_resnet_basic_block(const DM_Block *in, DM_Block *out, int out_c, int stride, unsigned int seed) {',
    'int dm_resnet_basic_block(DM_WeightCache *cache, const DM_Block *in, DM_Block *out, int out_c, int stride, unsigned int seed) {'
)

# Replace w1 creation in basic block
text = re.sub(
    r'float \*w1 = make_weights\(\(size_t\)out_c \* DM_NCHW_C\(in\) \* 3 \* 3, seed \+ 1, 0\.08f\);\n\s*if \(\!w1\) return -1;\n\s*int rc = dm_conv2d_same\(in, &x1, w1, NULL, out_c, 3, stride\);\n\s*free\(w1\);',
    r'DM_Block *w1 = dm_weight_cache_get(cache, 4, (int64_t[]){3, 3, DM_NCHW_C(in), out_c}, seed + 1, 0.08f);\n    int rc = dm_conv2d_same(in, &x1, w1, NULL, out_c, 3, stride);',
    text
)

# Replace BN1
text = re.sub(
    r'float \*gamma1 = make_weights\(out_c, seed \+ 2, 1\.0f\);\n\s*float \*beta1 = make_weights\(out_c, seed \+ 3, 0\.0f\);\n\s*float \*mean1 = make_weights\(out_c, seed \+ 4, 0\.0f\);\n\s*float \*var1 = make_weights\(out_c, seed \+ 5, 1\.0f\);\n\s*if \(\!gamma1 \|\| \!beta1 \|\| \!mean1 \|\| \!var1\) \{\n\s*free\(gamma1\); free\(beta1\); free\(mean1\); free\(var1\);\n\s*dm_block_free\(&x1\); return -1;\n\s*\}\n\s*rc = dm_batch_norm\(&x1, gamma1, beta1, mean1, var1, 1e-5f\);\n\s*free\(gamma1\); free\(beta1\); free\(mean1\); free\(var1\);',
    r'DM_Block *gamma1 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 2, 1.0f);\n    DM_Block *beta1 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 3, 0.0f);\n    DM_Block *mean1 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 4, 0.0f);\n    DM_Block *var1 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 5, 1.0f);\n    rc = dm_batch_norm(&x1, (float*)gamma1->data, (float*)beta1->data, (float*)mean1->data, (float*)var1->data, 1e-5f);',
    text
)

# Replace w2 creation
text = re.sub(
    r'float \*w2 = make_weights\(\(size_t\)out_c \* out_c \* 3 \* 3, seed \+ 6, 0\.08f\);\n\s*if \(\!w2\) \{ dm_block_free\(&x1\); return -1; \}\n\s*rc = dm_conv2d_same\(&x1, &x2, w2, NULL, out_c, 3, 1\);\n\s*free\(w2\);',
    r'DM_Block *w2 = dm_weight_cache_get(cache, 4, (int64_t[]){3, 3, out_c, out_c}, seed + 6, 0.08f);\n    rc = dm_conv2d_same(&x1, &x2, w2, NULL, out_c, 3, 1);',
    text
)

# Replace BN2
text = re.sub(
    r'float \*gamma2 = make_weights\(out_c, seed \+ 7, 1\.0f\);\n\s*float \*beta2 = make_weights\(out_c, seed \+ 8, 0\.0f\);\n\s*float \*mean2 = make_weights\(out_c, seed \+ 9, 0\.0f\);\n\s*float \*var2 = make_weights\(out_c, seed \+ 10, 1\.0f\);\n\s*if \(\!gamma2 \|\| \!beta2 \|\| \!mean2 \|\| \!var2\) \{\n\s*free\(gamma2\); free\(beta2\); free\(mean2\); free\(var2\);\n\s*dm_block_free\(&x2\); return -1;\n\s*\}\n\s*rc = dm_batch_norm\(&x2, gamma2, beta2, mean2, var2, 1e-5f\);\n\s*free\(gamma2\); free\(beta2\); free\(mean2\); free\(var2\);',
    r'DM_Block *gamma2 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 7, 1.0f);\n    DM_Block *beta2 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 8, 0.0f);\n    DM_Block *mean2 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 9, 0.0f);\n    DM_Block *var2 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 10, 1.0f);\n    rc = dm_batch_norm(&x2, (float*)gamma2->data, (float*)beta2->data, (float*)mean2->data, (float*)var2->data, 1e-5f);',
    text
)

# Replace shortcut w_short
text = re.sub(
    r'float \*w_short = make_weights\(\(size_t\)out_c \* DM_NCHW_C\(in\) \* 1 \* 1, seed \+ 11, 0\.08f\);\n\s*if \(\!w_short\) \{ dm_block_free\(&x2\); return -1; \}\n\s*rc = dm_conv2d_same\(in, &shortcut, w_short, NULL, out_c, 1, stride\);\n\s*free\(w_short\);',
    r'DM_Block *w_short = dm_weight_cache_get(cache, 4, (int64_t[]){1, 1, DM_NCHW_C(in), out_c}, seed + 11, 0.08f);\n        rc = dm_conv2d_same(in, &shortcut, w_short, NULL, out_c, 1, stride);',
    text
)

# Replace BN shortcut
text = re.sub(
    r'float \*gamma_s = make_weights\(out_c, seed \+ 12, 1\.0f\);\n\s*float \*beta_s = make_weights\(out_c, seed \+ 13, 0\.0f\);\n\s*float \*mean_s = make_weights\(out_c, seed \+ 14, 0\.0f\);\n\s*float \*var_s = make_weights\(out_c, seed \+ 15, 1\.0f\);\n\s*if \(\!gamma_s \|\| \!beta_s \|\| \!mean_s \|\| \!var_s\) \{\n\s*free\(gamma_s\); free\(beta_s\); free\(mean_s\); free\(var_s\);\n\s*dm_block_free\(&x2\); dm_block_free\(&shortcut\); return -1;\n\s*\}\n\s*rc = dm_batch_norm\(&shortcut, gamma_s, beta_s, mean_s, var_s, 1e-5f\);\n\s*free\(gamma_s\); free\(beta_s\); free\(mean_s\); free\(var_s\);',
    r'DM_Block *gamma_s = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 12, 1.0f);\n        DM_Block *beta_s = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 13, 0.0f);\n        DM_Block *mean_s = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 14, 0.0f);\n        DM_Block *var_s = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 15, 1.0f);\n        rc = dm_batch_norm(&shortcut, (float*)gamma_s->data, (float*)beta_s->data, (float*)mean_s->data, (float*)var_s->data, 1e-5f);',
    text
)

# Now refactor dm_resnet18_forward
text = text.replace(
    'int dm_resnet18_forward(const DM_Block *input, DM_Block *logits, int classes, unsigned int seed) {',
    'int dm_resnet18_forward(const DM_Block *input, DM_Block *logits, int classes, unsigned int seed) {\n    DM_WeightCache *cache = dm_weight_cache_new();\n    if (!cache) return -1;'
)

# Replace conv1
text = re.sub(
    r'float \*w_conv1 = make_weights\(\(size_t\)64 \* DM_NCHW_C\(input\) \* 7 \* 7, s\+\+, 0\.08f\);\n\s*if \(\!w_conv1\) return -1;\n\s*int rc = dm_conv2d_same\(input, &t1, w_conv1, NULL, 64, 7, 2\);\n\s*free\(w_conv1\);',
    r'DM_Block *w_conv1 = dm_weight_cache_get(cache, 4, (int64_t[]){7, 7, DM_NCHW_C(input), 64}, s++, 0.08f);\n    int rc = dm_conv2d_same(input, &t1, w_conv1, NULL, 64, 7, 2);',
    text
)

# Replace BN1 in forward
text = re.sub(
    r'float \*g1 = make_weights\(64, s\+\+, 1\.0f\);\n\s*float \*b1 = make_weights\(64, s\+\+, 0\.0f\);\n\s*float \*m1 = make_weights\(64, s\+\+, 0\.0f\);\n\s*float \*v1 = make_weights\(64, s\+\+, 1\.0f\);\n\s*if \(\!g1 \|\| \!b1 \|\| \!m1 \|\| \!v1\) \{\n\s*free\(g1\); free\(b1\); free\(m1\); free\(v1\);\n\s*dm_block_free\(&t1\); return -1;\n\s*\}\n\s*rc = dm_batch_norm\(&t1, g1, b1, m1, v1, 1e-5f\);\n\s*free\(g1\); free\(b1\); free\(m1\); free\(v1\);',
    r'DM_Block *g1 = dm_weight_cache_get(cache, 1, (int64_t[]){64}, s++, 1.0f);\n    DM_Block *b1 = dm_weight_cache_get(cache, 1, (int64_t[]){64}, s++, 0.0f);\n    DM_Block *m1 = dm_weight_cache_get(cache, 1, (int64_t[]){64}, s++, 0.0f);\n    DM_Block *v1 = dm_weight_cache_get(cache, 1, (int64_t[]){64}, s++, 1.0f);\n    rc = dm_batch_norm(&t1, (float*)g1->data, (float*)b1->data, (float*)m1->data, (float*)v1->data, 1e-5f);',
    text
)

# Fix dm_resnet_basic_block calls
text = re.sub(r'rc = dm_resnet_basic_block\(&(t\d+), &(t\d+), (\d+), (\d+), s\);', r'rc = dm_resnet_basic_block(cache, &\1, &\2, \3, \4, s);', text)

# Replace FC/Linear
text = re.sub(
    r'float \*w_fc = make_weights\(\(size_t\)classes \* 512, s\+\+, 0\.08f\);\n\s*float \*b_fc = make_weights\(classes, s\+\+, 0\.0f\);\n\s*if \(\!w_fc \|\| \!b_fc\) \{\n\s*free\(w_fc\); free\(b_fc\);\n\s*dm_block_free\(&t1\); return -1;\n\s*\}\n\s*rc = dm_linear\(&t1, logits, w_fc, b_fc, classes\);\n\s*free\(w_fc\); free\(b_fc\);',
    r'DM_Block *w_fc = dm_weight_cache_get(cache, 2, (int64_t[]){classes, 512}, s++, 0.08f);\n    DM_Block *b_fc = dm_weight_cache_get(cache, 1, (int64_t[]){classes}, s++, 0.0f);\n    rc = dm_linear(&t1, logits, (float*)w_fc->data, (float*)b_fc->data, classes);',
    text
)

# And dm_linear is expecting float* for weights currently... wait!
# Let's check dm_linear signature...
text = re.sub(r'return 0;\n\}', r'dm_weight_cache_free(cache);\n    return 0;\n}', text, count=1)
text = text.replace('dm_block_free(&t1); return -1;', 'dm_weight_cache_free(cache); dm_block_free(&t1); return -1;')

with open("src/models/vision/resnet.c", "w") as f:
    f.write(text)
