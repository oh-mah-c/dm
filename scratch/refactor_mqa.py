import re

with open('src/models/vision/mobilenet_tiny.c', 'r') as f:
    text = f.read()

text = text.replace(
    'float *wq = NULL, *wk = NULL, *wv = NULL, *wo = NULL;',
    'DM_Block *wq_b = NULL, *wk_b = NULL, *wv_b = NULL, *wo_b = NULL;\n    float *wq = NULL, *wk = NULL, *wv = NULL, *wo = NULL;'
)
text = text.replace(
    'wq = make_weights((size_t)heads * key_dim * DM_NCHW_C(in), seed + 22, 0.05f);',
    'wq_b = dm_weight_cache_get(cache, 2, (int64_t[]){heads * key_dim, DM_NCHW_C(in)}, seed + 22, 0.05f);\n    wq = (float *)wq_b->data;'
)
text = text.replace(
    'wk = make_weights((size_t)key_dim * DM_NCHW_C(in), seed + 23, 0.05f);',
    'wk_b = dm_weight_cache_get(cache, 2, (int64_t[]){key_dim, DM_NCHW_C(in)}, seed + 23, 0.05f);\n    wk = (float *)wk_b->data;'
)
text = text.replace(
    'wv = make_weights((size_t)key_dim * DM_NCHW_C(in), seed + 24, 0.05f);',
    'wv_b = dm_weight_cache_get(cache, 2, (int64_t[]){key_dim, DM_NCHW_C(in)}, seed + 24, 0.05f);\n    wv = (float *)wv_b->data;'
)
text = text.replace(
    'wo = make_weights((size_t)DM_NCHW_C(in) * heads * key_dim, seed + 25, 0.05f);',
    'wo_b = dm_weight_cache_get(cache, 2, (int64_t[]){DM_NCHW_C(in), heads * key_dim}, seed + 25, 0.05f);\n    wo = (float *)wo_b->data;'
)
text = re.sub(
    r'if \(!wq \|\| !wk \|\| !wv \|\| !wo \|\| !q \|\| !k \|\| !v \|\| !cat\) goto fail;',
    'if (!wq_b || !wk_b || !wv_b || !wo_b || !q || !k || !v || !cat) goto fail;',
    text
)
text = text.replace(
    'dm_block_free(&kv_in); free(wq); free(wk); free(wv); free(wo); free(q); free(k); free(v); free(cat);',
    'dm_block_free(&kv_in); free(q); free(k); free(v); free(cat);'
)

# Also refactor mobilenet_features to take and pass cache
text = text.replace(
    'static int mobilenet_features(const DM_Block *input, DM_Block *features, unsigned int seed) {',
    'static int mobilenet_features(const DM_Block *input, DM_Block *features, DM_WeightCache *cache, unsigned int seed) {'
)
text = text.replace('fused_ib(input, &x1, 16, 16, 3, 2, seed + 100)', 'fused_ib(input, &x1, cache, 16, 16, 3, 2, seed + 100)')
text = text.replace('dm_uib_block(&x1, &x2, DM_UIB_EXTRADW, 64, 24, 3, 3, 2, seed + 200)', 'dm_uib_block(&x1, &x2, cache, DM_UIB_EXTRADW, 64, 24, 3, 3, 2, seed + 200)')
text = text.replace('dm_uib_block(&x2, &x3, DM_UIB_IB, 96, 24, 3, 3, 1, seed + 300)', 'dm_uib_block(&x2, &x3, cache, DM_UIB_IB, 96, 24, 3, 3, 1, seed + 300)')
text = text.replace('dm_uib_block(&x3, &x4, DM_UIB_CONVNEXT, 96, 32, 5, 3, 2, seed + 400)', 'dm_uib_block(&x3, &x4, cache, DM_UIB_CONVNEXT, 96, 32, 5, 3, 2, seed + 400)')
text = text.replace('dm_mobile_mqa_block(&x4, &x5, 4, 8, 1, seed + 500)', 'dm_mobile_mqa_block(&x4, &x5, cache, 4, 8, 1, seed + 500)')
text = text.replace('pointwise_relu(&x5, &x6, 64, seed + 600, 1)', 'pointwise_relu(&x5, &x6, cache, 64, seed + 600, 1)')
text = text.replace(
    'int dm_mobilenet_tiny_forward(const DM_Block *input, DM_Block *logits, int classes, unsigned int seed) {',
    'int dm_mobilenet_tiny_forward(const DM_Block *input, DM_Block *logits, DM_WeightCache *cache, int classes, unsigned int seed) {'
)
text = text.replace('mobilenet_features(input, &features, seed)', 'mobilenet_features(input, &features, cache, seed)')

# dm_linear inside dm_mobilenet_tiny_forward
text = text.replace('float *w = NULL;', 'DM_Block *w_b = NULL;')
text = text.replace(
    'w = make_weights((size_t)classes * DM_NCHW_C(&features), seed + 700, 0.05f);',
    'w_b = dm_weight_cache_get(cache, 2, (int64_t[]){classes, DM_NCHW_C(&features)}, seed + 700, 0.05f);'
)
text = text.replace('if (!w) goto done;', 'if (!w_b) goto done;')
text = text.replace(
    'if (dm_linear(&features, logits, w, NULL, classes) != 0) goto done;',
    'if (dm_linear(&features, logits, w_b, NULL, classes) != 0) goto done;'
)
text = text.replace('free(w);', '')

with open('src/models/vision/mobilenet_tiny.c', 'w') as f:
    f.write(text)

