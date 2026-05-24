import re

# 1. Update include/models/vision/mobilenet_tiny.h
with open('include/models/vision/mobilenet_tiny.h', 'r') as f:
    h_text = f.read()

cache_struct = """typedef struct {
    DM_Block **blocks;
    uint32_t *seeds;
    int count;
    int capacity;
} DM_WeightCache;
"""

if "DM_WeightCache;" not in h_text:
    h_text = h_text.replace(
        "typedef enum {",
        cache_struct + "\ntypedef enum {"
    )

h_text = h_text.replace(
    "int dm_uib_block(const DM_Block *in, DM_Block *out, DM_UIBKind kind,",
    "int dm_uib_block(const DM_Block *in, DM_Block *out, DM_WeightCache *cache, DM_UIBKind kind,"
)

h_text = h_text.replace(
    "int dm_mobile_mqa_block(const DM_Block *in, DM_Block *out, int heads, int key_dim,",
    "int dm_mobile_mqa_block(const DM_Block *in, DM_Block *out, DM_WeightCache *cache, int heads, int key_dim,"
)

h_text = h_text.replace(
    "int dm_mobilenet_tiny_forward(const DM_Block *input, DM_Block *logits, int classes, unsigned int seed);",
    "int dm_mobilenet_tiny_forward(const DM_Block *input, DM_Block *logits, DM_WeightCache *cache, int classes, unsigned int seed);"
)

with open('include/models/vision/mobilenet_tiny.h', 'w') as f:
    f.write(h_text)

# 2. Update src/models/vision/mobilenet_tiny.c
with open('src/models/vision/mobilenet_tiny.c', 'r') as f:
    c_text = f.read()

# Remove the typedef from .c since it's now in .h
c_text = re.sub(r'typedef struct \{\n    DM_Block \*\*blocks;\n    uint32_t \*seeds;\n    int count;\n    int capacity;\n\} DM_WeightCache;\n', '', c_text)

# Fix dm_weight_cache_free(&cache); in cmd_train and cmd_infer where cache is undeclared
c_text = c_text.replace(
    'if (!fp) { head_free(&head); dm_weight_cache_free(&cache); return 1; }',
    'if (!fp) { head_free(&head); return 1; }'
)

c_text = c_text.replace(
    'if (!grad_w || !grad_b) { free(grad_w); free(grad_b); fclose(fp); head_free(&head); dm_weight_cache_free(&cache); return 1; }',
    'if (!grad_w || !grad_b) { free(grad_w); free(grad_b); fclose(fp); head_free(&head); return 1; }'
)

c_text = c_text.replace(
    'if (!raw) { dm_block_free(&feat); head_free(&head); dm_block_free(&logits); return 1; }',
    'if (!raw) { dm_block_free(&feat); head_free(&head); dm_block_free(&logits); return 1; }'
)
# Wait, let's just use regex to remove any dm_weight_cache_free(&cache); before it is declared.
# In cmd_train, cache is declared inside the epoch loop!
# Wait, actually, let me just find those specific lines and fix them.
c_text = re.sub(r'head_free\(&head\);\s*dm_weight_cache_free\(&cache\);\s*return 1;', 'head_free(&head); return 1;', c_text)

with open('src/models/vision/mobilenet_tiny.c', 'w') as f:
    f.write(c_text)


# 3. Update src/core/dm_engine.c
with open('src/core/dm_engine.c', 'r') as f:
    e_text = f.read()

e_text = e_text.replace(
    '{ TFE_DeleteTensorHandle(h_out); goto conv2d_err; }',
    '{ TFE_DeleteTensorHandle(h_out); goto pw_err; }' # Only on pointwise! Wait, I will use regex.
)
# Specifically in dm_pointwise_conv2d
e_text = re.sub(
    r'int dm_pointwise_conv2d(.*?)goto conv2d_err(.*?)}',
    r'int dm_pointwise_conv2d\1goto pw_err\2}',
    e_text, flags=re.DOTALL
)

with open('src/core/dm_engine.c', 'w') as f:
    f.write(e_text)
