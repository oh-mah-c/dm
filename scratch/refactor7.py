import re

with open('src/core/dm_engine.c', 'r') as f:
    text = f.read()

# Replace dm_tensor_count(X) -> (X)->count
text = re.sub(r'dm_tensor_count\(\s*([^)]+)\s*\)', r'(\1)->count', text)

# Replace dm_tensor_fill(X, 0.0f) -> memset(((float*)(X)->data), 0, (X)->count * sizeof(float))
text = re.sub(r'dm_tensor_fill\(\s*([^,]+)\s*,\s*0\.0f\s*\)', r'memset(((float*)(\1)->data), 0, (\1)->count * sizeof(float))', text)

# Replace dm_tensor_get(t, n, c, y, x) -> ((float*)(t)->data)[(((size_t)n * DM_NCHW_C(t) + c) * DM_NCHW_H(t) + y) * DM_NCHW_W(t) + x]
def repl_get(m):
    t, n, c, y, x = m.groups()
    return f'((float*)({t})->data)[(((size_t){n} * DM_NCHW_C({t}) + {c}) * DM_NCHW_H({t}) + {y}) * DM_NCHW_W({t}) + {x}]'

text = re.sub(r'dm_tensor_get\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', repl_get, text)

# Replace dm_tensor_set(t, n, c, y, x, v) -> ((float*)(t)->data)[...] = v
def repl_set(m):
    t, n, c, y, x, v = m.groups()
    return f'((float*)({t})->data)[(((size_t){n} * DM_NCHW_C({t}) + {c}) * DM_NCHW_H({t}) + {y}) * DM_NCHW_W({t}) + {x}] = {v}'

text = re.sub(r'dm_tensor_set\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', repl_set, text)

# Replace dm_tensor_alloc(t, n, c, h, w) -> dm_block_create(t, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){n, c, h, w})
def repl_alloc(m):
    t, n, c, h, w = m.groups()
    return f'dm_block_create({t}, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){{{n}, {c}, {h}, {w}}})'

text = re.sub(r'dm_tensor_alloc\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', repl_alloc, text)

with open('src/core/dm_engine.c', 'w') as f:
    f.write(text)

print("Fixed internal usages of dm_tensor_*")
