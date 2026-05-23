import re

with open('src/core/dm_engine.c', 'r') as f:
    text = f.read()

# Replace all `var->data[idx]` with `((float*)var->data)[idx]`
# Be careful: `->data` could be an assignment, we only want to replace array access.
text = re.sub(r'([a-zA-Z0-9_]+)->data\[(.*?)\]', r'((float*)\1->data)[\2]', text)

# Fix dm_tensor_count, dm_tensor_fill, dm_tensor_get, dm_tensor_set definition
text = re.sub(r'dm_tensor_count\s*\(\s*const\s+DM_Tensor\s*\*', r'dm_tensor_count(const DM_Block *', text)
text = re.sub(r'dm_tensor_fill\s*\(\s*DM_Tensor\s*\*', r'dm_tensor_fill(DM_Block *', text)
text = re.sub(r'dm_tensor_get\s*\(\s*const\s+DM_Tensor\s*\*', r'dm_tensor_get(const DM_Block *', text)
text = re.sub(r'dm_tensor_set\s*\(\s*DM_Tensor\s*\*', r'dm_tensor_set(DM_Block *', text)
text = re.sub(r'dm_tensor_alloc\s*\(\s*DM_Tensor\s*\*', r'dm_tensor_alloc(DM_Block *', text)
text = re.sub(r'dm_tensor_free\s*\(\s*DM_Tensor\s*\*', r'dm_tensor_free(DM_Block *', text)

with open('src/core/dm_engine.c', 'w') as f:
    f.write(text)

print("Fixed void* indexing and dm_tensor_* signatures.")
