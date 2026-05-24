import os

filepath = 'src/lib/dm_lib.c'
with open(filepath, 'r') as f:
    lines = f.readlines()

out = []
for line in lines:
    if 'dm_tensor_alloc' in line and 'extern ' in line:
        continue
    if 'dm_tensor_free' in line and 'extern ' in line:
        continue
    if 'dm_tensor_fill' in line and 'extern ' in line:
        continue
    if 'dm_tensor_get' in line and 'extern ' in line:
        continue
    if 'dm_tensor_set' in line and 'extern ' in line:
        continue
    if 'dm_tensor_count' in line and 'extern ' in line:
        continue
    out.append(line)

with open(filepath, 'w') as f:
    f.writelines(out)

print("dm_lib.c cleaned.")
