import re
import os

h_files = [
    'include/models/vision/mobilenet_tiny.h',
    'include/models/vision/resnet.h',
    'include/models/vision/vit.h',
    'include/models/gan.h',
    'include/models/vae.h',
    'include/encoding/image_patchify.h'
]

c_files = [
    'src/models/gan.c',
    'src/models/vae.c',
    'src/models/vision/resnet.c',
    'src/models/vision/vit.c',
    'src/models/vision/mobilenet_tiny.c',
    'src/encoding/image_patchify.c'
]

def replace_tensor_alloc(m):
    t, n, c, h, w = m.groups()
    return f'dm_block_create({t}, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){{{n}, {c}, {h}, {w}}})'

for filepath in h_files + c_files:
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, 'r') as f:
        text = f.read()

    # Replace DM_Tensor with DM_Block
    text = text.replace('DM_Tensor', 'DM_Block')
    text = text.replace('dm_tensor_free', 'dm_block_free')

    if filepath in c_files:
        # Replace dm_tensor_alloc
        text = re.sub(r'dm_tensor_alloc\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', replace_tensor_alloc, text)
        
        # Replace ->n, ->c, ->h, ->w
        text = re.sub(r'([a-zA-Z0-9_]+)->n', r'DM_NCHW_N(\1)', text)
        text = re.sub(r'([a-zA-Z0-9_]+)->c', r'DM_NCHW_C(\1)', text)
        text = re.sub(r'([a-zA-Z0-9_]+)->h', r'DM_NCHW_H(\1)', text)
        text = re.sub(r'([a-zA-Z0-9_]+)->w', r'DM_NCHW_W(\1)', text)

        # For image patchify, replace dm_tensor_get and set
        def repl_get(m):
            t, n, c, y, x = m.groups()
            return f'((float*)({t})->data)[(((size_t){n} * DM_NCHW_C({t}) + {c}) * DM_NCHW_H({t}) + {y}) * DM_NCHW_W({t}) + {x}]'

        text = re.sub(r'dm_tensor_get\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', repl_get, text)

        def repl_set(m):
            t, n, c, y, x, v = m.groups()
            return f'((float*)({t})->data)[(((size_t){n} * DM_NCHW_C({t}) + {c}) * DM_NCHW_H({t}) + {y}) * DM_NCHW_W({t}) + {x}] = {v}'

        text = re.sub(r'dm_tensor_set\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', repl_set, text)

        # Re-fix any .h that got incorrectly mapped to DM_NCHW_H if any. (But we used ->h, not \.h so include lines are safe)

    with open(filepath, 'w') as f:
        f.write(text)

print("All models fully refactored to DM_Block.")
