import re
import os

files_to_process = [
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

for filepath in files_to_process:
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, 'r') as f:
        text = f.read()

    # Replace dm_tensor_alloc
    text = re.sub(r'dm_tensor_alloc\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', replace_tensor_alloc, text)
    
    # Replace ->n, ->c, ->h, ->w
    text = re.sub(r'([a-zA-Z0-9_]+)->n', r'DM_NCHW_N(\1)', text)
    text = re.sub(r'([a-zA-Z0-9_]+)->c', r'DM_NCHW_C(\1)', text)
    text = re.sub(r'([a-zA-Z0-9_]+)->h', r'DM_NCHW_H(\1)', text)
    text = re.sub(r'([a-zA-Z0-9_]+)->w', r'DM_NCHW_W(\1)', text)

    # For image patchify, replace dm_tensor_set with direct array access
    # Wait, image_patchify has dm_tensor_get and set. It's safer to just replace them:
    def repl_get(m):
        t, n, c, y, x = m.groups()
        return f'((float*)({t})->data)[(((size_t){n} * DM_NCHW_C({t}) + {c}) * DM_NCHW_H({t}) + {y}) * DM_NCHW_W({t}) + {x}]'

    text = re.sub(r'dm_tensor_get\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', repl_get, text)

    def repl_set(m):
        t, n, c, y, x, v = m.groups()
        return f'((float*)({t})->data)[(((size_t){n} * DM_NCHW_C({t}) + {c}) * DM_NCHW_H({t}) + {y}) * DM_NCHW_W({t}) + {x}] = {v}'

    text = re.sub(r'dm_tensor_set\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', repl_set, text)

    # Some variables like `input->n` might be `model->n`. But `resnet.c`, `gan.c` don't use `->n` on models. Let's just run it.

    with open(filepath, 'w') as f:
        f.write(text)

print("Models refactored.")
