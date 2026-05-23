import re
import os

files_to_process = [
    'include/models/vision/mobilenet_tiny.h',
    'include/models/vision/resnet.h',
    'include/models/vision/vit.h',
    'include/models/gan.h',
    'include/models/vae.h',
    'include/encoding/image_patchify.h',
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
    
    # Replace dm_tensor_free
    text = re.sub(r'dm_tensor_free\(\s*([^)]+)\s*\)', r'dm_block_free(\1)', text)
    
    # Replace ->n, ->c, ->h, ->w
    text = re.sub(r'([a-zA-Z0-9_]+)->n', r'DM_NCHW_N(\1)', text)
    text = re.sub(r'([a-zA-Z0-9_]+)->c', r'DM_NCHW_C(\1)', text)
    text = re.sub(r'([a-zA-Z0-9_]+)->h', r'DM_NCHW_H(\1)', text)
    text = re.sub(r'([a-zA-Z0-9_]+)->w', r'DM_NCHW_W(\1)', text)

    # Replace .n, .c, .h, .w
    text = re.sub(r'([a-zA-Z0-9_]+)\.n', r'DM_NCHW_N(&\1)', text)
    text = re.sub(r'([a-zA-Z0-9_]+)\.c', r'DM_NCHW_C(&\1)', text)
    text = re.sub(r'([a-zA-Z0-9_]+)\.h', r'DM_NCHW_H(&\1)', text)
    text = re.sub(r'([a-zA-Z0-9_]+)\.w', r'DM_NCHW_W(&\1)', text)

    # Replace .data[idx] with ((float*)x.data)[idx]
    text = re.sub(r'([a-zA-Z0-9_]+)\.data\[(.*?)\]', r'((float*)\1.data)[\2]', text)
    text = re.sub(r'([a-zA-Z0-9_]+)->data\[(.*?)\]', r'((float*)\1->data)[\2]', text)

    # Replace DM_Tensor with DM_Block
    text = re.sub(r'\bDM_Tensor\b', 'DM_Block', text)

    # Also there might be dm_tensor_get and dm_tensor_set in patchify or models
    # We will replace them with manual macro or just let them stay if they use DM_Tensor alias? No, we replace them.
    # We can replace them via simple regex if needed, but let's see if it compiles.
    
    with open(filepath, 'w') as f:
        f.write(text)

print("Phase 2 python script completed.")
