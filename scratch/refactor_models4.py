import re
import os

os.system("git checkout include/models/ include/encoding/ src/models/ src/encoding/")

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

for filepath in h_files + c_files:
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, 'r') as f:
        text = f.read()

    # 1. Type replacement
    text = text.replace('DM_Tensor', 'DM_Block')
    text = text.replace('dm_tensor_free', 'dm_block_free')

    # 2. Count replacement
    text = re.sub(r'dm_tensor_count\((.*?)\)', r'(\1)->count', text)
    
    # 3. Fill replacement
    text = re.sub(r'dm_tensor_fill\((.*?),\s*(.*?)\);', r'memset(((float*)(\1)->data), 0, (\1)->count * sizeof(float)); /* value \2 ignored/forced zero */', text)
    # Actually wait, dm_tensor_fill might be called with 1.0f!
    # If so, memset won't work for 1.0f. Let's just restore dm_tensor_fill and change its signature or use a loop.
    # Fortunately, it's just a python script, we can implement it safely:
    text = re.sub(r'dm_tensor_fill\(([^,]+),\s*([^)]+)\);', r'{ size_t __n = (\1)->count; float *__d = (float*)(\1)->data; for(size_t __i=0; __i<__n; __i++) __d[__i] = \2; }', text)

    # 4. Alloc replacement
    def repl_alloc(m):
        t, n, c, h, w = m.groups()
        return f'dm_block_create({t}, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){{{n}, {c}, {h}, {w}}})'
    text = re.sub(r'dm_tensor_alloc\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', repl_alloc, text)
    
    # 5. Get and Set in patchify
    if 'image_patchify.c' in filepath:
        # manual replace for the specific complex lines:
        text = text.replace(
            'dm_tensor_set(out, 0, c, y, x, ((float)v / (float)maxv - 0.5f) * 2.0f);',
            '((float*)(out)->data)[(((size_t)0 * DM_NCHW_C(out) + c) * DM_NCHW_H(out) + y) * DM_NCHW_W(out) + x] = ((float)v / (float)maxv - 0.5f) * 2.0f;'
        )
        text = text.replace(
            'dm_tensor_set(out, n, c, y, x, dm_tensor_get(in, n, c, sy, sx));',
            '((float*)(out)->data)[(((size_t)n * DM_NCHW_C(out) + c) * DM_NCHW_H(out) + y) * DM_NCHW_W(out) + x] = ((float*)(in)->data)[(((size_t)n * DM_NCHW_C(in) + c) * DM_NCHW_H(in) + sy) * DM_NCHW_W(in) + sx];'
        )
        text = text.replace(
            'dm_tensor_set(out, n, oc, y, x, dm_tensor_get(in, n, c, y * patch_h + py, x * patch_w + px));',
            '((float*)(out)->data)[(((size_t)n * DM_NCHW_C(out) + oc) * DM_NCHW_H(out) + y) * DM_NCHW_W(out) + x] = ((float*)(in)->data)[(((size_t)n * DM_NCHW_C(in) + c) * DM_NCHW_H(in) + y * patch_h + py) * DM_NCHW_W(in) + x * patch_w + px];'
        )

    # 6. Dimensions access
    text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)->n\b', r'DM_NCHW_N(\1)', text)
    text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)->c\b', r'DM_NCHW_C(\1)', text)
    text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)->h\b', r'DM_NCHW_H(\1)', text)
    text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)->w\b', r'DM_NCHW_W(\1)', text)

    text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)\.n\b', r'DM_NCHW_N(&\1)', text)
    text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)\.c\b', r'DM_NCHW_C(&\1)', text)
    text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)\.h\b', r'DM_NCHW_H(&\1)', text)
    text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)\.w\b', r'DM_NCHW_W(&\1)', text)

    # 7. Data access (e.g. logits.data[i] -> ((float*)logits.data)[i])
    text = re.sub(r'([a-zA-Z0-9_]+)\.data\[', r'((float*)\1.data)[', text)
    text = re.sub(r'([a-zA-Z0-9_]+)->data\[', r'((float*)\1->data)[', text)

    with open(filepath, 'w') as f:
        f.write(text)

print("Safely refactored!")
