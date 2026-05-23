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

def repl_alloc(m):
    t, n, c, h, w = m.groups()
    return f'dm_block_create({t}, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){{{n}, {c}, {h}, {w}}})'

for filepath in h_files + c_files:
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, 'r') as f:
        lines = f.readlines()

    out_lines = []
    for line in lines:
        if line.strip().startswith('#include'):
            out_lines.append(line)
            continue

        text = line
        text = text.replace('DM_Tensor', 'DM_Block')
        text = text.replace('dm_tensor_free', 'dm_block_free')

        text = re.sub(r'dm_tensor_count\((.*?)\)', r'(\1)->count', text)
        
        text = re.sub(r'dm_tensor_fill\(([^,]+),\s*([^)]+)\);', r'{ size_t __n = (\1)->count; float *__d = (float*)(\1)->data; for(size_t __i=0; __i<__n; __i++) __d[__i] = \2; }', text)

        text = re.sub(r'dm_tensor_alloc\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)', repl_alloc, text)
        
        if 'image_patchify.c' in filepath:
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

        # Skip replacement on anything called head or h or cfg or model
        def check_valid(m, dim):
            var = m.group(1)
            if var in ['head', 'h', 'cfg', 'model', 'opt', 'params', 'ctx']:
                return m.group(0) # don't replace
            
            if m.group(0).endswith('->'+dim):
                return f'DM_NCHW_{dim.upper()}({var})'
            else:
                return f'DM_NCHW_{dim.upper()}(&{var})'

        text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)->n\b', lambda m: check_valid(m, 'n'), text)
        text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)->c\b', lambda m: check_valid(m, 'c'), text)
        text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)->h\b', lambda m: check_valid(m, 'h'), text)
        text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)->w\b', lambda m: check_valid(m, 'w'), text)

        text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)\.n\b', lambda m: check_valid(m, 'n'), text)
        text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)\.c\b', lambda m: check_valid(m, 'c'), text)
        text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)\.h\b', lambda m: check_valid(m, 'h'), text)
        text = re.sub(r'(?<![a-zA-Z0-9_])([a-zA-Z0-9_]+)\.w\b', lambda m: check_valid(m, 'w'), text)

        text = re.sub(r'([a-zA-Z0-9_]+)\.data\[', r'((float*)\1.data)[', text)
        text = re.sub(r'([a-zA-Z0-9_]+)->data\[', r'((float*)\1->data)[', text)

        # For assignments without brackets like: float *p = d_out.data;
        text = re.sub(r'([a-zA-Z0-9_]+)\.data(?!\[)', r'((float*)\1.data)', text)
        text = re.sub(r'([a-zA-Z0-9_]+)->data(?!\[)', r'((float*)\1->data)', text)

        out_lines.append(text)

    with open(filepath, 'w') as f:
        f.writelines(out_lines)

print("Safely refactored specific blocks with blacklist!")
