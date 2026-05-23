import re

with open('src/core/dm_engine.c', 'r') as f:
    text = f.read()

# Add layout validation to the beginning of dm_conv2d_same
conv2d = r'(int dm_conv2d_same\([^{]*\{)'
replacement = r'\1\n    if (!dm_block_is_nchw4(in) || !dm_block_is_nchw4(out)) return DM_ERR_INCOMPATIBLE;\n'
text = re.sub(conv2d, replacement, text)

# Add layout validation to dm_depthwise_conv2d_same
dw_conv2d = r'(int dm_depthwise_conv2d_same\([^{]*\{)'
replacement = r'\1\n    if (!dm_block_is_nchw4(in) || !dm_block_is_nchw4(out)) return DM_ERR_INCOMPATIBLE;\n'
text = re.sub(dw_conv2d, replacement, text)

with open('src/core/dm_engine.c', 'w') as f:
    f.write(text)

print("Validations injected.")
