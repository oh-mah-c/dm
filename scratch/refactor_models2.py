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

for filepath in files_to_process:
    if not os.path.exists(filepath):
        continue
        
    with open(filepath, 'r') as f:
        text = f.read()

    text = text.replace('DM_Tensor', 'DM_Block')
    text = text.replace('dm_tensor_free', 'dm_block_free')
    
    with open(filepath, 'w') as f:
        f.write(text)

print("Replaced DM_Tensor with DM_Block.")
