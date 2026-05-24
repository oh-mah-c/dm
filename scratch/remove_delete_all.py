import re

with open("src/core/dm_engine.c", "r") as f: text = f.read()

lines = text.split("\n")
for i, line in enumerate(lines):
    if "TFE_DeleteTensorHandle" in line:
        # Check if the variable being deleted was created by dm_to_tf
        # For dm_tensor_add: h_out and h_in are from dm_to_tf
        # Wait, I already removed it in dm_tensor_add, but it failed compilation earlier, let's make sure
        pass

# I'll just use regex for the known functions:
text = re.sub(r'TFE_DeleteTensorHandle\(in_h\);', r'/* TFE_DeleteTensorHandle(in_h); */', text)
# dm_tensor_add:
text = re.sub(r'TFE_DeleteTensorHandle\(h_in\);', r'/* TFE_DeleteTensorHandle(h_in); */', text)
text = re.sub(r'TFE_DeleteTensorHandle\(h_out\);(.*?\n.*?return 0;.*?dm_tensor_add)', r'/* TFE_DeleteTensorHandle(h_out); */\1', text, flags=re.DOTALL)
# wait, h_out is used in dm_conv2d_same, dm_linear, dm_max_pool2d_same...
# BUT in dm_conv2d_same, h_out is returned by execute_tf_transpose! So it MUST be deleted!
# In dm_linear, h_out is returned by execute_tf_matmul. So it MUST be deleted!
# Only in dm_tensor_add, h_out is dm_to_tf(out), so it MUST NOT be deleted!

with open("src/core/dm_engine.c", "w") as f: f.write(text)

