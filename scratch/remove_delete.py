import re

with open("src/core/dm_engine.c", "r") as f: text = f.read()

# We need to remove TFE_DeleteTensorHandle(h_in) and h_out if they came from dm_to_tf.
# The safest way is to regex out TFE_DeleteTensorHandle(xxx) where xxx = h_in, in_h, h_out (when it came from dm_to_tf), h_x, etc.
# Wait, let's just do it explicitly for the ones we found!

lines = text.split("\n")
for i, line in enumerate(lines):
    if "TFE_DeleteTensorHandle(in_h);" in line:
        lines[i] = line.replace("TFE_DeleteTensorHandle(in_h);", "// TFE_DeleteTensorHandle(in_h);")
    elif "TFE_DeleteTensorHandle(h_in);" in line:
        lines[i] = line.replace("TFE_DeleteTensorHandle(h_in);", "// TFE_DeleteTensorHandle(h_in);")
    elif "TFE_DeleteTensorHandle(h_x);" in line:
        lines[i] = line.replace("TFE_DeleteTensorHandle(h_x);", "// TFE_DeleteTensorHandle(h_x);")
    elif "TFE_DeleteTensorHandle(h_out);" in line and "TFE_TensorHandle *h_out = dm_to_tf(out);" in "\n".join(lines[max(0,i-20):i]):
        # Check if h_out was created via dm_to_tf or op
        # In dm_tensor_add, h_out is dm_to_tf(out)
        lines[i] = line.replace("TFE_DeleteTensorHandle(h_out);", "// TFE_DeleteTensorHandle(h_out);")

with open("src/core/dm_engine.c", "w") as f: f.write("\n".join(lines))
