import re

with open('src/core/dm_engine.c', 'r') as f:
    text = f.read()

# 1. Replace dm_to_tf and tf_to_dm with dm_lower_block / dm_raise_block
# This requires manual rewriting.
# I will first replace the signature DM_Tensor * -> DM_Block *
text = re.sub(r'dm_conv2d_same\s*\(\s*const\s+DM_Tensor\s+\*in,\s*DM_Tensor\s+\*out,', 'dm_conv2d_same(const DM_Block *in, DM_Block *out,', text)
text = re.sub(r'dm_depthwise_conv2d_same\s*\(\s*const\s+DM_Tensor\s+\*in,\s*DM_Tensor\s+\*out,', 'dm_depthwise_conv2d_same(const DM_Block *in, DM_Block *out,', text)
text = re.sub(r'dm_pointwise_conv2d\s*\(\s*const\s+DM_Tensor\s+\*in,\s*DM_Tensor\s+\*out,', 'dm_pointwise_conv2d(const DM_Block *in, DM_Block *out,', text)

text = re.sub(r'dm_linear\s*\(\s*const\s+DM_Tensor\s+\*in,\s*DM_Tensor\s+\*out,', 'dm_linear(const DM_Block *in, DM_Block *out,', text)

text = re.sub(r'dm_max_pool2d_same\s*\(\s*const\s+DM_Tensor\s+\*in,\s*DM_Tensor\s+\*out,', 'dm_max_pool2d_same(const DM_Block *in, DM_Block *out,', text)
text = re.sub(r'dm_global_avg_pool\s*\(\s*const\s+DM_Tensor\s+\*in,\s*DM_Tensor\s+\*out\)', 'dm_global_avg_pool(const DM_Block *in, DM_Block *out)', text)

text = re.sub(r'dm_batch_norm\s*\(\s*DM_Tensor\s+\*t,', 'dm_batch_norm(DM_Block *t,', text)
text = re.sub(r'dm_tensor_add\s*\(\s*DM_Tensor\s+\*out,\s*const\s+DM_Tensor\s+\*in\)', 'dm_tensor_add(DM_Block *out, const DM_Block *in)', text)

text = re.sub(r'dm_relu6\s*\(\s*DM_Tensor\s+\*t\)', 'dm_relu6(DM_Block *t)', text)
text = re.sub(r'dm_relu\s*\(\s*DM_Tensor\s+\*t\)', 'dm_relu(DM_Block *t)', text)
text = re.sub(r'dm_tanh_inplace\s*\(\s*DM_Tensor\s+\*t\)', 'dm_tanh_inplace(DM_Block *t)', text)
text = re.sub(r'dm_sigmoid_inplace\s*\(\s*DM_Tensor\s+\*t\)', 'dm_sigmoid_inplace(DM_Block *t)', text)
text = re.sub(r'dm_softmax\s*\(\s*DM_Tensor\s+\*t\)', 'dm_softmax(DM_Block *t)', text)

# For backward passes:
text = re.sub(r'dm_linear_backward\s*\(\s*const\s+DM_Tensor\s+\*in,\s*const\s+DM_Tensor\s+\*grad_out,\s*DM_Tensor\s+\*grad_in,', 'dm_linear_backward(const DM_Block *in, const DM_Block *grad_out,\n                         DM_Block *grad_in,', text)
text = re.sub(r'dm_tanh_backward\s*\(\s*const\s+DM_Tensor\s+\*out,\s*const\s+DM_Tensor\s+\*grad_out,\s*DM_Tensor\s+\*grad_in\)', 'dm_tanh_backward(const DM_Block *out, const DM_Block *grad_out,\n                         DM_Block *grad_in)', text)
text = re.sub(r'dm_relu_backward\s*\(\s*const\s+DM_Tensor\s+\*in,\s*const\s+DM_Tensor\s+\*grad_out,\s*DM_Tensor\s+\*grad_in\)', 'dm_relu_backward(const DM_Block *in, const DM_Block *grad_out,\n                         DM_Block *grad_in)', text)

text = re.sub(r'dm_maxout\s*\(\s*const\s+DM_Tensor\s+\*in,\s*DM_Tensor\s+\*out,', 'dm_maxout(const DM_Block *in,  DM_Block *out,', text)
text = re.sub(r'dm_maxout_backward\s*\(\s*const\s+DM_Tensor\s+\*grad_out,\s*DM_Tensor\s+\*grad_in,', 'dm_maxout_backward(const DM_Block *grad_out, DM_Block *grad_in,', text)

text = re.sub(r'dm_dropout\s*\(\s*const\s+DM_Tensor\s+\*in,\s*DM_Tensor\s+\*out,', 'dm_dropout(const DM_Block *in,  DM_Block *out,', text)
text = re.sub(r'dm_dropout_backward\s*\(\s*const\s+DM_Tensor\s+\*grad_out,\s*DM_Tensor\s+\*grad_in,', 'dm_dropout_backward(const DM_Block *grad_out, DM_Block *grad_in,', text)

with open('src/core/dm_engine.c', 'w') as f:
    f.write(text)

print("Signatures updated.")
