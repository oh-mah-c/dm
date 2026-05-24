import re

# Update dm_engine.h
with open('include/core/dm_engine.h', 'r') as f:
    h_content = f.read()

h_content = h_content.replace(
    'const float *w, const float *b',
    'const DM_Block *w, const DM_Block *b'
)

with open('include/core/dm_engine.h', 'w') as f:
    f.write(h_content)

# Update dm_engine.c
with open('src/core/dm_engine.c', 'r') as f:
    c_content = f.read()

# Replace the signatures
c_content = c_content.replace(
    'const float *w, const float *b',
    'const DM_Block *w, const DM_Block *b'
)

c_content = c_content.replace(
    'dm_matmul_nt(const float *A, const float *B',
    'dm_matmul_nt(const DM_Block *A, const DM_Block *B'
)

c_content = c_content.replace(
    'dm_matmul_nn(const float *A, const float *B',
    'dm_matmul_nn(const DM_Block *A, const DM_Block *B'
)

c_content = c_content.replace(
    'dm_linear_backward(const DM_Block *in, const DM_Block *grad_out,\n                         DM_Block *grad_in,\n                        float *grad_w, float *grad_b,\n                        const float *w, int out_c)',
    'dm_linear_backward(const DM_Block *in, const DM_Block *grad_out,\n                         DM_Block *grad_in,\n                        float *grad_w, float *grad_b,\n                        const DM_Block *w, int out_c)'
)

with open('src/core/dm_engine.c', 'w') as f:
    f.write(c_content)
