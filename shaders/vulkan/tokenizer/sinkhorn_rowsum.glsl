/*
 * sinkhorn_rowsum.glsl — Final Sinkhorn row-sum pass
 *
 * Computes row_sums[i] = u[i] * (K[i,:] @ v) after Sinkhorn convergence.
 * This is a copy of sinkhorn_spmv.glsl with mode hard-coded to 1 for clarity.
 *
 * Compile: glslc --target-env=vulkan1.1 -O sinkhorn_rowsum.glsl -o sinkhorn_rowsum.spv
 */

#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PC {
    uint n_rows;
    uint _pad;
} pc;

layout(set = 0, binding = 0) readonly buffer RowPtr  { uint  row_ptr[]; };
layout(set = 0, binding = 1) readonly buffer ColIdx  { uint  col_idx[]; };
layout(set = 0, binding = 2) readonly buffer Vals    { float vals[];    };
layout(set = 0, binding = 3) readonly buffer V       { float v[];       };
layout(set = 0, binding = 4) readonly buffer U       { float u[];       };
layout(set = 0, binding = 5)          buffer RowSums { float row_sums[];};

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.n_rows) return;

    uint start = row_ptr[row];
    uint end   = row_ptr[row + 1u];

    float dot = 0.0;
    for (uint k = start; k < end; k++) {
        dot += vals[k] * v[col_idx[k]];
    }
    row_sums[row] = u[row] * dot;
}
