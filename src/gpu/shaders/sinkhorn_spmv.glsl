/*
 * sinkhorn_spmv.glsl — Sparse matrix-vector product for Sinkhorn iterations
 *
 * mode 0 (u/v update):  vec_out[i] = p[i] / (K[i,:] @ vec_in)
 * mode 1 (row sums):    vec_out[i] = p[i] * (K[i,:] @ vec_in)
 *
 * Used twice per Sinkhorn iteration:
 *   K  CSR (n_tok  x n_char): computes u = p_tok  / (K   @ v)
 *   Kt CSR (n_char x n_tok) : computes v = p_char / (K^T @ u)
 * And once at the end for row_sums (mode 1, p = u).
 *
 * Compile: glslc --target-env=vulkan1.1 -O sinkhorn_spmv.glsl -o sinkhorn_spmv.spv
 */

#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PC {
    uint n_rows; /* rows in THIS matrix (n_tok or n_char) */
    uint mode;   /* 0 = divide, 1 = multiply              */
} pc;

layout(set = 0, binding = 0) readonly buffer RowPtr { uint row_ptr[]; };
layout(set = 0, binding = 1) readonly buffer ColIdx { uint col_idx[]; };
layout(set = 0, binding = 2) readonly buffer Vals   { float vals[];   };
layout(set = 0, binding = 3) readonly buffer VecIn  { float vec_in[]; };
layout(set = 0, binding = 4) readonly buffer P      { float p[];      };
layout(set = 0, binding = 5)          buffer VecOut { float vec_out[];};

void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= pc.n_rows) return;

    uint start = row_ptr[row];
    uint end   = row_ptr[row + 1u];

    float dot = 0.0;
    for (uint k = start; k < end; k++) {
        dot += vals[k] * vec_in[col_idx[k]];
    }

    if (pc.mode == 0u) {
        /* u/v update: out = p / dot */
        vec_out[row] = (dot > 1e-30) ? p[row] / dot : 0.0;
    } else {
        /* row-sums: out = p (=u) * dot */
        vec_out[row] = p[row] * dot;
    }
}
