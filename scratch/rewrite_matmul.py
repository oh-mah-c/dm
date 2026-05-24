import re

with open("src/core/dm_engine.c", "r") as f: text = f.read()

# Replace dm_matmul_nt
old_nt = """void dm_matmul_nt(const DM_Block *A, const DM_Block *B, float *C, int M, int N, int K) {
    DM_Block tf_A, tf_B;
    if (dm_lower_block(A, &tf_A, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(B, &tf_B, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return;
    TFE_TensorHandle *hA = (TFE_TensorHandle *)tf_A.handle;
    TFE_TensorHandle *hB = (TFE_TensorHandle *)tf_B.handle;
    if (!hA || !hB) {
        if (hA) TFE_DeleteTensorHandle(hA);
        if (hB) TFE_DeleteTensorHandle(hB);
        return;
    }
    TFE_TensorHandle *res = execute_tf_matmul(hA, hB, false, true);
    if (res) { tf_to_raw(res, C); TFE_DeleteTensorHandle(res); }
    TFE_DeleteTensorHandle(hA);
    TFE_DeleteTensorHandle(hB);
}"""

new_nt = """void dm_matmul_nt(const DM_Block *A, const DM_Block *B, DM_Block *C) {
    if (!A || !B || !C) return;
    DM_Block tf_A, tf_B;
    if (dm_lower_block(A, &tf_A, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(B, &tf_B, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return;
    TFE_TensorHandle *hA = (TFE_TensorHandle *)tf_A.handle;
    TFE_TensorHandle *hB = (TFE_TensorHandle *)tf_B.handle;
    if (!hA || !hB) return;
    TFE_TensorHandle *res = execute_tf_matmul(hA, hB, false, true);
    if (res) { tf_to_dm(res, C); TFE_DeleteTensorHandle(res); }
}"""

old_nn = """void dm_matmul_nn(const DM_Block *A, const DM_Block *B, float *C, int M, int K, int N) {
    DM_Block tf_A, tf_B;
    if (dm_lower_block(A, &tf_A, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(B, &tf_B, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return;
    TFE_TensorHandle *hA = (TFE_TensorHandle *)tf_A.handle;
    TFE_TensorHandle *hB = (TFE_TensorHandle *)tf_B.handle;
    if (!hA || !hB) {
        if (hA) TFE_DeleteTensorHandle(hA);
        if (hB) TFE_DeleteTensorHandle(hB);
        return;
    }
    TFE_TensorHandle *res = execute_tf_matmul(hA, hB, false, false);
    if (res) { tf_to_raw(res, C); TFE_DeleteTensorHandle(res); }
    TFE_DeleteTensorHandle(hA);
    TFE_DeleteTensorHandle(hB);
}"""

new_nn = """void dm_matmul_nn(const DM_Block *A, const DM_Block *B, DM_Block *C) {
    if (!A || !B || !C) return;
    DM_Block tf_A, tf_B;
    if (dm_lower_block(A, &tf_A, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(B, &tf_B, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return;
    TFE_TensorHandle *hA = (TFE_TensorHandle *)tf_A.handle;
    TFE_TensorHandle *hB = (TFE_TensorHandle *)tf_B.handle;
    if (!hA || !hB) return;
    TFE_TensorHandle *res = execute_tf_matmul(hA, hB, false, false);
    if (res) { tf_to_dm(res, C); TFE_DeleteTensorHandle(res); }
}"""

text = text.replace(old_nt, new_nt)
text = text.replace(old_nn, new_nn)

with open("src/core/dm_engine.c", "w") as f: f.write(text)
