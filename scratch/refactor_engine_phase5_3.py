import re

with open('src/core/dm_engine.c', 'r') as f:
    content = f.read()

# For depthwise_conv2d
content = re.sub(
    r'TFE_DeleteTensorHandle\(h_b\);\s*TFE_DeleteTensorHandle\(h_out\);\s*h_out = h_out2;\s*}\s*if \(h_out\) \{\s*tf_to_dm\(h_out, out\);\s*TFE_DeleteTensorHandle\(h_out\);\s*}\s*TFE_DeleteTensorHandle\(h_in\);\s*TFE_DeleteTensorHandle\(h_w\);\s*return 0;\s*dw_err:\s*if \(h_in\) TFE_DeleteTensorHandle\(h_in\);\s*if \(h_w\)  TFE_DeleteTensorHandle\(h_w\);\s*return -1;',
    r'''TFE_DeleteTensorHandle(h_out);
        h_out = h_out2;
    }
    if (h_out) {
        tf_to_dm(h_out, out);
        TFE_DeleteTensorHandle(h_out);
    }
    return 0;

dw_err:
    return -1;''',
    content
)

# For pointwise_conv2d
content = re.sub(
    r'TFE_DeleteTensorHandle\(h_b\);\s*TFE_DeleteTensorHandle\(h_out\);\s*h_out = h_out2;\s*}\s*if \(h_out\) \{\s*tf_to_dm\(h_out, out\);\s*TFE_DeleteTensorHandle\(h_out\);\s*}\s*TFE_DeleteTensorHandle\(h_in\);\s*TFE_DeleteTensorHandle\(h_w\);\s*return 0;\s*pw_err:\s*if \(h_in\) TFE_DeleteTensorHandle\(h_in\);\s*if \(h_w\)  TFE_DeleteTensorHandle\(h_w\);\s*return -1;',
    r'''TFE_DeleteTensorHandle(h_out);
        h_out = h_out2;
    }
    if (h_out) {
        tf_to_dm(h_out, out);
        TFE_DeleteTensorHandle(h_out);
    }
    return 0;

pw_err:
    return -1;''',
    content
)

# For dm_linear
content = re.sub(
    r'TFE_DeleteTensorHandle\(h_b\);\s*TFE_DeleteTensorHandle\(h_out\);\s*h_out = h_out2;\s*}\s*if \(h_out\) \{\s*tf_to_dm\(h_out, out\);\s*TFE_DeleteTensorHandle\(h_out\);\s*}\s*TFE_DeleteTensorHandle\(h_in\);\s*TFE_DeleteTensorHandle\(h_w\);\s*return 0;\s*ln_err:\s*if \(h_in\) TFE_DeleteTensorHandle\(h_in\);\s*if \(h_w\)  TFE_DeleteTensorHandle\(h_w\);\s*return -1;',
    r'''TFE_DeleteTensorHandle(h_out);
        h_out = h_out2;
    }
    if (h_out) {
        tf_to_dm(h_out, out);
        TFE_DeleteTensorHandle(h_out);
    }
    return 0;

ln_err:
    return -1;''',
    content
)

with open('src/core/dm_engine.c', 'w') as f:
    f.write(content)
