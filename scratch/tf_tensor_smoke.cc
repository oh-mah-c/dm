#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

#include <iostream>

int main() {
    tensorflow::Tensor x(tensorflow::DT_FLOAT, tensorflow::TensorShape({2, 3}));
    auto flat = x.flat<float>();
    for (int i = 0; i < flat.size(); ++i) {
        flat(i) = static_cast<float>(i);
    }
    std::cout << x.shape().DebugString() << " " << flat(5) << "\n";
    return 0;
}
