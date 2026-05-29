#pragma once

#include <torch/torch.h>
#include "algorithms/ohm_feynman_kernel.h"

namespace dm {
namespace modules {

struct OhmFEYNMANImpl : torch::nn::Module {
    int num_paths_;
    int steps_per_path_;

    OhmFEYNMANImpl(int num_paths = 1000000, int steps_per_path = 100) 
        : num_paths_(num_paths), steps_per_path_(steps_per_path) {}

    // Computes the probability amplitude from start_x to end_x
    // Returns a tensor of shape [2] containing {Real, Imag}
    torch::Tensor forward(float start_x, float end_x) {
        float prob_real = 0.0f;
        float prob_imag = 0.0f;

        // Bỏ qua Autograd Dispatcher, chạy thẳng xuống CPU thuần!
        dm::algorithm::feynman_path_integral_cpu(
            num_paths_, steps_per_path_, 
            start_x, end_x, 
            &prob_real, &prob_imag
        );

        return torch::tensor({prob_real, prob_imag}, torch::kFloat32);
    }
};

TORCH_MODULE(OhmFEYNMAN);

} // namespace modules
} // namespace dm
