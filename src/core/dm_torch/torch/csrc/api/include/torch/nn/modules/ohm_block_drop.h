#pragma once

#include <torch/nn/module.h>
#include <torch/types.h>

#include <cstddef>

namespace torch::nn {

struct TORCH_API OhmBlockDropImpl : torch::nn::Module {
    double drop_prob_;
    int64_t block_size_;
    int64_t dim_;

    OhmBlockDropImpl(double drop_prob, int64_t block_size, int64_t dim = 1);

    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(OhmBlockDrop);

} // namespace torch::nn
