#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/container/any.h>
#include <torch/types.h>

#include <cstddef>
#include <vector>
#include <string>

namespace torch::nn {

struct TORCH_API OhmHardRouterImpl : torch::nn::Module {
    torch::nn::Linear gate{nullptr};
    std::vector<torch::nn::AnyModule> experts_;
    int64_t num_experts_;

    OhmHardRouterImpl(int64_t input_dim, int64_t num_experts, const std::vector<torch::nn::AnyModule>& experts);

    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(OhmHardRouter);

} // namespace torch::nn
