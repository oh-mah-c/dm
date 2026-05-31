#pragma once

#include <torch/nn/module.h>
#include <torch/types.h>

#include <cstddef>
#include <algorithm>

namespace torch::nn {

struct TORCH_API OhmRingKVImpl : torch::nn::Module {
    torch::Tensor cache_tensor;
    int64_t head_ptr;
    int64_t max_seq_len_;

    OhmRingKVImpl(int64_t batch_size, int64_t max_seq_len, int64_t num_heads, int64_t head_dim, torch::Device device = torch::kCPU);

    torch::Tensor forward(const torch::Tensor& x_t);
};
TORCH_MODULE(OhmRingKV);

} // namespace torch::nn
