#pragma once

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

namespace torch {
namespace nn {

// Cấu hình siêu tham số cho OhmQPA
struct TORCH_API OhmQPAOptions {
    OhmQPAOptions(int64_t hidden_dim);
    TORCH_ARG(int64_t, hidden_dim);
};

// OhmQPA: Quantum Parametric Attention Scoring Function
// Thay thế cho hàm Scaled Dot-Product của Transformer.
class TORCH_API OhmQPAImpl : public torch::nn::Module {
public:
    OhmQPAImpl(const OhmQPAOptions& options_);

    void reset();
    void reset_parameters();

    // Forward pass: Tính toán xác suất lượng tử P(|00>) + P(|11>) làm điểm Attention
    // query: [Batch, Heads, SeqLen_Q, HeadDim]
    // key:   [Batch, Heads, SeqLen_K, HeadDim]
    // return: Attention Scores [Batch, Heads, SeqLen_Q, SeqLen_K]
    torch::Tensor forward(const torch::Tensor& query, const torch::Tensor& key);

    OhmQPAOptions options;

    // Trainable Quantum Parameters
    torch::Tensor lambda1;
    torch::Tensor lambda2;
    torch::Tensor alpha;
    torch::Tensor beta;
};

TORCH_MODULE(OhmQPA);

} // namespace nn
} // namespace torch
