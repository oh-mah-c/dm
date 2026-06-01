#include <torch/nn/modules/ohm_block_drop.h>

namespace torch::nn {

OhmBlockDropImpl::OhmBlockDropImpl(double drop_prob, int64_t block_size, int64_t dim)
    : drop_prob_(drop_prob), block_size_(block_size), dim_(dim) {}

torch::Tensor OhmBlockDropImpl::forward(const torch::Tensor& x) {
    // Forward Pass (Inference): Simply return x
    if (!is_training()) {
        return x;
    }
    
    // Forward Pass (Training): Uniform random number r
    double r = torch::rand({1}, torch::TensorOptions().dtype(torch::kFloat32)).item<float>();
    if (r > drop_prob_) {
        return x;
    }

    int64_t L = x.size(dim_);
    if (L <= block_size_) {
        return torch::zeros_like(x);
    }

    // Generate random starting index idx
    int64_t idx = torch::randint(0, L - block_size_ + 1, {1}).item<int64_t>();
    
    // Clone to protect autograd graph from in-place view modification of inputs
    auto out = x.clone();
    
    // Contiguous Erasure in-place on the output slice
    out.slice(dim_, idx, idx + block_size_).zero_();
    
    return out;
}

} // namespace torch::nn
