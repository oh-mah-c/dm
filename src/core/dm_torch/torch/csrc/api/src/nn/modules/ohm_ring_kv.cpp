#include <torch/nn/modules/ohm_ring_kv.h>

namespace torch::nn {

OhmRingKVImpl::OhmRingKVImpl(int64_t batch_size, int64_t max_seq_len, int64_t num_heads, int64_t head_dim, torch::Device device) {
    max_seq_len_ = max_seq_len;
    head_ptr = 0;
    // Pre-allocate zeroed tensor mapped to device
    cache_tensor = register_buffer("cache_tensor", torch::zeros({batch_size, max_seq_len, num_heads, head_dim}, torch::TensorOptions().device(device)));
}

torch::Tensor OhmRingKVImpl::forward(const torch::Tensor& x_t) {
    // Calculate insertion index using modulo arithmetic
    int64_t i = head_ptr % max_seq_len_;
    
    // In-place memory write (strictly prevents reallocation)
    cache_tensor.slice(1, i, i + 1).copy_(x_t);
    
    head_ptr++;
    
    // Return valid window of the cache
    int64_t valid_len = std::min(head_ptr, max_seq_len_);
    return cache_tensor.slice(1, 0, valid_len);
}

} // namespace torch::nn
