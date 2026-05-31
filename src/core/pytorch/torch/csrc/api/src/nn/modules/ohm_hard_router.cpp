#include <torch/nn/modules/ohm_hard_router.h>

namespace torch::nn {

OhmHardRouterImpl::OhmHardRouterImpl(int64_t input_dim, int64_t num_experts, const std::vector<torch::nn::AnyModule>& experts) 
    : num_experts_(num_experts) {
    gate = register_module("gate", torch::nn::Linear(input_dim, num_experts));
    TORCH_CHECK(experts.size() == static_cast<size_t>(num_experts), "Mismatch between num_experts and experts vector size");
    for (int64_t i = 0; i < num_experts; ++i) {
        experts_.push_back(experts[i]);
        register_module("expert_" + std::to_string(i), experts_.back().ptr());
    }
}

torch::Tensor OhmHardRouterImpl::forward(const torch::Tensor& x) {
    // Gate calculation
    auto z = gate->forward(x);
    
    // Calculate the absolute winner index i^*
    // We ensure robust scalar extraction for batched z logits
    int64_t i_star = z.view({-1, num_experts_}).mean(0).argmax(-1).item<int64_t>();
    
    // C++ native control flow to execute ONLY the chosen expert module
    return experts_[i_star].forward(x);
}

} // namespace torch::nn
