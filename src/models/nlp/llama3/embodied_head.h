#pragma once
#include <torch/torch.h>

namespace ohm {
namespace nlp {

// StateEncoder maps continuous MuJoCo states [nq] -> Llama3 hidden dimension [dim]
struct StateEncoderImpl : torch::nn::Module {
    torch::nn::Linear layer1{nullptr};
    torch::nn::Linear layer2{nullptr};
    
    StateEncoderImpl(int64_t state_dim, int64_t hidden_dim) {
        layer1 = register_module("layer1", torch::nn::Linear(state_dim, 256));
        layer2 = register_module("layer2", torch::nn::Linear(256, hidden_dim));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(layer1->forward(x));
        return layer2->forward(x);
    }
};
TORCH_MODULE(StateEncoder);

// ActionHead maps Llama3 hidden dimension [dim] -> MuJoCo controls [nu]
struct ActionHeadImpl : torch::nn::Module {
    torch::nn::Linear layer1{nullptr};
    torch::nn::Linear layer2{nullptr};
    
    ActionHeadImpl(int64_t hidden_dim, int64_t action_dim) {
        layer1 = register_module("layer1", torch::nn::Linear(hidden_dim, 256));
        layer2 = register_module("layer2", torch::nn::Linear(256, action_dim));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(layer1->forward(x));
        // Scale to [-1, 1] range which is typical for MuJoCo motor controllers
        return torch::tanh(layer2->forward(x)); 
    }
};
TORCH_MODULE(ActionHead);

} // namespace nlp
} // namespace ohm
