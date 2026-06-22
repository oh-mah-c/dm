#pragma once
#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <ggml.h>
#include <gguf.h>

namespace ohm {
namespace nlp {

class GGUFLoader {
public:
    GGUFLoader(const std::string& filepath);
    ~GGUFLoader();

    // Load and dequantize all tensors into a dictionary of PyTorch tensors
    std::unordered_map<std::string, at::Tensor> load();

private:
    std::string filepath_;
    struct ggml_context* ctx_ggml_ = nullptr;
    struct gguf_context* ctx_gguf_ = nullptr;
};

} // namespace nlp
} // namespace ohm
