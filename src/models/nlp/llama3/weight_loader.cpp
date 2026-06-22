#include "weight_loader.h"
#include <iostream>
#include <stdexcept>
#include <vector>

// ggml-quants for dequantization functions
#include "../../../core/dm_ggml/src/ggml-quants.h"

namespace ohm {
namespace nlp {

GGUFLoader::GGUFLoader(const std::string& filepath) : filepath_(filepath) {
    struct gguf_init_params params = {
        .no_alloc = false,
        .ctx      = &ctx_ggml_,
    };

    ctx_gguf_ = gguf_init_from_file(filepath.c_str(), params);
    if (!ctx_gguf_) {
        throw std::runtime_error("Failed to load GGUF file: " + filepath);
    }
    std::cout << "[GGUFLoader] Successfully parsed GGUF: " << filepath << "\n";
}

GGUFLoader::~GGUFLoader() {
    if (ctx_gguf_) {
        gguf_free(ctx_gguf_);
    }
    if (ctx_ggml_) {
        ggml_free(ctx_ggml_);
    }
}

std::unordered_map<std::string, at::Tensor> GGUFLoader::load() {
    std::unordered_map<std::string, at::Tensor> tensors;
    
    // ggml context contains the allocated tensors from the GGUF file
    struct ggml_tensor* t = ggml_get_first_tensor(ctx_ggml_);
    
    while (t != nullptr) {
        std::string name = t->name;
        
        // ggml dimensions are reversed compared to PyTorch
        // ggml: [width, height, channels, batches] -> PyTorch: [batches, channels, height, width]
        std::vector<int64_t> shape;
        int n_dims = ggml_n_dims(t);
        for (int i = n_dims - 1; i >= 0; --i) {
            shape.push_back(t->ne[i]);
        }
        
        size_t n_elements = ggml_nelements(t);
        
        if (t->type == GGML_TYPE_F32) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            at::Tensor tensor = torch::from_blob(t->data, shape, options).clone();
            tensors[name] = tensor;
        } 
        else if (t->type == GGML_TYPE_F16) {
            auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
            at::Tensor tensor = torch::from_blob(t->data, shape, options).clone();
            tensors[name] = tensor;
        }
        else if (t->type == GGML_TYPE_Q4_0) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            at::Tensor temp = torch::empty(shape, options);
            const block_q4_0* src = (const block_q4_0*) t->data;
            dequantize_row_q4_0(src, temp.data_ptr<float>(), n_elements);
            tensors[name] = temp.to(torch::kFloat16);
        }
        else if (t->type == GGML_TYPE_Q8_0) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            at::Tensor temp = torch::empty(shape, options);
            const block_q8_0* src = (const block_q8_0*) t->data;
            dequantize_row_q8_0(src, temp.data_ptr<float>(), n_elements);
            tensors[name] = temp.to(torch::kFloat16);
        }
        else if (t->type == GGML_TYPE_Q5_0) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            at::Tensor temp = torch::empty(shape, options);
            const block_q5_0* src = (const block_q5_0*) t->data;
            dequantize_row_q5_0(src, temp.data_ptr<float>(), n_elements);
            tensors[name] = temp.to(torch::kFloat16);
        }
        else if (t->type == GGML_TYPE_Q5_1) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            at::Tensor temp = torch::empty(shape, options);
            const block_q5_1* src = (const block_q5_1*) t->data;
            dequantize_row_q5_1(src, temp.data_ptr<float>(), n_elements);
            tensors[name] = temp.to(torch::kFloat16);
        }
        else if (t->type == GGML_TYPE_Q6_K) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            at::Tensor temp = torch::empty(shape, options);
            const block_q6_K* src = (const block_q6_K*) t->data;
            dequantize_row_q6_K(src, temp.data_ptr<float>(), n_elements);
            tensors[name] = temp.to(torch::kFloat16);
        }
        else if (t->type == GGML_TYPE_Q4_K) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            at::Tensor temp = torch::empty(shape, options);
            const block_q4_K* src = (const block_q4_K*) t->data;
            dequantize_row_q4_K(src, temp.data_ptr<float>(), n_elements);
            tensors[name] = temp.to(torch::kFloat16);
        }
        else if (t->type == GGML_TYPE_Q5_K) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            at::Tensor temp = torch::empty(shape, options);
            const block_q5_K* src = (const block_q5_K*) t->data;
            dequantize_row_q5_K(src, temp.data_ptr<float>(), n_elements);
            tensors[name] = temp.to(torch::kFloat16);
        }
        else {
            std::cout << "[WARNING] Unhandled quantization type " << t->type << " for tensor " << name << "\n";
        }
        
        t = ggml_get_next_tensor(ctx_ggml_, t);
    }
    
    return tensors;
}

} // namespace nlp
} // namespace ohm
