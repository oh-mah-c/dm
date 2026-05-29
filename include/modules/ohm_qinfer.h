#pragma once

#include <torch/torch.h>
#include "algorithms/ohm_qinfer_kernel.h"
#include <memory>

namespace dm {
namespace modules {

struct OhmQINFERImpl : torch::nn::Module {
    int num_universes_;
    int grid_size_;
    std::unique_ptr<dm::algorithm::OhmQInferArena> arena_;

    OhmQINFERImpl(int num_universes = 1000, int grid_size = 100) 
        : num_universes_(num_universes), grid_size_(grid_size) {
        
        arena_ = std::make_unique<dm::algorithm::OhmQInferArena>(num_universes, grid_size);
        arena_->init_random_universes();
    }

    // Nhận một điểm dữ liệu (tọa độ electron đập vào màn hình) và suy diễn
    void forward(int X_measured) {
        arena_->deduce_physics(X_measured);
    }

    // Trả về bức tường thế năng V(x) đã suy diễn được
    torch::Tensor get_deduced_barrier() {
        const float* best_v = arena_->get_best_V_barrier();
        
        // Tạo Tensor từ mảng C++ (Copy dữ liệu ra ngoài)
        return torch::from_blob((void*)best_v, {grid_size_}, torch::kFloat32).clone();
    }
};

TORCH_MODULE(OhmQINFER);

} // namespace modules
} // namespace dm
