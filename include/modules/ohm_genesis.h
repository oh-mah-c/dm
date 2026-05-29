#pragma once

#include <torch/torch.h>
#include "algorithms/ohm_genesis_kernel.h"
#include <vector>

namespace dm {
namespace modules {

struct OhmGENESISImpl : torch::nn::Module {
    int width_, height_;
    
    // Ping-Pong Buffers (2 mảng cho A, 2 mảng cho B)
    torch::Tensor A_[2];
    torch::Tensor B_[2];
    
    int current_idx_; // 0 hoặc 1

    // Thông số sinh học (Mặc định cho San Hô - Corals)
    float D_a_, D_b_, feed_, kill_;

    OhmGENESISImpl(int width = 100, int height = 100) 
        : width_(width), height_(height), current_idx_(0),
          D_a_(1.0f), D_b_(0.5f), feed_(0.0545f), kill_(0.0620f) {
        
        // Cấp phát 2 bộ đệm tĩnh (Zero-Allocation runtime)
        for (int i = 0; i < 2; ++i) {
            // Hóa chất A đổ đầy khắp nơi (nồng độ 1.0)
            A_[i] = register_buffer("A_" + std::to_string(i), torch::ones({height, width}, torch::kFloat32));
            // Hóa chất B ban đầu trống rỗng (nồng độ 0.0)
            B_[i] = register_buffer("B_" + std::to_string(i), torch::zeros({height, width}, torch::kFloat32));
        }
    }

    // Nhỏ một "Hạt giống" (Seed) hóa chất B vào giữa lưới
    void plant_seed(int size = 5) {
        auto b_acc = B_[current_idx_].accessor<float, 2>();
        int cx = width_ / 2;
        int cy = height_ / 2;
        
        for (int y = cy - size; y < cy + size; ++y) {
            for (int x = cx - size; x < cx + size; ++x) {
                if (x >= 0 && x < width_ && y >= 0 && y < height_) {
                    b_acc[y][x] = 1.0f;
                }
            }
        }
    }

    // Chạy mô phỏng qua nhiều bước (Ping-Pong Swap)
    void forward(int steps = 10, float dt = 1.0f) {
        for (int s = 0; s < steps; ++s) {
            int next_idx = 1 - current_idx_;
            
            float* A_old = A_[current_idx_].data_ptr<float>();
            float* B_old = B_[current_idx_].data_ptr<float>();
            float* A_new = A_[next_idx].data_ptr<float>();
            float* B_new = B_[next_idx].data_ptr<float>();

            // Gọi thẳng C++ Kernel
            dm::algorithm::reaction_diffusion_step_cpu(
                A_old, B_old, A_new, B_new,
                width_, height_,
                D_a_, D_b_, feed_, kill_, dt
            );

            // Hoán đổi mảng (Pointer Swap O(1))
            current_idx_ = next_idx;
        }
    }

    torch::Tensor get_biology_pattern() {
        return B_[current_idx_].clone();
    }
};

TORCH_MODULE(OhmGENESIS);

} // namespace modules
} // namespace dm
