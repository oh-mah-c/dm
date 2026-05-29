#pragma once

#include <vector>

namespace dm {
namespace nlp {
namespace mla {

// Cấu hình (Config) của Multi-Head Latent Attention
struct MLAConfig {
    int num_heads;
    int head_dim;
    int latent_dim;    // Chiều của Vector Tiềm ẩn c_t (Thường rất nhỏ so với num_heads * head_dim)
    int rope_dim;      // Chiều của Decoupled 3D-RoPE
    
    // Dung lượng (Byte) của 1 Token trong KV Cache truyền thống (Standard MHA)
    // 2 (Key, Value) * num_heads * head_dim * sizeof(float)
    size_t standard_kv_cache_bytes_per_token() const {
        return 2 * num_heads * head_dim * sizeof(float);
    }
    
    // Dung lượng (Byte) của 1 Token trong VideoMLA Cache
    // latent_dim (c_t) + rope_dim (k_t^R) * sizeof(float)
    size_t mla_cache_bytes_per_token() const {
        return (latent_dim + rope_dim) * sizeof(float);
    }
};

// Cấu trúc lưu trữ MLACache (Được phẳng hóa để tối ưu L3 Cache)
struct MLACache {
    std::vector<float> data; // Array 1D lưu trữ [c_t | k_t^R] cho mọi tokens
    int num_tokens;
    int token_stride;
    
    MLACache(int max_tokens, const MLAConfig& config) {
        token_stride = config.latent_dim + config.rope_dim;
        data.resize(max_tokens * token_stride, 0.0f);
        num_tokens = 0;
    }
    
    // Thêm một Token mới vào Cache (Compression / Down-Projection)
    // - input_kv: Dữ liệu KV gốc (Dài num_heads * head_dim * 2)
    // - Thực tế ở đây chúng ta giả lập phép Down-Projection bằng cách nén trực tiếp
    void append_token(const std::vector<float>& compressed_latent, const std::vector<float>& decoupled_rope) {
        int offset = num_tokens * token_stride;
        for(size_t i = 0; i < compressed_latent.size(); ++i) {
            data[offset + i] = compressed_latent[i];
        }
        for(size_t i = 0; i < decoupled_rope.size(); ++i) {
            data[offset + compressed_latent.size() + i] = decoupled_rope[i];
        }
        num_tokens++;
    }
};

// Động cơ VideoMLA
class VideoMLAEngine {
private:
    MLAConfig config;
    
    // Ma trận trọng số (Chỉ mô phỏng kích thước để tính toán)
    // Trong thực tế, các ma trận này sẽ được nhân để Up-Project từ latent_dim lên num_heads * head_dim
    std::vector<float> W_UK; // Trọng số khôi phục Key
    std::vector<float> W_UV; // Trọng số khôi phục Value

public:
    VideoMLAEngine(const MLAConfig& cfg) : config(cfg) {}
    
    const MLAConfig& get_config() const { return config; }
    
    // Thực hiện Up-Projection: Khôi phục Key từ Latent Vector (Chỉ giả lập trả về kích thước)
    // (Bản chất: k_c = c_t * W_UK)
    std::vector<float> up_project_key(const MLACache& cache, int token_idx) const;
    
    // Thực hiện Up-Projection: Khôi phục Value từ Latent Vector
    // (Bản chất: v = c_t * W_UV)
    std::vector<float> up_project_value(const MLACache& cache, int token_idx) const;
};

} // namespace mla
} // namespace nlp
} // namespace dm
