#pragma once
// -----------------------------------------------------------------------------
// OhmC2 — CPU-first dual-path LLM
//
// Uses parallel RealFormer attention and per-token hard MoE KAN experts.
// Inference keeps preallocated K/V and RealFormer score caches; ordinary
// LibTorch temporary tensors are still allowed.
// -----------------------------------------------------------------------------

#include <torch/torch.h>
#include <torch/nn/modules/kan_linear.h>
#include <torch/nn/modules/ohm_block_drop.h>
#include <torch/nn/modules/ohm_hard_router.h>
#include <torch/nn/modules/ohm_ring_kv.h>

#include <cstdint>
#include <string>
#include <vector>

namespace dm {
namespace models {
namespace nlp {

struct OhmC2Config {
    int64_t dim = 512;
    int64_t ffn_dim = 2048;
    int64_t n_layers = 16;
    int64_t n_heads = 8;
    int64_t n_experts = 4;
    int64_t kan_grid = 5;
    int64_t kan_order = 3;
    int64_t vocab_size = 64000;
    int64_t max_seq = 2048;
    std::string ffn_type = "swiglu";  // swiglu, kan, kan_moe
    double norm_eps = 1e-6;
    double dropout = 0.0;
    double rope_theta = 10000.0;
    bool use_rope = true;
    bool realformer_mean = true;

    int64_t head_dim() const { return dim / n_heads; }

    static OhmC2Config tiny();
    static OhmC2Config small();
    static OhmC2Config medium();
    static OhmC2Config large();
    static OhmC2Config cpu_quality();
};

struct OhmC2GenerationState {
    std::vector<torch::nn::OhmRingKV> k_ring;
    std::vector<torch::nn::OhmRingKV> v_ring;
    std::vector<torch::Tensor> score_cache;  // [B,H,max_seq]
    int64_t pos = 0;
    int64_t valid_len = 0;
    int64_t max_seq = 0;
    int64_t batch_size = 0;

    void reset();
};

struct OhmC2RMSNormImpl : torch::nn::Module {
    torch::Tensor weight;
    double eps;

    OhmC2RMSNormImpl(int64_t dim, double eps = 1e-6);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(OhmC2RMSNorm);

struct OhmC2DualGateBlockImpl : torch::nn::Module {
    OhmC2Config cfg;
    int64_t layer_index = 0;

    OhmC2RMSNorm norm_attn_in{nullptr};
    OhmC2RMSNorm norm_attn_out{nullptr};
    OhmC2RMSNorm norm_ffn_in{nullptr};
    OhmC2RMSNorm norm_ffn_out{nullptr};

    torch::nn::Linear wq{nullptr}, wk{nullptr}, wv{nullptr}, wo{nullptr};
    torch::nn::Linear router{nullptr};
    torch::nn::ModuleList experts{nullptr};
    torch::nn::OhmHardRouter ohm_router{nullptr};
    torch::nn::Linear ffn_up{nullptr}, ffn_gate_proj{nullptr};
    torch::nn::Linear w_cross{nullptr}, w2{nullptr};
    torch::nn::OhmBlockDrop block_drop{nullptr};

    torch::Tensor gate_attn;
    torch::Tensor gate_ffn;
    torch::Tensor last_route_counts;  // [E], updated by ffn_route()

    OhmC2DualGateBlockImpl(const OhmC2Config& cfg, int64_t layer_index);

    std::pair<torch::Tensor, torch::Tensor>
    forward(torch::Tensor x, torch::Tensor prev_scores = {});

    torch::Tensor forward_step(torch::Tensor x,
                               OhmC2GenerationState& state);

    torch::Tensor ffn_route(torch::Tensor norm_f);
    torch::Tensor ffn_forward(torch::Tensor norm_f);

private:
    std::pair<torch::Tensor, torch::Tensor>
    attention_full(torch::Tensor norm_a, torch::Tensor prev_scores);
};
TORCH_MODULE(OhmC2DualGateBlock);

struct OhmC2LLMImpl : torch::nn::Module {
    OhmC2Config cfg;

    torch::nn::Embedding tok_embeddings{nullptr};
    torch::nn::ModuleList layers{nullptr};
    OhmC2RMSNorm norm{nullptr};
    torch::nn::Linear lm_head{nullptr};
    torch::Tensor last_loss;
    std::vector<torch::Tensor> last_prev_scores;

    explicit OhmC2LLMImpl(const OhmC2Config& cfg = OhmC2Config::tiny());

    torch::Tensor forward(torch::Tensor tokens,
                          torch::Tensor targets = {});
    torch::Tensor forward_step(torch::Tensor token_ids,
                               OhmC2GenerationState& state);
    OhmC2GenerationState init_generation_state(int64_t batch_size = 1) const;
    std::vector<int64_t> generate(const std::vector<int64_t>& prompt_ids,
                                  int64_t max_new_tokens = 32,
                                  float temperature = 0.0f,
                                  float top_p = 0.9f,
                                  int64_t eos_id = 1);
};
TORCH_MODULE(OhmC2LLM);

OhmC2LLM make_ohmc2_tiny();
OhmC2LLM make_ohmc2_small();
OhmC2LLM make_ohmc2_medium();
OhmC2LLM make_ohmc2_large();

struct OhmC2TrainConfig {
    double lr = 3e-4;
    double min_lr = 1e-5;
    double weight_decay = 0.1;
    double beta1 = 0.9;
    double beta2 = 0.95;
    double eps = 1e-8;
    double grad_clip = 1.0;
    int64_t warmup_iters = 100;
    int64_t max_iters = 2000;
    torch::Device device = torch::kCPU;
};

torch::optim::AdamW make_ohmc2_optimizer(OhmC2LLM& model,
                                          const OhmC2TrainConfig& cfg);
double ohmc2_lr_schedule(int64_t iter, const OhmC2TrainConfig& cfg);
float ohmc2_train_step(OhmC2LLM& model,
                       torch::optim::AdamW& optimizer,
                       torch::Tensor tokens,
                       const OhmC2TrainConfig& cfg,
                       int64_t iter);

}  // namespace nlp
}  // namespace models
}  // namespace dm
