// OhmC1 training CLI
// oh-mah-c, dm/OhmC1, 2026. [151]
//
// Usage:
//   ./ohmc1_train [--size tiny|small|medium|large] [--seqlen N] [--epochs N]
//                 [--batch N] [--lr F] [--minlr F] [--warmup N] [--maxiter N]
//                 [--save PATH] [--cuda]
//
// Recommended per-size settings:
//   tiny:   --lr 3e-3 --warmup 100  --maxiter 2000
//   small:  --lr 3e-4 --warmup 2000 --maxiter 100000
//   medium: --lr 2e-4 --warmup 4000 --maxiter 250000
//   large:  --lr 1e-4 --warmup 8000 --maxiter 500000

#include "models/nlp/ohmc1/ohmc1.h"

#include <torch/torch.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>

using namespace dm::models::nlp;

static void usage(const char* prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --size   <tiny|small|medium|large>  model preset (default: tiny)\n"
        "  --seqlen <int>                      context window override\n"
        "  --epochs <int>                      training epochs (default: 5)\n"
        "  --batch  <int>                      batch size (default: 4)\n"
        "  --lr     <float>                    peak learning rate (default: 3e-4)\n"
        "  --minlr  <float>                    min lr after cosine decay (default: 1e-5)\n"
        "  --warmup <int>                      warmup steps (default: 100)\n"
        "  --maxiter<int>                      total gradient steps (default: 2000)\n"
        "  --save   <path>                     checkpoint path (default: ohmc1_best.pt)\n"
        "  --cuda                              use CUDA if available\n",
        prog);
}

int main(int argc, char** argv) {
    std::string size_str  = "tiny";
    int64_t epochs        = 5;
    int64_t batch_size    = 4;
    int64_t seqlen_override = 0;
    std::string save_path = "ohmc1_best.pt";
    bool use_cuda         = false;

    OhmC1TrainConfig train_cfg;
    train_cfg.lr           = 3e-4;
    train_cfg.min_lr       = 1e-5;
    train_cfg.warmup_iters = 100;
    train_cfg.max_iters    = 2000;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        else if (arg == "--size"    && i + 1 < argc) size_str           = argv[++i];
        else if (arg == "--seqlen"  && i + 1 < argc) seqlen_override    = std::atoi(argv[++i]);
        else if (arg == "--epochs"  && i + 1 < argc) epochs             = std::atoi(argv[++i]);
        else if (arg == "--batch"   && i + 1 < argc) batch_size         = std::atoi(argv[++i]);
        else if (arg == "--lr"      && i + 1 < argc) train_cfg.lr       = std::atof(argv[++i]);
        else if (arg == "--minlr"   && i + 1 < argc) train_cfg.min_lr   = std::atof(argv[++i]);
        else if (arg == "--warmup"  && i + 1 < argc) train_cfg.warmup_iters = std::atoi(argv[++i]);
        else if (arg == "--maxiter" && i + 1 < argc) train_cfg.max_iters    = std::atoi(argv[++i]);
        else if (arg == "--save"    && i + 1 < argc) save_path          = argv[++i];
        else if (arg == "--cuda")                    use_cuda           = true;
        else { std::fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    // Build model config
    OhmC1Config model_cfg;
    if      (size_str == "tiny")   model_cfg = OhmC1Config::tiny();
    else if (size_str == "small")  model_cfg = OhmC1Config::small();
    else if (size_str == "medium") model_cfg = OhmC1Config::medium();
    else if (size_str == "large")  model_cfg = OhmC1Config::large();
    else {
        std::fprintf(stderr, "Unknown size: %s\n", size_str.c_str());
        return 1;
    }
    if (seqlen_override > 0) model_cfg.seq_len = seqlen_override;

    // Device
    torch::Device device = torch::kCPU;
    if (use_cuda) {
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::printf("Using CUDA.\n");
        } else {
            std::printf("CUDA not available, using CPU.\n");
        }
    }
    train_cfg.device = device;

    // Print config
    std::printf("OhmC1 — sandwich-norm + QKNorm decoder LLM\n");
    std::printf("  size:       %s\n", size_str.c_str());
    std::printf("  dim:        %lld\n", (long long)model_cfg.dim);
    std::printf("  n_layers:   %lld\n", (long long)model_cfg.n_layers);
    std::printf("  n_heads:    %lld  (kv=%lld, rep=%lld)\n",
        (long long)model_cfg.n_heads,
        (long long)model_cfg.n_kv_heads,
        (long long)model_cfg.n_rep());
    std::printf("  seq_len:    %lld\n", (long long)model_cfg.seq_len);
    std::printf("  vocab_size: %lld\n", (long long)model_cfg.vocab_size);
    std::printf("  rope_theta: %.0f\n", (double)model_cfg.rope_theta);
    std::printf("  epochs:     %lld\n", (long long)epochs);
    std::printf("  batch:      %lld\n", (long long)batch_size);
    std::printf("  lr:         %.2e -> %.2e\n", train_cfg.lr, train_cfg.min_lr);
    std::printf("  warmup:     %lld  max_iter: %lld\n\n",
        (long long)train_cfg.warmup_iters, (long long)train_cfg.max_iters);

    // Build model and move to device
    auto model = OhmC1Model(model_cfg);
    model->to(device);

    int64_t total_params = 0;
    for (auto& p : model->parameters()) total_params += p.numel();
    std::printf("Total parameters: %lld (%.1fM)\n\n",
        (long long)total_params, (double)total_params / 1e6);

    auto optimizer = make_ohmc1_optimizer(model, train_cfg);

    float best_loss = 1e30f;
    int64_t global_step = 0;

    for (int64_t ep = 0; ep < epochs; ++ep) {
        // Training with synthetic random tokens (smoke test)
        int64_t steps_per_epoch = std::max((int64_t)1,
            train_cfg.max_iters / std::max(epochs, (int64_t)1));

        for (int64_t step = 0; step < steps_per_epoch; ++step) {
            auto tokens = torch::randint(0, model_cfg.vocab_size,
                {batch_size, model_cfg.seq_len + 1},
                torch::TensorOptions().dtype(torch::kLong).device(device));

            float loss = ohmc1_train_step(model, optimizer, tokens, train_cfg, global_step);
            ++global_step;

            if (step % 10 == 0) {
                std::printf("epoch %lld / %lld  step %lld  loss %.4f  lr %.2e\n",
                    (long long)(ep + 1), (long long)epochs,
                    (long long)global_step, loss,
                    ohmc1_lr_schedule(global_step - 1, train_cfg));
            }

            if (loss < best_loss) {
                best_loss = loss;
                torch::save(model, save_path);
            }
        }
    }

    std::printf("\nTraining complete. Best loss: %.4f  Saved to: %s\n",
        best_loss, save_path.c_str());
    return 0;
}
