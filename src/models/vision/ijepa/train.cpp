// ─────────────────────────────────────────────────────────────────────────────
// I-JEPA pretraining entry point
//
// Usage:
//   ./ijepa_train [options]
//
//   --model    vit_b16|vit_l16|vit_h14     (default: vit_b16)
//   --img-size <int>                        (default: 224)
//   --epochs   <int>                        (default: 300)
//   --batch    <int>                        (default: 2048 — paper default)
//   --lr       <float>                      (default: 1e-4 → 1e-3 warmup, paper App.A)
//   --wd-start <float>                      (default: 0.04, linearly → 0.4)
//   --ema-start<float>                      (default: 0.996, linearly → 1.0)
//   --warmup   <int>                        warmup epochs (default: 15)
//   --data     <path>                       image folder root (default: ./data)
//   --save     <path>                       checkpoint (default: ijepa_ckpt.pt)
//   --cuda                                  use CUDA if available
//
// Demo mode (no --data): runs a smoke-test on random tensors for 3 steps.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/ijepa/ijepa.h"

#include <torch/torch.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cmath>

using namespace dm::models::vision;

// ── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    std::string model      = "vit_b16";
    int64_t     img_size   = 224;
    int64_t     epochs     = 300;
    int64_t     batch_size = 2048;
    double      lr         = 1e-4;    // initial; warmed to 1e-3 over warmup_epochs
    double      wd_start   = 0.04;
    double      wd_end     = 0.40;
    double      ema_start  = 0.996;
    int64_t     warmup_epochs = 15;
    std::string data_root  = "";
    std::string save_path  = "ijepa_ckpt.pt";
    bool        cuda       = false;
};

static Args parse_args(int argc, char *argv[]) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if      (s == "--model"    && i+1 < argc) a.model          = argv[++i];
        else if (s == "--img-size" && i+1 < argc) a.img_size       = std::stoll(argv[++i]);
        else if (s == "--epochs"   && i+1 < argc) a.epochs         = std::stoll(argv[++i]);
        else if (s == "--batch"    && i+1 < argc) a.batch_size     = std::stoll(argv[++i]);
        else if (s == "--lr"       && i+1 < argc) a.lr             = std::stod(argv[++i]);
        else if (s == "--wd-start" && i+1 < argc) a.wd_start       = std::stod(argv[++i]);
        else if (s == "--ema-start"&& i+1 < argc) a.ema_start      = std::stod(argv[++i]);
        else if (s == "--warmup"   && i+1 < argc) a.warmup_epochs  = std::stoll(argv[++i]);
        else if (s == "--data"     && i+1 < argc) a.data_root      = argv[++i];
        else if (s == "--save"     && i+1 < argc) a.save_path      = argv[++i];
        else if (s == "--cuda")                   a.cuda           = true;
    }
    return a;
}

// ── Cosine LR schedule (Appendix A) ─────────────────────────────────────────
// Linearly warm up from lr_start to lr_peak over warmup_epochs,
// then cosine decay to lr_min.

static double cosine_lr(double lr_start, double lr_peak, double lr_min,
                        int64_t epoch, int64_t warmup, int64_t total) {
    if (epoch < warmup) {
        return lr_start + (lr_peak - lr_start) * static_cast<double>(epoch) / warmup;
    }
    double progress = static_cast<double>(epoch - warmup) / (total - warmup);
    return lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + std::cos(M_PI * progress));
}

// ── EMA momentum schedule (Appendix A: linearly 0.996 → 1.0) ────────────────

static double ema_momentum(double ema_start, int64_t epoch, int64_t total) {
    return ema_start + (1.0 - ema_start) * static_cast<double>(epoch) / total;
}

// ── Weight decay schedule (linearly wd_start → wd_end) ──────────────────────

static double weight_decay_sched(double wd_start, double wd_end,
                                  int64_t epoch, int64_t total) {
    return wd_start + (wd_end - wd_start) * static_cast<double>(epoch) / total;
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    Args args = parse_args(argc, argv);

    // Device
    torch::Device device = torch::kCPU;
    if (args.cuda && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA.\n";
    }

    // Build model
    IJEPAConfig cfg;
    if      (args.model == "vit_b16") cfg = IJEPAConfig::vit_b16(args.img_size);
    else if (args.model == "vit_l16") cfg = IJEPAConfig::vit_l16(args.img_size);
    else if (args.model == "vit_h14") cfg = IJEPAConfig::vit_h14(args.img_size);
    else throw std::invalid_argument("Unknown model: " + args.model);

    IJEPA model(cfg);
    model->to(device);

    std::cout << "I-JEPA " << args.model
              << "  img=" << args.img_size
              << "  patches=" << model->num_patches
              << "  embed=" << cfg.embed_dim
              << "  pred_dim=" << cfg.pred_dim << "\n";

    // Count trainable parameters (context encoder + predictor only)
    int64_t n_params = 0;
    for (auto &p : model->context_encoder->parameters()) n_params += p.numel();
    for (auto &p : model->predictor->parameters())       n_params += p.numel();
    std::cout << "Trainable parameters: " << n_params / 1'000'000.0 << "M\n";

    // Optimizer: AdamW on context-encoder + predictor (not target-encoder)
    std::vector<torch::Tensor> opt_params;
    for (auto &p : model->context_encoder->parameters()) opt_params.push_back(p);
    for (auto &p : model->predictor->parameters())       opt_params.push_back(p);

    auto opt = torch::optim::AdamW(opt_params,
        torch::optim::AdamWOptions(args.lr).weight_decay(args.wd_start));

    // ── Demo / smoke-test mode (no data path) ────────────────────────────────
    if (args.data_root.empty()) {
        std::cout << "\nNo --data provided. Running smoke-test (3 steps).\n";
        model->train();
        for (int step = 0; step < 3; step++) {
            auto img = torch::randn({2, 3, args.img_size, args.img_size},
                                     torch::TensorOptions().device(device));
            opt.zero_grad();
            auto loss = model(img);
            loss.backward();
            opt.step();
            model->update_target_encoder(cfg.ema_momentum);
            std::cout << "  step " << step
                      << "  loss=" << loss.item<float>() << "\n";
        }
        std::cout << "Smoke-test complete.\n";
        return 0;
    }

    // ── Real training loop ────────────────────────────────────────────────────

    // ImageFolder dataset with basic normalisation (ImageNet mean/std).
    auto transform = torch::data::transforms::Compose<>({
        torch::data::transforms::Normalize<>(
            {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})
    });

    auto dataset = torch::data::datasets::ImageFolder(args.data_root)
                       .map(torch::data::transforms::Stack<>());
    auto loader  = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
                       std::move(dataset),
                       torch::data::DataLoaderOptions()
                           .batch_size(args.batch_size)
                           .workers(4));

    for (int64_t epoch = 0; epoch < args.epochs; epoch++) {
        model->train();
        double lr  = cosine_lr(args.lr, args.lr * 10.0, 1e-6,
                                epoch, args.warmup_epochs, args.epochs);
        double wd  = weight_decay_sched(args.wd_start, args.wd_end,
                                         epoch, args.epochs);
        double ema = ema_momentum(args.ema_start, epoch, args.epochs);

        // Update optimizer LR and WD
        for (auto &pg : opt.param_groups()) {
            static_cast<torch::optim::AdamWOptions &>(pg.options()).lr(lr);
            static_cast<torch::optim::AdamWOptions &>(pg.options()).weight_decay(wd);
        }

        double epoch_loss = 0.0;
        int64_t steps = 0;
        for (auto &batch : *loader) {
            auto img = batch.data.to(device);
            opt.zero_grad();
            auto loss = model(img);
            loss.backward();
            // Gradient clip (common in ViT training)
            torch::nn::utils::clip_grad_norm_(opt_params, 1.0);
            opt.step();
            model->update_target_encoder(ema);
            epoch_loss += loss.item<double>();
            steps++;
        }

        std::cout << "epoch " << epoch
                  << "  loss=" << epoch_loss / steps
                  << "  lr="  << lr
                  << "  wd="  << wd
                  << "  ema=" << ema << "\n";

        // Save checkpoint
        torch::save(model, args.save_path);
    }

    std::cout << "Training complete. Checkpoint: " << args.save_path << "\n";
    return 0;
}
