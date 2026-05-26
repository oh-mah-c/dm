// ─────────────────────────────────────────────────────────────────────────────
// VICReg pretraining entry point
//
// Usage:
//   ./vicreg_train [options]
//
//   --repr-dim  <int>     encoder output dimension (default: 2048)
//   --exp-dim   <int>     expander dimension (default: 8192)
//   --lambda    <float>   invariance weight (default: 25)
//   --mu        <float>   variance weight   (default: 25)
//   --nu        <float>   covariance weight (default: 1)
//   --epochs    <int>     training epochs   (default: 1000)
//   --batch     <int>     batch size        (default: 2048)
//   --lr        <float>   base learning rate (lr = batch/256 × base_lr, default base: 0.2)
//   --wd        <float>   weight decay (default: 1e-6)
//   --warmup    <int>     warmup epochs (default: 10)
//   --data      <path>    image folder root (default: ./data)
//   --save      <path>    checkpoint path (default: vicreg_ckpt.pt)
//   --cuda                use CUDA if available
//
// Demo mode (no --data): runs a smoke-test on random tensors for 5 steps.
//
// References:
//   Section 4.2 (training protocol):
//     LARS optimizer, lr = batch/256 × base_lr, cosine decay from lr to 0.002,
//     weight-decay=1e-6, 1000 epochs, warmup=10 epochs, batch=2048.
//   LibTorch does not ship LARS; we use AdamW as a practical substitute.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/vicreg/vicreg.h"

#include <torch/torch.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cmath>

using namespace dm::models::vision;

// ── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    int64_t repr_dim    = 2048;
    int64_t exp_dim     = 8192;
    double  lambda_inv  = 25.0;
    double  mu_var      = 25.0;
    double  nu_cov      = 1.0;
    int64_t epochs      = 1000;
    int64_t batch_size  = 2048;
    double  base_lr     = 0.2;      // scaled: lr = batch/256 × base_lr
    double  wd          = 1e-6;
    int64_t warmup      = 10;
    std::string data_root  = "";
    std::string save_path  = "vicreg_ckpt.pt";
    bool    cuda           = false;
    int64_t in_dim         = 3*224*224;  // for MLP demo mode
};

static Args parse_args(int argc, char *argv[]) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if      (s == "--repr-dim" && i+1 < argc) a.repr_dim   = std::stoll(argv[++i]);
        else if (s == "--exp-dim"  && i+1 < argc) a.exp_dim    = std::stoll(argv[++i]);
        else if (s == "--lambda"   && i+1 < argc) a.lambda_inv = std::stod(argv[++i]);
        else if (s == "--mu"       && i+1 < argc) a.mu_var     = std::stod(argv[++i]);
        else if (s == "--nu"       && i+1 < argc) a.nu_cov     = std::stod(argv[++i]);
        else if (s == "--epochs"   && i+1 < argc) a.epochs     = std::stoll(argv[++i]);
        else if (s == "--batch"    && i+1 < argc) a.batch_size = std::stoll(argv[++i]);
        else if (s == "--lr"       && i+1 < argc) a.base_lr    = std::stod(argv[++i]);
        else if (s == "--wd"       && i+1 < argc) a.wd         = std::stod(argv[++i]);
        else if (s == "--warmup"   && i+1 < argc) a.warmup     = std::stoll(argv[++i]);
        else if (s == "--data"     && i+1 < argc) a.data_root  = argv[++i];
        else if (s == "--save"     && i+1 < argc) a.save_path  = argv[++i];
        else if (s == "--cuda")                   a.cuda       = true;
    }
    return a;
}

// ── Cosine LR schedule with warmup (Section 4.2) ─────────────────────────────
// Warm up linearly from 0 to lr_peak over warmup epochs.
// Then cosine decay from lr_peak to lr_min.

static double cosine_lr_warmup(double lr_peak, double lr_min,
                                int64_t epoch, int64_t warmup, int64_t total) {
    if (epoch < warmup) {
        return lr_peak * static_cast<double>(epoch + 1) / warmup;
    }
    double progress = static_cast<double>(epoch - warmup) / (total - warmup);
    return lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + std::cos(M_PI * progress));
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
    Args args = parse_args(argc, argv);

    // Effective learning rate (Section 4.2)
    double lr_peak = args.base_lr * static_cast<double>(args.batch_size) / 256.0;
    double lr_min  = 0.002 * lr_peak;  // paper: final lr = 0.002 × base_lr

    // Device
    torch::Device device = torch::kCPU;
    if (args.cuda && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA.\n";
    }

    // Build model (MLP encoder — replace with ResNet backbone for real training)
    VICRegConfig cfg;
    cfg.repr_dim     = args.repr_dim;
    cfg.expander_dim = args.exp_dim;
    cfg.lambda_inv   = args.lambda_inv;
    cfg.mu_var       = args.mu_var;
    cfg.nu_cov       = args.nu_cov;

    auto model = make_vicreg_mlp(args.in_dim, args.repr_dim, args.exp_dim);
    model->cfg = cfg;
    model->to(device);

    // Count parameters
    int64_t n_params = 0;
    for (auto &p : model->parameters()) n_params += p.numel();
    std::cout << "VICReg  repr_dim=" << args.repr_dim
              << "  exp_dim=" << args.exp_dim
              << "  λ=" << cfg.lambda_inv
              << "  μ=" << cfg.mu_var
              << "  ν=" << cfg.nu_cov << "\n";
    std::cout << "Total parameters: " << n_params / 1'000'000.0 << "M\n";
    std::cout << "Effective lr_peak: " << lr_peak
              << "  lr_min: " << lr_min << "\n";

    // Optimizer: AdamW (paper uses LARS; AdamW is a practical LibTorch substitute)
    auto opt = torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(lr_peak).weight_decay(args.wd));

    // ── Demo / smoke-test mode ────────────────────────────────────────────────
    if (args.data_root.empty()) {
        std::cout << "\nNo --data provided. Running smoke-test (5 steps).\n";
        int64_t flat_dim = args.in_dim;
        model->train();
        for (int step = 0; step < 5; step++) {
            auto x  = torch::randn({4, flat_dim},
                                   torch::TensorOptions().device(device));
            auto xp = torch::randn({4, flat_dim},
                                   torch::TensorOptions().device(device));
            opt.zero_grad();
            auto r = model(x, xp);
            r.total.backward();
            opt.step();
            std::cout << "  step " << step
                      << "  loss=" << r.total.item<float>()
                      << "  inv="  << r.inv.item<float>()
                      << "  var="  << r.var.item<float>()
                      << "  cov="  << r.cov.item<float>() << "\n";
        }
        std::cout << "Smoke-test complete.\n";
        return 0;
    }

    // ── Real training loop ────────────────────────────────────────────────────
    // NOTE: torch::data::datasets::ImageFolder is not available in this
    // LibTorch distribution.  Replace the data-loading block below with your
    // preferred data pipeline (e.g. custom Dataset subclass + DataLoader).
    // The training logic (scheduler, optimizer step, checkpoint) is complete.

    // Placeholder: iterate over synthetic batches to demonstrate the loop.
    // In real use, replace the inner loop body's `x` / `xp` tensors with
    // batches from an actual image dataset with two random augmentation views.

    for (int64_t epoch = 0; epoch < args.epochs; epoch++) {
        model->train();

        double lr = cosine_lr_warmup(lr_peak, lr_min,
                                     epoch, args.warmup, args.epochs);
        for (auto &pg : opt.param_groups())
            static_cast<torch::optim::AdamWOptions &>(pg.options()).lr(lr);

        // ── Replace this block with a real DataLoader ─────────────────────
        // Simulated single batch per epoch for skeleton completeness.
        auto x  = torch::randn({static_cast<int>(args.batch_size > 32 ? 32 : args.batch_size),
                                 static_cast<int>(args.in_dim)},
                                torch::TensorOptions().device(device));
        auto xp = torch::randn_like(x);
        // ─────────────────────────────────────────────────────────────────

        opt.zero_grad();
        auto r = model(x, xp);
        r.total.backward();
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        opt.step();

        float etotal = r.total.item<float>();
        float einv   = r.inv.item<float>();
        float evar   = r.var.item<float>();
        float ecov   = r.cov.item<float>();

        std::cout << "epoch " << epoch
                  << "  loss="  << etotal
                  << "  inv="   << einv
                  << "  var="   << evar
                  << "  cov="   << ecov
                  << "  lr="    << lr << "\n";

        torch::save(model, args.save_path);
    }

    std::cout << "Training complete. Checkpoint: " << args.save_path << "\n";
    return 0;
}
