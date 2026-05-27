// ─────────────────────────────────────────────────────────────────────────────
// CCNet — Training Binary
// Huang et al., IEEE TPAMI 2020  (arXiv:1811.11721v2)
//
// Usage:
//   ccnet_train [OPTIONS]
//
// Options:
//   --epochs N          Number of training epochs (default: 40)
//   --batch-size N      Batch size (default: 2)
//   --lr LR             Initial learning rate (default: 0.01)
//   --weight-decay WD   Weight decay (default: 0.0001)
//   --momentum M        SGD momentum (default: 0.9)
//   --rcca-ch N         RCCA feature channels (default: 512)
//   --key-ch N          Key/Query channels for RCCA (default: 64)
//   --mid-ch N          Head mid channels (default: 512)
//   --loops N           RCCA recurrent loops (default: 2)
//   --num-classes N     Number of segmentation classes (default: 19)
//   --ccl               Enable Category Consistent Loss (default: off)
//   --ccl-alpha F       CCL alpha weight (default: 1.0)
//   --ccl-beta F        CCL beta weight (default: 1.0)
//   --ccl-gamma F       CCL gamma weight (default: 0.001)
//   --ignore-index N    Ignore label (default: 255)
//   --crop-size N       Input crop size (default: 64 for demo)
//   --save PATH         Checkpoint save path (default: ccnet_best.pt)
//   --device cpu|cuda   Device (default: cuda if available else cpu)
//
// Demo: runs on random tensors simulating Cityscapes-like inputs.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/ccnet/ccnet.h"

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace dm::models::vision;

// ─── Minimal backbone stub for the demo ─────────────────────────────────────
// In a real training pipeline, replace with dilated ResNet-101.
// This stub: 3 -> backbone_out_ch with a single strided conv.

struct StubBackboneImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};

    StubBackboneImpl(int64_t out_ch) {
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, out_ch, 3)
                .padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_ch));
    }
    torch::Tensor forward(torch::Tensor x) {
        return torch::relu(bn1->forward(conv1->forward(x)));
    }
};
TORCH_MODULE(StubBackbone);

// ─── Poly LR schedule ───────────────────────────────────────────────────────

static double poly_lr(double base_lr, int64_t iter, int64_t max_iter, double power) {
    return base_lr * std::pow(1.0 - static_cast<double>(iter) / max_iter, power);
}

// ─── CLI parsing helpers ─────────────────────────────────────────────────────

static std::string get_arg(const std::vector<std::string>& args,
                           const std::string& key,
                           const std::string& def) {
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == key) return args[i + 1];
    return def;
}

static bool has_flag(const std::vector<std::string>& args, const std::string& key) {
    for (auto& a : args) if (a == key) return true;
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    // ── Hyper-parameters ──────────────────────────────────────────────────
    int64_t epochs      = std::stoll(get_arg(args, "--epochs",      "40"));
    int64_t batch_size  = std::stoll(get_arg(args, "--batch-size",  "2"));
    double  lr          = std::stod (get_arg(args, "--lr",          "0.01"));
    double  weight_decay= std::stod (get_arg(args, "--weight-decay","0.0001"));
    double  momentum    = std::stod (get_arg(args, "--momentum",    "0.9"));
    int64_t rcca_ch     = std::stoll(get_arg(args, "--rcca-ch",     "512"));
    int64_t key_ch      = std::stoll(get_arg(args, "--key-ch",      "64"));
    int64_t mid_ch      = std::stoll(get_arg(args, "--mid-ch",      "512"));
    int     loops       = std::stoi (get_arg(args, "--loops",       "2"));
    int64_t num_classes = std::stoll(get_arg(args, "--num-classes", "19"));
    int64_t backbone_ch = std::stoll(get_arg(args, "--backbone-ch", "2048"));
    bool    use_ccl     = has_flag(args, "--ccl");
    float   ccl_alpha   = std::stof (get_arg(args, "--ccl-alpha",   "1.0"));
    float   ccl_beta    = std::stof (get_arg(args, "--ccl-beta",    "1.0"));
    float   ccl_gamma   = std::stof (get_arg(args, "--ccl-gamma",   "0.001"));
    int64_t ignore_idx  = std::stoll(get_arg(args, "--ignore-index","255"));
    int64_t crop_size   = std::stoll(get_arg(args, "--crop-size",   "64"));
    std::string save_path = get_arg(args, "--save", "ccnet_best.pt");
    std::string device_str= get_arg(args, "--device",
        torch::cuda::is_available() ? "cuda" : "cpu");

    torch::Device device(device_str);

    std::cout << "CCNet training demo\n"
              << "  device:      " << device_str << "\n"
              << "  backbone_ch: " << backbone_ch << "\n"
              << "  rcca_ch:     " << rcca_ch << "\n"
              << "  key_ch:      " << key_ch  << "\n"
              << "  mid_ch:      " << mid_ch  << "\n"
              << "  loops:       " << loops   << "\n"
              << "  num_classes: " << num_classes << "\n"
              << "  crop_size:   " << crop_size << "\n"
              << "  CCL:         " << (use_ccl ? "on" : "off") << "\n"
              << "  epochs:      " << epochs << "\n"
              << "  batch_size:  " << batch_size << "\n"
              << "  lr:          " << lr << "\n"
              << std::flush;

    // ── Build model ───────────────────────────────────────────────────────
    auto backbone = StubBackbone(backbone_ch);
    CCNet model(torch::nn::AnyModule(backbone), backbone_ch,
                num_classes, rcca_ch, key_ch, mid_ch, loops);
    model->to(device);
    model->train();

    // ── Optimizer: SGD with momentum ──────────────────────────────────────
    torch::optim::SGD optimizer(
        model->parameters(),
        torch::optim::SGDOptions(lr)
            .momentum(momentum)
            .weight_decay(weight_decay));

    // ── Poly LR schedule parameters ───────────────────────────────────────
    // Simulating ~100 iters/epoch
    int64_t iters_per_epoch = 100;
    int64_t max_iter = epochs * iters_per_epoch;

    CCLLossOptions ccl_opts;
    ccl_opts.alpha = ccl_alpha;
    ccl_opts.beta  = ccl_beta;
    ccl_opts.gamma = ccl_gamma;

    float best_loss = std::numeric_limits<float>::infinity();
    int64_t global_iter = 0;

    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss  = 0.0;
        double epoch_seg   = 0.0;
        double epoch_ccl_v = 0.0;
        int64_t n_batches  = iters_per_epoch;

        for (int64_t step = 0; step < n_batches; ++step, ++global_iter) {
            // Poly LR update
            double cur_lr = poly_lr(lr, global_iter, max_iter, 0.9);
            for (auto& group : optimizer.param_groups())
                static_cast<torch::optim::SGDOptions&>(group.options()).lr(cur_lr);

            // ── Simulated batch (random tensors) ──────────────────────────
            // In production: replace with DataLoader over Cityscapes.
            auto imgs = torch::randn({batch_size, 3, crop_size, crop_size},
                                      torch::TensorOptions().device(device));
            auto lbls = torch::randint(0, num_classes,
                                       {batch_size, crop_size, crop_size},
                                       torch::TensorOptions()
                                           .dtype(torch::kLong)
                                           .device(device));

            optimizer.zero_grad();

            // Forward
            auto logits = model->forward(imgs); // [B, num_classes, H, W]

            // Segmentation loss (cross-entropy, ignore_index=255)
            namespace F = torch::nn::functional;
            auto seg_loss = F::cross_entropy(
                logits, lbls,
                F::CrossEntropyFuncOptions().ignore_index(ignore_idx));

            torch::Tensor total_loss = seg_loss;

            // CCL loss (on RCCA output)
            if (use_ccl) {
                // For CCL we need the RCCA feature map; forward the same
                // backbone+reduction+RCCA sub-path explicitly.
                // We reuse the already-run backbone if we restructure forward;
                // here we duplicate forward (demo only).
                auto feat = model->rcca->forward(
                    torch::relu(
                        model->reduction_bn->forward(
                            model->reduction->forward(
                                model->backbone.forward<torch::Tensor>(imgs)))));
                auto ccl = ccl_loss(feat, lbls, ccl_opts, ignore_idx);
                total_loss = total_loss + ccl;
                epoch_ccl_v += ccl.item<double>();
            }

            total_loss.backward();
            optimizer.step();

            epoch_loss += total_loss.item<double>();
            epoch_seg  += seg_loss.item<double>();
        }

        double avg_loss = epoch_loss / n_batches;
        double avg_seg  = epoch_seg  / n_batches;

        std::cout << "Epoch [" << (epoch + 1) << "/" << epochs << "]"
                  << "  loss=" << avg_loss
                  << "  seg=" << avg_seg;
        if (use_ccl)
            std::cout << "  ccl=" << (epoch_ccl_v / n_batches);
        std::cout << "\n" << std::flush;

        // Save best checkpoint
        if (avg_loss < best_loss) {
            best_loss = static_cast<float>(avg_loss);
            torch::save(model, save_path);
            std::cout << "  => Saved checkpoint to " << save_path << "\n";
        }
    }

    std::cout << "Training complete. Best loss: " << best_loss << "\n";
    return 0;
}
