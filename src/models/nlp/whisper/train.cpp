// ─────────────────────────────────────────────────────────────────────────────
// Whisper training entry point — Radford et al., ICML 2023 (radford23a)
//
// Usage:
//   ./whisper_train [options]
//   --size     <tiny|base|small|medium|large>  (default: base)
//   --epochs   <int>    (default: 10)
//   --batch    <int>    (default: 256)
//   --lr       <float>  (default: 1e-3)
//   --warmup   <int>    warmup steps (default: 2048)
//   --data     <path>   dataset root (default: ./data)
//   --save     <path>   checkpoint (default: whisper_best.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/whisper/whisper.h"

#include <torch/torch.h>
#include <iostream>
#include <string>

using namespace dm::models::nlp;

struct Args {
    std::string size       = "base";
    int64_t     epochs     = 10;
    int64_t     batch      = 256;
    double      lr         = 1e-3;
    int64_t     warmup     = 2048;
    std::string data       = "./data";
    std::string save       = "whisper_best.pt";
    bool        cuda       = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--size"   && i+1<argc) a.size    = argv[++i];
        else if (k == "--epochs" && i+1<argc) a.epochs  = std::stoll(argv[++i]);
        else if (k == "--batch"  && i+1<argc) a.batch   = std::stoll(argv[++i]);
        else if (k == "--lr"     && i+1<argc) a.lr      = std::stod(argv[++i]);
        else if (k == "--warmup" && i+1<argc) a.warmup  = std::stoll(argv[++i]);
        else if (k == "--data"   && i+1<argc) a.data    = argv[++i];
        else if (k == "--save"   && i+1<argc) a.save    = argv[++i];
        else if (k == "--cuda")               a.cuda    = true;
    }
    return a;
}

static WhisperModel build_model(const std::string& size) {
    if (size == "tiny")   return make_whisper_tiny();
    if (size == "small")  return make_whisper_small();
    if (size == "medium") return make_whisper_medium();
    if (size == "large")  return make_whisper_large();
    return make_whisper_base(); // default
}

int main(int argc, char* argv[]) {
    auto args   = parse_args(argc, argv);
    auto device = (args.cuda && torch::cuda::is_available())
                  ? torch::kCUDA : torch::kCPU;
    std::cout << "[Whisper] size=" << args.size
              << "  device=" << device << "\n";

    auto model = build_model(args.size);
    model->to(device);

    int64_t n_params = 0;
    for (auto& p : model->parameters()) n_params += p.numel();
    std::cout << "[Whisper] params=" << n_params / 1'000'000 << "."
              << (n_params / 100'000) % 10 << "M\n";

    // AdamW — Section 2.2
    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(args.lr)
            .betas({0.9, 0.999})
            .weight_decay(1e-6));

    // Placeholder batches
    struct Batch {
        torch::Tensor mel;        // [B, 80, T]
        torch::Tensor tokens_in;  // [B, L]
        torch::Tensor tokens_tgt; // [B, L]
    };
    std::vector<Batch> train_batches, val_batches;

    try {
        torch::Tensor mels, tokens;
        torch::load(mels,   args.data + "/mels.pt");
        torch::load(tokens, args.data + "/tokens.pt");
        int64_t total = mels.size(0);
        int64_t val_n = std::max<int64_t>(1, total / 10);
        for (int64_t i = 0; i + args.batch <= total - val_n; i += args.batch) {
            auto m = mels.slice(0, i, i + args.batch).to(device);
            auto t = tokens.slice(0, i, i + args.batch).to(device);
            // teacher-forced: input is tokens[:-1], target is tokens[1:]
            train_batches.push_back({m,
                t.slice(1, 0, t.size(1) - 1),
                t.slice(1, 1, t.size(1))});
        }
        for (int64_t i = total - val_n; i + args.batch <= total; i += args.batch) {
            auto m = mels.slice(0, i, i + args.batch).to(device);
            auto t = tokens.slice(0, i, i + args.batch).to(device);
            val_batches.push_back({m,
                t.slice(1, 0, t.size(1) - 1),
                t.slice(1, 1, t.size(1))});
        }
        std::cout << "[Whisper] data: " << train_batches.size()
                  << " train + " << val_batches.size() << " val batches\n";
    } catch (...) {
        std::cout << "[Whisper] no data at " << args.data
                  << " — model construction verified, skipping training\n";
    }

    float best_val = std::numeric_limits<float>::max();
    int64_t global_step = 0;

    for (int64_t ep = 0; ep < args.epochs; ++ep) {
        // Train
        double train_loss = 0.0;
        int64_t steps = 0;
        for (auto& b : train_batches) {
            // Linear lr warmup (Section 2.2)
            double lr_scale = 1.0;
            if (global_step < args.warmup)
                lr_scale = (double)(global_step + 1) / (double)args.warmup;
            whisper_train_step(model, optimizer, b.mel, b.tokens_in, b.tokens_tgt, lr_scale);
            ++global_step;
            ++steps;
        }
        // Validate
        float val_loss = 0.0f;
        if (!val_batches.empty()) {
            model->eval();
            torch::NoGradGuard ng;
            double vl = 0.0;
            for (auto& b : val_batches) {
                auto logits = model->forward(b.mel, b.tokens_in);
                int64_t B = logits.size(0), L = logits.size(1), V = logits.size(2);
                auto loss = torch::nn::functional::cross_entropy(
                    logits.view({B*L, V}), b.tokens_tgt.view({B*L}));
                vl += loss.item<double>();
            }
            val_loss = (float)(vl / (double)val_batches.size());
        }
        std::cout << "[Whisper] epoch " << ep + 1 << "/" << args.epochs
                  << "  val_loss=" << val_loss << "\n";

        if (val_loss < best_val) {
            best_val = val_loss;
            torch::serialize::OutputArchive ar;
            model->save(ar);
            ar.save_to(args.save);
            std::cout << "[Whisper] saved → " << args.save << "\n";
        }
    }
    return 0;
}
