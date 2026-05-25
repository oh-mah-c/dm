// ─────────────────────────────────────────────────────────────────────────────
// Llama 2 training entry point — Touvron et al., arXiv:2307.09288, 2023
//
// Usage:
//   ./llama2_train [options]
//   --size    <stories110k|7b|13b|70b>    (default: stories110k)
//   --dim     <int>     override hidden dim (overrides --size)
//   --layers  <int>     override n_layers
//   --heads   <int>     override n_heads
//   --kvheads <int>     override n_kv_heads (GQA)
//   --seqlen  <int>     context length      (default: 256)
//   --epochs  <int>                         (default: 10)
//   --batch   <int>     batch size          (default: 64)
//   --lr      <float>   peak learning rate  (default: 3e-4)
//   --warmup  <int>     warmup iters        (default: 100)
//   --maxiter <int>     total iters         (default: 100000)
//   --data    <path>    token dataset root  (default: ./data)
//   --save    <path>    checkpoint path     (default: llama2_best.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/llama2/llama2.h"

#include <torch/torch.h>
#include <iostream>
#include <string>

using namespace dm::models::nlp;

struct Args {
    std::string size      = "stories110k";
    int64_t     dim       = 0;
    int64_t     layers    = 0;
    int64_t     heads     = 0;
    int64_t     kvheads   = 0;
    int64_t     seqlen    = 0;
    int64_t     epochs    = 10;
    int64_t     batch     = 64;
    double      lr        = 3e-4;
    int64_t     warmup    = 100;
    int64_t     maxiter   = 100000;
    std::string data      = "./data";
    std::string save      = "llama2_best.pt";
    bool        cuda      = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--size"    && i+1<argc) a.size    = argv[++i];
        else if (k == "--dim"     && i+1<argc) a.dim     = std::stoll(argv[++i]);
        else if (k == "--layers"  && i+1<argc) a.layers  = std::stoll(argv[++i]);
        else if (k == "--heads"   && i+1<argc) a.heads   = std::stoll(argv[++i]);
        else if (k == "--kvheads" && i+1<argc) a.kvheads = std::stoll(argv[++i]);
        else if (k == "--seqlen"  && i+1<argc) a.seqlen  = std::stoll(argv[++i]);
        else if (k == "--epochs"  && i+1<argc) a.epochs  = std::stoll(argv[++i]);
        else if (k == "--batch"   && i+1<argc) a.batch   = std::stoll(argv[++i]);
        else if (k == "--lr"      && i+1<argc) a.lr      = std::stod(argv[++i]);
        else if (k == "--warmup"  && i+1<argc) a.warmup  = std::stoll(argv[++i]);
        else if (k == "--maxiter" && i+1<argc) a.maxiter = std::stoll(argv[++i]);
        else if (k == "--data"    && i+1<argc) a.data    = argv[++i];
        else if (k == "--save"    && i+1<argc) a.save    = argv[++i];
        else if (k == "--cuda")                a.cuda    = true;
    }
    return a;
}

static Llama2Config build_config(const Args& a) {
    Llama2Config cfg;
    if      (a.size == "7b")  cfg = Llama2Config::llama_7b();
    else if (a.size == "13b") cfg = Llama2Config::llama_13b();
    else if (a.size == "70b") cfg = Llama2Config::llama_70b();
    else                       cfg = Llama2Config::stories110k();

    // Per-arg overrides
    if (a.dim     > 0) cfg.dim        = a.dim;
    if (a.layers  > 0) cfg.n_layers   = a.layers;
    if (a.heads   > 0) cfg.n_heads    = a.heads;
    if (a.kvheads > 0) cfg.n_kv_heads = a.kvheads;
    if (a.seqlen  > 0) cfg.seq_len    = a.seqlen;
    return cfg;
}

int main(int argc, char* argv[]) {
    auto args   = parse_args(argc, argv);
    auto device = (args.cuda && torch::cuda::is_available())
                  ? torch::kCUDA : torch::kCPU;
    auto cfg    = build_config(args);

    std::cout << "[Llama2] size=" << args.size
              << "  dim=" << cfg.dim
              << "  n_layers=" << cfg.n_layers
              << "  n_heads=" << cfg.n_heads
              << "  n_kv_heads=" << cfg.n_kv_heads
              << "  seq_len=" << cfg.seq_len
              << "  device=" << device << "\n";

    auto model = Llama2Model(cfg);
    model->to(device);

    int64_t n_params = 0;
    for (auto& p : model->parameters()) n_params += p.numel();
    std::cout << "[Llama2] params=" << n_params / 1'000'000 << "."
              << (n_params / 100'000) % 10 << "M\n";

    Llama2TrainConfig tcfg;
    tcfg.lr           = args.lr;
    tcfg.warmup_iters = args.warmup;
    tcfg.max_iters    = args.maxiter;
    tcfg.batch_size   = args.batch;
    tcfg.device       = device;

    auto optimizer = make_llama2_optimizer(model, tcfg);

    // Load token data
    std::vector<torch::Tensor> train_batches, val_batches;
    try {
        torch::Tensor tokens;
        torch::load(tokens, args.data + "/tokens.pt");  // [N] int64 flat token stream
        int64_t total = tokens.size(0);
        int64_t chunk = cfg.seq_len + 1;               // +1 for target shift
        int64_t val_chunks = std::max<int64_t>(1, total / chunk / 10);
        int64_t n_chunks   = total / chunk;

        for (int64_t i = 0; i < n_chunks - val_chunks; ++i) {
            auto seg = tokens.slice(0, i * chunk, (i + 1) * chunk).unsqueeze(0);
            if ((int64_t)train_batches.size() < args.batch) {
                train_batches.push_back(seg);
            }
        }
        for (int64_t i = n_chunks - val_chunks; i < n_chunks; ++i) {
            val_batches.push_back(
                tokens.slice(0, i * chunk, (i + 1) * chunk).unsqueeze(0));
        }
        std::cout << "[Llama2] data: " << train_batches.size()
                  << " train + " << val_batches.size() << " val chunks\n";
    } catch (...) {
        std::cout << "[Llama2] no data at " << args.data
                  << " — model construction verified, skipping training\n";
    }

    float best_val = std::numeric_limits<float>::max();
    int64_t global_iter = 0;

    for (int64_t ep = 0; ep < args.epochs; ++ep) {
        double train_loss_sum = 0.0;
        int64_t steps = 0;

        for (auto& batch : train_batches) {
            auto data = batch.to(device);
            float loss = llama2_train_step(model, optimizer, data, tcfg, global_iter);
            train_loss_sum += loss;
            ++steps;
            ++global_iter;
        }

        // Validate
        float val_loss = 0.0f;
        if (!val_batches.empty()) {
            model->eval();
            torch::NoGradGuard ng;
            double vl = 0.0;
            for (auto& batch : val_batches) {
                auto data    = batch.to(device);
                auto tokens  = data.slice(1, 0, data.size(1) - 1);
                auto targets = data.slice(1, 1, data.size(1));
                model->forward(tokens, targets);
                vl += model->last_loss.item<double>();
            }
            val_loss = (float)(vl / (double)val_batches.size());
        }

        double train_loss_avg = steps > 0 ? train_loss_sum / steps : 0.0;
        std::cout << "[Llama2] epoch " << ep + 1 << "/" << args.epochs
                  << "  iter=" << global_iter
                  << "  train_loss=" << train_loss_avg
                  << "  val_loss=" << val_loss << "\n";

        if (val_loss < best_val || val_batches.empty()) {
            best_val = val_loss;
            torch::serialize::OutputArchive ar;
            model->save(ar);
            ar.save_to(args.save);
            std::cout << "[Llama2] saved → " << args.save << "\n";
        }
    }
    return 0;
}
