// ─────────────────────────────────────────────────────────────────────────────
// Transformer-XL training entry point — Dai et al., ACL 2019
//
// Usage:
//   ./transformer_xl_train [options]
//   --size    <tiny|base|large|xl|small>  (default: tiny)
//   --layers  <int>     override n_layers
//   --dim     <int>     override d_model
//   --heads   <int>     override n_heads
//   --dhead   <int>     override d_head
//   --dff     <int>     override d_inner (FFN)
//   --seglen  <int>     segment length           (default: 128)
//   --memlen  <int>     memory length            (default: same as seglen)
//   --epochs  <int>                              (default: 10)
//   --batch   <int>     batch size               (default: 32)
//   --lr      <float>   peak learning rate       (default: 2.5e-4)
//   --clip    <float>   gradient clip norm       (default: 0.25)
//   --warmup  <int>     warmup steps             (default: 0)
//   --maxiter <int>     total training steps     (default: 400000)
//   --data    <path>    dataset root             (default: ./data)
//   --save    <path>    checkpoint path          (default: transformer_xl_best.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/transformer_xl/transformer_xl.h"

#include <torch/torch.h>
#include <iostream>
#include <string>

using namespace dm::models::nlp;

struct Args {
    std::string size     = "tiny";
    int64_t     layers   = 0;
    int64_t     dim      = 0;
    int64_t     heads    = 0;
    int64_t     dhead    = 0;
    int64_t     dff      = 0;
    int64_t     seglen   = 0;
    int64_t     memlen   = 0;
    int64_t     epochs   = 10;
    int64_t     batch    = 32;
    double      lr       = 2.5e-4;
    double      clip     = 0.25;
    int64_t     warmup   = 0;
    int64_t     maxiter  = 400000;
    std::string data     = "./data";
    std::string save     = "transformer_xl_best.pt";
    bool        cuda     = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--size"    && i+1<argc) a.size    = argv[++i];
        else if (k == "--layers"  && i+1<argc) a.layers  = std::stoll(argv[++i]);
        else if (k == "--dim"     && i+1<argc) a.dim     = std::stoll(argv[++i]);
        else if (k == "--heads"   && i+1<argc) a.heads   = std::stoll(argv[++i]);
        else if (k == "--dhead"   && i+1<argc) a.dhead   = std::stoll(argv[++i]);
        else if (k == "--dff"     && i+1<argc) a.dff     = std::stoll(argv[++i]);
        else if (k == "--seglen"  && i+1<argc) a.seglen  = std::stoll(argv[++i]);
        else if (k == "--memlen"  && i+1<argc) a.memlen  = std::stoll(argv[++i]);
        else if (k == "--epochs"  && i+1<argc) a.epochs  = std::stoll(argv[++i]);
        else if (k == "--batch"   && i+1<argc) a.batch   = std::stoll(argv[++i]);
        else if (k == "--lr"      && i+1<argc) a.lr      = std::stod(argv[++i]);
        else if (k == "--clip"    && i+1<argc) a.clip    = std::stod(argv[++i]);
        else if (k == "--warmup"  && i+1<argc) a.warmup  = std::stoll(argv[++i]);
        else if (k == "--maxiter" && i+1<argc) a.maxiter = std::stoll(argv[++i]);
        else if (k == "--data"    && i+1<argc) a.data    = argv[++i];
        else if (k == "--save"    && i+1<argc) a.save    = argv[++i];
        else if (k == "--cuda")                a.cuda    = true;
    }
    return a;
}

static TransformerXLConfig build_config(const Args& a) {
    TransformerXLConfig cfg;
    if      (a.size == "base")  cfg = TransformerXLConfig::base();
    else if (a.size == "small") cfg = TransformerXLConfig::small();
    else if (a.size == "large") cfg = TransformerXLConfig::large();
    else if (a.size == "xl")    cfg = TransformerXLConfig::xl();
    else                        cfg = TransformerXLConfig::tiny();

    if (a.layers > 0) cfg.n_layers = a.layers;
    if (a.dim    > 0) cfg.d_model  = a.dim;
    if (a.heads  > 0) cfg.n_heads  = a.heads;
    if (a.dhead  > 0) cfg.d_head   = a.dhead;
    if (a.dff    > 0) cfg.d_inner  = a.dff;
    if (a.seglen > 0) cfg.seg_len  = a.seglen;
    if (a.memlen > 0) cfg.mem_len  = a.memlen;
    else              cfg.mem_len  = cfg.seg_len;  // default: M = L
    return cfg;
}

int main(int argc, char* argv[]) {
    auto a   = parse_args(argc, argv);
    auto cfg = build_config(a);

    torch::Device device = a.cuda && torch::cuda::is_available()
                         ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << "\n";

    auto model = TransformerXLModel(cfg);
    model->to(device);

    int64_t n_params = 0;
    for (auto& item : model->named_parameters())
        n_params += item.value().numel();
    std::cout << "Params: " << n_params / 1e6 << "M\n";
    std::cout << "Layers: " << cfg.n_layers
              << "  d_model: " << cfg.d_model
              << "  n_heads: " << cfg.n_heads
              << "  d_head: "  << cfg.d_head
              << "  d_inner: " << cfg.d_inner << "\n";
    std::cout << "seg_len: " << cfg.seg_len
              << "  mem_len: " << cfg.mem_len << "\n";

    TransformerXLTrainConfig tcfg;
    tcfg.lr          = a.lr;
    tcfg.clip        = a.clip;
    tcfg.warmup      = a.warmup;
    tcfg.total_steps = a.maxiter;

    auto opt = make_txl_optimizer(model, tcfg);

    // Synthetic data loop (replace with real data loader for actual training)
    double best_loss = 1e9;
    int64_t step     = 0;

    for (int64_t epoch = 0; epoch < a.epochs && step < a.maxiter; ++epoch) {
        // Each epoch: simulate 100 batches of random tokens
        // In real training, iterate over a corpus in segment chunks.
        bool new_doc = true;
        for (int64_t batch_i = 0; batch_i < 100 && step < a.maxiter;
             ++batch_i, ++step) {
            auto tokens = torch::randint(
                0, cfg.vocab_size,
                {a.batch, cfg.seg_len + 1},
                torch::TensorOptions().dtype(torch::kLong).device(device));

            auto result = txl_train_step(model, opt, tokens, tcfg, step, new_doc);
            new_doc     = false;
            double loss = result.second;

            if (step % 20 == 0) {
                std::printf("epoch %lld  step %lld  loss %.4f  bpc %.4f  lr %.2e\n",
                    static_cast<long long>(epoch),
                    static_cast<long long>(step),
                    loss,
                    loss / std::log(2.0),
                    txl_lr_schedule(step, tcfg));
                std::fflush(stdout);
            }

            if (loss < best_loss) {
                best_loss = loss;
                torch::save(model, a.save);
            }
        }
    }

    std::printf("Training complete. Best loss: %.4f  Saved: %s\n",
                best_loss, a.save.c_str());
    return 0;
}
