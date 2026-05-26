// ─────────────────────────────────────────────────────────────────────────────
// LoRA fine-tuning entry point — Hu et al., arXiv:2106.09685v2, 2021
//
// Demonstrates LoRA adaptation on a small language model (toy GPT-like decoder).
// In production, replace ToyLM with a real pre-trained model (GPT-2, LLaMA, etc.)
// and load pre-trained weights before calling inject_lora().
//
// Usage:
//   ./lora_train [options]
//   --rank    <int>     LoRA rank r                  (default: 4)
//   --alpha   <float>   LoRA alpha α                 (default: 4.0)
//   --layers  <int>     toy model transformer layers (default: 4)
//   --dim     <int>     toy model d_model            (default: 128)
//   --seqlen  <int>     training segment length      (default: 64)
//   --epochs  <int>                                  (default: 5)
//   --batch   <int>     batch size                   (default: 8)
//   --lr      <float>   peak learning rate           (default: 1e-4)
//   --warmup  <int>     warmup steps                 (default: 100)
//   --maxiter <int>     total training steps         (default: 2000)
//   --targets <str>     comma-separated name patterns to apply LoRA to
//                       (default: "W_q,W_v")
//   --save    <path>    checkpoint path (LoRA weights only) (default: lora_best.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/lora/lora.h"

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>

using namespace dm::models::nlp;

// ─── Toy decoder-only LM with LoRA on W_q and W_v ────────────────────────
// Build the model with LoRALinear directly for W_q and W_v (rather than
// injecting post-hoc), which avoids the member-holder aliasing issue and
// is the pattern recommended for new models.  W_k and W_o are plain Linear
// and remain frozen.
struct ToyAttentionImpl : torch::nn::Module {
    ToyAttentionImpl(int64_t dim, int64_t heads, const LoRAConfig& lora_cfg)
        : n_heads(heads), d_head(dim/heads) {
        W_q = register_module("W_q", LoRALinear(dim, dim, lora_cfg));
        W_v = register_module("W_v", LoRALinear(dim, dim, lora_cfg));
        W_k = register_module("W_k",
                  torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
        W_o = register_module("W_o",
                  torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
        // Freeze W_k and W_o
        W_k->weight.set_requires_grad(false);
        W_o->weight.set_requires_grad(false);
    }
    torch::Tensor forward(torch::Tensor x) {
        int64_t B = x.size(0), T = x.size(1), D = x.size(2);
        auto q = W_q->forward(x).view({B, T, n_heads, d_head}).transpose(1,2);
        auto k = W_k->forward(x).view({B, T, n_heads, d_head}).transpose(1,2);
        auto v = W_v->forward(x).view({B, T, n_heads, d_head}).transpose(1,2);
        auto scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(d_head)));
        auto attn  = q.matmul(k.transpose(-2,-1)) * scale;
        auto mask  = torch::triu(torch::ones({T,T}, x.options()), 1).to(torch::kBool);
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9f);
        attn = torch::softmax(attn, -1);
        auto out = attn.matmul(v).transpose(1,2).contiguous().view({B, T, D});
        return W_o->forward(out);
    }
    LoRALinear W_q{nullptr}, W_v{nullptr};
    torch::nn::Linear W_k{nullptr}, W_o{nullptr};
    int64_t n_heads, d_head;
};
TORCH_MODULE(ToyAttention);

struct ToyBlockImpl : torch::nn::Module {
    ToyBlockImpl(int64_t dim, int64_t heads, const LoRAConfig& lora_cfg) {
        ln1 = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
        ln2 = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
        attn = register_module("attn", ToyAttention(dim, heads, lora_cfg));
        fc1  = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(dim, 4*dim).bias(true)));
        fc2  = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(4*dim, dim).bias(true)));
        // Freeze FFN
        fc1->weight.set_requires_grad(false);
        fc1->bias.set_requires_grad(false);
        fc2->weight.set_requires_grad(false);
        fc2->bias.set_requires_grad(false);
    }
    torch::Tensor forward(torch::Tensor x) {
        x = x + attn->forward(ln1->forward(x));
        x = x + fc2->forward(torch::gelu(fc1->forward(ln2->forward(x))));
        return x;
    }
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    ToyAttention attn{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};
TORCH_MODULE(ToyBlock);

struct ToyLMImpl : torch::nn::Module {
    ToyLMImpl(int64_t vocab, int64_t dim, int64_t layers, int64_t heads,
              const LoRAConfig& lora_cfg) {
        embed = register_module("embed", torch::nn::Embedding(vocab, dim));
        embed->weight.set_requires_grad(false);  // freeze embedding
        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int64_t i = 0; i < layers; ++i)
            blocks->push_back(ToyBlock(dim, heads, lora_cfg));
        ln_out = register_module("ln_out", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
        head   = register_module("head", torch::nn::Linear(torch::nn::LinearOptions(dim, vocab).bias(false)));
        head->weight.set_requires_grad(false);
    }
    torch::Tensor forward(torch::Tensor tokens) {
        auto x = embed->forward(tokens);
        for (int64_t i = 0; i < (int64_t)blocks->size(); ++i)
            x = blocks->at<ToyBlockImpl>(i).forward(x);
        return head->forward(ln_out->forward(x));
    }
    torch::nn::Embedding   embed{nullptr};
    torch::nn::ModuleList  blocks{nullptr};
    torch::nn::LayerNorm   ln_out{nullptr};
    torch::nn::Linear      head{nullptr};
};
TORCH_MODULE(ToyLM);

// ─── CLI args ──────────────────────────────────────────────────────────────
struct Args {
    int64_t     rank    = 4;
    double      alpha   = 4.0;
    int64_t     layers  = 4;
    int64_t     dim     = 128;
    int64_t     heads   = 4;
    int64_t     vocab   = 1000;
    int64_t     seqlen  = 64;
    int64_t     epochs  = 5;
    int64_t     batch   = 8;
    double      lr      = 1e-4;
    int64_t     warmup  = 100;
    int64_t     maxiter = 2000;
    std::string targets = "W_q,W_v";
    std::string save    = "lora_best.pt";
    bool        cuda    = false;
};

static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(tok);
    return out;
}

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--rank"    && i+1<argc) a.rank    = std::stoll(argv[++i]);
        else if (k == "--alpha"   && i+1<argc) a.alpha   = std::stod(argv[++i]);
        else if (k == "--layers"  && i+1<argc) a.layers  = std::stoll(argv[++i]);
        else if (k == "--dim"     && i+1<argc) a.dim     = std::stoll(argv[++i]);
        else if (k == "--seqlen"  && i+1<argc) a.seqlen  = std::stoll(argv[++i]);
        else if (k == "--epochs"  && i+1<argc) a.epochs  = std::stoll(argv[++i]);
        else if (k == "--batch"   && i+1<argc) a.batch   = std::stoll(argv[++i]);
        else if (k == "--lr"      && i+1<argc) a.lr      = std::stod(argv[++i]);
        else if (k == "--warmup"  && i+1<argc) a.warmup  = std::stoll(argv[++i]);
        else if (k == "--maxiter" && i+1<argc) a.maxiter = std::stoll(argv[++i]);
        else if (k == "--targets" && i+1<argc) a.targets = argv[++i];
        else if (k == "--save"    && i+1<argc) a.save    = argv[++i];
        else if (k == "--cuda")                a.cuda    = true;
    }
    return a;
}

int main(int argc, char* argv[]) {
    auto a = parse_args(argc, argv);

    torch::Device device = a.cuda && torch::cuda::is_available()
                         ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << "\n";

    // Build toy model with LoRA baked into W_q and W_v
    LoRAConfig lora_cfg;
    lora_cfg.rank    = a.rank;
    lora_cfg.alpha   = a.alpha;
    lora_cfg.dropout = 0.0;

    auto model = ToyLM(a.vocab, a.dim, a.layers, a.heads, lora_cfg);
    model->to(device);

    int64_t total_params = 0, lora_params = 0;
    for (auto& item : model->named_parameters()) {
        total_params += item.value().numel();
        if (item.key().find("lora_A") != std::string::npos ||
            item.key().find("lora_B") != std::string::npos) {
            lora_params += item.value().numel();
        }
    }
    std::printf("Total params: %.3fM  LoRA trainable: %lld  (%.2f%%)\n",
        total_params / 1e6,
        static_cast<long long>(lora_params),
        100.0 * lora_params / total_params);

    // Collect only LoRA parameters for optimizer
    std::vector<torch::Tensor> lora_params_vec;
    for (auto& item : model->named_parameters()) {
        if (item.key().find("lora_A") != std::string::npos ||
            item.key().find("lora_B") != std::string::npos) {
            lora_params_vec.push_back(item.value());
        }
    }

    LoRATrainConfig tcfg;
    tcfg.lr          = a.lr;
    tcfg.warmup      = a.warmup;
    tcfg.total_steps = a.maxiter;

    auto opt = make_lora_optimizer(lora_params_vec, tcfg);

    double  best_loss = 1e9;
    int64_t step      = 0;

    model->train();
    for (int64_t epoch = 0; epoch < a.epochs && step < a.maxiter; ++epoch) {
        for (int64_t bi = 0; bi < 100 && step < a.maxiter; ++bi, ++step) {
            auto tokens = torch::randint(
                0, a.vocab,
                {a.batch, a.seqlen + 1},
                torch::TensorOptions().dtype(torch::kLong).device(device));

            // Update LR
            double lr_now = lora_lr_schedule(step, tcfg);
            for (auto& pg : opt.param_groups())
                static_cast<torch::optim::AdamOptions&>(pg.options()).lr(lr_now);

            opt.zero_grad();
            auto inp    = tokens.slice(1, 0, a.seqlen);
            auto tgt    = tokens.slice(1, 1, a.seqlen + 1);
            auto logits = model->forward(inp);
            auto loss   = torch::nn::functional::cross_entropy(
                              logits.reshape({-1, a.vocab}),
                              tgt.reshape({-1}));
            loss.backward();
            opt.step();

            double loss_val = loss.item().toDouble();

            if (step % 20 == 0) {
                std::printf("epoch %lld  step %lld  loss %.4f  bpc %.4f  lr %.2e\n",
                    static_cast<long long>(epoch),
                    static_cast<long long>(step),
                    loss_val,
                    loss_val / std::log(2.0),
                    lr_now);
                std::fflush(stdout);
            }

            if (loss_val < best_loss) {
                best_loss = loss_val;
                // Save LoRA params as a vector of tensors
                std::vector<torch::Tensor> lora_tensors;
                for (auto& item : model->named_parameters()) {
                    if (item.key().find("lora_A") != std::string::npos ||
                        item.key().find("lora_B") != std::string::npos) {
                        lora_tensors.push_back(item.value().detach().cpu());
                    }
                }
                torch::save(lora_tensors, a.save);
            }
        }
    }

    std::printf("Training complete. Best loss: %.4f  Saved: %s\n",
                best_loss, a.save.c_str());
    return 0;
}
