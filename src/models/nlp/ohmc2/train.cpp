#include "models/nlp/ohmc2/ohmc2.h"

#include <torch/torch.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace dm::models::nlp;

static void usage(const char* prog) {
    std::printf(
        "Usage: %s [--size tiny|small|medium|large] [--seqlen N] [--epochs N] "
        "[--steps N] [--max-steps N] [--batch N] [--lr F] [--dataset PATH] "
        "[--save PATH] [--resume PATH] [--eval-interval N] [--save-interval N]\n"
        "       Optional CPU sizing: [--dim N] [--layers N] [--heads N] "
        "[--ffn N] [--experts N] [--kan-grid N] [--kan-order N] [--vocab-size N]\n"
        "       Quality options: [--ffn-type swiglu|kan|kan_moe] [--rope 0|1] [--rope-theta F]\n",
        prog);
}

static void write_metadata(const std::string& path, const OhmC2Config& cfg) {
    std::ofstream f(path + ".json");
    f << "{\n"
      << "  \"format\": \"ohmc2_checkpoint_metadata_v1\",\n"
      << "  \"dim\": " << cfg.dim << ",\n"
      << "  \"layers\": " << cfg.n_layers << ",\n"
      << "  \"heads\": " << cfg.n_heads << ",\n"
      << "  \"ffn\": " << cfg.ffn_dim << ",\n"
      << "  \"ffn_type\": \"" << cfg.ffn_type << "\",\n"
      << "  \"experts\": " << cfg.n_experts << ",\n"
      << "  \"kan_grid\": " << cfg.kan_grid << ",\n"
      << "  \"kan_order\": " << cfg.kan_order << ",\n"
      << "  \"vocab_size\": " << cfg.vocab_size << ",\n"
      << "  \"max_seq\": " << cfg.max_seq << ",\n"
      << "  \"use_rope\": " << (cfg.use_rope ? "true" : "false") << ",\n"
      << "  \"rope_theta\": " << cfg.rope_theta << "\n"
      << "}\n";
}

int main(int argc, char** argv) {
    std::string size = "tiny";
    std::string dataset;
    std::string save = "ohmc2_best.pt";
    int64_t epochs = 1;
    int64_t batch = 2;
    int64_t seqlen = 0;
    int64_t steps_per_epoch = 2;
    int64_t max_steps = 0;
    int64_t eval_interval = 0;
    int64_t save_interval = 0;
    int64_t override_dim = 0;
    int64_t override_layers = 0;
    int64_t override_heads = 0;
    int64_t override_ffn = 0;
    int64_t override_experts = 0;
    int64_t override_kan_grid = 0;
    int64_t override_kan_order = 0;
    int64_t override_vocab = 0;
    int64_t override_rope = -1;
    double override_rope_theta = 0.0;
    std::string override_ffn_type;
    std::string resume;
    OhmC2TrainConfig train_cfg;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else if (a == "--size" && i + 1 < argc) size = argv[++i];
        else if (a == "--seqlen" && i + 1 < argc) seqlen = std::atoll(argv[++i]);
        else if (a == "--epochs" && i + 1 < argc) epochs = std::atoll(argv[++i]);
        else if (a == "--steps" && i + 1 < argc) steps_per_epoch = std::atoll(argv[++i]);
        else if (a == "--max-steps" && i + 1 < argc) max_steps = std::atoll(argv[++i]);
        else if (a == "--eval-interval" && i + 1 < argc) eval_interval = std::atoll(argv[++i]);
        else if (a == "--save-interval" && i + 1 < argc) save_interval = std::atoll(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) batch = std::atoll(argv[++i]);
        else if (a == "--lr" && i + 1 < argc) train_cfg.lr = std::atof(argv[++i]);
        else if (a == "--dataset" && i + 1 < argc) dataset = argv[++i];
        else if (a == "--save" && i + 1 < argc) save = argv[++i];
        else if (a == "--resume" && i + 1 < argc) resume = argv[++i];
        else if (a == "--dim" && i + 1 < argc) override_dim = std::atoll(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) override_layers = std::atoll(argv[++i]);
        else if (a == "--heads" && i + 1 < argc) override_heads = std::atoll(argv[++i]);
        else if (a == "--ffn" && i + 1 < argc) override_ffn = std::atoll(argv[++i]);
        else if (a == "--experts" && i + 1 < argc) override_experts = std::atoll(argv[++i]);
        else if (a == "--kan-grid" && i + 1 < argc) override_kan_grid = std::atoll(argv[++i]);
        else if (a == "--kan-order" && i + 1 < argc) override_kan_order = std::atoll(argv[++i]);
        else if (a == "--vocab-size" && i + 1 < argc) override_vocab = std::atoll(argv[++i]);
        else if (a == "--ffn-type" && i + 1 < argc) override_ffn_type = argv[++i];
        else if (a == "--rope" && i + 1 < argc) override_rope = std::atoll(argv[++i]);
        else if (a == "--rope-theta" && i + 1 < argc) override_rope_theta = std::atof(argv[++i]);
        else { std::fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    OhmC2Config cfg;
    if (size == "cpu_quality" || size == "cpu-quality") cfg = OhmC2Config::cpu_quality();
    else if (size == "tiny") cfg = OhmC2Config::tiny();
    else if (size == "small") cfg = OhmC2Config::small();
    else if (size == "medium") cfg = OhmC2Config::medium();
    else if (size == "large") cfg = OhmC2Config::large();
    else { std::fprintf(stderr, "Unknown size: %s\n", size.c_str()); return 1; }
    if (seqlen > 0) cfg.max_seq = seqlen;
    if (override_dim > 0) cfg.dim = override_dim;
    if (override_layers > 0) cfg.n_layers = override_layers;
    if (override_heads > 0) cfg.n_heads = override_heads;
    if (override_ffn > 0) cfg.ffn_dim = override_ffn;
    if (override_experts > 0) cfg.n_experts = override_experts;
    if (override_kan_grid > 0) cfg.kan_grid = override_kan_grid;
    if (override_kan_order > 0) cfg.kan_order = override_kan_order;
    if (override_vocab > 0) cfg.vocab_size = override_vocab;
    if (!override_ffn_type.empty()) cfg.ffn_type = override_ffn_type;
    if (override_rope >= 0) cfg.use_rope = override_rope != 0;
    if (override_rope_theta > 0.0) cfg.rope_theta = override_rope_theta;
    if (cfg.dim <= 0 || cfg.n_heads <= 0 || cfg.dim % cfg.n_heads != 0) {
        std::fprintf(stderr, "Invalid config: dim must be divisible by heads\n");
        return 1;
    }
    if (cfg.ffn_type != "swiglu" && cfg.ffn_type != "kan" && cfg.ffn_type != "kan_moe") {
        std::fprintf(stderr, "Invalid config: ffn_type must be swiglu, kan, or kan_moe\n");
        return 1;
    }
    if (cfg.ffn_dim <= 0 || cfg.n_layers <= 0 || cfg.n_experts <= 0 ||
        cfg.kan_grid <= 0 || cfg.kan_order <= 0 || cfg.max_seq <= 1) {
        std::fprintf(stderr, "Invalid config: positive ffn/layers/experts/KAN/seq required\n");
        return 1;
    }

    std::vector<int64_t> data;
    if (!dataset.empty()) {
        FILE* f = std::fopen(dataset.c_str(), "rb");
        if (!f) { std::perror(dataset.c_str()); return 1; }
        
        if (dataset.length() >= 4 && dataset.substr(dataset.length() - 4) == ".bin") {
            int32_t token;
            while (std::fread(&token, sizeof(int32_t), 1, f) == 1) {
                data.push_back((int64_t)token);
            }
            std::printf("Loaded %zu subword tokens from binary file\n", data.size());
        } else {
            int c;
            while ((c = std::fgetc(f)) != EOF) data.push_back((unsigned char)c);
            if (override_vocab <= 0) cfg.vocab_size = 256;
            std::printf("Loaded %zu byte-level tokens\n", data.size());
        }
        std::fclose(f);
    }

    std::printf("OhmC2 CPU-first LLM size=%s dim=%lld layers=%lld heads=%lld ffn=%lld ffn_type=%s experts=%lld G=%lld vocab=%lld seq=%lld rope=%d\n",
        size.c_str(), (long long)cfg.dim, (long long)cfg.n_layers,
        (long long)cfg.n_heads, (long long)cfg.ffn_dim, cfg.ffn_type.c_str(),
        (long long)cfg.n_experts,
        (long long)cfg.kan_grid, (long long)cfg.vocab_size,
        (long long)cfg.max_seq, cfg.use_rope ? 1 : 0);

    auto model = OhmC2LLM(cfg);
    if (!resume.empty()) {
        try {
            torch::load(model, resume);
            std::printf("Resumed model from %s\n", resume.c_str());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Failed to resume: %s\n", e.what());
            return 1;
        }
    }
    auto opt = make_ohmc2_optimizer(model, train_cfg);
    int64_t step = 0;
    float best = 1e30f;

    int64_t train_start = 0;
    int64_t train_end = (int64_t)data.size();
    int64_t val_start = train_end;
    if (!data.empty() && data.size() > (size_t)(cfg.max_seq + 2) * 10) {
        val_start = std::max<int64_t>(cfg.max_seq + 2, (int64_t)(data.size() * 0.95));
        train_end = val_start;
    }

    auto sample_tokens = [&](int64_t start_min, int64_t end_max) {
        if (!data.empty()) {
            auto t = torch::empty({batch, cfg.max_seq + 1}, torch::kLong);
            auto* p = t.data_ptr<int64_t>();
            int64_t max_start = end_max - cfg.max_seq - 1;
            for (int64_t b = 0; b < batch; ++b) {
                int64_t span = std::max<int64_t>(1, max_start - start_min);
                int64_t start = start_min + (std::rand() % span);
                for (int64_t i = 0; i <= cfg.max_seq; ++i)
                    p[b * (cfg.max_seq + 1) + i] = data[start + i];
            }
            return t;
        }
        return torch::randint(0, cfg.vocab_size, {batch, cfg.max_seq + 1},
            torch::TensorOptions().dtype(torch::kLong));
    };

    auto eval_loss = [&]() {
        torch::NoGradGuard ng;
        model->eval();
        auto tokens = sample_tokens(val_start, (int64_t)data.size());
        int64_t T = tokens.size(1) - 1;
        model->forward(tokens.slice(1, 0, T), tokens.slice(1, 1, T + 1));
        return model->last_loss.item<float>();
    };

    const int64_t total_steps = max_steps > 0 ? max_steps : epochs * steps_per_epoch;
    for (int64_t s = 0; s < total_steps; ++s) {
            torch::Tensor tokens;
            if (!data.empty()) {
                if ((int64_t)data.size() <= cfg.max_seq + 1) {
                    std::fprintf(stderr, "Dataset too small for seq_len\n");
                    return 1;
                }
                tokens = sample_tokens(train_start, train_end);
            } else {
                tokens = sample_tokens(0, 0);
            }
            float loss = ohmc2_train_step(model, opt, tokens, train_cfg, step++);
            std::printf("step %lld train_loss %.4f lr %.2e",
                (long long)step, loss, ohmc2_lr_schedule(step, train_cfg));
            if (eval_interval > 0 && step % eval_interval == 0 && val_start < (int64_t)data.size())
                std::printf(" val_loss %.4f", eval_loss());
            std::printf("\n");
            if (loss < best) {
                best = loss;
                torch::save(model, save);
                write_metadata(save, cfg);
            } else if (save_interval > 0 && step % save_interval == 0) {
                torch::save(model, save);
                write_metadata(save, cfg);
            }
    }
    std::printf("Training complete. best=%.4f saved=%s\n", best, save.c_str());
    return 0;
}
