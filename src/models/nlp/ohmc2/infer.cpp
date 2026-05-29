#include "models/nlp/ohmc2/ohmc2.h"

#include <torch/torch.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace dm::models::nlp;

static void usage(const char* prog) {
    std::printf(
        "Usage: %s [--size tiny|small|medium|large] [--model PATH] "
        "[--prompt 1,2,3|text] [--tokens N] [--temp F] [--top-p F] "
        "[--vocab-size N] [--seqlen N] [--ids]\n"
        "       Optional CPU sizing: [--dim N] [--layers N] [--heads N] "
        "[--ffn N] [--experts N] [--kan-grid N] [--kan-order N]\n"
        "       Quality options: [--ffn-type swiglu|kan|kan_moe] [--rope 0|1] [--rope-theta F]\n",
        prog);
}

static OhmC2Config pick_config(const std::string& size) {
    if (size == "tiny") return OhmC2Config::tiny();
    if (size == "small") return OhmC2Config::small();
    if (size == "medium") return OhmC2Config::medium();
    if (size == "large") return OhmC2Config::large();
    throw std::runtime_error("unknown OhmC2 size: " + size);
}

static bool read_text(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

static int64_t meta_int(const std::string& s, const std::string& key, int64_t fallback) {
    auto p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return fallback;
    p = s.find(':', p);
    if (p == std::string::npos) return fallback;
    return std::stoll(s.substr(p + 1));
}

static double meta_double(const std::string& s, const std::string& key, double fallback) {
    auto p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return fallback;
    p = s.find(':', p);
    if (p == std::string::npos) return fallback;
    return std::stod(s.substr(p + 1));
}

static bool meta_bool(const std::string& s, const std::string& key, bool fallback) {
    auto p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return fallback;
    p = s.find(':', p);
    if (p == std::string::npos) return fallback;
    auto v = s.substr(p + 1, 8);
    if (v.find("true") != std::string::npos) return true;
    if (v.find("false") != std::string::npos) return false;
    return fallback;
}

static std::string meta_string(const std::string& s, const std::string& key,
                               const std::string& fallback) {
    auto p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return fallback;
    p = s.find(':', p);
    if (p == std::string::npos) return fallback;
    auto a = s.find('"', p + 1);
    if (a == std::string::npos) return fallback;
    auto b = s.find('"', a + 1);
    if (b == std::string::npos) return fallback;
    return s.substr(a + 1, b - a - 1);
}

static OhmC2Config config_from_metadata(const std::string& text) {
    auto c = OhmC2Config::cpu_quality();
    c.dim = meta_int(text, "dim", c.dim);
    c.n_layers = meta_int(text, "layers", c.n_layers);
    c.n_heads = meta_int(text, "heads", c.n_heads);
    c.ffn_dim = meta_int(text, "ffn", c.ffn_dim);
    c.ffn_type = meta_string(text, "ffn_type", c.ffn_type);
    c.n_experts = meta_int(text, "experts", c.n_experts);
    c.kan_grid = meta_int(text, "kan_grid", c.kan_grid);
    c.kan_order = meta_int(text, "kan_order", c.kan_order);
    c.vocab_size = meta_int(text, "vocab_size", c.vocab_size);
    c.max_seq = meta_int(text, "max_seq", c.max_seq);
    c.use_rope = meta_bool(text, "use_rope", c.use_rope);
    c.rope_theta = meta_double(text, "rope_theta", c.rope_theta);
    return c;
}

static std::vector<int64_t> parse_prompt(const std::string& s) {
    std::vector<int64_t> ids;
    bool numeric = !s.empty();
    for (char c : s) {
        if (!(std::isdigit(static_cast<unsigned char>(c)) || c == ',' ||
              c == '-' || std::isspace(static_cast<unsigned char>(c)))) {
            numeric = false;
            break;
        }
    }
    if (numeric) {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) ids.push_back(std::stoll(item));
        }
    } else {
        for (unsigned char c : s) ids.push_back(static_cast<int64_t>(c));
    }
    if (ids.empty()) ids = {1, 2, 3};
    return ids;
}

int main(int argc, char** argv) {
    std::string size = "tiny";
    std::string model_path;
    std::string prompt = "1,2,3";
    int64_t tokens = 16;
    int64_t vocab_size = 0;
    int64_t seqlen = 0;
    int64_t override_dim = 0;
    int64_t override_layers = 0;
    int64_t override_heads = 0;
    int64_t override_ffn = 0;
    int64_t override_experts = 0;
    int64_t override_kan_grid = 0;
    int64_t override_kan_order = 0;
    int64_t override_rope = -1;
    double override_rope_theta = 0.0;
    std::string override_ffn_type;
    float temp = 0.0f;
    float top_p = 0.9f;
    bool print_ids = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else if (a == "--size" && i + 1 < argc) size = argv[++i];
        else if ((a == "--model" || a == "--checkpoint") && i + 1 < argc) model_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (a == "--tokens" && i + 1 < argc) tokens = std::atoll(argv[++i]);
        else if (a == "--temp" && i + 1 < argc) temp = std::atof(argv[++i]);
        else if (a == "--top-p" && i + 1 < argc) top_p = std::atof(argv[++i]);
        else if (a == "--vocab-size" && i + 1 < argc) vocab_size = std::atoll(argv[++i]);
        else if (a == "--seqlen" && i + 1 < argc) seqlen = std::atoll(argv[++i]);
        else if (a == "--dim" && i + 1 < argc) override_dim = std::atoll(argv[++i]);
        else if (a == "--layers" && i + 1 < argc) override_layers = std::atoll(argv[++i]);
        else if (a == "--heads" && i + 1 < argc) override_heads = std::atoll(argv[++i]);
        else if (a == "--ffn" && i + 1 < argc) override_ffn = std::atoll(argv[++i]);
        else if (a == "--experts" && i + 1 < argc) override_experts = std::atoll(argv[++i]);
        else if (a == "--kan-grid" && i + 1 < argc) override_kan_grid = std::atoll(argv[++i]);
        else if (a == "--kan-order" && i + 1 < argc) override_kan_order = std::atoll(argv[++i]);
        else if (a == "--ffn-type" && i + 1 < argc) override_ffn_type = argv[++i];
        else if (a == "--rope" && i + 1 < argc) override_rope = std::atoll(argv[++i]);
        else if (a == "--rope-theta" && i + 1 < argc) override_rope_theta = std::atof(argv[++i]);
        else if (a == "--ids") print_ids = true;
        else { std::fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    OhmC2Config cfg;
    std::string meta;
    bool has_meta = !model_path.empty() && read_text(model_path + ".json", meta);
    if (has_meta) {
        cfg = config_from_metadata(meta);
        auto mismatch = [&](bool set, bool ok, const char* name) {
            if (set && !ok) {
                std::fprintf(stderr, "Checkpoint metadata mismatch for %s\n", name);
                std::exit(1);
            }
        };
        mismatch(vocab_size > 0, vocab_size == cfg.vocab_size, "vocab_size");
        mismatch(seqlen > 0, seqlen == cfg.max_seq, "max_seq");
        mismatch(override_dim > 0, override_dim == cfg.dim, "dim");
        mismatch(override_layers > 0, override_layers == cfg.n_layers, "layers");
        mismatch(override_heads > 0, override_heads == cfg.n_heads, "heads");
        mismatch(override_ffn > 0, override_ffn == cfg.ffn_dim, "ffn");
        mismatch(override_experts > 0, override_experts == cfg.n_experts, "experts");
        mismatch(override_kan_grid > 0, override_kan_grid == cfg.kan_grid, "kan_grid");
        mismatch(override_kan_order > 0, override_kan_order == cfg.kan_order, "kan_order");
        mismatch(!override_ffn_type.empty(), override_ffn_type == cfg.ffn_type, "ffn_type");
        mismatch(override_rope >= 0, (override_rope != 0) == cfg.use_rope, "use_rope");
        mismatch(override_rope_theta > 0.0,
                 std::abs(override_rope_theta - cfg.rope_theta) < 1e-9,
                 "rope_theta");
    } else {
        cfg = (size == "cpu_quality" || size == "cpu-quality")
            ? OhmC2Config::cpu_quality()
            : pick_config(size);
        if (vocab_size > 0) cfg.vocab_size = vocab_size;
        if (seqlen > 0) cfg.max_seq = seqlen;
        if (override_dim > 0) cfg.dim = override_dim;
        if (override_layers > 0) cfg.n_layers = override_layers;
        if (override_heads > 0) cfg.n_heads = override_heads;
        if (override_ffn > 0) cfg.ffn_dim = override_ffn;
        if (override_experts > 0) cfg.n_experts = override_experts;
        if (override_kan_grid > 0) cfg.kan_grid = override_kan_grid;
        if (override_kan_order > 0) cfg.kan_order = override_kan_order;
        if (!override_ffn_type.empty()) cfg.ffn_type = override_ffn_type;
        if (override_rope >= 0) cfg.use_rope = override_rope != 0;
        if (override_rope_theta > 0.0) cfg.rope_theta = override_rope_theta;
    }
    if (cfg.dim <= 0 || cfg.n_heads <= 0 || cfg.dim % cfg.n_heads != 0) {
        std::fprintf(stderr, "Invalid config: dim must be divisible by heads\n");
        return 1;
    }
    if (cfg.ffn_type != "swiglu" && cfg.ffn_type != "kan" && cfg.ffn_type != "kan_moe") {
        std::fprintf(stderr, "Invalid config: ffn_type must be swiglu, kan, or kan_moe\n");
        return 1;
    }

    auto model = OhmC2LLM(cfg);
    if (!model_path.empty()) {
        try {
            torch::load(model, model_path);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Failed to load model: %s\n", e.what());
            return 1;
        }
    }

    auto ids = parse_prompt(prompt);
    for (auto& id : ids) {
        if (id < 0) id = 0;
        id %= cfg.vocab_size;
    }

    torch::NoGradGuard ng;
    auto out = model->generate(ids, tokens, temp, top_p, -1);
    if (!print_ids && cfg.vocab_size == 256) {
        std::printf("OhmC2 generated text:\n");
        for (int64_t id : out) {
            unsigned char c = static_cast<unsigned char>(id & 0xff);
            std::putchar(static_cast<int>(c));
        }
        std::printf("\n");
    } else {
        std::printf("OhmC2 generated ids:");
        for (int64_t id : out) std::printf(" %lld", (long long)id);
        std::printf("\n");
    }
    return 0;
}
