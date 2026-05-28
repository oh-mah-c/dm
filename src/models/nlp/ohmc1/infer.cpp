// OhmC1 inference CLI
// oh-mah-c, dm/OhmC1, 2026. [151]
//
// Usage:
//   ./ohmc1_infer --model ohmc1_best.pt --prompt "First Citizen:" [--tokens 200] [--temp 0.8]

#include "models/nlp/ohmc1/ohmc1.h"

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>

using namespace dm::models::nlp;

static void usage(const char* prog) {
    std::printf(
        "Usage: %s [options]\n"
        "  --model  <path>     path to saved model (e.g., ohmc1_best.pt)\n"
        "  --size   <tiny|small|medium|large>  model preset (default: tiny)\n"
        "  --prompt <string>   initial text to condition on\n"
        "  --tokens <int>      number of tokens to generate (default: 200)\n"
        "  --temp   <float>    temperature (default: 0.8)\n"
        "  --cuda              use CUDA if available\n",
        prog);
}

int main(int argc, char** argv) {
    std::string model_path = "ohmc1_best.pt";
    std::string size_str = "tiny";
    std::string prompt_str = "First Citizen:\n";
    int64_t max_tokens = 200;
    float temperature = 0.8f;
    bool use_cuda = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { usage(argv[0]); return 0; }
        else if (arg == "--model"  && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--size"   && i + 1 < argc) size_str = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) prompt_str = argv[++i];
        else if (arg == "--tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (arg == "--temp"   && i + 1 < argc) temperature = std::atof(argv[++i]);
        else if (arg == "--cuda")                   use_cuda = true;
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
    // For char-level, vocab was changed to 256 in training if we didn't use BPE
    model_cfg.vocab_size = 256;

    torch::Device device = torch::kCPU;
    if (use_cuda && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::printf("Using CUDA.\n");
    }

    std::printf("Loading model from %s...\n", model_path.c_str());
    
    // Load the entire model object
    OhmC1Model model(model_cfg);
    try {
        torch::load(model, model_path);
        model->to(device);
        model->eval();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Failed to load model: %s\n", e.what());
        return 1;
    }

    std::printf("Model loaded successfully.\n");

    // Convert prompt string to character-level tokens
    std::vector<int64_t> prompt_tokens;
    for (char c : prompt_str) {
        prompt_tokens.push_back(static_cast<int64_t>(static_cast<unsigned char>(c)));
    }

    std::printf("\n--- Generating ---\n");
    std::printf("%s", prompt_str.c_str());

    // Generate
    torch::NoGradGuard no_grad;
    auto generated_tokens = model->generate(prompt_tokens, max_tokens, temperature, 0.9f, -1);

    // Skip the prompt part that is returned, or if `generate` returns new tokens
    // Wait, the `generate` function typically returns *all* tokens or *new* tokens?
    // Usually it returns all tokens (prompt + generated). Let's print the newly generated ones.
    // If it returns only new tokens, we just print them.
    // We can just decode all generated tokens. Let's see if generate returns the full sequence.
    // We will just decode the ones after the prompt.

    // Let's assume it returns only the newly generated tokens, or it returns the full sequence.
    // I will decode the returned vector, but only starting from prompt_tokens.size() if the prefix matches,
    // otherwise I'll decode the whole vector. To be safe, let's just decode everything.
    
    std::string output_str;
    for (size_t i = 0; i < generated_tokens.size(); ++i) {
        output_str += static_cast<char>(generated_tokens[i]);
    }
    
    // If output starts with prompt, strip it so we don't print twice
    if (output_str.substr(0, prompt_str.size()) == prompt_str) {
        std::printf("%s", output_str.c_str() + prompt_str.size());
    } else {
        std::printf("%s", output_str.c_str());
    }
    
    std::printf("\n------------------\n");

    return 0;
}
