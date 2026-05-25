// ─────────────────────────────────────────────────────────────────────────────
// VAE training entry point
//
// Usage:
//   ./vae_train [options]
//   --input-dim  <int>   (default: 784  — 28×28 MNIST)
//   --hidden-dim <int>   (default: 500  — Section 5)
//   --latent-dim <int>   (default: 20   — Section 5)
//   --epochs     <int>   (default: 100)
//   --lr         <float> (default: 1e-3)
//   --batch      <int>   (default: 100  — Section 5: M=100)
//   --data       <path>  MNIST root (default: ./data)
//   --save       <path>  checkpoint (default: vae_best.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/generative/vae/vae.h"

#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace dm::models::generative;

struct Args {
    int64_t  input_dim  = 784;
    int64_t  hidden_dim = 500;
    int64_t  latent_dim = 20;
    int64_t  epochs     = 100;
    double   lr         = 1e-3;
    int64_t  batch_size = 100;
    std::string data_root  = "./data";
    std::string save_path  = "vae_best.pt";
    bool     cuda       = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--input-dim"  && i+1<argc) a.input_dim  = std::stoll(argv[++i]);
        else if (k == "--hidden-dim" && i+1<argc) a.hidden_dim = std::stoll(argv[++i]);
        else if (k == "--latent-dim" && i+1<argc) a.latent_dim = std::stoll(argv[++i]);
        else if (k == "--epochs"     && i+1<argc) a.epochs     = std::stoll(argv[++i]);
        else if (k == "--lr"         && i+1<argc) a.lr         = std::stod(argv[++i]);
        else if (k == "--batch"      && i+1<argc) a.batch_size = std::stoll(argv[++i]);
        else if (k == "--data"       && i+1<argc) a.data_root  = argv[++i];
        else if (k == "--save"       && i+1<argc) a.save_path  = argv[++i];
        else if (k == "--cuda")                   a.cuda       = true;
    }
    return a;
}

int main(int argc, char* argv[]) {
    auto args   = parse_args(argc, argv);
    auto device = (args.cuda && torch::cuda::is_available())
                  ? torch::kCUDA : torch::kCPU;
    std::cout << "[VAE] device: " << device << "\n";

    auto model = VAE(args.input_dim, args.hidden_dim, args.latent_dim);
    model->to(device);

    int64_t n_params = 0;
    for (auto& p : model->parameters()) n_params += p.numel();
    std::cout << "[VAE] input=" << args.input_dim
              << "  hidden=" << args.hidden_dim
              << "  latent=" << args.latent_dim
              << "  params=" << n_params / 1000 << "K\n";

    // Load MNIST (binarised on the fly by thresholding)
    auto make_batches = [&](torch::data::datasets::MNIST::Mode mode) {
        std::vector<torch::Tensor> batches;
        try {
            auto ds = torch::data::datasets::MNIST(args.data_root, mode)
                          .map(torch::data::transforms::Stack<>());
            auto loader = torch::data::make_data_loader<
                              torch::data::samplers::RandomSampler>(
                          std::move(ds), args.batch_size);
            for (auto& batch : *loader) {
                // Normalise: [0,255] uint8 → [0,1] float, then flatten
                auto x = batch.data.to(torch::kFloat) / 255.0f;
                x = x.view({x.size(0), -1});
                batches.push_back(x);
            }
        } catch (const c10::Error&) {
            // MNIST data not present — return empty
        }
        return batches;
    };

    std::cout << "[VAE] loading MNIST from " << args.data_root << " ...\n";
    auto train_batches = make_batches(torch::data::datasets::MNIST::Mode::kTrain);
    auto val_batches   = make_batches(torch::data::datasets::MNIST::Mode::kTest);
    std::cout << "[VAE] train=" << train_batches.size()
              << "  val=" << val_batches.size() << " batches\n";

    VAETrainConfig cfg;
    cfg.lr         = args.lr;
    cfg.batch_size = args.batch_size;
    cfg.max_epochs = args.epochs;
    cfg.device     = device;

    vae_train(model, cfg, train_batches, val_batches, args.save_path);
    return 0;
}
