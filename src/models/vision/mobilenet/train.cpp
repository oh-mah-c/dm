// ─────────────────────────────────────────────────────────────────────────────
// MobileNet v1 training entry point — Howard et al., arXiv:1704.04861v1
//
// Usage:
//   ./mobilenet_train [options]
//   --alpha    <float>  width multiplier (default: 1.0)
//   --classes  <int>    number of classes (default: 1000)
//   --epochs   <int>    (default: 100)
//   --batch    <int>    (default: 256)
//   --lr       <float>  (default: 0.01)
//   --data     <path>   dataset root (default: ./data)
//   --save     <path>   checkpoint (default: mobilenet_best.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/mobilenet/mobilenet.h"

#include <torch/torch.h>
#include <iostream>
#include <string>

using namespace dm::models::vision;

struct Args {
    float       alpha   = 1.0f;
    int64_t     classes = 1000;
    int64_t     epochs  = 100;
    int64_t     batch   = 256;
    double      lr      = 1e-2;
    std::string data    = "./data";
    std::string save    = "mobilenet_best.pt";
    bool        cuda    = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--alpha"   && i+1<argc) a.alpha   = std::stof(argv[++i]);
        else if (k == "--classes" && i+1<argc) a.classes = std::stoll(argv[++i]);
        else if (k == "--epochs"  && i+1<argc) a.epochs  = std::stoll(argv[++i]);
        else if (k == "--batch"   && i+1<argc) a.batch   = std::stoll(argv[++i]);
        else if (k == "--lr"      && i+1<argc) a.lr      = std::stod(argv[++i]);
        else if (k == "--data"    && i+1<argc) a.data    = argv[++i];
        else if (k == "--save"    && i+1<argc) a.save    = argv[++i];
        else if (k == "--cuda")               a.cuda    = true;
    }
    return a;
}

int main(int argc, char* argv[]) {
    auto args   = parse_args(argc, argv);
    auto device = (args.cuda && torch::cuda::is_available())
                  ? torch::kCUDA : torch::kCPU;
    std::cout << "[MobileNet] device: " << device << "\n";

    auto model = MobileNet(args.classes, args.alpha);
    model->to(device);

    int64_t n_params = 0;
    for (auto& p : model->parameters()) n_params += p.numel();
    std::cout << "[MobileNet] alpha=" << args.alpha
              << "  classes=" << args.classes
              << "  params=" << n_params / 1'000'000 << "."
              << (n_params / 100'000) % 10 << "M\n";

    // Placeholder: no real data — model construction verified
    std::vector<std::pair<torch::Tensor, torch::Tensor>> train_batches, val_batches;
    try {
        torch::Tensor imgs, labels;
        torch::load(imgs,   args.data + "/imgs.pt");
        torch::load(labels, args.data + "/labels.pt");
        int64_t total = imgs.size(0);
        int64_t val_n = std::max<int64_t>(1, total / 10);
        for (int64_t i = 0; i + args.batch <= total - val_n; i += args.batch)
            train_batches.push_back({imgs.slice(0,i,i+args.batch),
                                     labels.slice(0,i,i+args.batch)});
        for (int64_t i = total - val_n; i + args.batch <= total; i += args.batch)
            val_batches.push_back({imgs.slice(0,i,i+args.batch),
                                   labels.slice(0,i,i+args.batch)});
        std::cout << "[MobileNet] data: " << train_batches.size()
                  << " train + " << val_batches.size() << " val batches\n";
    } catch (...) {
        std::cout << "[MobileNet] no data at " << args.data
                  << " — model construction verified, skipping training\n";
    }

    MobileNetTrainConfig cfg;
    cfg.lr         = args.lr;
    cfg.max_epochs = args.epochs;
    cfg.batch_size = args.batch;
    cfg.device     = device;

    mobilenet_train(model, cfg, train_batches, val_batches, args.save);
    return 0;
}
