// ─────────────────────────────────────────────────────────────────────────────
// ResNet training entry point
//
// Usage:
//   ./resnet_train [options]
//
//   --model    resnet18|resnet34|resnet50|resnet101|resnet152   (default: resnet50)
//   --epochs   <int>      (default: 90)
//   --lr       <float>    (default: 0.1  — paper Section 3.4)
//   --batch    <int>      (default: 128)
//   --classes  <int>      (default: 10   — MNIST demo)
//   --data     <path>     MNIST data root (default: ./data)
//   --save     <path>     checkpoint file (default: resnet_best.pt)
//   --cuda                use CUDA if available
//
// This file uses MNIST as a runnable out-of-the-box demo.
// For ImageNet, replace the dataset/loader construction below.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/resnet/resnet.h"

#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace dm::models::vision;

// ── CLI ──────────────────────────────────────────────────────────────────────
struct Args {
    std::string model      = "resnet50";
    int64_t     num_classes= 10;
    int64_t     epochs     = 90;
    double      lr         = 0.1;
    int64_t     batch_size = 128;
    std::string save_path  = "resnet_best.pt";
    std::string data_root  = "./data";
    bool        cuda       = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--model"   && i+1<argc) a.model       = argv[++i];
        else if (k == "--classes" && i+1<argc) a.num_classes  = std::stoll(argv[++i]);
        else if (k == "--epochs"  && i+1<argc) a.epochs       = std::stoll(argv[++i]);
        else if (k == "--lr"      && i+1<argc) a.lr           = std::stod (argv[++i]);
        else if (k == "--batch"   && i+1<argc) a.batch_size   = std::stoll(argv[++i]);
        else if (k == "--save"    && i+1<argc) a.save_path    = argv[++i];
        else if (k == "--data"    && i+1<argc) a.data_root    = argv[++i];
        else if (k == "--cuda")                a.cuda         = true;
    }
    return a;
}

// ── Data helpers ─────────────────────────────────────────────────────────────
// Collect DataLoader batches into a plain vector and resize 1ch28×28 → 3ch224×224
// so ResNet's conv1 (7×7 /2 → 112×112 → pool /2 → 56×56) makes sense.
// Replace this with your own dataset for real training.
template <typename Loader>
static std::vector<std::pair<torch::Tensor, torch::Tensor>>
collect_and_resize(Loader& loader) {
    std::vector<std::pair<torch::Tensor, torch::Tensor>> out;
    for (auto& batch : loader) {
        auto img = batch.data;
        // 1-channel → 3-channel
        if (img.size(1) == 1)
            img = img.expand({img.size(0), 3, img.size(2), img.size(3)}).contiguous();
        // resize to 224×224 (paper: 224×224 crops for ImageNet)
        img = torch::nn::functional::interpolate(
            img,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{224, 224})
                .mode(torch::kBilinear)
                .align_corners(false));
        out.emplace_back(img, batch.target);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    auto args   = parse_args(argc, argv);
    auto device = (args.cuda && torch::cuda::is_available())
                  ? torch::kCUDA : torch::kCPU;
    std::cout << "[ResNet] device: " << device << "\n";

    // ── Build model and wrap in AnyModule ────────────────────────────────────
    torch::nn::AnyModule model;
    if      (args.model == "resnet18")  model = torch::nn::AnyModule(make_resnet18 (args.num_classes));
    else if (args.model == "resnet34")  model = torch::nn::AnyModule(make_resnet34 (args.num_classes));
    else if (args.model == "resnet50")  model = torch::nn::AnyModule(make_resnet50 (args.num_classes));
    else if (args.model == "resnet101") model = torch::nn::AnyModule(make_resnet101(args.num_classes));
    else if (args.model == "resnet152") model = torch::nn::AnyModule(make_resnet152(args.num_classes));
    else throw std::invalid_argument("Unknown model: " + args.model);

    model.ptr()->to(device);

    int64_t n_params = 0;
    for (const auto& p : model.ptr()->parameters()) n_params += p.numel();
    std::cout << "[ResNet] " << args.model
              << "  classes=" << args.num_classes
              << "  params=" << n_params / 1'000'000 << "M\n";

    // ── Data — MNIST as runnable default ─────────────────────────────────────
    auto train_ds = torch::data::datasets::MNIST(
                        args.data_root,
                        torch::data::datasets::MNIST::Mode::kTrain)
                    .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<
                            torch::data::samplers::RandomSampler>(
                        std::move(train_ds), args.batch_size);

    auto val_ds   = torch::data::datasets::MNIST(
                        args.data_root,
                        torch::data::datasets::MNIST::Mode::kTest)
                    .map(torch::data::transforms::Stack<>());
    auto val_loader = torch::data::make_data_loader<
                          torch::data::samplers::SequentialSampler>(
                      std::move(val_ds), args.batch_size);

    std::cout << "[ResNet] loading data from " << args.data_root << " ...\n";
    auto train_batches = collect_and_resize(*train_loader);
    auto val_batches   = collect_and_resize(*val_loader);
    std::cout << "[ResNet] train=" << train_batches.size()
              << " batches  val=" << val_batches.size() << " batches\n";

    // ── Training config (paper Section 3.4) ──────────────────────────────────
    ResNetTrainConfig cfg;
    cfg.lr            = args.lr;
    cfg.batch_size    = args.batch_size;
    cfg.max_epochs    = args.epochs;
    cfg.device        = device;
    // Scale milestones proportionally: paper uses epochs 30,60 out of 90
    cfg.lr_milestones = {args.epochs * 30 / 90, args.epochs * 60 / 90};

    // ── Train ─────────────────────────────────────────────────────────────────
    resnet_train(model, cfg, train_batches, val_batches, args.save_path);

    return 0;
}
