// ─────────────────────────────────────────────────────────────────────────────
// Swin Transformer training entry point
//
// Usage:
//   ./swin_train [options]
//   --model  swin_t|swin_s|swin_b|swin_l   (default: swin_t)
//   --epochs <int>       (default: 300 — paper Section 4.1)
//   --lr     <float>     (default: 0.001)
//   --batch  <int>       (default: 64 — reduced from paper's 1024 for local use)
//   --classes <int>      (default: 10  — MNIST demo)
//   --img-size <int>     (default: 224)
//   --data   <path>      MNIST root    (default: ./data)
//   --save   <path>      checkpoint    (default: swin_best.pt)
//   --cuda               use CUDA if available
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/swin/swin.h"

#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace dm::models::vision;

struct Args {
    std::string model      = "swin_t";
    int64_t     num_classes= 10;
    int64_t     epochs     = 300;
    double      lr         = 1e-3;
    int64_t     batch_size = 64;
    int64_t     img_size   = 224;
    std::string save_path  = "swin_best.pt";
    std::string data_root  = "./data";
    bool        cuda       = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--model"    && i+1<argc) a.model       = argv[++i];
        else if (k == "--classes"  && i+1<argc) a.num_classes  = std::stoll(argv[++i]);
        else if (k == "--epochs"   && i+1<argc) a.epochs       = std::stoll(argv[++i]);
        else if (k == "--lr"       && i+1<argc) a.lr           = std::stod (argv[++i]);
        else if (k == "--batch"    && i+1<argc) a.batch_size   = std::stoll(argv[++i]);
        else if (k == "--img-size" && i+1<argc) a.img_size     = std::stoll(argv[++i]);
        else if (k == "--save"     && i+1<argc) a.save_path    = argv[++i];
        else if (k == "--data"     && i+1<argc) a.data_root    = argv[++i];
        else if (k == "--cuda")                 a.cuda         = true;
    }
    return a;
}

template <typename Loader>
static std::vector<std::pair<torch::Tensor, torch::Tensor>>
collect_and_resize(Loader& loader, int64_t img_size) {
    std::vector<std::pair<torch::Tensor, torch::Tensor>> out;
    for (auto& batch : loader) {
        auto img = batch.data;
        if (img.size(1) == 1)
            img = img.expand({img.size(0), 3, img.size(2), img.size(3)}).contiguous();
        img = torch::nn::functional::interpolate(
            img,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{img_size, img_size})
                .mode(torch::kBilinear)
                .align_corners(false));
        out.emplace_back(img, batch.target);
    }
    return out;
}

int main(int argc, char* argv[]) {
    auto args   = parse_args(argc, argv);
    auto device = (args.cuda && torch::cuda::is_available())
                  ? torch::kCUDA : torch::kCPU;
    std::cout << "[Swin] device: " << device << "\n";

    // Build model
    torch::nn::AnyModule model;
    if      (args.model == "swin_t") model = torch::nn::AnyModule(make_swin_t(args.num_classes, args.img_size));
    else if (args.model == "swin_s") model = torch::nn::AnyModule(make_swin_s(args.num_classes, args.img_size));
    else if (args.model == "swin_b") model = torch::nn::AnyModule(make_swin_b(args.num_classes, args.img_size));
    else if (args.model == "swin_l") model = torch::nn::AnyModule(make_swin_l(args.num_classes, args.img_size));
    else throw std::invalid_argument("Unknown model: " + args.model);

    model.ptr()->to(device);

    int64_t n_params = 0;
    for (const auto& p : model.ptr()->parameters()) n_params += p.numel();
    std::cout << "[Swin] built " << args.model
              << "  classes=" << args.num_classes
              << "  params=" << n_params / 1'000'000 << "M\n";

    // MNIST as runnable default
    auto train_ds = torch::data::datasets::MNIST(
                        args.data_root,
                        torch::data::datasets::MNIST::Mode::kTrain)
                    .map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<
                            torch::data::samplers::RandomSampler>(
                        std::move(train_ds), args.batch_size);

    auto val_ds = torch::data::datasets::MNIST(
                        args.data_root,
                        torch::data::datasets::MNIST::Mode::kTest)
                  .map(torch::data::transforms::Stack<>());
    auto val_loader = torch::data::make_data_loader<
                          torch::data::samplers::SequentialSampler>(
                      std::move(val_ds), args.batch_size);

    std::cout << "[Swin] loading data from " << args.data_root << " ...\n";
    auto train_batches = collect_and_resize(*train_loader, args.img_size);
    auto val_batches   = collect_and_resize(*val_loader,   args.img_size);
    std::cout << "[Swin] train=" << train_batches.size()
              << "  val=" << val_batches.size() << " batches\n";

    SwinTrainConfig cfg;
    cfg.lr           = args.lr;
    cfg.batch_size   = args.batch_size;
    cfg.max_epochs   = args.epochs;
    cfg.device       = device;

    swin_train(model, cfg, train_batches, val_batches, args.save_path);
    return 0;
}
