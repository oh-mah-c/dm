// ─────────────────────────────────────────────────────────────────────────────
// VGG training entry point
//
// Usage:
//   ./vgg_train [options]
//   --model  vgg_a|vgg_b|vgg_c|vgg16|vgg19   (default: vgg16)
//   --epochs <int>     (default: 74  — paper Section 3.1)
//   --lr     <float>   (default: 0.01)
//   --batch  <int>     (default: 64  — reduced from paper's 256 for local use)
//   --classes <int>    (default: 10  — MNIST demo)
//   --data   <path>    MNIST root    (default: ./data)
//   --save   <path>    checkpoint    (default: vgg_best.pt)
//   --cuda             use CUDA if available
//   --bn               enable BatchNorm (not in original paper)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/vgg/vgg.h"

#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace dm::models::vision;

struct Args {
    std::string model      = "vgg16";
    int64_t     num_classes= 10;
    int64_t     epochs     = 74;
    double      lr         = 0.01;
    int64_t     batch_size = 64;
    std::string save_path  = "vgg_best.pt";
    std::string data_root  = "./data";
    bool        cuda       = false;
    bool        bn         = false;
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
        else if (k == "--bn")                  a.bn           = true;
    }
    return a;
}

template <typename Loader>
static std::vector<std::pair<torch::Tensor, torch::Tensor>>
collect_and_resize(Loader& loader) {
    std::vector<std::pair<torch::Tensor, torch::Tensor>> out;
    for (auto& batch : loader) {
        auto img = batch.data;
        // 1-channel → 3-channel
        if (img.size(1) == 1)
            img = img.expand({img.size(0), 3, img.size(2), img.size(3)}).contiguous();
        // Paper: "input to our ConvNets is a fixed-size 224×224 RGB image"
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

int main(int argc, char* argv[]) {
    auto args   = parse_args(argc, argv);
    auto device = (args.cuda && torch::cuda::is_available())
                  ? torch::kCUDA : torch::kCPU;
    std::cout << "[VGG] device: " << device << "\n";

    // Build model
    torch::nn::AnyModule model;
    if      (args.model == "vgg_a")  model = torch::nn::AnyModule(make_vgg_a (args.num_classes, args.bn));
    else if (args.model == "vgg_b")  model = torch::nn::AnyModule(make_vgg_b (args.num_classes, args.bn));
    else if (args.model == "vgg_c")  model = torch::nn::AnyModule(make_vgg_c (args.num_classes, args.bn));
    else if (args.model == "vgg16")  model = torch::nn::AnyModule(make_vgg16 (args.num_classes, args.bn));
    else if (args.model == "vgg19")  model = torch::nn::AnyModule(make_vgg19 (args.num_classes, args.bn));
    else throw std::invalid_argument("Unknown model: " + args.model);

    model.ptr()->to(device);

    int64_t n_params = 0;
    for (const auto& p : model.ptr()->parameters()) n_params += p.numel();
    std::cout << "[VGG] built " << args.model
              << "  classes=" << args.num_classes
              << "  params=" << n_params / 1'000'000 << "M\n";

    // MNIST as runnable default (replace with ImageNet loader for real training)
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

    std::cout << "[VGG] loading data from " << args.data_root << " ...\n";
    auto train_batches = collect_and_resize(*train_loader);
    auto val_batches   = collect_and_resize(*val_loader);
    std::cout << "[VGG] train=" << train_batches.size()
              << "  val=" << val_batches.size() << " batches\n";

    // Training config (paper Section 3.1 defaults)
    VGGTrainConfig cfg;
    cfg.lr            = args.lr;
    cfg.batch_size    = args.batch_size;
    cfg.max_epochs    = args.epochs;
    cfg.device        = device;
    cfg.lr_milestones = {args.epochs * 25 / 74, args.epochs * 50 / 74};

    vgg_train(model, cfg, train_batches, val_batches, args.save_path);
    return 0;
}
