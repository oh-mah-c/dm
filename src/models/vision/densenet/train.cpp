// ─────────────────────────────────────────────────────────────────────────────
// DenseNet training entry point
//
// Usage:
//   ./densenet_train [options]
//
//   --model    densenet121|densenet169|densenet201|densenet264|
//              cifar-L<depth>-k<growth>[bc]   (default: densenet121)
//              Examples:
//                cifar-L40-k12        plain DenseNet L=40 k=12
//                cifar-L100-k12bc     DenseNet-BC L=100 k=12
//   --classes  <int>      (default: 10  — MNIST/CIFAR demo)
//   --epochs   <int>      (default: 90  — paper ImageNet; 300 for CIFAR)
//   --lr       <float>    (default: 0.1 — paper Section 4.2)
//   --batch    <int>      (default: 64  — paper CIFAR batch)
//   --data     <path>     MNIST data root (default: ./data)
//   --save     <path>     checkpoint file (default: densenet_best.pt)
//   --cuda                use CUDA if available
//
// Uses MNIST as an out-of-the-box runnable demo.
// For CIFAR-10/100 or ImageNet, replace the dataset loader section.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/densenet/densenet.h"

#include <torch/torch.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace dm::models::vision;

// ── CLI ──────────────────────────────────────────────────────────────────────
struct Args {
    std::string model      = "densenet121";
    int64_t     num_classes= 10;
    int64_t     epochs     = 90;
    double      lr         = 0.1;
    int64_t     batch_size = 64;
    std::string data_root  = "./data";
    std::string save_path  = "densenet_best.pt";
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
        else if (k == "--data"    && i+1<argc) a.data_root    = argv[++i];
        else if (k == "--save"    && i+1<argc) a.save_path    = argv[++i];
        else if (k == "--cuda")                a.cuda         = true;
    }
    return a;
}

// ── Model factory ─────────────────────────────────────────────────────────────
static std::shared_ptr<DenseNetImpl> build_model(const Args& a) {
    const std::string& m = a.model;
    if (m == "densenet121") return make_densenet121(a.num_classes);
    if (m == "densenet169") return make_densenet169(a.num_classes);
    if (m == "densenet201") return make_densenet201(a.num_classes);
    if (m == "densenet264") return make_densenet264(a.num_classes);

    // cifar-L<depth>-k<growth>[bc]
    if (m.substr(0, 6) == "cifar-") {
        auto spec = m.substr(6);  // e.g. "L40-k12" or "L100-k12bc"
        bool bc   = spec.size() >= 2 && spec.substr(spec.size() - 2) == "bc";
        if (bc) spec = spec.substr(0, spec.size() - 2);

        // Parse L<depth>-k<growth>
        auto lpos = spec.find('L');
        auto kpos = spec.find('k');
        if (lpos == std::string::npos || kpos == std::string::npos)
            throw std::invalid_argument("CIFAR model spec must be cifar-L<D>-k<K>[bc]");
        int64_t depth = std::stoll(spec.substr(lpos + 1, kpos - lpos - 2));
        int64_t k     = std::stoll(spec.substr(kpos + 1));
        return make_densenet_cifar(depth, k, bc, a.num_classes);
    }

    throw std::invalid_argument("Unknown model: " + m);
}

// ── Data helpers ──────────────────────────────────────────────────────────────
// Collect MNIST batches and resize to match expected input.
// ImageNet models (not small_input) expect 224×224.
// CIFAR models (small_input) expect 32×32.
static bool is_cifar_model(const std::string& m) {
    return m.substr(0, 6) == "cifar-";
}

template <typename Loader>
static std::vector<std::pair<torch::Tensor, torch::Tensor>>
collect_and_resize(Loader& loader, bool cifar_size) {
    const int64_t sz = cifar_size ? 32 : 224;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> out;
    for (auto& batch : loader) {
        auto img = batch.data;
        if (img.size(1) == 1)
            img = img.expand({img.size(0), 3, img.size(2), img.size(3)}).contiguous();
        img = torch::nn::functional::interpolate(
            img,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{sz, sz})
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
    std::cout << "[DenseNet] device: " << device << "\n";

    auto net = build_model(args);
    net->to(device);

    int64_t n_params = 0;
    for (const auto& p : net->parameters()) n_params += p.numel();
    std::cout << "[DenseNet] " << args.model
              << "  classes=" << args.num_classes
              << "  params=" << n_params / 1'000'000 << "M\n";

    // ── Data (MNIST as runnable demo) ────────────────────────────────────────
    bool cifar = is_cifar_model(args.model);
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

    std::cout << "[DenseNet] loading data from " << args.data_root << " ...\n";
    auto train_batches = collect_and_resize(*train_loader, cifar);
    auto val_batches   = collect_and_resize(*val_loader,   cifar);
    std::cout << "[DenseNet] train=" << train_batches.size()
              << " batches  val=" << val_batches.size() << " batches\n";

    // ── Optimiser — SGD, paper Section 4.2 ──────────────────────────────────
    // weight_decay=1e-4, momentum=0.9, initial lr=0.1
    torch::optim::SGD optim(net->parameters(),
        torch::optim::SGDOptions(args.lr)
            .momentum(0.9)
            .weight_decay(1e-4)
            .nesterov(true));

    // LR schedule: divide by 10 at 50% and 75% of total epochs (paper).
    auto adjust_lr = [&](int64_t epoch) {
        double factor = 1.0;
        if (epoch >= args.epochs * 3 / 4) factor = 0.01;
        else if (epoch >= args.epochs / 2) factor = 0.1;
        auto& g = optim.param_groups()[0].options();
        static_cast<torch::optim::SGDOptions&>(g).lr(args.lr * factor);
    };

    // ── Training loop ────────────────────────────────────────────────────────
    float best_acc = 0.0f;
    for (int64_t epoch = 1; epoch <= args.epochs; ++epoch) {
        adjust_lr(epoch);

        net->train();
        double train_loss = 0.0;
        int64_t n_train   = 0;
        for (auto& [data, target] : train_batches) {
            data   = data.to(device);
            target = target.to(device);
            optim.zero_grad();
            auto out  = net->forward(data);
            auto loss = torch::cross_entropy_loss(out, target);
            loss.backward();
            optim.step();
            train_loss += loss.item<double>() * data.size(0);
            n_train    += data.size(0);
        }

        net->eval();
        int64_t correct = 0, total = 0;
        {
            torch::NoGradGuard ng;
            for (auto& [data, target] : val_batches) {
                data   = data.to(device);
                target = target.to(device);
                auto pred = net->forward(data).argmax(1);
                correct  += pred.eq(target).sum().item<int64_t>();
                total    += data.size(0);
            }
        }
        float acc = static_cast<float>(correct) / static_cast<float>(total);
        std::cout << "epoch " << epoch << "/" << args.epochs
                  << "  loss=" << train_loss / n_train
                  << "  val_acc=" << acc << "\n";

        if (acc > best_acc) {
            best_acc = acc;
            torch::save(net, args.save_path);
        }
    }
    std::cout << "[DenseNet] best val_acc=" << best_acc
              << "  saved to " << args.save_path << "\n";
    return 0;
}
