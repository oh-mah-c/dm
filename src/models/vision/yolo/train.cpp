// ─────────────────────────────────────────────────────────────────────────────
// YOLO v1 training entry point — Redmon et al., CVPR 2016
//
// Usage:
//   ./yolo_train [options]
//   --S        <int>    grid size          (default: 7)
//   --B        <int>    boxes per cell     (default: 2)
//   --C        <int>    number of classes  (default: 20)
//   --epochs   <int>    total epochs       (default: 135)
//   --batch    <int>    batch size         (default: 64)
//   --lr       <float>  peak learning rate (default: 1e-2)
//   --data     <path>   dataset root       (default: ./data)
//   --save     <path>   checkpoint path    (default: yolo_best.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/yolo/yolo.h"

#include <torch/torch.h>
#include <iostream>
#include <string>

using namespace dm::models::vision;

struct Args {
    int64_t S        = 7;
    int64_t B        = 2;
    int64_t C        = 20;
    int64_t epochs   = 135;
    int64_t batch    = 64;
    double  lr       = 1e-2;
    std::string data = "./data";
    std::string save = "yolo_best.pt";
    bool    cuda     = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--S"      && i+1<argc) a.S      = std::stoll(argv[++i]);
        else if (k == "--B"      && i+1<argc) a.B      = std::stoll(argv[++i]);
        else if (k == "--C"      && i+1<argc) a.C      = std::stoll(argv[++i]);
        else if (k == "--epochs" && i+1<argc) a.epochs = std::stoll(argv[++i]);
        else if (k == "--batch"  && i+1<argc) a.batch  = std::stoll(argv[++i]);
        else if (k == "--lr"     && i+1<argc) a.lr     = std::stod(argv[++i]);
        else if (k == "--data"   && i+1<argc) a.data   = argv[++i];
        else if (k == "--save"   && i+1<argc) a.save   = argv[++i];
        else if (k == "--cuda")               a.cuda   = true;
    }
    return a;
}

int main(int argc, char* argv[]) {
    auto args   = parse_args(argc, argv);
    auto device = (args.cuda && torch::cuda::is_available())
                  ? torch::kCUDA : torch::kCPU;
    std::cout << "[YOLO] device: " << device << "\n";

    auto model = YOLO(args.S, args.B, args.C);
    model->to(device);

    int64_t n_params = 0;
    for (auto& p : model->parameters()) n_params += p.numel();
    std::cout << "[YOLO] S=" << args.S << " B=" << args.B
              << " C=" << args.C
              << " params=" << n_params / 1'000'000 << "M\n";

    // ── Data loading ────────────────────────────────────────────────────────
    // Placeholder: generate random batches for smoke-testing.
    // Real training requires a PASCAL VOC / COCO data loader that produces:
    //   imgs:    [batch, 3, 448, 448] float in [0,1]
    //   targets: [batch, S, S, B*5+C] float — encoded ground truth
    std::vector<std::pair<torch::Tensor, torch::Tensor>> train_batches, val_batches;

    try {
        // Attempt to load from data path — expect pre-encoded .pt files
        // Layout: imgs.pt [N, 3, 448, 448], targets.pt [N, S, S, B*5+C]
        torch::Tensor imgs, targets;
        torch::load(imgs,    args.data + "/imgs.pt");
        torch::load(targets, args.data + "/targets.pt");
        int64_t total = imgs.size(0);
        int64_t val_n = std::max<int64_t>(1, total / 10);
        for (int64_t i = 0; i + args.batch <= total - val_n; i += args.batch)
            train_batches.push_back({imgs.slice(0,i,i+args.batch),
                                     targets.slice(0,i,i+args.batch)});
        for (int64_t i = total-val_n; i + args.batch <= total; i += args.batch)
            val_batches.push_back({imgs.slice(0,i,i+args.batch),
                                   targets.slice(0,i,i+args.batch)});
        std::cout << "[YOLO] loaded data from " << args.data
                  << ": " << train_batches.size() << " train batches, "
                  << val_batches.size()  << " val batches\n";
    } catch (...) {
        std::cout << "[YOLO] data not found at " << args.data
                  << " — running with empty batches (model construction verified)\n";
    }

    YOLOTrainConfig cfg;
    cfg.max_epochs   = args.epochs;
    cfg.lr_stage1    = args.lr;
    cfg.device       = device;

    yolo_train(model, cfg, train_batches, val_batches, args.save);
    return 0;
}
