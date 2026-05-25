#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// YOLO v1 — You Only Look Once: Unified, Real-Time Object Detection
// J. Redmon, S. Divvala, R. Girshick, A. Farhadi — CVPR 2016
//
// ── Architecture (Section 2.1, Figure 3) ────────────────────────────────────
//
// 24 convolutional layers followed by 2 fully connected layers.
// Alternating 1×1 reduction layers followed by 3×3 conv layers,
// inspired by GoogLeNet but using 1×1 reductions instead of inception modules.
//
// Convolutional backbone (Figure 3):
//   Conv  7×7,  64, s=2  → MaxPool 2×2, s=2
//   Conv  3×3, 192        → MaxPool 2×2, s=2
//   Conv  1×1, 128
//   Conv  3×3, 256
//   Conv  1×1, 256
//   Conv  3×3, 512        → MaxPool 2×2, s=2
//   ×4: Conv 1×1,256 / Conv 3×3,512
//   Conv  1×1, 512
//   Conv  3×3, 1024       → MaxPool 2×2, s=2
//   ×2: Conv 1×1,512 / Conv 3×3,1024
//   Conv  3×3, 1024
//   Conv  3×3, 1024, s=2
//   Conv  3×3, 1024
//   Conv  3×3, 1024
// FC 4096 → FC S*S*(B*5+C)
//
// ── Detection head (Section 2) ──────────────────────────────────────────────
//
// Grid: S×S cells, each predicts B bounding boxes + C class probs.
// Output tensor: [batch, S, S, B*5 + C]
//   Box encoding: [x, y, w, h, conf]  (5 values per box)
//     x, y: centre offset relative to grid cell, normalised to [0,1]
//     w, h: width/height relative to full image, normalised to [0,1]
//     conf: Pr(Object) × IoU^truth_pred
//   Class probs: C conditional probs Pr(Class_i | Object), one per cell
//
// PASCAL VOC: S=7, B=2, C=20 → output [batch, 7, 7, 30]
//
// ── Activation (Section 2.2, Eq. 2) ────────────────────────────────────────
// Leaky ReLU: φ(x) = x if x > 0, else 0.1x
// Final FC layer uses linear (no activation).
//
// ── Loss function (Section 2.2, Eq. 3) ──────────────────────────────────────
// Multi-part sum-squared error:
//   λ_coord * Σ_cells Σ_boxes 𝟙^obj_ij [(x-x̂)² + (y-ŷ)²]         xy
//   λ_coord * Σ_cells Σ_boxes 𝟙^obj_ij [(√w-√ŵ)² + (√h-√ĥ)²]     wh (sqrt)
//   + Σ_cells Σ_boxes 𝟙^obj_ij (C-Ĉ)²                              obj conf
//   + λ_noobj * Σ_cells Σ_boxes 𝟙^noobj_ij (C-Ĉ)²                  noobj conf
//   + Σ_cells 𝟙^obj_i Σ_classes (p-p̂)²                             class
//
// λ_coord = 5,  λ_noobj = 0.5
// Only the "responsible" box (highest IoU with GT) is penalised for coords.
//
// ── Hyperparameters (Section 2.2) ───────────────────────────────────────────
// Input: 448×448 (pretrain on 224×224)
// Epochs: ~135 total
// Batch: 64, momentum: 0.9, weight_decay: 0.0005
// LR schedule: warm up 1e-3→1e-2, then 1e-2(75ep), 1e-3(30ep), 1e-4(30ep)
// Dropout: 0.5 after first FC layer
// Data aug: random scale/translate ≤20% of image size,
//           random HSV saturation/exposure ±1.5
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <string>
#include <vector>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// Default PASCAL VOC detection constants (Section 2, Figure 3)
// ─────────────────────────────────────────────────────────────────────────────
constexpr int64_t YOLO_S = 7;    // grid size
constexpr int64_t YOLO_B = 2;    // boxes per cell
constexpr int64_t YOLO_C = 20;   // PASCAL VOC classes

// ─────────────────────────────────────────────────────────────────────────────
// YOLOImpl — full detection network (Figure 3)
// ─────────────────────────────────────────────────────────────────────────────
struct YOLOImpl : torch::nn::Module {
    // Convolutional backbone (24 conv layers, Figure 3)
    torch::nn::Sequential backbone{nullptr};

    // Detection head: FC4096 → FC(S*S*(B*5+C))
    torch::nn::Sequential detector{nullptr};

    // Configuration
    int64_t S;   // grid size
    int64_t B;   // boxes per cell
    int64_t C;   // number of classes

    YOLOImpl(int64_t S = YOLO_S,
             int64_t B = YOLO_B,
             int64_t C = YOLO_C);

    // Forward pass.
    // Input:  x [batch, 3, 448, 448]
    // Output: pred [batch, S, S, B*5+C]
    torch::Tensor forward(torch::Tensor x);

    // Decode raw output to boxes with NMS applied.
    // Returns vector of [N, 6] tensors (x1,y1,x2,y2,score,class_id) per image.
    std::vector<torch::Tensor> decode(const torch::Tensor& pred,
                                      float conf_thresh = 0.25f,
                                      float nms_thresh  = 0.45f) const;

};
TORCH_MODULE(YOLO);

// ─────────────────────────────────────────────────────────────────────────────
// YOLO detection loss  (Section 2.2, Eq. 3)
//
// pred:   [batch, S, S, B*5+C]  — raw network output (sigmoid applied inside)
// target: [batch, S, S, B*5+C]  — ground-truth in same encoding
//
// Returns scalar loss.
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor yolo_loss(const torch::Tensor& pred,
                        const torch::Tensor& target,
                        int64_t S     = YOLO_S,
                        int64_t B     = YOLO_B,
                        int64_t C     = YOLO_C,
                        float lambda_coord = 5.0f,
                        float lambda_noobj = 0.5f);

// ─────────────────────────────────────────────────────────────────────────────
// IoU between two sets of boxes (xywh format, all values in [0,1])
// pred_boxes:  [N, 4]
// gt_box:      [4]   (x, y, w, h)
// Returns:     [N]
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor box_iou_xywh(const torch::Tensor& pred_boxes,
                           const torch::Tensor& gt_box);

// ─────────────────────────────────────────────────────────────────────────────
// Non-maximum suppression  (Section 2.3)
// boxes:  [N, 6]  (x1, y1, x2, y2, score, class_id)
// Returns filtered [M, 6]
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor nms(const torch::Tensor& boxes, float iou_thresh = 0.45f);

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration  (Section 2.2)
// ─────────────────────────────────────────────────────────────────────────────
struct YOLOTrainConfig {
    int64_t batch_size   = 64;
    int64_t max_epochs   = 135;
    double  momentum     = 0.9;
    double  weight_decay = 5e-4;
    // LR schedule: warmup → 1e-2(75ep) → 1e-3(30ep) → 1e-4(30ep)
    double  lr_warmup_start = 1e-3;
    double  lr_warmup_end   = 1e-2;
    int64_t warmup_epochs   = 1;    // ramp lr in the first epoch
    double  lr_stage1       = 1e-2;  int64_t epochs_stage1 = 75;
    double  lr_stage2       = 1e-3;  int64_t epochs_stage2 = 30;
    double  lr_stage3       = 1e-4;  int64_t epochs_stage3 = 30;
    double  dropout         = 0.5;
    float   lambda_coord    = 5.0f;
    float   lambda_noobj    = 0.5f;
    torch::Device device    = torch::kCPU;
};

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────────────────────────────────────
float yolo_train_epoch(YOLO& model,
                       torch::optim::SGD& optimizer,
                       torch::Device device,
                       const std::vector<std::pair<torch::Tensor,
                                                   torch::Tensor>>& batches,
                       float lambda_coord = 5.0f,
                       float lambda_noobj = 0.5f);

float yolo_evaluate(YOLO& model,
                    torch::Device device,
                    const std::vector<std::pair<torch::Tensor,
                                                torch::Tensor>>& batches,
                    float lambda_coord = 5.0f,
                    float lambda_noobj = 0.5f);

void yolo_train(YOLO& model,
                const YOLOTrainConfig& cfg,
                const std::vector<std::pair<torch::Tensor, torch::Tensor>>& train_batches,
                const std::vector<std::pair<torch::Tensor, torch::Tensor>>& val_batches,
                const std::string& save_path = "yolo_best.pt");

} // namespace vision
} // namespace models
} // namespace dm
