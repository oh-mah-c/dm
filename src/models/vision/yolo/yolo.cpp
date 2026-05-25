// ─────────────────────────────────────────────────────────────────────────────
// YOLO v1 implementation — Redmon et al., CVPR 2016
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/yolo/yolo.h"

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: LeakyReLU activation (Section 2.2, Eq. 2)
// φ(x) = x if x > 0, else 0.1x
// ─────────────────────────────────────────────────────────────────────────────
static inline torch::nn::LeakyReLU leaky() {
    return torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true));
}

// ─────────────────────────────────────────────────────────────────────────────
// push_conv: push Conv2d + BatchNorm2d + LeakyReLU directly into a Sequential.
// Cannot nest Sequential inside Sequential via AnyModule in LibTorch.
// ─────────────────────────────────────────────────────────────────────────────
static void push_conv(torch::nn::Sequential& seq,
                      int64_t in_ch, int64_t out_ch,
                      int64_t ksize, int64_t stride = 1, int64_t pad = -1) {
    if (pad < 0) pad = ksize / 2;
    seq->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_ch, out_ch, ksize)
            .stride(stride).padding(pad).bias(false)));
    seq->push_back(torch::nn::BatchNorm2d(out_ch));
    seq->push_back(torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(0.1).inplace(true)));
}

// ─────────────────────────────────────────────────────────────────────────────
// YOLOImpl constructor — 24-layer backbone + 2-layer FC head  (Figure 3)
// ─────────────────────────────────────────────────────────────────────────────
YOLOImpl::YOLOImpl(int64_t S_, int64_t B_, int64_t C_)
    : S(S_), B(B_), C(C_) {

    // ── Backbone ────────────────────────────────────────────────────────────
    torch::nn::Sequential bb;

    // Block 1: Conv 7×7/64/s2 → MaxPool 2×2/s2
    push_conv(bb, 3,   64,  7, 2);                      // 448→224
    bb->push_back(torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(2).stride(2)));      // 224→112

    // Block 2: Conv 3×3/192 → MaxPool
    push_conv(bb, 64,  192, 3);
    bb->push_back(torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(2).stride(2)));      // 112→56

    // Block 3: 1×1/128, 3×3/256, 1×1/256, 3×3/512 → MaxPool
    push_conv(bb, 192, 128, 1, 1, 0);
    push_conv(bb, 128, 256, 3);
    push_conv(bb, 256, 256, 1, 1, 0);
    push_conv(bb, 256, 512, 3);
    bb->push_back(torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(2).stride(2)));      // 56→28

    // Block 4: ×4 (1×1/256, 3×3/512) then 1×1/512, 3×3/1024 → MaxPool
    for (int i = 0; i < 4; ++i) {
        push_conv(bb, 512, 256, 1, 1, 0);
        push_conv(bb, 256, 512, 3);
    }
    push_conv(bb, 512,  512,  1, 1, 0);
    push_conv(bb, 512,  1024, 3);
    bb->push_back(torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(2).stride(2)));      // 28→14

    // Block 5: ×2 (1×1/512, 3×3/1024) then 3×3/1024, 3×3/1024/s2
    for (int i = 0; i < 2; ++i) {
        push_conv(bb, 1024, 512,  1, 1, 0);
        push_conv(bb, 512,  1024, 3);
    }
    push_conv(bb, 1024, 1024, 3);
    push_conv(bb, 1024, 1024, 3, 2);                    // 14→7

    // Block 6: 3×3/1024, 3×3/1024
    push_conv(bb, 1024, 1024, 3);
    push_conv(bb, 1024, 1024, 3);

    backbone = register_module("backbone", bb);

    // ── Detection head ───────────────────────────────────────────────────────
    // FC4096 (leaky) → Dropout(0.5) → FC(S*S*(B*5+C)) (linear)
    int64_t flat    = 1024 * S * S;   // after backbone: [B, 1024, S, S]
    int64_t out_dim = S * S * (B * 5 + C);

    torch::nn::Sequential det;
    det->push_back(torch::nn::Flatten());
    det->push_back(torch::nn::Linear(flat, 4096));
    det->push_back(leaky());
    det->push_back(torch::nn::Dropout(
        torch::nn::DropoutOptions().p(0.5)));
    det->push_back(torch::nn::Linear(4096, out_dim));
    // No activation on final layer — sigmoid applied per-component in loss

    detector = register_module("detector", det);
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward  — returns raw [batch, S, S, B*5+C]
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor YOLOImpl::forward(torch::Tensor x) {
    auto feat = backbone->forward(x);           // [B, 1024, S, S]
    auto raw  = detector->forward(feat);        // [B, S*S*(B*5+C)]
    return raw.view({x.size(0), S, S, B * 5 + C});
}

// ─────────────────────────────────────────────────────────────────────────────
// box_iou_xywh  (used by loss and decode)
// pred_boxes: [N, 4] (x, y, w, h) — centre + size, normalised [0,1]
// gt_box:     [4]
// Returns:    [N]
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor box_iou_xywh(const torch::Tensor& pred_boxes,
                           const torch::Tensor& gt_box) {
    // Convert xywh → xyxy
    auto px1 = pred_boxes.select(1,0) - pred_boxes.select(1,2) * 0.5f;
    auto py1 = pred_boxes.select(1,1) - pred_boxes.select(1,3) * 0.5f;
    auto px2 = pred_boxes.select(1,0) + pred_boxes.select(1,2) * 0.5f;
    auto py2 = pred_boxes.select(1,1) + pred_boxes.select(1,3) * 0.5f;

    auto gx1 = gt_box[0] - gt_box[2] * 0.5f;
    auto gy1 = gt_box[1] - gt_box[3] * 0.5f;
    auto gx2 = gt_box[0] + gt_box[2] * 0.5f;
    auto gy2 = gt_box[1] + gt_box[3] * 0.5f;

    auto ix1 = torch::max(px1, gx1);
    auto iy1 = torch::max(py1, gy1);
    auto ix2 = torch::min(px2, gx2);
    auto iy2 = torch::min(py2, gy2);

    auto inter_w = torch::clamp(ix2 - ix1, 0.f);
    auto inter_h = torch::clamp(iy2 - iy1, 0.f);
    auto inter   = inter_w * inter_h;

    auto area_p  = (px2 - px1) * (py2 - py1);
    auto area_g  = (gx2 - gx1) * (gy2 - gy1);
    return inter / (area_p + area_g - inter + 1e-6f);
}

// ─────────────────────────────────────────────────────────────────────────────
// nms — non-maximum suppression  (Section 2.3)
// boxes: [N, 6]  (x1, y1, x2, y2, score, class_id)
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor nms(const torch::Tensor& boxes, float iou_thresh) {
    if (boxes.size(0) == 0) return boxes;
    // Sort by score descending
    auto order = std::get<1>(boxes.select(1, 4).sort(0, /*descending=*/true));
    std::vector<int64_t> keep;
    std::vector<bool> suppressed(boxes.size(0), false);
    auto boxes_cpu = boxes.cpu();

    for (int64_t i = 0; i < (int64_t)order.size(0); ++i) {
        int64_t idx = order[i].item<int64_t>();
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        auto b1 = boxes_cpu[idx];   // [6]
        for (int64_t j = i + 1; j < (int64_t)order.size(0); ++j) {
            int64_t jdx = order[j].item<int64_t>();
            if (suppressed[jdx]) continue;
            auto b2 = boxes_cpu[jdx];
            // IoU xyxy
            float ix1 = std::max(b1[0].item<float>(), b2[0].item<float>());
            float iy1 = std::max(b1[1].item<float>(), b2[1].item<float>());
            float ix2 = std::min(b1[2].item<float>(), b2[2].item<float>());
            float iy2 = std::min(b1[3].item<float>(), b2[3].item<float>());
            float iw  = std::max(0.f, ix2 - ix1);
            float ih  = std::max(0.f, iy2 - iy1);
            float inter = iw * ih;
            float a1 = (b1[2]-b1[0]).item<float>() * (b1[3]-b1[1]).item<float>();
            float a2 = (b2[2]-b2[0]).item<float>() * (b2[3]-b2[1]).item<float>();
            float iou = inter / (a1 + a2 - inter + 1e-6f);
            if (iou > iou_thresh) suppressed[jdx] = true;
        }
    }

    if (keep.empty()) return torch::zeros({0, 6}, boxes.options());
    auto idx_t = torch::tensor(keep, torch::kLong);
    return boxes.index_select(0, idx_t);
}

// ─────────────────────────────────────────────────────────────────────────────
// decode — raw pred [batch, S, S, B*5+C] → list of [N, 6] per image
// ─────────────────────────────────────────────────────────────────────────────
std::vector<torch::Tensor> YOLOImpl::decode(const torch::Tensor& pred_raw,
                                             float conf_thresh,
                                             float nms_thresh) const {
    // Apply sigmoid to xy, confidence, class probs; keep wh raw (they are
    // already predicted as normalised values; paper uses linear final layer
    // but treats wh as normalised to image — we clamp to [0,1])
    auto pred = pred_raw.clone();
    std::vector<torch::Tensor> results;
    int64_t batch = pred.size(0);

    // Build grid offsets for xy decoding
    auto device = pred.device();
    auto grid_x = torch::arange(S, torch::TensorOptions().dtype(torch::kFloat).device(device));
    auto grid_y = torch::arange(S, torch::TensorOptions().dtype(torch::kFloat).device(device));
    auto gy = grid_y.view({S, 1, 1}).expand({S, S, 1});
    auto gx = grid_x.view({1, S, 1}).expand({S, S, 1});

    for (int64_t b = 0; b < batch; ++b) {
        auto p = pred[b];   // [S, S, B*5+C]
        std::vector<torch::Tensor> all_boxes;

        for (int64_t bi = 0; bi < B; ++bi) {
            int64_t off = bi * 5;
            // x, y: sigmoid + grid offset → absolute [0,S], then /S → [0,1]
            auto tx   = torch::sigmoid(p.slice(2, off+0, off+1));   // [S,S,1]
            auto ty   = torch::sigmoid(p.slice(2, off+1, off+2));
            auto tw   = torch::clamp(p.slice(2, off+2, off+3), 0.f, 1.f);
            auto th   = torch::clamp(p.slice(2, off+3, off+4), 0.f, 1.f);
            auto conf = torch::sigmoid(p.slice(2, off+4, off+5));

            auto abs_x = (tx + gx) / (float)S;
            auto abs_y = (ty + gy) / (float)S;

            // class probs (per cell, shared over boxes)
            auto cls_probs = torch::softmax(p.slice(2, B*5, B*5+C), 2);  // [S,S,C]
            auto cls_score = cls_probs * conf;   // [S,S,C]

            auto max_result = cls_score.max(2);
            auto scores  = std::get<0>(max_result);   // [S,S]
            auto cls_ids = std::get<1>(max_result);   // [S,S]

            // Convert to xyxy
            auto x1 = abs_x.squeeze(2) - tw.squeeze(2) * 0.5f;
            auto y1 = abs_y.squeeze(2) - th.squeeze(2) * 0.5f;
            auto x2 = abs_x.squeeze(2) + tw.squeeze(2) * 0.5f;
            auto y2 = abs_y.squeeze(2) + th.squeeze(2) * 0.5f;

            // Filter by threshold
            auto mask = scores > conf_thresh;
            if (!mask.any().item<bool>()) continue;

            auto bx1  = x1.masked_select(mask);
            auto by1  = y1.masked_select(mask);
            auto bx2  = x2.masked_select(mask);
            auto by2  = y2.masked_select(mask);
            auto bsc  = scores.masked_select(mask);
            auto bcls = cls_ids.masked_select(mask).to(torch::kFloat);

            // Stack to [N, 6]
            auto box_t = torch::stack({bx1, by1, bx2, by2, bsc, bcls}, 1);
            all_boxes.push_back(box_t);
        }

        if (all_boxes.empty()) {
            results.push_back(torch::zeros({0, 6}));
        } else {
            auto combined = torch::cat(all_boxes, 0);
            results.push_back(nms(combined, nms_thresh));
        }
    }
    return results;
}

// ─────────────────────────────────────────────────────────────────────────────
// yolo_loss  (Section 2.2, Eq. 3)
//
// pred:   [batch, S, S, B*5+C]  — raw network output
// target: [batch, S, S, B*5+C]  — ground truth (same layout)
//
// Target encoding:
//   For cells that contain an object: box values filled, conf=1, class=one-hot
//   For cells with no object:         box values 0,    conf=0, class=0
//
// Loss terms (Eq. 3):
//   xy:    λ_coord * Σ_obj [(x-x̂)² + (y-ŷ)²]
//   wh:    λ_coord * Σ_obj [(√w-√ŵ)² + (√h-√ĥ)²]   (sqrt trick)
//   conf obj:   Σ_obj   (C-Ĉ)²     where Ĉ = sigmoid(raw_conf)
//   conf noobj: λ_noobj * Σ_noobj (C-Ĉ)²
//   class:      Σ_obj   (p-p̂)²
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor yolo_loss(const torch::Tensor& pred_raw,
                        const torch::Tensor& target,
                        int64_t S, int64_t B, int64_t C,
                        float lambda_coord, float lambda_noobj) {
    using namespace torch;

    // Apply sigmoid to xy + conf components; classes computed with MSE on raw
    // (paper: "sum-squared error in the output"; we apply sigmoid to conf+xy
    //  for numerical stability; wh kept as-is since targets are in [0,1])
    int64_t batch = pred_raw.size(0);

    auto loss = torch::zeros({1}, pred_raw.options());

    for (int64_t bi = 0; bi < B; ++bi) {
        int64_t off = bi * 5;
        // Raw predictions for this box
        auto pred_x    = pred_raw.slice(3, off+0, off+1).squeeze(3);  // [N,S,S]
        auto pred_y    = pred_raw.slice(3, off+1, off+2).squeeze(3);
        auto pred_w    = pred_raw.slice(3, off+2, off+3).squeeze(3);
        auto pred_h    = pred_raw.slice(3, off+3, off+4).squeeze(3);
        auto pred_conf = pred_raw.slice(3, off+4, off+5).squeeze(3);

        // GT
        auto tgt_x    = target.slice(3, off+0, off+1).squeeze(3);
        auto tgt_y    = target.slice(3, off+1, off+2).squeeze(3);
        auto tgt_w    = target.slice(3, off+2, off+3).squeeze(3);
        auto tgt_h    = target.slice(3, off+3, off+4).squeeze(3);
        auto tgt_conf = target.slice(3, off+4, off+5).squeeze(3);  // 0 or 1

        // Object mask: cells where GT says there IS an object
        auto obj_mask  = tgt_conf;            // [N,S,S]  1.0 or 0.0
        auto noobj_mask = 1.0f - tgt_conf;

        // ── xy loss (Eq. 3, term 1) ──────────────────────────────────────
        // sigmoid applied to pred_x/y so output is in [0,1] (cell-relative)
        auto sig_x = torch::sigmoid(pred_x);
        auto sig_y = torch::sigmoid(pred_y);
        loss = loss + lambda_coord *
               (obj_mask * ((sig_x - tgt_x).pow(2) + (sig_y - tgt_y).pow(2))).sum();

        // ── wh loss (Eq. 3, term 2): sqrt trick ─────────────────────────
        // Clamp to avoid sqrt of negative (predictions are unconstrained)
        auto pw = torch::clamp(pred_w, 0.f);
        auto tw = torch::clamp(tgt_w,  0.f);
        auto ph = torch::clamp(pred_h, 0.f);
        auto th = torch::clamp(tgt_h,  0.f);
        loss = loss + lambda_coord *
               (obj_mask * ((pw.sqrt() - tw.sqrt()).pow(2) +
                            (ph.sqrt() - th.sqrt()).pow(2))).sum();

        // ── Confidence loss (terms 3 + 4) ────────────────────────────────
        auto conf_pred = torch::sigmoid(pred_conf);
        loss = loss +     (obj_mask   * (conf_pred - tgt_conf).pow(2)).sum();
        loss = loss + lambda_noobj *
                          (noobj_mask * (conf_pred - tgt_conf).pow(2)).sum();
    }

    // ── Class loss (term 5): only for cells with an object ────────────────
    // Use obj_conf from box-0 as the cell-level mask
    auto cell_obj_mask = target.slice(3, 4, 5).squeeze(3);  // [N,S,S]

    auto pred_cls = pred_raw.slice(3, B*5, B*5+C);    // [N,S,S,C]
    auto tgt_cls  = target.slice(3,  B*5, B*5+C);

    // cell_obj_mask broadcast: [N,S,S,1] * [N,S,S,C]
    loss = loss +
        (cell_obj_mask.unsqueeze(3) * (pred_cls - tgt_cls).pow(2)).sum();

    return loss / (float)batch;
}

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────────────────────────────────────
float yolo_train_epoch(YOLO& model,
                       torch::optim::SGD& optimizer,
                       torch::Device device,
                       const std::vector<std::pair<torch::Tensor,
                                                   torch::Tensor>>& batches,
                       float lambda_coord,
                       float lambda_noobj) {
    model->train();
    double total = 0.0;
    int64_t steps = 0;
    for (auto& [imgs, targets] : batches) {
        auto x = imgs.to(device);
        auto t = targets.to(device);
        optimizer.zero_grad();
        auto pred = model->forward(x);
        auto loss = yolo_loss(pred, t,
                              model->S, model->B, model->C,
                              lambda_coord, lambda_noobj);
        loss.backward();
        optimizer.step();
        total += loss.item<double>();
        ++steps;
    }
    return steps > 0 ? (float)(total / steps) : 0.f;
}

float yolo_evaluate(YOLO& model,
                    torch::Device device,
                    const std::vector<std::pair<torch::Tensor,
                                                torch::Tensor>>& batches,
                    float lambda_coord,
                    float lambda_noobj) {
    model->eval();
    torch::NoGradGuard ng;
    double total = 0.0;
    int64_t steps = 0;
    for (auto& [imgs, targets] : batches) {
        auto x = imgs.to(device);
        auto t = targets.to(device);
        auto pred = model->forward(x);
        auto loss = yolo_loss(pred, t,
                              model->S, model->B, model->C,
                              lambda_coord, lambda_noobj);
        total += loss.item<double>();
        ++steps;
    }
    return steps > 0 ? (float)(total / steps) : 0.f;
}

void yolo_train(YOLO& model,
                const YOLOTrainConfig& cfg,
                const std::vector<std::pair<torch::Tensor, torch::Tensor>>& train_batches,
                const std::vector<std::pair<torch::Tensor, torch::Tensor>>& val_batches,
                const std::string& save_path) {
    model->to(cfg.device);

    torch::optim::SGD optimizer(model->parameters(),
        torch::optim::SGDOptions(cfg.lr_stage1)
            .momentum(cfg.momentum)
            .weight_decay(cfg.weight_decay));

    float best_val = std::numeric_limits<float>::max();
    int64_t epoch  = 0;

    // Stage 0: warmup (epoch 0) — ramp lr from lr_warmup_start to lr_warmup_end
    // Stage 1: lr_stage1 for epochs_stage1
    // Stage 2: lr_stage2 for epochs_stage2
    // Stage 3: lr_stage3 for epochs_stage3
    auto set_lr = [&](double lr) {
        for (auto& pg : optimizer.param_groups())
            static_cast<torch::optim::SGDOptions&>(pg.options()).lr(lr);
    };

    for (; epoch < cfg.max_epochs; ++epoch) {
        // LR schedule
        if (epoch == 0)
            set_lr(cfg.lr_warmup_start);
        else if (epoch == cfg.warmup_epochs)
            set_lr(cfg.lr_stage1);
        else if (epoch == cfg.warmup_epochs + cfg.epochs_stage1)
            set_lr(cfg.lr_stage2);
        else if (epoch == cfg.warmup_epochs + cfg.epochs_stage1 + cfg.epochs_stage2)
            set_lr(cfg.lr_stage3);

        float train_loss = yolo_train_epoch(model, optimizer, cfg.device,
                                            train_batches,
                                            cfg.lambda_coord, cfg.lambda_noobj);
        float val_loss   = yolo_evaluate(model, cfg.device,
                                         val_batches,
                                         cfg.lambda_coord, cfg.lambda_noobj);

        std::cout << "[YOLO] epoch " << epoch + 1 << "/" << cfg.max_epochs
                  << "  train_loss=" << train_loss
                  << "  val_loss="   << val_loss << "\n";

        if (val_loss < best_val) {
            best_val = val_loss;
            torch::serialize::OutputArchive ar;
            model->save(ar);
            ar.save_to(save_path);
            std::cout << "[YOLO] saved checkpoint → " << save_path << "\n";
        }
    }
}

} // namespace vision
} // namespace models
} // namespace dm
