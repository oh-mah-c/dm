// ─────────────────────────────────────────────────────────────────────────────
// Swin Transformer implementation
// Ze Liu et al., ICCV 2021
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/swin/swin.h"

#include <torch/torch.h>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: partition feature map into non-overlapping windows
//
// x: [B, H, W, C]  →  windows: [num_windows*B, M, M, C]
// ─────────────────────────────────────────────────────────────────────────────
static torch::Tensor window_partition(torch::Tensor x, int64_t window_size) {
    auto B = x.size(0), H = x.size(1), W = x.size(2), C = x.size(3);
    int64_t M = window_size;
    // [B, H/M, M, W/M, M, C]
    x = x.view({B, H / M, M, W / M, M, C});
    // [B, H/M, W/M, M, M, C]  →  [num_windows*B, M, M, C]
    x = x.permute({0, 1, 3, 2, 4, 5}).contiguous();
    return x.view({-1, M, M, C});
}

// Reverse: windows → feature map
// windows: [num_windows*B, M, M, C]  →  x: [B, H, W, C]
static torch::Tensor window_reverse(torch::Tensor windows, int64_t window_size,
                                    int64_t H, int64_t W) {
    int64_t M = window_size;
    int64_t B = windows.size(0) / ((H / M) * (W / M));
    auto x = windows.view({B, H / M, W / M, M, M, -1});
    x = x.permute({0, 1, 3, 2, 4, 5}).contiguous();
    return x.view({B, H, W, -1});
}

// ─────────────────────────────────────────────────────────────────────────────
// SwinMlp
// ─────────────────────────────────────────────────────────────────────────────
SwinMlpImpl::SwinMlpImpl(int64_t in_features, int64_t hidden_features, double dropout) {
    fc1  = register_module("fc1",  torch::nn::Linear(in_features, hidden_features));
    fc2  = register_module("fc2",  torch::nn::Linear(hidden_features, in_features));
    drop = register_module("drop", torch::nn::Dropout(dropout));
}

torch::Tensor SwinMlpImpl::forward(torch::Tensor x) {
    x = fc1->forward(x);
    x = torch::gelu(x);      // Section 3.1: GELU nonlinearity
    x = drop->forward(x);
    x = fc2->forward(x);
    x = drop->forward(x);
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// WindowAttention — relative position bias (Section 3.2, Eq. 4)
// ─────────────────────────────────────────────────────────────────────────────
WindowAttentionImpl::WindowAttentionImpl(int64_t dim_, int64_t window_size_,
                                         int64_t num_heads_,
                                         double attn_drop_, double proj_drop_)
    : dim(dim_), window_size(window_size_), num_heads(num_heads_) {
    scale = 1.0 / std::sqrt(static_cast<double>(dim_ / num_heads_));

    // Relative position bias table: (2M-1)*(2M-1) entries per head
    int64_t M = window_size;
    relative_position_bias_table = register_parameter(
        "relative_position_bias_table",
        torch::zeros({(2*M - 1) * (2*M - 1), num_heads_}));
    torch::nn::init::normal_(relative_position_bias_table, 0.0, 0.02);

    _build_relative_position_index();

    qkv       = register_module("qkv",       torch::nn::Linear(
                                   torch::nn::LinearOptions(dim_, dim_ * 3).bias(false)));
    proj      = register_module("proj",      torch::nn::Linear(dim_, dim_));
    attn_drop = register_module("attn_drop", torch::nn::Dropout(attn_drop_));
    proj_drop = register_module("proj_drop", torch::nn::Dropout(proj_drop_));
}

void WindowAttentionImpl::_build_relative_position_index() {
    // Build pairwise relative position index for M×M patches in a window
    int64_t M = window_size;
    // coords: [2, M, M]
    auto coords_h = torch::arange(M);
    auto coords_w = torch::arange(M);
    // [2, M, M]
    auto coords   = torch::stack(torch::meshgrid({coords_h, coords_w}, /*indexing=*/"ij"));
    // [2, M²]
    auto coords_flatten = torch::flatten(coords, 1);
    // [2, M², M²]: relative coords
    auto rel = coords_flatten.unsqueeze(2) - coords_flatten.unsqueeze(1);
    // [M², M², 2]
    rel = rel.permute({1, 2, 0}).contiguous();
    rel.select(2, 0) += M - 1;   // shift to start from 0
    rel.select(2, 1) += M - 1;
    rel.select(2, 0) *= (2 * M - 1);
    // [M², M²]
    auto index = rel.sum(-1);
    relative_position_index = register_buffer("relative_position_index", index);
}

torch::Tensor WindowAttentionImpl::forward(torch::Tensor x,
                                            torch::optional<torch::Tensor> mask) {
    auto B_  = x.size(0);  // num_windows * batch
    auto N   = x.size(1);  // M*M
    auto C   = x.size(2);

    // qkv: [B_, N, 3*C]  →  [3, B_, num_heads, N, head_dim]
    auto qkv_out = qkv->forward(x).reshape({B_, N, 3, num_heads, C / num_heads})
                      .permute({2, 0, 3, 1, 4});
    auto q = qkv_out[0], k = qkv_out[1], v = qkv_out[2];

    // Scaled dot-product  (matmul handles 4-D batch: [B_, nh, N, head_dim])
    auto attn = torch::matmul(q * scale, k.transpose(-2, -1));  // [B_, nh, N, N]

    // Relative position bias
    // index: [N, N]  →  table lookup  →  [N, N, num_heads]  →  [1, nh, N, N]
    int64_t M2 = window_size * window_size;
    auto idx  = relative_position_index.view({-1});              // [M²*M²]
    auto bias = relative_position_bias_table.index_select(0, idx) // [M²*M², nh]
                    .view({M2, M2, num_heads})
                    .permute({2, 0, 1})   // [nh, M², M²]
                    .unsqueeze(0);        // [1, nh, M², M²]
    attn = attn + bias;

    // Apply shifted-window mask if provided
    if (mask.has_value()) {
        auto& msk = mask.value();
        int64_t nW = msk.size(0);
        // attn: [B_/nW, nW, nh, N, N]
        attn = attn.view({B_ / nW, nW, num_heads, N, N})
               + msk.unsqueeze(1).unsqueeze(0);
        attn = attn.view({-1, num_heads, N, N});
    }

    attn = torch::softmax(attn, -1);
    attn = attn_drop->forward(attn);

    // Aggregate values  (matmul handles 4-D: [B_, nh, N, N] x [B_, nh, N, head_dim])
    auto out = torch::matmul(attn, v)                      // [B_, nh, N, head_dim]
                   .transpose(1, 2).contiguous()           // [B_, N, nh, head_dim]
                   .view({B_, N, C});
    out = proj->forward(out);
    out = proj_drop->forward(out);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// SwinBlock
// ─────────────────────────────────────────────────────────────────────────────
SwinBlockImpl::SwinBlockImpl(int64_t dim_, int64_t res_h, int64_t res_w,
                              int64_t num_heads, int64_t window_size_,
                              int64_t shift_size_, double mlp_ratio,
                              double dropout, double attn_drop)
    : dim(dim_), input_resolution_h(res_h), input_resolution_w(res_w),
      window_size(window_size_), shift_size(shift_size_) {

    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim_})));
    attn  = register_module("attn",  WindowAttention(dim_, window_size_, num_heads,
                                                     attn_drop, dropout));
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim_})));
    int64_t mlp_hidden = static_cast<int64_t>(dim_ * mlp_ratio);
    mlp   = register_module("mlp",   SwinMlp(dim_, mlp_hidden, dropout));

    if (shift_size > 0) {
        _compute_attn_mask();
    }
}

void SwinBlockImpl::_compute_attn_mask() {
    // Build attention mask for shifted windows (Fig. 4 / Section 3.2)
    // img_mask: [1, H, W, 1]
    auto img_mask = torch::zeros({1, input_resolution_h, input_resolution_w, 1});
    int64_t M   = window_size;
    int64_t sh  = shift_size;
    int64_t H   = input_resolution_h;
    int64_t W   = input_resolution_w;

    // Assign unique region labels (0..8) using slice indexing
    int64_t cnt = 0;
    std::vector<std::pair<int64_t,int64_t>> h_slices = {
        {0, H - M}, {H - M, H - sh}, {H - sh, H}};
    std::vector<std::pair<int64_t,int64_t>> w_slices = {
        {0, W - M}, {W - M, W - sh}, {W - sh, W}};
    for (auto [hs, he] : h_slices) {
        for (auto [ws, we] : w_slices) {
            img_mask.slice(1, hs, he).slice(2, ws, we).fill_(cnt);
            ++cnt;
        }
    }

    // Partition into windows: [num_windows, M, M, 1]
    auto mask_windows = window_partition(img_mask, M);
    mask_windows = mask_windows.view({-1, M * M});

    // attn_mask: [num_windows, M², M²]  — 0 where same region, -100 elsewhere
    auto raw = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2);
    raw = raw.masked_fill(raw != 0, -100.0f);
    raw = raw.masked_fill(raw == 0, 0.0f);
    attn_mask = register_buffer("attn_mask", raw);
}

torch::Tensor SwinBlockImpl::forward(torch::Tensor x) {
    int64_t B = x.size(0);
    int64_t L = x.size(1);
    int64_t C = x.size(2);
    int64_t H = input_resolution_h;
    int64_t W = input_resolution_w;

    // Self-attention branch
    auto shortcut = x;
    x = norm1->forward(x);
    x = x.view({B, H, W, C});

    // Cyclic shift (Eq. 3 / Fig. 4)
    torch::Tensor shifted_x;
    if (shift_size > 0) {
        shifted_x = torch::roll(x, {-shift_size, -shift_size}, {1, 2});
    } else {
        shifted_x = x;
    }

    // Partition into windows
    auto x_windows = window_partition(shifted_x, window_size);
    x_windows = x_windows.view({-1, window_size * window_size, C});

    // W-MSA / SW-MSA
    torch::optional<torch::Tensor> mask_opt = torch::nullopt;
    if (shift_size > 0 && attn_mask.defined()) {
        mask_opt = attn_mask.to(x.device());
    }
    auto attn_windows = attn->forward(x_windows, mask_opt);

    // Merge windows back
    attn_windows = attn_windows.view({-1, window_size, window_size, C});
    auto shifted_out = window_reverse(attn_windows, window_size, H, W);

    // Reverse cyclic shift
    torch::Tensor x_out;
    if (shift_size > 0) {
        x_out = torch::roll(shifted_out, {shift_size, shift_size}, {1, 2});
    } else {
        x_out = shifted_out;
    }
    x_out = x_out.view({B, L, C});

    // Residual + MLP
    x = shortcut + x_out;
    x = x + mlp->forward(norm2->forward(x));
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// PatchMerging
// ─────────────────────────────────────────────────────────────────────────────
PatchMergingImpl::PatchMergingImpl(int64_t res_h, int64_t res_w, int64_t dim_)
    : input_resolution_h(res_h), input_resolution_w(res_w), dim(dim_) {
    reduction = register_module("reduction",
                    torch::nn::Linear(
                        torch::nn::LinearOptions(4 * dim_, 2 * dim_).bias(false)));
    norm = register_module("norm",
               torch::nn::LayerNorm(torch::nn::LayerNormOptions({4 * dim_})));
}

torch::Tensor PatchMergingImpl::forward(torch::Tensor x) {
    // x: [B, H*W, C]
    int64_t B = x.size(0);
    int64_t H = input_resolution_h;
    int64_t W = input_resolution_w;
    int64_t C = dim;

    x = x.view({B, H, W, C});

    // Subsample 2×2 neighbours
    auto x0 = x.slice(1, 0, H, 2).slice(2, 0, W, 2);  // [B, H/2, W/2, C]
    auto x1 = x.slice(1, 1, H, 2).slice(2, 0, W, 2);
    auto x2 = x.slice(1, 0, H, 2).slice(2, 1, W, 2);
    auto x3 = x.slice(1, 1, H, 2).slice(2, 1, W, 2);

    x = torch::cat({x0, x1, x2, x3}, -1);  // [B, H/2, W/2, 4C]
    x = x.view({B, -1, 4 * C});            // [B, H/2*W/2, 4C]
    x = norm->forward(x);
    x = reduction->forward(x);             // [B, H/2*W/2, 2C]
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// SwinStage
// ─────────────────────────────────────────────────────────────────────────────
SwinStageImpl::SwinStageImpl(int64_t dim, int64_t res_h, int64_t res_w,
                              int64_t depth, int64_t num_heads,
                              int64_t window_size, double mlp_ratio,
                              double dropout, double attn_drop,
                              bool use_downsample) {
    blocks = register_module("blocks", torch::nn::Sequential());

    int64_t M  = window_size;
    // Ensure shift size does not exceed resolution
    int64_t sh = (res_h <= M && res_w <= M) ? 0 : M / 2;

    for (int64_t i = 0; i < depth; ++i) {
        int64_t shift = (i % 2 == 0) ? 0 : sh;   // alternate W-MSA / SW-MSA
        blocks->push_back(SwinBlock(dim, res_h, res_w, num_heads,
                                    window_size, shift, mlp_ratio,
                                    dropout, attn_drop));
    }

    if (use_downsample) {
        downsample = register_module("downsample",
                         PatchMerging(res_h, res_w, dim));
    }
}

std::tuple<torch::Tensor, int64_t, int64_t>
SwinStageImpl::forward(torch::Tensor x) {
    for (size_t bi = 0; bi < blocks->size(); ++bi) {
        x = blocks->ptr(bi)->as<SwinBlockImpl>()->forward(x);
    }
    int64_t H = input_resolution_h();
    int64_t W = input_resolution_w();
    if (downsample) {
        x = downsample->forward(x);
        H /= 2; W /= 2;
    }
    return {x, H, W};
}

// helper to retrieve stored resolution from first block
int64_t SwinStageImpl::input_resolution_h() const {
    if (blocks->size() > 0) {
        return blocks->ptr(0)->as<SwinBlockImpl>()->input_resolution_h;
    }
    return 0;
}
int64_t SwinStageImpl::input_resolution_w() const {
    if (blocks->size() > 0) {
        return blocks->ptr(0)->as<SwinBlockImpl>()->input_resolution_w;
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// PatchEmbed
// ─────────────────────────────────────────────────────────────────────────────
PatchEmbedImpl::PatchEmbedImpl(int64_t img_size, int64_t patch_size_,
                                int64_t in_chans, int64_t embed_dim_)
    : patch_size(patch_size_), embed_dim(embed_dim_) {
    num_patches_h = img_size / patch_size_;
    num_patches_w = img_size / patch_size_;

    // Combine patch partition + linear embedding via stride-4 conv
    proj = register_module("proj",
               torch::nn::Conv2d(
                   torch::nn::Conv2dOptions(in_chans, embed_dim_, patch_size_)
                   .stride(patch_size_)));
    norm = register_module("norm",
               torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_})));
}

torch::Tensor PatchEmbedImpl::forward(torch::Tensor x) {
    // x: [B, C, H, W]  →  [B, embed_dim, H/P, W/P]
    x = proj->forward(x);
    auto B  = x.size(0);
    auto C  = x.size(1);
    auto Hp = x.size(2);
    auto Wp = x.size(3);
    // [B, H/P * W/P, embed_dim]
    x = x.flatten(2).transpose(1, 2);
    x = norm->forward(x);
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// SwinTransformer
// ─────────────────────────────────────────────────────────────────────────────
SwinTransformerImpl::SwinTransformerImpl(
    int64_t img_size, int64_t patch_size, int64_t in_chans, int64_t num_classes,
    int64_t embed_dim, std::vector<int64_t> depths, std::vector<int64_t> num_heads,
    int64_t window_size, double mlp_ratio, double dropout, double attn_drop)
{
    // Patch embedding
    patch_embed = register_module("patch_embed",
                      PatchEmbed(img_size, patch_size, in_chans, embed_dim));
    pos_drop    = register_module("pos_drop", torch::nn::Dropout(dropout));

    stages      = register_module("stages", torch::nn::ModuleList());

    int64_t num_stages = static_cast<int64_t>(depths.size());  // 4
    int64_t cur_dim    = embed_dim;
    int64_t cur_H      = img_size / patch_size;
    int64_t cur_W      = img_size / patch_size;

    stage_resolutions.clear();
    for (int64_t i = 0; i < num_stages; ++i) {
        bool last = (i == num_stages - 1);
        auto stage = std::make_shared<SwinStageImpl>(
            cur_dim, cur_H, cur_W, depths[i], num_heads[i],
            window_size, mlp_ratio, dropout, attn_drop,
            /*use_downsample=*/!last);
        stage_resolutions.push_back({cur_H, cur_W});
        stages->push_back(stage);

        if (!last) {
            cur_H    /= 2;
            cur_W    /= 2;
            cur_dim  *= 2;
        }
    }

    _num_features = cur_dim;   // 8C after 3 patch-merging layers

    norm = register_module("norm",
               torch::nn::LayerNorm(torch::nn::LayerNormOptions({_num_features})));
    head = register_module("head",
               torch::nn::Linear(_num_features, num_classes));

    // Weight initialisation (standard for vision Transformers)
    for (auto& m : modules(/*include_self=*/false)) {
        if (auto* lin = m->as<torch::nn::LinearImpl>()) {
            torch::nn::init::normal_(lin->weight, 0.0, 0.02);
            if (lin->bias.defined())
                torch::nn::init::constant_(lin->bias, 0.0);
        } else if (auto* ln = m->as<torch::nn::LayerNormImpl>()) {
            torch::nn::init::constant_(ln->bias,   0.0);
            torch::nn::init::constant_(ln->weight, 1.0);
        }
    }
    // head bias = 0 (classifier)
    torch::nn::init::constant_(head->bias, 0.0);
}

torch::Tensor SwinTransformerImpl::forward(torch::Tensor x) {
    // Patch embedding: [B, 3, H, W] → [B, H/P * W/P, C]
    x = patch_embed->forward(x);
    x = pos_drop->forward(x);

    // 4 stages
    for (size_t i = 0; i < stages->size(); ++i) {
        auto  stage = stages->ptr(i)->as<SwinStageImpl>();
        auto [out, out_H, out_W] = stage->forward(x);
        x = out;
        (void)out_H; (void)out_W;
    }

    // Global average pool + head
    x = norm->forward(x);            // [B, L, 8C]
    x = x.mean(1);                   // [B, 8C]  — global avg over sequence
    x = head->forward(x);            // [B, num_classes]
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory functions
// ─────────────────────────────────────────────────────────────────────────────
std::shared_ptr<SwinTransformerImpl> make_swin_t(int64_t num_classes, int64_t img_size) {
    return std::make_shared<SwinTransformerImpl>(
        img_size, 4, 3, num_classes,
        /*embed_dim=*/96,
        /*depths=*/std::vector<int64_t>{2, 2, 6, 2},
        /*num_heads=*/std::vector<int64_t>{3, 6, 12, 24},
        /*window_size=*/7);
}

std::shared_ptr<SwinTransformerImpl> make_swin_s(int64_t num_classes, int64_t img_size) {
    return std::make_shared<SwinTransformerImpl>(
        img_size, 4, 3, num_classes,
        /*embed_dim=*/96,
        /*depths=*/std::vector<int64_t>{2, 2, 18, 2},
        /*num_heads=*/std::vector<int64_t>{3, 6, 12, 24},
        /*window_size=*/7);
}

std::shared_ptr<SwinTransformerImpl> make_swin_b(int64_t num_classes, int64_t img_size) {
    return std::make_shared<SwinTransformerImpl>(
        img_size, 4, 3, num_classes,
        /*embed_dim=*/128,
        /*depths=*/std::vector<int64_t>{2, 2, 18, 2},
        /*num_heads=*/std::vector<int64_t>{4, 8, 16, 32},
        /*window_size=*/7);
}

std::shared_ptr<SwinTransformerImpl> make_swin_l(int64_t num_classes, int64_t img_size) {
    return std::make_shared<SwinTransformerImpl>(
        img_size, 4, 3, num_classes,
        /*embed_dim=*/192,
        /*depths=*/std::vector<int64_t>{2, 2, 18, 2},
        /*num_heads=*/std::vector<int64_t>{6, 12, 24, 48},
        /*window_size=*/7);
}

// ─────────────────────────────────────────────────────────────────────────────
// Training / evaluation helpers
// ─────────────────────────────────────────────────────────────────────────────
float swin_train_epoch(torch::nn::AnyModule&  model,
                       torch::optim::AdamW&   optimizer,
                       torch::Device          device,
                       const std::vector<std::pair<torch::Tensor,torch::Tensor>>& batches) {
    model.ptr()->train();
    double total_loss = 0.0;
    int64_t steps = 0;
    for (auto& [data, target] : batches) {
        auto x = data.to(device);
        auto y = target.to(device);
        optimizer.zero_grad();
        auto out  = model.forward<torch::Tensor>(x);
        auto loss = torch::nn::functional::cross_entropy(out, y);
        loss.backward();
        // Gradient clipping (common for Transformers)
        torch::nn::utils::clip_grad_norm_(model.ptr()->parameters(), 5.0);
        optimizer.step();
        total_loss += loss.item<double>();
        ++steps;
    }
    return steps > 0 ? static_cast<float>(total_loss / steps) : 0.f;
}

std::pair<float,float> swin_evaluate(
    torch::nn::AnyModule& model,
    torch::Device         device,
    const std::vector<std::pair<torch::Tensor,torch::Tensor>>& batches) {
    model.ptr()->eval();
    torch::NoGradGuard no_grad;
    int64_t correct1 = 0, correct5 = 0, total = 0;
    for (auto& [data, target] : batches) {
        auto x = data.to(device);
        auto y = target.to(device);
        auto out  = model.forward<torch::Tensor>(x);
        auto pred = out.topk(5, 1, true, true);
        auto indices = std::get<1>(pred);  // [N, 5]
        auto y_exp = y.unsqueeze(1).expand_as(indices);
        auto correct = indices.eq(y_exp);
        correct1 += correct.select(1, 0).sum().item<int64_t>();
        correct5 += correct.any(1).sum().item<int64_t>();
        total    += x.size(0);
    }
    if (total == 0) return {0.f, 0.f};
    return {static_cast<float>(correct1) / total,
            static_cast<float>(correct5) / total};
}

void swin_train(torch::nn::AnyModule&   model,
                const SwinTrainConfig&  cfg,
                const std::vector<std::pair<torch::Tensor,torch::Tensor>>& train_batches,
                const std::vector<std::pair<torch::Tensor,torch::Tensor>>& val_batches,
                const std::string&      save_path) {
    model.ptr()->to(cfg.device);

    // AdamW optimizer (paper Section 4.1)
    auto opt_opts = torch::optim::AdamWOptions(cfg.lr)
                        .weight_decay(cfg.weight_decay);
    torch::optim::AdamW optimizer(model.ptr()->parameters(), opt_opts);

    float best_top1 = 0.f;
    for (int64_t epoch = 0; epoch < cfg.max_epochs; ++epoch) {
        // Cosine LR schedule with linear warm-up (Section 4.1)
        double lr;
        if (epoch < cfg.warmup_epochs) {
            lr = cfg.lr * (epoch + 1.0) / cfg.warmup_epochs;
        } else {
            double progress = static_cast<double>(epoch - cfg.warmup_epochs) /
                              (cfg.max_epochs - cfg.warmup_epochs);
            lr = cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) *
                 (1.0 + std::cos(M_PI * progress));
        }
        for (auto& pg : optimizer.param_groups())
            pg.options().set_lr(lr);

        float loss = swin_train_epoch(model, optimizer, cfg.device, train_batches);
        auto [top1, top5] = swin_evaluate(model, cfg.device, val_batches);
        std::cout << "[Swin] epoch " << epoch + 1 << "/" << cfg.max_epochs
                  << "  loss=" << loss
                  << "  top1=" << top1 * 100.f << "%"
                  << "  top5=" << top5 * 100.f << "%"
                  << "  lr="   << lr << "\n";

        if (top1 > best_top1) {
            best_top1 = top1;
            torch::serialize::OutputArchive archive;
            model.ptr()->save(archive);
            archive.save_to(save_path);
            std::cout << "[Swin] saved checkpoint → " << save_path << "\n";
        }
    }
}

} // namespace vision
} // namespace models
} // namespace dm
