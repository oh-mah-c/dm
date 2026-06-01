#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Criss-Cross Attention  (dm::prim)
// Huang et al., "CCNet: Criss-Cross Attention for Semantic Segmentation"
// IEEE TPAMI 2020  (arXiv:1811.11721v2)
//
// Overview (Section 3.2, Fig. 3):
//   Given a local feature map H in R^{C x W x H}, for each spatial position u:
//     1. Affinity: Q and K projections (1x1 conv, C' channels) yield a sparse
//        attention score over the (H+W-1) positions on the criss-cross path of u.
//        d_{i,u} = Q_u * Omega_{i,u}^T   (Eq. 1)
//        A = softmax(D, dim=channel)
//     2. Aggregation: V projection (1x1 conv, C channels) collects context:
//        H'_u = sum_{i=0}^{H+W-1} A_{i,u} * Phi_{i,u} + H_u  (Eq. 2)
//
// Usage:
//   dm::prim::CrissCrossAttention attn(256, 64);  // in_channels=256, key_channels=64
//   torch::Tensor out = attn(x);                  // x: [B,C,H,W] -> [B,C,H,W]
//
// For full RCCA (R=2 recurrent), apply the module twice and share parameters:
//   h1 = attn(x);
//   h2 = attn(h1);
// The CCNetImpl model below wraps this with a loop.
//
// Complexity: O(N*sqrt(N)) vs O(N^2) for non-local block (Section 1).
// Memory: ~11x less GPU memory than non-local at the same mIoU (Table 5).
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include <torch/csrc/Export.h>

#include <cstdint>
#include <stdexcept>

namespace dm { namespace prim {

// ─────────────────────────────────────────────────────────────────────────────
// CrissCrossAttentionOptions
// ─────────────────────────────────────────────────────────────────────────────
struct TORCH_API CrissCrossAttentionOptions {
    CrissCrossAttentionOptions(int64_t in_channels, int64_t key_channels)
        : in_channels_(in_channels),
          key_channels_(key_channels) {}

    TORCH_ARG(int64_t, in_channels);
    TORCH_ARG(int64_t, key_channels);
};

// ─────────────────────────────────────────────────────────────────────────────
// CrissCrossAttentionImpl
//
// Forward:
//   x   : [B, C, H, W]   (= H in paper notation)
//   out : [B, C, H, W]   (= H' in paper notation, includes residual H_u)
//
// Internal tensors during forward:
//   Q, K : [B, C', H, W]
//   V    : [B, C,  H, W]
//   D    : [B, (H+W-1), H*W]  — criss-cross affinity scores before softmax
//   A    : softmax(D, dim=0)
//   out  : aggregation via A over criss-cross positions + residual
// ─────────────────────────────────────────────────────────────────────────────
class TORCH_API CrissCrossAttentionImpl : public torch::nn::Module {
 public:
    explicit CrissCrossAttentionImpl(int64_t in_channels, int64_t key_channels)
        : CrissCrossAttentionImpl(
              CrissCrossAttentionOptions(in_channels, key_channels)) {}
    explicit CrissCrossAttentionImpl(CrissCrossAttentionOptions options_);

    void reset();
    void pretty_print(std::ostream& stream) const override;

    // x: [B, C, H, W] -> [B, C, H, W]
    torch::Tensor forward(const torch::Tensor& x);

    CrissCrossAttentionOptions options;

    torch::nn::Conv2d query_conv{nullptr};   // 1x1: C -> C'
    torch::nn::Conv2d key_conv{nullptr};     // 1x1: C -> C'
    torch::nn::Conv2d value_conv{nullptr};   // 1x1: C -> C
    torch::nn::BatchNorm2d bn{nullptr};      // BN on value output (before residual)

 private:
    // Gather the H+W-1 criss-cross feature vectors for each spatial position.
    // x_gather: [B, C', H, W] -> [B, C', (H+W-1), H*W]
    static torch::Tensor _gather_criss_cross(const torch::Tensor& x);

    // Scatter the aggregated context back into [B, C, H, W].
    // context: [B, C, (H+W-1), H*W], A: [B, (H+W-1), H*W]
    static torch::Tensor _aggregate(const torch::Tensor& value,
                                    const torch::Tensor& A);
};

TORCH_MODULE(CrissCrossAttention);

// ─────────────────────────────────────────────────────────────────────────────
// Inline implementations
// ─────────────────────────────────────────────────────────────────────────────

inline CrissCrossAttentionImpl::CrissCrossAttentionImpl(
    CrissCrossAttentionOptions options_)
    : options(std::move(options_)) {
    reset();
}

inline void CrissCrossAttentionImpl::reset() {
    int64_t C  = options.in_channels();
    int64_t Ck = options.key_channels();

    query_conv = register_module("query_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(C, Ck, 1).bias(false)));
    key_conv   = register_module("key_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(C, Ck, 1).bias(false)));
    value_conv = register_module("value_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(C, C, 1).bias(false)));
    bn = register_module("bn", torch::nn::BatchNorm2d(C));
}

inline void CrissCrossAttentionImpl::pretty_print(std::ostream& stream) const {
    stream << "CrissCrossAttention(in=" << options.in_channels()
           << ", key=" << options.key_channels() << ")";
}

// Gather the (H+W-1) criss-cross neighbour features for every position.
//
// For each position (h, w), the criss-cross path consists of:
//   - all W positions in row h:     (h, 0), ..., (h, W-1)
//   - all H positions in col w:     (0, w), ..., (H-1, w)
//   minus the overlap at (h, w) itself (counted once).
//
// Returns a tensor of shape [B, C', (H+W-1), H*W].
// Indexing convention: first W entries = same row, next H-1 = same col
// (excluding the self position which is at index w in the row slice).
inline torch::Tensor CrissCrossAttentionImpl::_gather_criss_cross(
    const torch::Tensor& x)
{
    // x: [B, C', H, W]
    const int64_t B  = x.size(0);
    const int64_t Ck = x.size(1);
    const int64_t H  = x.size(2);
    const int64_t W  = x.size(3);

    // Row context: for each row h, all W columns.
    // x_row[b, c, h, w, col] = x[b, c, h, col]
    // Shape: [B, Ck, H, W, W] but we want per-query-position W neighbours.
    // Use expand: for query at (h,w), row slice is x[:, :, h, :] -> [B,Ck,W]
    // We broadcast x along the query-position dim (W) and extract the W
    // entries from the same row. This is simply x itself (each column
    // has all W entries in that row available via the spatial dims).

    // Simpler approach: build the (H+W-1) criss-cross context tensor directly.
    // For query position (h, w):
    //   - row entries: x[:, :, h, 0..W-1]           -> W vectors
    //   - col entries: x[:, :, 0..H-1, w] excluding h -> H-1 vectors
    // Concatenate to get H+W-1 vectors.
    //
    // We do this for ALL positions simultaneously using tensor ops:
    //
    // Row part [B, Ck, H, W, W]:
    //   x_row = x.unsqueeze(4).expand(-1,-1,-1,-1,W)   <- wrong, think again
    //
    // The standard efficient implementation:
    //   row_ctx: [B, Ck, H, W] -> for each (h,w): take x[:,:,h,:]
    //     => unsqueeze(3): [B,Ck,H,1,W] tiled to [B,Ck,H,W,W]
    //   col_ctx: [B, Ck, H, W] -> for each (h,w): take x[:,:,:,w] minus self
    //     => unsqueeze(4): [B,Ck,H,W,1] ... careful
    //
    // We return [B, Ck, (H+W-1), N] where N=H*W.

    // Row context: x[:, :, h, :] for each query (h,w)
    // Replicate across the W-dimension of queries: [B,Ck,H,W] each row h
    // For all queries in row h: they share the same W row values.
    // Layout: [B, Ck, H, W, W] but that's large; use gather instead.
    //
    // Efficient: use torch::repeat and index.
    // x_row: [B, Ck, H, 1, W] -> expand [B, Ck, H, W, W]
    auto x_row = x.unsqueeze(3)
                  .expand({B, Ck, H, W, W});  // [B,Ck,H,W,W] - row ctx per position
    // x_col: [B, Ck, H, W, 1] -> for col context we want x[:,:,:,w] for each w
    // => x.permute(0,1,3,2) gives [B,Ck,W,H], then unsqueeze(3) expand
    auto x_col = x.permute({0, 1, 3, 2})
                  .unsqueeze(3)
                  .expand({B, Ck, W, H, H});  // [B,Ck,W,H,H] - col ctx per col-position
    // x_col[b,c,w,h,:] = x[b,c,:,w]

    // Build output [B, Ck, H+W-1, H*W].
    // For position idx = h*W + w:
    //   first W  entries: x_row[b,c,h,w,:] = x[b,c,h,:] (row)
    //   next H-1 entries: x[b,c,0..H-1 except h, w]     (col without self)
    //
    // This is complex to express as one contiguous operation.
    // We use the "pad + unfold" approach common in open-source CCNet impls:

    // Horizontal path: [B, Ck, H, W] -> for each row, all W positions
    // h_feat[b,c,u,i] = x[b,c, u//W, i]  where i in [0,W)
    auto h_feat = x.view({B, Ck, H, 1, W})
                   .expand({B, Ck, H, W, W})
                   .contiguous()
                   .view({B, Ck, H * W, W}); // [B,Ck,N,W]

    // Vertical path: [B, Ck, H, W] -> for each col, all H positions
    // We need, for query at position h, the OTHER H-1 rows in the same column.
    // Trick: gather all H positions then mask/remove self with circular shift.
    // Standard approach: pad the column vector and extract a window of H-1.
    //
    // v_feat[b,c,n,j] = x[b,c,j, n%W]  j != n//W
    // We include ALL H rows and then exclude the self position later
    // (standard CCNet: include self in col, resulting in H+W entries, then
    //  subtract self from D at aggregation). Actually the paper's official
    //  code keeps H+W-1 by excluding self from col.
    //
    // Use the "gather via index" method: for each column, roll the H entries
    // so the self position is last, then take first H-1.
    //
    // col_all[b,c,w,h] = x[b,c,h,w] — shape [B,Ck,W,H]
    auto col_all = x.permute({0, 1, 3, 2}).contiguous(); // [B,Ck,W,H]
    // For query at (h,w): we want col_all[b,c,w, 0..H-1 except h].
    // Use circular roll trick: roll col_all by (H-h) for each h, take [0..H-2].
    // This is per-query and hard to batch; use the simpler "include+exclude"
    // approach via unfold on a padded tensor:
    //
    // pad col dimension on both sides by H-1, unfold window H, stride 1
    // => for position h in [0,H): window is col_padded[h..h+H-1] of length H
    //    then drop the self element at index H-1 (since padding = H-1 before).
    auto col_pad = torch::nn::functional::pad(
        col_all,
        torch::nn::functional::PadFuncOptions({0, 0, 0, H - 1})); // [B,Ck,W,2H-1]
    // unfold last dim: size=H, step=1 -> [B,Ck,W,H,H]  (H windows of length H)
    auto col_unf = col_pad.unfold(/*dim=*/3, /*size=*/H, /*step=*/1); // [B,Ck,W,H,H]
    // col_unf[b,c,w,h,:] = col_all[b,c,w, h..h+H-1] which contains self at 0
    // We want to exclude the self-position (index 0 in the window) and keep H-1.
    // Actually with this padding, self is at index 0 of each window.
    // Drop index 0: col_unf[:,:,:,:,1:H] -> [B,Ck,W,H,H-1]
    // But we want chronological order: positions before h, then after h.
    // The paper doesn't require any specific order — just aggregation.
    // Simplest: keep indices [0..H-2] which corresponds to rows [h..h+H-2]
    // (excludes row h itself since self is at col_pad[h]).
    // Actually with the pad: col_pad[:,:,w,h] .. col_pad[:,:,w,h+H-1]
    // self is at col_all[:,:,w,h] which lands at index 0 (since pad=H-1 before).
    // Wait: pad({0,0,0,H-1}) pads H-1 zeros AFTER, not before.
    // col_pad[b,c,w, 0..H-1] = col_all[b,c,w, 0..H-1], [H..2H-2] = 0.
    // window for h: col_pad[b,c,w, h..h+H-1].  self = col_pad[b,c,w,h] at offset 0.
    // So drop offset 0: take offsets [1..H-1]. That gives H-1 entries:
    //   rows h+1, h+2, ..., h+H-1 (modulo/padded with zeros).
    // That's not correct — we lose rows 0..h-1.
    //
    // Correct padding: pad H-1 zeros BEFORE too, to enable a sliding window
    // that, for each h, places h at the CENTER and we can take all H-1 neighbours.
    // But this gives 2H-1 pad. Let's use a well-known technique:
    // pad H-1 on each side, unfold window=H, giving 2H-1 windows of length H.
    // Window h+H-1 covers positions [h..h+H-1] in the padded tensor, but that
    // includes positions outside [0,H).
    //
    // Cleanest approach (used in official CCNet): build two halves manually.

    // RESTART with the clean official approach:
    // For each query position n = h*W+w:
    //   criss-cross = row h (W entries) + col w minus self (H-1 entries)
    //   total H+W-1 entries.
    // Shape: [B, Ck, H+W-1, N] = [B, Ck, H+W-1, H*W]

    // Vertical part (H-1 entries per position, excluding self):
    // For col w, query row h: all rows except h.
    // Approach: gather all H rows, then roll so that the self row is last,
    // drop last. Using torch::roll:
    //   for each (b,c,w,h): roll col by -h, drop last element
    // This needs per-element roll which is not vectorizable directly.
    //
    // Use the "inf-diagonal" approach: build [B,Ck,W,H,H] then mask diagonal.
    // col_all: [B,Ck,W,H]
    // expanded: col_all.unsqueeze(3).expand(B,Ck,W,H,H): for each query-h,
    //   repeat all H col entries. Then mask diagonal (self) and gather H-1.
    //   => col_exp[b,c,w,h,j] = col_all[b,c,w,j]
    auto col_exp = col_all.unsqueeze(3).expand({B, Ck, W, H, H}); // [B,Ck,W,H,H]
    // Create index that for each query-h picks all j != h (H-1 entries).
    // Use a gather index: for row h, pick [h+1, h+2, ..., H-1, 0, 1, ..., h-1]
    // (circular shift by h+1). Shape: [H, H-1].
    auto arange = torch::arange(H, x.options().dtype(torch::kLong));
    // For each h, indices are arange rolled by -(h+1), then take [0..H-2].
    // Build [H, H-1] index tensor.
    auto idx_col = torch::zeros({H, H - 1}, x.options().dtype(torch::kLong));
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t j = 0; j < H - 1; ++j) {
            idx_col[h][j] = (h + 1 + j) % H;
        }
    }
    // col_exp[:,:,w,h,:] gathered with idx_col[h,:] -> [B,Ck,W,H,H-1]
    // idx_col: [H, H-1] -> broadcast to [B,Ck,W,H,H-1]
    auto idx_col_exp = idx_col.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                              .expand({B, Ck, W, H, H - 1}); // [B,Ck,W,H,H-1]
    auto v_feat_raw = col_exp.gather(/*dim=*/4, idx_col_exp); // [B,Ck,W,H,H-1]
    // Reshape to [B,Ck,N,H-1]: transpose W,H dims then reshape
    auto v_feat = v_feat_raw.permute({0, 1, 3, 2, 4})
                            .contiguous()
                            .view({B, Ck, H * W, H - 1}); // [B,Ck,N,H-1]

    // Concatenate row (W) and col-without-self (H-1) -> [B,Ck,N,H+W-1]
    auto feat = torch::cat({h_feat, v_feat}, /*dim=*/3); // [B,Ck,N,H+W-1]
    // Transpose to [B,Ck,H+W-1,N]
    return feat.permute({0, 1, 3, 2}).contiguous(); // [B,Ck,H+W-1,N]
}

// Compute aggregation: sum_i A_{i,u} * Phi_{i,u}
// value_cc: [B, C, H+W-1, N]  (criss-cross V features)
// A:        [B, H+W-1, N]     (attention weights, softmaxed)
// returns:  [B, C, N]
inline torch::Tensor CrissCrossAttentionImpl::_aggregate(
    const torch::Tensor& value_cc,
    const torch::Tensor& A)
{
    // Expand A for broadcasting: [B, 1, H+W-1, N]
    auto A_exp = A.unsqueeze(1); // [B,1,H+W-1,N]
    // Weighted sum over H+W-1 dimension: [B,C,N]
    return (value_cc * A_exp).sum(/*dim=*/2);
}

inline torch::Tensor CrissCrossAttentionImpl::forward(const torch::Tensor& x) {
    // x: [B, C, H, W]
    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t N = H * W;

    // Q, K: [B, C', H, W]
    auto Q = query_conv->forward(x); // [B,Ck,H,W]
    auto K = key_conv->forward(x);   // [B,Ck,H,W]
    // V: [B, C, H, W]
    auto V = value_conv->forward(x); // [B,C,H,W]

    // Gather criss-cross features: [B, Ck, H+W-1, N]
    auto Q_flat = Q.view({B, options.key_channels(), 1, N})
                   .expand({-1, -1, H + W - 1, -1}); // [B,Ck,H+W-1,N]
    auto K_cc = _gather_criss_cross(K); // [B,Ck,H+W-1,N]

    // Affinity (Eq. 1): d_{i,u} = Q_u * Omega_{i,u}^T
    // Sum over channel dimension C': [B, H+W-1, N]
    auto D = (Q_flat * K_cc).sum(/*dim=*/1); // [B,H+W-1,N]
    // Softmax over the H+W-1 positions for each query (dim=1)
    auto A = torch::softmax(D, /*dim=*/1); // [B,H+W-1,N]

    // Gather V on criss-cross path: [B, C, H+W-1, N]
    auto V_cc = _gather_criss_cross(V); // [B,C,H+W-1,N]

    // Aggregation (Eq. 2): sum_i A_{i,u} * Phi_{i,u}
    auto out = _aggregate(V_cc, A); // [B,C,N]
    out = out.view({B, C, H, W});   // [B,C,H,W]

    // BN + residual (H_u added back per Eq. 2)
    out = bn->forward(out) + x;

    return out;
}

}} // namespace dm::prim
