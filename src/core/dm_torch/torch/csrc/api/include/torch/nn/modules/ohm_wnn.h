#pragma once
/*
 * OhmWeightlessNet (Ohm-WNN) -- Zero-FLOP Weightless Neural Network (dm::prim)
 *
 * A WiSARD/RAM-based discriminant network where inference is purely O(1)
 * integer memory addressing.  No matrix multiplication, no floating-point
 * arithmetic in the forward pass.
 *
 * Algorithm (address-decoding / RAM-lookup):
 *
 *   The input x [B, n_inputs] is partitioned into n_discriminators chunks of K
 *   positions each.  Each chunk is bit-packed into an integer address:
 *
 *     idx[b, d] = sum_{j=0}^{K-1}  x[b, d*K+j] << j        (Eq. 1)
 *
 *   For every output class o, the arena cell at (o, d, idx[b,d]) is read and
 *   all n_discriminators votes are summed:
 *
 *     y[b, o] = sum_{d=0}^{n_disc-1}  arena[o, d, idx[b,d]]  (Eq. 2)
 *
 * Training (WiSARD-style discrete write, no backprop):
 *
 *     arena[label[b], d, idx[b,d]]   += 1   for all b, d
 *     arena[o,        d, idx[b,d]]   -= 1   for all b, d, o != label[b]
 *                                            (only when use_bleaching=true)
 *
 * Memory layout: arena [n_outputs, n_discriminators, 1<<K] int32
 *   -- registered as a buffer (not a parameter), so it is serialised via
 *      state_dict() / torch::save / torch::load but receives no gradient.
 *
 * Usage:
 *   dm::prim::OhmWeightlessNet wnn(128, 10);   // 128 inputs, 10 classes
 *   wnn->teach(x_batch, labels);               // discrete training
 *   auto votes = wnn->forward(x_batch);        // [B, 10] int32
 *   auto pred  = wnn->predict(x_batch);        // [B]     int64 argmax
 */

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include <torch/csrc/Export.h>

#include <cstdint>
#include <ostream>
#include <stdexcept>

namespace dm {
namespace prim {

// -----------------------------------------------------------------------------
// Options
// -----------------------------------------------------------------------------

struct OhmWeightlessNetOptions {
    OhmWeightlessNetOptions(int64_t n_inputs, int64_t n_outputs)
        : n_inputs_(n_inputs), n_outputs_(n_outputs) {}

    TORCH_ARG(int64_t, n_inputs);
    TORCH_ARG(int64_t, n_outputs);
    // Chunk size in bits: each discriminator addresses 2^K cells (default 8).
    TORCH_ARG(int64_t, K) = 8;
    // Bits per input value: 1=binary, 2/4/8=quantized (default 1).
    TORCH_ARG(int64_t, bits_per_input) = 1;
    // Decrement non-target classes during teach() (default true).
    TORCH_ARG(bool, use_bleaching) = true;

 public:
    int64_t n_discriminators() const { return (n_inputs() + K() - 1) / K(); }
    int64_t padded_width()     const { return n_discriminators() * K(); }
};

// -----------------------------------------------------------------------------
// Implementation class
// -----------------------------------------------------------------------------

class TORCH_API OhmWeightlessNetImpl : public torch::nn::Module {
 public:
    explicit OhmWeightlessNetImpl(int64_t n_inputs, int64_t n_outputs)
        : OhmWeightlessNetImpl(OhmWeightlessNetOptions(n_inputs, n_outputs)) {}

    explicit OhmWeightlessNetImpl(OhmWeightlessNetOptions options_);

    void reset();
    void pretty_print(std::ostream& stream) const override;

    // x: [B, n_inputs] int32 or bool  ->  [B, n_outputs] int32
    torch::Tensor forward(const torch::Tensor& x);

    // WiSARD discrete training.  labels: [B] int64 in [0, n_outputs).
    void teach(const torch::Tensor& x, const torch::Tensor& labels);

    // Returns argmax of forward(x): [B] int64.
    torch::Tensor predict(const torch::Tensor& x);

    OhmWeightlessNetOptions options;

    // [n_outputs, n_discriminators, 1<<K] int32 — registered buffer.
    torch::Tensor arena;

 private:
    // Returns bit-packed discriminator addresses: [B, n_disc_] int32.
    torch::Tensor _pack(const torch::Tensor& x) const;

    int64_t n_disc_;
    int64_t addr_sz_;
};

TORCH_MODULE(OhmWeightlessNet);

// -----------------------------------------------------------------------------
// Inline implementations
// -----------------------------------------------------------------------------

inline OhmWeightlessNetImpl::OhmWeightlessNetImpl(
    OhmWeightlessNetOptions options_)
    : options(std::move(options_))
{
    if (options.n_inputs()  <= 0)
        throw std::invalid_argument("OhmWeightlessNet: n_inputs must be > 0");
    if (options.n_outputs() <= 0)
        throw std::invalid_argument("OhmWeightlessNet: n_outputs must be > 0");
    if (options.K() < 1 || options.K() > 30)
        throw std::invalid_argument("OhmWeightlessNet: K must be in [1, 30]");
    if (options.bits_per_input() < 1 || options.bits_per_input() > 8)
        throw std::invalid_argument("OhmWeightlessNet: bits_per_input must be in [1, 8]");

    n_disc_  = options.n_discriminators();
    addr_sz_ = int64_t{1} << options.K();
    reset();
}

inline void OhmWeightlessNetImpl::reset() {
    n_disc_  = options.n_discriminators();
    addr_sz_ = int64_t{1} << options.K();

    arena = register_buffer("arena",
        torch::zeros(
            {options.n_outputs(), n_disc_, addr_sz_},
            torch::TensorOptions().dtype(torch::kInt32)));
}

inline void OhmWeightlessNetImpl::pretty_print(std::ostream& stream) const {
    stream << "dm::prim::OhmWeightlessNet("
           << "n_inputs="          << options.n_inputs()
           << ", n_outputs="       << options.n_outputs()
           << ", K="               << options.K()
           << ", n_discriminators=" << n_disc_
           << ", arena_cells="     << (options.n_outputs() * n_disc_ * addr_sz_)
           << ", bleaching="       << (options.use_bleaching() ? "on" : "off")
           << ")";
}

inline torch::Tensor OhmWeightlessNetImpl::_pack(
    const torch::Tensor& x) const
{
    const int64_t B      = x.size(0);
    const int64_t K      = options.K();
    const int64_t padded = options.padded_width();

    auto xi = x.to(torch::kInt32);

    // Zero-pad if n_inputs is not a multiple of K.
    if (xi.size(1) < padded) {
        auto pad = torch::zeros(
            {B, padded - xi.size(1)},
            torch::TensorOptions().dtype(torch::kInt32).device(xi.device()));
        xi = torch::cat({xi, pad}, /*dim=*/1);
    }

    // [B, padded] -> [B, n_disc_, K]
    xi = xi.view({B, n_disc_, K});

    // Clamp to valid range for the configured quantisation.
    const int32_t max_val =
        static_cast<int32_t>((int64_t{1} << options.bits_per_input()) - 1);
    xi = xi.clamp(0, max_val);

    // Branchless bit-pack: idx[b,d] = sum_j( x[b,d,j] << j )
    // shifts: [K] -> broadcast [1, 1, K] over [B, n_disc_, K]
    auto shifts = torch::arange(K,
        torch::TensorOptions().dtype(torch::kInt32).device(xi.device()))
        .view({1, 1, K});

    return torch::bitwise_left_shift(xi, shifts).sum(/*dim=*/-1).to(torch::kInt32);  // [B, n_disc_]
}

inline torch::Tensor OhmWeightlessNetImpl::forward(
    const torch::Tensor& x)
{
    const int64_t B = x.size(0);
    const int64_t O = options.n_outputs();

    auto idx = _pack(x);   // [B, n_disc_] int32

    // Expand idx over the output dimension and arena over the batch dimension
    // (both are zero-copy view expansions).
    //
    // idx_g:      [B, O, n_disc_, 1]  int64  -- gather index
    // arena_exp:  [B, O, n_disc_, addr_sz_]  -- zero-copy expand
    auto idx_g = idx.unsqueeze(1)
                    .expand({B, O, n_disc_})
                    .unsqueeze(-1)
                    .to(torch::kInt64);                        // [B,O,n_disc_,1]

    auto arena_exp = arena.unsqueeze(0)
                         .expand({B, O, n_disc_, addr_sz_});   // zero-copy

    // gathered[b,o,d,0] = arena[o, d, idx[b,d]]
    auto gathered = arena_exp.gather(/*dim=*/3, idx_g)
                             .squeeze(-1);                      // [B, O, n_disc_]

    // Sum votes over discriminators.
    return gathered.sum(/*dim=*/2);   // [B, n_outputs] int32
}

inline void OhmWeightlessNetImpl::teach(
    const torch::Tensor& x,
    const torch::Tensor& labels)
{
    torch::NoGradGuard ng;

    const int64_t B          = x.size(0);
    const int64_t O          = options.n_outputs();
    const int64_t out_stride = n_disc_ * addr_sz_;

    auto idx = _pack(x);  // [B, n_disc_] int32

    // Work on a 1D view of the arena for scatter_add_.
    auto arena_flat = arena.view({-1});   // [O * n_disc_ * addr_sz_]

    auto dev = x.device();

    // Discriminator base offsets: [1, n_disc_] int64
    auto disc_off = torch::arange(n_disc_,
        torch::TensorOptions().dtype(torch::kInt64).device(dev))
        .unsqueeze(0) * addr_sz_;

    auto idx_long = idx.to(torch::kInt64);  // [B, n_disc_]

    // Flat index for correct class: [B, n_disc_] -> [B * n_disc_]
    auto labels_off = labels.view({B, 1}).expand({B, n_disc_}) * out_stride;
    auto correct_flat = (labels_off + disc_off + idx_long).view({-1});

    auto ones = torch::ones(
        {B * n_disc_},
        torch::TensorOptions().dtype(torch::kInt32).device(dev));

    arena_flat.scatter_add_(0, correct_flat, ones);

    if (options.use_bleaching()) {
        // All-class flat indices: [B, O, n_disc_]
        auto all_out_off = torch::arange(O,
            torch::TensorOptions().dtype(torch::kInt64).device(dev))
            .view({1, O, 1}) * out_stride;

        auto all_flat = (all_out_off
                        + disc_off.unsqueeze(1)    // [B, 1, n_disc_]
                        + idx_long.unsqueeze(1))   // [B, 1, n_disc_]
                        .view({B, O, n_disc_});     // [B, O, n_disc_]

        // Mask: true where output class != label[b]  -> [B, O, n_disc_]
        auto out_range = torch::arange(O,
            torch::TensorOptions().dtype(torch::kInt64).device(dev))
            .view({1, O, 1}).expand({B, O, n_disc_});
        auto wrong_mask = out_range.ne(labels.view({B, 1, 1}).expand({B, O, n_disc_}));

        auto wrong_flat = all_flat.masked_select(wrong_mask).to(torch::kInt64);
        if (wrong_flat.numel() > 0) {
            auto neg_ones = torch::full(
                {wrong_flat.numel()}, -1,
                torch::TensorOptions().dtype(torch::kInt32).device(dev));
            arena_flat.scatter_add_(0, wrong_flat, neg_ones);
        }
    }
}

inline torch::Tensor OhmWeightlessNetImpl::predict(
    const torch::Tensor& x)
{
    return forward(x).argmax(/*dim=*/1);  // [B] int64
}

}  // namespace prim
}  // namespace dm
