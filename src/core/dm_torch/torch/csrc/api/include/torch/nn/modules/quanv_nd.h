#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// QuanvNd — Quantum-Inspired Convolutional Layer  (dm::prim)
//
// Implements a Virtual Quantum Circuit (VQC) operating inside a non-overlapping
// sliding window of kernel=2, stride=2 across 1-D, 2-D, or 3-D spatial grids.
//
// Architectural summary:
//   D=1  window 2×1  → 2 virtual qubits  → 2 output channels / input channel
//   D=2  window 2×2  → 4 virtual qubits  → 4 output channels / input channel
//   D=3  window 2×2×2→ 8 virtual qubits  → 8 output channels / input channel
//
// Three-phase VQC per cell (all operations are differentiable LibTorch ops):
//   1. Quantum Embedding   — RY rotation: |ψ_i⟩ = cos(x_i)|0⟩ + sin(x_i)|1⟩
//   2. Quantum Entanglement — parameterised CX-like rotation gate (δθ ∈ ℝ,
//                             learned via back-prop): rotates paired amplitudes
//   3. Quantum Measurement  — Born-rule collapse: P(|1⟩) = sin²(β_i) = β²
//
// Memory contract (forward pass, no heap allocation):
//   - All intermediate tensors are views / in-place operations on the
//     pre-allocated `unfold`-ed patch tensor.  No std::vector resizing occurs
//     on the hot path.
//   - The layer is safe to call inside a `torch::NoGradGuard` context or with
//     autograd enabled; the computational graph is preserved end-to-end.
//
// Output tensor shapes:
//   D=1: [B, C_in * 2,        L/2]
//   D=2: [B, C_in * 4,        H/2, W/2]
//   D=3: [B, C_in * 8,        D/2, H/2, W/2]
//
// Usage (dm project):
//   #include <torch/nn/modules/quanv_nd.h>
//
//   dm::prim::QuanvNd<2> qconv(3);  // 3 input channels, 2-D
//   auto y = qconv(x);              // x: [B, 3, H, W]
//
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include <torch/csrc/Export.h>

#include <cstddef>
#include <string>

namespace dm {
namespace prim {

// ── Options ───────────────────────────────────────────────────────────────────

struct QuanvNdOptions {
    explicit QuanvNdOptions(int64_t in_channels) : in_channels_(in_channels) {}

    /// Number of input channels (C_in).
    TORCH_ARG(int64_t, in_channels);

    /// If true, inputs are assumed to already lie in [-π, π]; no tanh
    /// normalisation is applied.  Default: false (apply π·tanh(x) pre-scaling).
    TORCH_ARG(bool, assume_normalised) = false;
};

// ── Implementation ────────────────────────────────────────────────────────────

/// QuanvNdImpl<D> — template over spatial dimension D ∈ {1, 2, 3}.
///
/// Inherits torch::nn::Module so it registers parameters, participates in
/// `to()`, `save()`/`load()`, and autograd automatically.
template <size_t D>
class TORCH_API QuanvNdImpl : public torch::nn::Module {
 public:
  static_assert(D >= 1 && D <= 3,
                "QuanvNd only supports D ∈ {1, 2, 3}.");

  /// Number of virtual qubits = 2^D.
  static constexpr int64_t N_QUBITS = (1LL << D);

  /// Sliding-window kernel / stride (always 2, non-overlapping).
  static constexpr int64_t KERNEL   = 2;
  static constexpr int64_t STRIDE   = 2;

  explicit QuanvNdImpl(int64_t in_channels)
      : QuanvNdImpl(QuanvNdOptions(in_channels)) {}

  explicit QuanvNdImpl(QuanvNdOptions options_);

  /// Reset / re-initialise learnable rotation angles δθ.
  void reset();

  /// Pretty-print layer description.
  void pretty_print(std::ostream& stream) const override;

  /// Forward pass.  Input `x` must have D+2 dimensions: [B, C_in, spatial...].
  torch::Tensor forward(const torch::Tensor& x);

  /// Public config.
  QuanvNdOptions options;

  // ── Learnable parameters ───────────────────────────────────────────────────
  //
  // delta_theta: [C_in, N_QUBITS]
  //   One rotation angle per (input channel, qubit).  These are the only
  //   trainable weights; the VQC has no biases (phase symmetry is broken by
  //   δθ alone).
  torch::Tensor delta_theta;

 private:
  // ── Private VQC primitives ─────────────────────────────────────────────────

  /// Phase 1 — Quantum Embedding.
  ///
  /// Maps a real scalar x_i → qubit amplitudes (α_i, β_i):
  ///   α_i = cos(x_i)   →  amplitude of |0⟩
  ///   β_i = sin(x_i)   →  amplitude of |1⟩
  ///
  /// Inputs:
  ///   patches — [B, C_in, N_QUBITS, N_cells]  extracted window values,
  ///             pre-normalised to (-π/2, π/2).
  /// Outputs:
  ///   alpha   — [B, C_in, N_QUBITS, N_cells]  cos(patches)
  ///   beta    — [B, C_in, N_QUBITS, N_cells]  sin(patches)
  std::pair<torch::Tensor, torch::Tensor>
  vqc_embed(const torch::Tensor& patches) const;

  /// Phase 2 — Quantum Entanglement.
  ///
  /// Applies a parameterised two-qubit rotation gate to adjacent pairs.
  /// The gate models the action of a CX-like unitary with a learned angle δθ:
  ///
  ///   For a pair (α_i, β_i), (α_j, β_j):
  ///     new_β_i = β_i · cos(δθ_{c,i}) − β_j · sin(δθ_{c,i})
  ///     new_β_j = β_j · cos(δθ_{c,i}) + β_i · sin(δθ_{c,i})
  ///   (α amplitudes are left invariant — the gate acts on the |1⟩ subspace.)
  ///
  ///   D=1: entangle qubit pairs (0,1)
  ///   D=2: entangle pairs (0,1) then (2,3)  [first-order ZZ couplings]
  ///   D=3: entangle pairs (0,1),(2,3),(4,5),(6,7) then (0,2),(1,3),(4,6),(5,7)
  ///         [two-level cascade covering all nearest-neighbour edges of a cube]
  ///
  /// Modifies beta in-place (alpha stays unchanged after embedding).
  torch::Tensor vqc_entangle(torch::Tensor beta) const;

  /// Phase 3 — Quantum Measurement (Born rule).
  ///
  ///   P_i(|1⟩) = |β_i|²
  ///
  /// This collapses the complex qubit state to a real-valued feature.
  /// The pow(2) is differentiable; gradients flow back through β to δθ.
  ///
  /// Returns [B, C_in, N_QUBITS, N_cells].
  torch::Tensor vqc_measure(const torch::Tensor& beta) const;

  // ── Spatial helpers ────────────────────────────────────────────────────────

  /// Tile unfold: extracts all non-overlapping 2^D windows into a flat patch
  /// tensor of shape [B, C_in, N_QUBITS, N_cells].
  ///
  /// Uses torch::Tensor::unfold() chained across each spatial axis so the
  /// entire extraction is a zero-copy strided view on the original tensor.
  torch::Tensor extract_patches(const torch::Tensor& x) const;

  /// Reconstruct spatial output shape from patch tensor.
  /// Returns [B, C_in * N_QUBITS, spatial_out...].
  torch::Tensor reshape_output(const torch::Tensor& measured,
                               const torch::Tensor& x) const;
};

// ── TORCH_MODULE macro generates the ModuleHolder wrapper ─────────────────────
// We must instantiate for each D because TORCH_MODULE cannot be templated.

/// 1-D Quantum Convolutional Layer  (window 2, stride 2, output 2 ch/input).
class TORCH_API Quanv1dImpl final : public QuanvNdImpl<1> {
 public:
  using QuanvNdImpl<1>::QuanvNdImpl;
};
TORCH_MODULE(Quanv1d);

/// 2-D Quantum Convolutional Layer  (window 2×2, stride 2, output 4 ch/input).
class TORCH_API Quanv2dImpl final : public QuanvNdImpl<2> {
 public:
  using QuanvNdImpl<2>::QuanvNdImpl;
};
TORCH_MODULE(Quanv2d);

/// 3-D Quantum Convolutional Layer  (window 2×2×2, stride 2, output 8 ch/input).
class TORCH_API Quanv3dImpl final : public QuanvNdImpl<3> {
 public:
  using QuanvNdImpl<3>::QuanvNdImpl;
};
TORCH_MODULE(Quanv3d);

/// Generic alias — preferred for template code inside dm modules.
template <size_t D>
using QuanvNd = typename std::conditional<D == 1,
                  Quanv1d,
                  typename std::conditional<D == 2,
                    Quanv2d,
                    Quanv3d
                  >::type
                >::type;

}  // namespace prim
}  // namespace dm
