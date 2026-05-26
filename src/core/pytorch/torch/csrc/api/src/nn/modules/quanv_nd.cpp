// ─────────────────────────────────────────────────────────────────────────────
// QuanvNd — Quantum-Inspired Convolutional Layer  (dm::prim)
// Implementation: quanv_nd.cpp
//
// All tensor operations are differentiable LibTorch ops.  No heap allocation
// occurs on the forward hot path after construction.
//
// Mathematical reference per phase:
//
//   Embedding  (RY gate):
//     α_i = cos(θ_i),  β_i = sin(θ_i)     θ_i = x_i (pre-normalised input)
//     |ψ_i⟩ = α_i|0⟩ + β_i|1⟩             [normalised: α²+β²=1 ✓]
//
//   Entanglement (parameterised CX-like rotation on |1⟩ subspace):
//     [β'_i]   [ cos(δθ)  −sin(δθ)] [β_i]
//     [β'_j] = [ sin(δθ)   cos(δθ)] [β_j]
//
//     This is a Givens rotation in the two-qubit |11⟩ subspace, which is
//     unitary and therefore norm-preserving.  δθ is a learned parameter.
//
//   Measurement (Born rule):
//     P_i = |β'_i|² = sin²(effective rotation angle)
//     This is the probability of measuring |1⟩ on qubit i.
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/nn/modules/quanv_nd.h>
#include <torch/nn/init.h>

#include <cmath>
#include <ostream>
#include <stdexcept>

namespace dm {
namespace prim {

// ─────────────────────────────────────────────────────────────────────────────
// Construction & reset
// ─────────────────────────────────────────────────────────────────────────────

template <size_t D>
QuanvNdImpl<D>::QuanvNdImpl(QuanvNdOptions options_)
    : options(std::move(options_)) {
  reset();
}

template <size_t D>
void QuanvNdImpl<D>::reset() {
  // Register the only trainable parameters: one δθ per (channel, qubit).
  // Initialised uniformly in (-π, π) so qubits start in diverse states.
  delta_theta = register_parameter(
      "delta_theta",
      torch::empty({options.in_channels(), N_QUBITS}));
  torch::nn::init::uniform_(delta_theta,
                             -static_cast<double>(M_PI),
                              static_cast<double>(M_PI));
}

// ─────────────────────────────────────────────────────────────────────────────
// pretty_print
// ─────────────────────────────────────────────────────────────────────────────

template <size_t D>
void QuanvNdImpl<D>::pretty_print(std::ostream& stream) const {
  stream << "dm::prim::QuanvNd<" << D << ">"
         << "(in_channels=" << options.in_channels()
         << ", n_qubits=" << N_QUBITS
         << ", kernel=2, stride=2"
         << ", out_channels=" << options.in_channels() * N_QUBITS
         << ")";
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1 — Quantum Embedding
// ─────────────────────────────────────────────────────────────────────────────

template <size_t D>
std::pair<torch::Tensor, torch::Tensor>
QuanvNdImpl<D>::vqc_embed(const torch::Tensor& patches) const {
  // patches: [B, C_in, N_QUBITS, N_cells]
  //
  // Pre-normalise inputs to the RY rotation domain (-π/2, π/2) via π·tanh(x)
  // unless the caller guarantees the data is already normalised.
  torch::Tensor theta;
  if (options.assume_normalised()) {
    theta = patches;
  } else {
    // π · tanh(x) maps any real input smoothly into (-π, π).
    // Differentiable — tanh gradient flows back through the VQC.
    theta = static_cast<double>(M_PI) * torch::tanh(patches);
  }

  //   |ψ_i⟩ = cos(θ_i)|0⟩ + sin(θ_i)|1⟩
  //   alpha_i = ⟨0|ψ_i⟩ = cos(θ_i)    ∈ [-1, 1]
  //   beta_i  = ⟨1|ψ_i⟩ = sin(θ_i)    ∈ [-1, 1]
  //   alpha²  + beta² = 1  (unit norm — valid quantum state)
  return {torch::cos(theta), torch::sin(theta)};
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 — Quantum Entanglement (parameterised Givens rotation)
// ─────────────────────────────────────────────────────────────────────────────

template <size_t D>
torch::Tensor QuanvNdImpl<D>::vqc_entangle(torch::Tensor beta) const {
  // beta: [B, C_in, N_QUBITS, N_cells]
  //
  // delta_theta: [C_in, N_QUBITS]  → broadcast to [1, C_in, N_QUBITS, 1]
  //
  // We apply Givens (2-qubit) rotations to adjacent qubit pairs.
  // The rotation matrix acting on the |1⟩ subspace of qubits (i, j):
  //
  //   R(δθ) = [[cos δθ, -sin δθ],
  //             [sin δθ,  cos δθ]]
  //
  // Gate derivation: this is the real part of the CNOT-Rz-CNOT decomposition
  // projected to the occupation subspace, giving a norm-preserving, unitary
  // operation that mixes adjacent qubit states via a learnable angle δθ.

  // Reshape δθ for broadcasting: [1, C_in, N_QUBITS, 1]
  auto dth = delta_theta.unsqueeze(0).unsqueeze(-1);
  auto c   = torch::cos(dth);  // [1, C_in, N_QUBITS, 1]
  auto s   = torch::sin(dth);  // [1, C_in, N_QUBITS, 1]

  // ── Helper: apply one layer of pairwise Givens rotations ─────────────────
  // pair_offset: which qubits to pair — 0 means pair (0,1),(2,3),(4,5),(6,7)
  //                                     2 means pair (0,2),(1,3),(4,6),(5,7)
  auto apply_givens = [&](torch::Tensor b,
                          int64_t pair_stride) -> torch::Tensor {
    // Split β into even/odd qubit indices at the given stride.
    // For pair_stride=1: even={0,2,4,6}, odd={1,3,5,7}
    // For pair_stride=2: even={0,1,4,5}, odd={2,3,6,7}
    auto even = b.slice(/*dim=*/2, /*start=*/0,
                        /*end=*/N_QUBITS, /*step=*/pair_stride * 2)
                 .contiguous();  // [B, C_in, N_QUBITS/2, N_cells]
    auto odd  = b.slice(/*dim=*/2, /*start=*/pair_stride,
                        /*end=*/N_QUBITS, /*step=*/pair_stride * 2)
                 .contiguous();

    // δθ slice for the even-qubit partners
    auto c_e = c.slice(/*dim=*/2, 0, N_QUBITS, pair_stride * 2);
    auto s_e = s.slice(/*dim=*/2, 0, N_QUBITS, pair_stride * 2);

    //   β'_even = β_even · cos(δθ) − β_odd  · sin(δθ)
    //   β'_odd  = β_odd  · cos(δθ) + β_even · sin(δθ)
    auto new_even = even * c_e - odd * s_e;
    auto new_odd  = odd  * c_e + even * s_e;

    // Scatter results back into the full qubit dimension.
    // We build a fresh tensor and use index_copy_ — still zero extra heap
    // alloc since sizes are compile-time fixed.
    auto out = b.clone();
    // Build index tensors at compile time via constexpr
    torch::Tensor idx_even = torch::arange(0, (int64_t)N_QUBITS,
                                           pair_stride * 2,
                                           torch::kLong);
    torch::Tensor idx_odd  = torch::arange(pair_stride, (int64_t)N_QUBITS,
                                           pair_stride * 2,
                                           torch::kLong);
    out.index_copy_(2, idx_even, new_even);
    out.index_copy_(2, idx_odd,  new_odd);
    return out;
  };

  // ── D=1: 2 qubits — one Givens layer on pair (q0, q1) ────────────────────
  if constexpr (D == 1) {
    // Layer 1: entangle adjacent pair (0,1)
    beta = apply_givens(beta, /*pair_stride=*/1);
  }

  // ── D=2: 4 qubits — two Givens layers covering a 2×2 lattice plaquette ───
  //
  //   Qubit layout maps to spatial positions:
  //     q0=(0,0)  q1=(0,1)
  //     q2=(1,0)  q3=(1,1)
  //
  //   Layer 1: horizontal pairs  (q0,q1), (q2,q3)  — stride=1
  //   Layer 2: vertical   pairs  (q0,q2), (q1,q3)  — stride=2
  //
  //   After two layers every qubit has "seen" all four inputs (full plaquette
  //   entanglement), analogous to a 2-qubit quantum lattice gate.
  if constexpr (D == 2) {
    beta = apply_givens(beta, 1);  // horizontal: (0,1), (2,3)
    beta = apply_givens(beta, 2);  // vertical:   (0,2), (1,3)
  }

  // ── D=3: 8 qubits — three Givens layers covering a 2×2×2 cube ────────────
  //
  //   Qubit layout (bit encoding of spatial index):
  //     q0=(0,0,0) q1=(0,0,1) q2=(0,1,0) q3=(0,1,1)
  //     q4=(1,0,0) q5=(1,0,1) q6=(1,1,0) q7=(1,1,1)
  //
  //   Layer 1  stride=1 : (0,1),(2,3),(4,5),(6,7)  — x-axis edges
  //   Layer 2  stride=2 : (0,2),(1,3),(4,6),(5,7)  — y-axis edges
  //   Layer 3  stride=4 : (0,4),(1,5),(2,6),(3,7)  — z-axis edges
  //
  //   This forms the full set of cube edges (12 edges → 3 layers × 4 pairs),
  //   achieving all-to-all entanglement through the cascade.
  if constexpr (D == 3) {
    beta = apply_givens(beta, 1);  // x-edges
    beta = apply_givens(beta, 2);  // y-edges
    beta = apply_givens(beta, 4);  // z-edges
  }

  return beta;
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3 — Quantum Measurement (Born rule projection)
// ─────────────────────────────────────────────────────────────────────────────

template <size_t D>
torch::Tensor QuanvNdImpl<D>::vqc_measure(const torch::Tensor& beta) const {
  // β: [B, C_in, N_QUBITS, N_cells]
  //
  // P_i(|1⟩) = |⟨1|ψ'_i⟩|² = β'_i²
  //
  // The pow(2) is the Born rule for a projective Z-measurement.  It is:
  //   • Real-valued (collapsing complex state → classical probability)
  //   • Differentiable w.r.t. β, hence w.r.t. δθ
  //   • Non-negative ∈ [0,1] (valid probability)
  //
  // Note: we use pow(2) rather than abs().pow(2) because β is always real in
  // this simulation; pow(2) avoids a branch and keeps gradients cleaner.
  return beta.pow(2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Spatial helpers
// ─────────────────────────────────────────────────────────────────────────────

template <size_t D>
torch::Tensor QuanvNdImpl<D>::extract_patches(const torch::Tensor& x) const {
  // x: [B, C_in, spatial...]   spatial dims = D axes each of size S_d
  //
  // Strategy: chain torch::Tensor::unfold() over each spatial axis.
  //   unfold(dim, size=2, step=2) yields a new trailing dimension of size 2
  //   and halves the original axis length.
  //
  // After D unfolds the shape is:
  //   [B, C_in, S1/2, S2/2, ..., SD/2, 2, 2, ..., 2]
  //             ↑── D output spatial dims ──↑  ↑ D fold dims ↑
  //
  // This is a zero-copy strided view — no data is copied by unfold itself.
  // Only the subsequent .contiguous() call (inside vqc_embed) copies once,
  // which is unavoidable for cache-friendly BLAS access patterns.

  auto out = x;

  // spatial axes in x are at positions 2, 3, …, 2+D-1
  for (size_t d = 0; d < D; ++d) {
    int64_t spatial_dim = static_cast<int64_t>(2 + d);
    out = out.unfold(spatial_dim, KERNEL, STRIDE);
    // After unfold at axis spatial_dim:
    //   original axis shrinks: S_d → S_d/2
    //   new trailing axis appended: size = KERNEL = 2
  }

  // Current shape: [B, C_in, S1/2, …, SD/2, 2^0, 2^1, …, 2^(D-1)]
  // (D trailing axes of size 2 from unfold)

  // Flatten the D trailing "fold" axes into a single qubit axis of size 2^D,
  // and flatten the D spatial axes into a single N_cells axis.
  //
  // Reshape target: [B, C_in, N_QUBITS, N_cells]
  //   N_cells = ∏_d (S_d / 2)
  //   N_QUBITS = 2^D (compile-time constant)

  int64_t B      = x.size(0);
  int64_t C_in   = x.size(1);

  // Compute N_cells = product of halved spatial dims
  int64_t N_cells = 1;
  for (size_t d = 0; d < D; ++d) {
    N_cells *= (x.size(static_cast<int64_t>(2 + d)) / 2);
  }

  // We need to interleave the spatial and fold axes before flattening.
  // After unfold the tensor layout is:
  //   axes:  0=B, 1=C, 2..2+D-1=spatial_out, 2+D..2+2D-1=fold_size
  //
  // For N_QUBITS we need the fold dimensions grouped together (or we can
  // reshape directly since unfold gives contiguous strides for size-2 steps).
  //
  // Flatten spatial axes [2 … 2+D-1] → one axis, fold axes [2+D … 2+2D-1] → one axis.

  // Permute so fold axes come immediately after channel:
  //   desired: [B, C_in, fold0, fold1, …, foldD-1, sp0, sp1, …, spD-1]
  // Then reshape to [B, C_in, N_QUBITS, N_cells].
  if constexpr (D == 1) {
    // Shape after unfold: [B, C_in, S/2, 2]
    // Permute: [B, C_in, 2, S/2]  → already ordered correctly
    out = out.permute({0, 1, 3, 2});
    // Reshape: [B, C_in, 2, S/2] — N_QUBITS=2, N_cells=S/2
    out = out.contiguous().view({B, C_in, N_QUBITS, N_cells});
  } else if constexpr (D == 2) {
    // Shape after unfold: [B, C_in, H/2, W/2, 2, 2]
    // Permute to: [B, C_in, foldH, foldW, spH, spW] = [B, C_in, 4, 5, 2, 3]
    out = out.permute({0, 1, 4, 5, 2, 3});
    // Reshape: [B, C_in, 4, N_cells]  N_cells=H/2 * W/2
    out = out.contiguous().view({B, C_in, N_QUBITS, N_cells});
  } else if constexpr (D == 3) {
    // Shape after unfold: [B, C_in, D/2, H/2, W/2, 2, 2, 2]
    // Permute to: [B, C_in, fd, fh, fw, sd, sh, sw]
    out = out.permute({0, 1, 5, 6, 7, 2, 3, 4});
    // Reshape: [B, C_in, 8, N_cells]
    out = out.contiguous().view({B, C_in, N_QUBITS, N_cells});
  }

  return out;  // [B, C_in, N_QUBITS, N_cells]
}

template <size_t D>
torch::Tensor QuanvNdImpl<D>::reshape_output(const torch::Tensor& measured,
                                              const torch::Tensor& x) const {
  // measured: [B, C_in, N_QUBITS, N_cells]
  //
  // Target output shape: [B, C_in * N_QUBITS, S1/2, S2/2, …, SD/2]
  //   where C_out = C_in * N_QUBITS (each input channel → N_QUBITS output ch)

  int64_t B    = x.size(0);
  int64_t C_in = x.size(1);
  int64_t C_out = C_in * N_QUBITS;

  if constexpr (D == 1) {
    int64_t L_out = x.size(2) / 2;
    // [B, C_in, 2, L_out] → [B, C_in*2, L_out]
    return measured.view({B, C_out, L_out});
  } else if constexpr (D == 2) {
    int64_t H_out = x.size(2) / 2;
    int64_t W_out = x.size(3) / 2;
    // [B, C_in, 4, H_out*W_out] → [B, C_in, 4, H_out, W_out]
    // then merge first two ch dims → [B, C_in*4, H_out, W_out]
    auto t = measured.view({B, C_in, N_QUBITS, H_out, W_out});
    // Permute so qubit axis follows C_in: [B, C_in, N_Q, H, W] → [B, N_Q, C_in, H, W]?
    // Actually we want C_out = C_in * N_Q as a flat channel dim.
    // Easiest: reshape(B, C_in, N_Q, H_out, W_out) then permute(0,1,2,3,4)
    // then view(B, C_out, H_out, W_out) — axes are already contiguous.
    return t.view({B, C_out, H_out, W_out});
  } else {
    int64_t D_out = x.size(2) / 2;
    int64_t H_out = x.size(3) / 2;
    int64_t W_out = x.size(4) / 2;
    auto t = measured.view({B, C_in, N_QUBITS, D_out, H_out, W_out});
    return t.view({B, C_out, D_out, H_out, W_out});
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Forward pass
// ─────────────────────────────────────────────────────────────────────────────

template <size_t D>
torch::Tensor QuanvNdImpl<D>::forward(const torch::Tensor& x) {
  // ── Validate input shape ──────────────────────────────────────────────────
  TORCH_CHECK(x.dim() == static_cast<int64_t>(D + 2),
              "QuanvNd<", D, ">: expected input with ", D + 2,
              " dimensions [B, C_in, spatial...], got ", x.dim());
  TORCH_CHECK(x.size(1) == options.in_channels(),
              "QuanvNd<", D, ">: expected in_channels=", options.in_channels(),
              " but got ", x.size(1));
  for (size_t d = 0; d < D; ++d) {
    TORCH_CHECK(x.size(static_cast<int64_t>(2 + d)) % 2 == 0,
                "QuanvNd<", D, ">: spatial dim ", d, " (size ",
                x.size(static_cast<int64_t>(2 + d)),
                ") must be divisible by kernel=2");
  }

  // ── Step 1: extract non-overlapping 2^D patches via unfold ───────────────
  // patches: [B, C_in, N_QUBITS, N_cells]
  // This is a zero-copy strided view (unfold) + one contiguous() call.
  // No std::vector or dynamic allocation on the hot path.
  auto patches = extract_patches(x);

  // ── Step 2: VQC Phase 1 — Quantum Embedding ───────────────────────────────
  // Map each patch value x_i → (α_i = cos(θ_i), β_i = sin(θ_i))
  // Both alpha and beta: [B, C_in, N_QUBITS, N_cells]
  auto embed = vqc_embed(patches);
  // alpha is not modified by entanglement — we keep it for a possible future
  // extension to full two-qubit measurement.  For now only beta matters.
  auto beta  = embed.second;   // [B, C_in, N_QUBITS, N_cells]

  // ── Step 3: VQC Phase 2 — Quantum Entanglement ───────────────────────────
  // Apply D layers of parameterised Givens rotations to β amplitudes.
  // Gradient flows through cos/sin of delta_theta.
  beta = vqc_entangle(beta);   // [B, C_in, N_QUBITS, N_cells]

  // ── Step 4: VQC Phase 3 — Quantum Measurement (Born rule) ─────────────────
  // P_i = β'_i²   ∈ [0,1]  per qubit
  auto measured = vqc_measure(beta);  // [B, C_in, N_QUBITS, N_cells]

  // ── Step 5: reshape to canonical output tensor ────────────────────────────
  // [B, C_in * N_QUBITS, spatial_out...]
  return reshape_output(measured, x);
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit instantiations — required because template bodies are in .cpp
// ─────────────────────────────────────────────────────────────────────────────

template class QuanvNdImpl<1>;
template class QuanvNdImpl<2>;
template class QuanvNdImpl<3>;

}  // namespace prim
}  // namespace dm
