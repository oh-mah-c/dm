#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// KANLinear — Kolmogorov-Arnold Network layer (dm::prim)
// Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025
//
// A single KAN layer Φ: ℝ^{n_in} → ℝ^{n_out}.
//
// Each of the n_in × n_out edges carries a learnable 1D activation function:
//
//   φ_{j,i}(x) = w_b · silu(x) + w_s · spline(x)           (Eq. 15-16)
//
//   spline(x) = Σ_{m=0}^{G+k-1}  c_{j,i,m} · B_m(x)        (Eq. 17)
//
// where B_m are B-spline basis functions of order k on G uniform intervals.
// Grid points are adaptive: initialised to [-1,1] and updated each forward
// pass to track the empirical input range (Section 2.5, Appendix I).
//
// Layer forward (Eq. 5):
//   x_{out,j} = Σ_i  φ_{j,i}(x_{in,i})
//
// Parameters
// ──────────
//   spline_weight : [n_out, n_in, G+k]   learnable B-spline coefficients c
//   w_b           : [n_out, n_in]         residual (SiLU) scale
//   w_s           : [n_out, n_in]         spline scale
//   grid          : [n_in,  G+2k+1]      B-spline knot vector (buffer, non-param)
//
// Sparsity loss (Eqs. 19-22)
// ──────────────────────────
// Call kan_linear_l1(layer, inputs) to get the L1+entropy regularisation term:
//   |Φ|_1 = Σ_{i,j} |φ_{j,i}|_1   where  |φ|_1 = (1/N_p) Σ_s |φ(x^(s))|
//   S(Φ)  = -Σ_{i,j} (|φ_{j,i}|_1 / |Φ|_1) · log(|φ_{j,i}|_1 / |Φ|_1)
//
// Grid extension (Eq. 18, Appendix L)
// ─────────────────────────────────────
// Call extend_grid(layer, new_G, x_samples) to refine the spline grid from
// the current G to new_G intervals, preserving the learned function via
// least-squares projection.
//
// Usage
// ─────
//   dm::prim::KANLinear layer(2, 5, /*G=*/5, /*k=*/3);
//   auto y = layer->forward(x);   // x: [B, 2]  →  y: [B, 5]
//
//   // Sparsity regularisation (add to training loss):
//   auto reg = dm::prim::kan_linear_sparsity(layer, x, /*mu1=*/1.0, /*mu2=*/1.0);
//
//   // Grid extension (coarser→finer grid):
//   dm::prim::kan_linear_extend_grid(layer, new_G=10, x_samples);
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include <torch/csrc/Export.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace dm { namespace prim {

// ─────────────────────────────────────────────────────────────────────────────
// Options
// ─────────────────────────────────────────────────────────────────────────────

struct TORCH_API KANLinearOptions {
    KANLinearOptions(int64_t in_features, int64_t out_features)
        : in_features_(in_features), out_features_(out_features) {}

    // n_in: input dimension
    TORCH_ARG(int64_t, in_features);
    // n_out: output dimension
    TORCH_ARG(int64_t, out_features);
    // G: number of spline intervals (grid size). Paper default: 5
    TORCH_ARG(int64_t, G) = 5;
    // k: B-spline order (piecewise polynomial degree = k-1). Paper default: 3
    TORCH_ARG(int64_t, k) = 3;
    // grid_range: [grid_min, grid_max] initial knot span
    TORCH_ARG(double, grid_min) = -1.0;
    TORCH_ARG(double, grid_max) =  1.0;
    // grid_eps: margin when updating grid from data (fraction of range)
    TORCH_ARG(double, grid_eps) = 0.02;
    // Whether to update the grid from input statistics each forward pass
    TORCH_ARG(bool, update_grid) = true;
    // w_b init std (Xavier-style controlled by this multiplier)
    TORCH_ARG(double, wb_scale) = 1.0;
    // w_s init value
    TORCH_ARG(double, ws_init) = 1.0;
    // spline init std (N(0, sigma^2))
    TORCH_ARG(double, sp_sigma) = 0.1;
};

// ─────────────────────────────────────────────────────────────────────────────
// B-spline helpers (anonymous namespace, implementation detail)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Build a uniform extended knot vector for B-splines of order k on G intervals.
// Returns tensor of shape [G + 2k + 1] covering [a, b] with k repeated knots
// at each end (clamped B-splines).
inline torch::Tensor _make_grid(int64_t G, int64_t k,
                                double a, double b,
                                torch::TensorOptions opts = {}) {
    // Interior knots: G+1 points uniformly in [a,b]
    auto interior = torch::linspace(a, b, G + 1, opts);
    // Step
    double h = (b - a) / G;
    // Prepend k repeated knots: a - k*h, ..., a - h
    // Append  k repeated knots: b + h,   ..., b + k*h
    std::vector<torch::Tensor> parts;
    parts.reserve(2 * k + 1);
    for (int64_t i = k; i >= 1; --i)
        parts.push_back(torch::full({1}, a - i * h, opts));
    parts.push_back(interior);
    for (int64_t i = 1; i <= k; ++i)
        parts.push_back(torch::full({1}, b + i * h, opts));
    return torch::cat(parts, 0); // [G + 2k + 1]
}

// Evaluate B-spline basis for all n_in input scalars simultaneously.
//
//   x    : [B, n_in]   — input values
//   grid : [n_in, G+2k+1]  — knot vectors (one per input feature)
//   k    : spline order
//
// Returns bases : [B, n_in, G+k]  — B-spline basis values B_{0..G+k-1}(x)
//
// Uses the Cox-de Boor recurrence:
//   B_{i,0}(x) = 1  if t_i <= x < t_{i+1}, else 0
//   B_{i,p}(x) = (x-t_i)/(t_{i+p}-t_i)·B_{i,p-1}(x)
//              + (t_{i+p+1}-x)/(t_{i+p+1}-t_{i+1})·B_{i+1,p-1}(x)
//
inline torch::Tensor _bspline_basis(
        const torch::Tensor& x,    // [B, n_in]
        const torch::Tensor& grid, // [n_in, G+2k+1]
        int64_t k)
{
    // x:    [B, n_in]
    // grid: [n_in, M]  where M = G+2k+1
    const int64_t M = grid.size(1);

    // x_exp: [B, n_in, 1]  broadcast over basis index
    auto x_exp = x.unsqueeze(2);

    // Order-0 basis: [B, n_in, M-1]
    // B_{i,0}(x) = 1  if  t_i <= x < t_{i+1}
    auto t_left  = grid.slice(1, 0, M - 1).unsqueeze(0);  // [1, n_in, M-1]
    auto t_right = grid.slice(1, 1, M    ).unsqueeze(0);  // [1, n_in, M-1]
    auto basis = ((x_exp >= t_left) & (x_exp < t_right)).to(x.dtype()); // [B, n_in, M-1]

    // De Boor recurrence for orders 1..k
    // After step p, basis has shape [B, n_in, M-1-p]
    for (int64_t p = 1; p <= k; ++p) {
        // n_out = number of order-p basis functions = (M-1) - p = M-p-1
        int64_t n_out = M - 1 - p;

        // For B_{i,p}: uses t_i, t_{i+p}, t_{i+1}, t_{i+p+1}
        // i ranges 0..n_out-1
        auto ti    = grid.slice(1, 0,   n_out    ).unsqueeze(0); // [1, n_in, n_out]
        auto ti_p  = grid.slice(1, p,   n_out + p).unsqueeze(0); // [1, n_in, n_out]
        auto ti_1  = grid.slice(1, 1,   n_out + 1).unsqueeze(0); // [1, n_in, n_out]
        auto ti_p1 = grid.slice(1, p+1, n_out+p+1).unsqueeze(0); // [1, n_in, n_out]

        // left:  (x - t_i) / (t_{i+p} - t_i)  *  B_{i,p-1}
        auto denom_l = (ti_p - ti).abs().clamp_min(1e-8f);
        auto c_left  = (x_exp - ti) / denom_l;  // [B, n_in, n_out]

        // right: (t_{i+p+1} - x) / (t_{i+p+1} - t_{i+1})  *  B_{i+1,p-1}
        auto denom_r = (ti_p1 - ti_1).abs().clamp_min(1e-8f);
        auto c_right = (ti_p1 - x_exp) / denom_r;  // [B, n_in, n_out]

        // basis (order p-1) has n_out+1 entries; pick adjacent pairs
        auto bl = basis.slice(2, 0, n_out    ); // [B, n_in, n_out]
        auto br = basis.slice(2, 1, n_out + 1); // [B, n_in, n_out]

        basis = c_left * bl + c_right * br; // [B, n_in, n_out]
    }
    // basis: [B, n_in, M-1-k] = [B, n_in, G+k]
    return basis;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// KANLinearImpl
// ─────────────────────────────────────────────────────────────────────────────

class TORCH_API KANLinearImpl : public torch::nn::Module {
 public:
    explicit KANLinearImpl(int64_t in_features, int64_t out_features)
        : KANLinearImpl(KANLinearOptions(in_features, out_features)) {}
    explicit KANLinearImpl(KANLinearOptions options_);

    void reset();
    void pretty_print(std::ostream& stream) const override;

    // Forward pass.
    // x: [B, in_features]  →  [B, out_features]
    torch::Tensor forward(const torch::Tensor& x);

    // Update grid from a batch of input activations.
    // Call before forward during grid-update steps (Section 2.5).
    void update_grid_from_data(const torch::Tensor& x);

    // Grid extension: refine grid from current G to new_G intervals (Eq. 18).
    // x_samples: [N, in_features] — representative input samples.
    void extend_grid(int64_t new_G, const torch::Tensor& x_samples);

    // Public accessors
    int64_t in_features()  const { return options.in_features(); }
    int64_t out_features() const { return options.out_features(); }
    int64_t G()            const { return options.G(); }
    int64_t k()            const { return options.k(); }

    KANLinearOptions options;

    // Learnable parameters
    // spline_weight: [out, in, G+k]  — B-spline coefficients c_{j,i,m}
    torch::Tensor spline_weight;
    // w_b: [out, in]  — residual (SiLU) scale
    torch::Tensor w_b;
    // w_s: [out, in]  — spline scale
    torch::Tensor w_s;

    // Non-trainable knot vectors
    // grid: [in, G+2k+1]  — one knot vector per input feature
    torch::Tensor grid;   // registered as buffer

    // Evaluate spline(x) = Σ_m c_{j,i,m} B_m(x_i)  for all (j,i) pairs.
    // x: [B, in]  →  spline_out: [B, out, in]
    torch::Tensor _spline_eval(const torch::Tensor& x) const;
};

TORCH_MODULE(KANLinear);

// ─────────────────────────────────────────────────────────────────────────────
// Inline implementations
// ─────────────────────────────────────────────────────────────────────────────

inline KANLinearImpl::KANLinearImpl(KANLinearOptions options_)
    : options(std::move(options_)) { reset(); }

inline void KANLinearImpl::reset() {
    const int64_t n  = options.in_features();
    const int64_t m  = options.out_features();
    const int64_t G  = options.G();
    const int64_t k_ = options.k();
    const int64_t n_coef = G + k_; // number of B-spline basis functions

    // Grid: [n, G+2k+1]
    auto g = _make_grid(G, k_, options.grid_min(), options.grid_max());
    // Broadcast to [n, G+2k+1]
    auto grid_init = g.unsqueeze(0).expand({n, -1}).contiguous();
    grid = register_buffer("grid", grid_init);

    // spline_weight: [m, n, G+k]  — small random init (Appendix I footnote 2)
    spline_weight = register_parameter("spline_weight",
        torch::randn({m, n, n_coef}) * static_cast<float>(options.sp_sigma()));

    // w_b: Xavier init (Appendix I)
    double fan_in  = static_cast<double>(n);
    double xavier_std = options.wb_scale() * std::sqrt(2.0 / fan_in);
    w_b = register_parameter("w_b",
        torch::randn({m, n}) * static_cast<float>(xavier_std));

    // w_s: init to ws_init (default 1.0)
    w_s = register_parameter("w_s",
        torch::full({m, n}, static_cast<float>(options.ws_init())));
}

inline void KANLinearImpl::pretty_print(std::ostream& stream) const {
    stream << "dm::prim::KANLinear("
           << options.in_features() << " -> " << options.out_features()
           << ", G=" << options.G()
           << ", k=" << options.k() << ")";
}

inline torch::Tensor KANLinearImpl::_spline_eval(const torch::Tensor& x) const {
    // x: [B, n_in]
    // grid: [n_in, G+2k+1]
    // spline_weight: [n_out, n_in, G+k]
    auto basis = _bspline_basis(x, grid, options.k()); // [B, n_in, G+k]

    // spline(x)_{j,i} = Σ_m spline_weight[j,i,m] * basis[B,i,m]
    // = einsum("bim, jim -> bji", basis, spline_weight)
    // We want output [B, n_out, n_in]
    // basis:         [B, n_in, G+k]
    // spline_weight: [n_out, n_in, G+k]
    // Result: for each b and each (j,i): dot over m
    // = (spline_weight: [n_out, n_in, G+k]) @ (basis^T: [B, n_in, G+k, 1])
    // Easiest: reshape and mm
    // basis_flat: [B*n_in, G+k]
    const int64_t B_   = x.size(0);
    const int64_t n_in = x.size(1);
    const int64_t G_k  = options.G() + options.k();

    auto basis_flat = basis.reshape({B_ * n_in, G_k}); // [B*n_in, G+k]
    // sw_flat: [n_out, n_in * (G+k)] then reshape
    // Better: use einsum equivalent via bmm
    // basis: [B, n_in, G+k]  ->  [B, n_in * (G+k), 1]  — too many reshapes
    // Cleanest: for each (j,i), dot with basis[:,i,:]
    // Use torch::einsum if available, else manual bmm:
    //   out[b,j,i] = sum_m spline_weight[j,i,m] * basis[b,i,m]
    //              = (basis[b,i,:] · spline_weight[j,i,:])
    // batch over b: einsum("bim,jim->bji")
    auto spline_out = torch::einsum("bim,jim->bji",
        {basis, spline_weight}); // [B, n_out, n_in]
    return spline_out;
}

inline void KANLinearImpl::update_grid_from_data(const torch::Tensor& x) {
    // x: [B, n_in]
    // Update each input feature's grid to span [min - eps*range, max + eps*range]
    torch::NoGradGuard ng;
    const int64_t G  = options.G();
    const int64_t k_ = options.k();
    double eps = options.grid_eps();

    // Per-feature min/max over batch: [n_in]
    auto x_min = std::get<0>(x.min(0));
    auto x_max = std::get<0>(x.max(0));
    auto range  = (x_max - x_min).clamp_min(1e-6f);
    auto a = x_min - eps * range;
    auto b = x_max + eps * range;

    // Rebuild grid row-by-row (per input feature)
    const int64_t n_in = x.size(1);
    const int64_t M    = G + 2 * k_ + 1;
    auto new_grid = grid.clone();
    for (int64_t i = 0; i < n_in; ++i) {
        double ai = a[i].item<double>();
        double bi = b[i].item<double>();
        auto g = _make_grid(G, k_, ai, bi, x.options());
        new_grid[i].copy_(g);
    }
    grid.copy_(new_grid);
}

inline void KANLinearImpl::extend_grid(int64_t new_G,
                                       const torch::Tensor& x_samples) {
    // Refine spline grid from current G to new_G (new_G > G).
    // Projects existing coefficients to the finer grid via least-squares (Eq. 18).
    torch::NoGradGuard ng;
    const int64_t old_G = options.G();
    const int64_t k_    = options.k();
    const int64_t n_in  = options.in_features();
    const int64_t n_out = options.out_features();

    if (new_G <= old_G)
        throw std::invalid_argument("extend_grid: new_G must be > current G");

    // Rebuild grid with new_G intervals using current grid range
    // (Use min/max of current grid interior as range)
    const int64_t M_old = grid.size(1);
    // Interior span: grid[i, k] .. grid[i, k+old_G]
    auto new_grid_t = torch::zeros({n_in, new_G + 2 * k_ + 1},
                                    grid.options());
    for (int64_t i = 0; i < n_in; ++i) {
        double ai = grid[i][k_    ].item<double>();
        double bi = grid[i][k_ + old_G].item<double>();
        auto g = _make_grid(new_G, k_, ai, bi, grid.options());
        new_grid_t[i].copy_(g);
    }

    // Evaluate old spline on x_samples to get supervised targets
    // x_samples: [N, n_in]
    const int64_t N = x_samples.size(0);
    auto basis_old  = _bspline_basis(x_samples, grid, k_); // [N, n_in, old_G+k]
    // old_spline[b,j,i] = Σ_m spline_weight[j,i,m]*basis_old[b,i,m]
    auto old_vals = torch::einsum("bim,jim->bji",
        {basis_old, spline_weight}); // [N, n_out, n_in]

    // Evaluate new basis on x_samples
    auto basis_new = _bspline_basis(x_samples, new_grid_t, k_); // [N, n_in, new_G+k]

    // Least-squares: for each (j,i) solve B_new * c_new ≈ old_vals
    // basis_new[:,i,:]: [N, new_G+k]  — per input feature i
    // old_vals[:,j,i]:  [N]           — per (j,i) edge
    const int64_t new_nc = new_G + k_;
    auto new_sw = torch::zeros({n_out, n_in, new_nc}, spline_weight.options());

    for (int64_t i = 0; i < n_in; ++i) {
        auto Bi = basis_new.select(1, i); // [N, new_G+k]
        // Solve: Bi @ c = old_vals[:, :, i]^T  →  c: [new_G+k, n_out]
        // old_vals[:, :, i]: [N, n_out]
        auto targets = old_vals.select(2, i); // [N, n_out]
        // lstsq: min ||Bi @ c - targets||^2
        // torch::linalg_lstsq returns (solution, residuals, rank, sv)
        auto result = std::get<0>(torch::linalg_lstsq(Bi, targets, std::nullopt,
                                                       std::nullopt)); // [new_nc, n_out]
        // result: [new_G+k, n_out]  → new_sw[:, i, :] = result^T
        new_sw.select(1, i).copy_(result.t()); // [n_out, new_G+k]
    }

    // Update state.
    // Buffer: set_() repoints the storage in-place.
    grid.set_(new_grid_t.storage(), 0, new_grid_t.sizes(), new_grid_t.strides());
    // Parameter: update via the protected parameters_ dict (no unregister API).
    // Assign a new leaf tensor; the member variable and the dict entry are the
    // same object (register_parameter returns a Tensor& stored there).
    auto new_sw_leaf = new_sw.clone().detach().requires_grad_(true);
    parameters_["spline_weight"] = new_sw_leaf;
    spline_weight = new_sw_leaf;
    options.G(new_G);
}

inline torch::Tensor KANLinearImpl::forward(const torch::Tensor& x) {
    // x: [B, n_in]
    if (x.dim() != 2 || x.size(1) != options.in_features())
        throw std::invalid_argument(
            "KANLinear::forward: expected [B, " +
            std::to_string(options.in_features()) + "] input");

    // Optionally update grid from this batch
    if (options.update_grid() && is_training())
        update_grid_from_data(x.detach());

    // Residual (SiLU) branch: silu(x) * w_b
    // x: [B, n_in]  →  silu_out: [B, n_out, n_in]
    auto x_silu = torch::silu(x); // [B, n_in]
    // w_b: [n_out, n_in]  →  broadcast: [1, n_out, n_in] * [B, 1, n_in]
    auto residual = x_silu.unsqueeze(1) * w_b.unsqueeze(0); // [B, n_out, n_in]

    // Spline branch: spline(x) * w_s
    // spline_eval: [B, n_out, n_in]
    auto spline_out = _spline_eval(x);                       // [B, n_out, n_in]
    auto scaled_sp  = spline_out * w_s.unsqueeze(0);         // [B, n_out, n_in]

    // φ(x) = residual + spline_out (per edge)
    auto phi = residual + scaled_sp; // [B, n_out, n_in]

    // Sum over input dimension (Eq. 5): x_{out,j} = Σ_i φ_{j,i}(x_i)
    return phi.sum(2); // [B, n_out]
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparsity regularisation helper (Eqs. 19-22)
//
//   kan_linear_sparsity(layer, x, mu1, mu2)
//
// Returns  mu1 * |Φ|_1 + mu2 * S(Φ)  as a scalar tensor.
// Call once per layer, sum across all layers with overall lambda weight:
//   l_total = l_pred + lambda * Σ_l  kan_linear_sparsity(layer_l, x_l, mu1, mu2)
// ─────────────────────────────────────────────────────────────────────────────

inline torch::Tensor kan_linear_sparsity(
        const KANLinear& layer,
        const torch::Tensor& x,   // [B, n_in] — pre-activation of this layer
        double mu1 = 1.0,
        double mu2 = 1.0)
{
    // Compute φ_{j,i}(x_s) for all (j,i,s) — detach x to avoid double-backward
    auto xd = x.detach();
    if (xd.dim() != 2 || xd.size(1) != layer->in_features())
        throw std::invalid_argument("kan_linear_sparsity: x shape mismatch");

    auto x_silu   = torch::silu(xd);
    auto residual = xd.unsqueeze(1) * layer->w_b.unsqueeze(0);
    auto sp_out   = layer->_spline_eval(xd);
    auto phi = residual + sp_out * layer->w_s.unsqueeze(0); // [B, n_out, n_in]

    // |φ_{j,i}|_1 = mean over B of |φ_{j,i}(x_s)|  →  [n_out, n_in]
    auto phi_l1 = phi.abs().mean(0); // [n_out, n_in]

    // |Φ|_1 = sum over all (j,i) of |φ_{j,i}|_1
    auto Phi_l1 = phi_l1.sum();

    // Entropy S(Φ) = -Σ_{j,i} (|φ_{j,i}|_1 / |Φ|_1) * log(|φ_{j,i}|_1 / |Φ|_1)
    auto p = phi_l1 / Phi_l1.clamp_min(1e-10f);
    auto entropy = -(p * (p.clamp_min(1e-10f).log())).sum();

    return static_cast<float>(mu1) * Phi_l1
         + static_cast<float>(mu2) * entropy;
}

}} // namespace dm::prim
