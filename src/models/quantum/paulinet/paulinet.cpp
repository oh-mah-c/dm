// ─────────────────────────────────────────────────────────────────────────────
// PauliNet implementation
// Hermann, Schätzle & Noé, arXiv:1909.08423v5, Nature Chemistry 2020
//
// Equations referenced:
//   Ansatz:           ψ = e^{J+γ} Σ_p c_p det↑_p det↓_p          (Eq. 1)
//   Antisymmetry:     ψ(...,r_i,...,r_j,...) = -ψ(...,r_j,...,r_i,...) (Eq. 2)
//   Jastrow/BF:       J = η(Σ x_i^L), f_i = κ(x_i^L)              (Eq. 5)
//   VMC energy:       E = E_{r~|ψ|²}[E_loc]                        (Eq. 7)
//   VMC gradient:     ∇L = 2E[(E_loc − Ē) ∇ ln|ψ|]                (Eq. 8)
//   Cusp factor:      γ = -Σ_{i<j} c_{ij}/(1+|r_i−r_j|)           (Eq. 9)
//   SchNet update:    x^{n+1} = x^n + Σ g(z^{n,±}) + g(z^n)       (Eq. 11)
//   RBF features:     e_k(r) = r²·exp(-(r-μ_k)²/σ_k²)             (Eq. 12-14)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/quantum/paulinet/paulinet.h"

#include <torch/torch.h>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <random>

namespace dm {
namespace models {
namespace quantum {

// ─────────────────────────────────────────────────────────────────────────────
// PauliNetConfig — convenience constructors
// ─────────────────────────────────────────────────────────────────────────────

PauliNetConfig PauliNetConfig::hydrogen() {
    PauliNetConfig c;
    c.n_up = 1; c.n_down = 0; c.n_nuclei = 1; c.n_det = 1;
    return c;
}

PauliNetConfig PauliNetConfig::h2() {
    PauliNetConfig c;
    c.n_up = 1; c.n_down = 1; c.n_nuclei = 2; c.n_det = 1;
    return c;
}

PauliNetConfig PauliNetConfig::lih() {
    PauliNetConfig c;
    c.n_up = 2; c.n_down = 2; c.n_nuclei = 2; c.n_det = 4;
    return c;
}

// ─────────────────────────────────────────────────────────────────────────────
// DistanceFeature  —  Eqs. 12–14
// e_k(r) = r² · exp(-(r - μ_k)² / σ_k²)
// Cuspless: e_k(0) = 0, (d/dr e_k)(0) = 0
// ─────────────────────────────────────────────────────────────────────────────

DistanceFeatureImpl::DistanceFeatureImpl(int64_t n_rbf, double r_cutoff) {
    // q_k uniformly spaced in (0,1)
    auto q = torch::linspace(1.0 / (n_rbf + 1), n_rbf * 1.0 / (n_rbf + 1), n_rbf);
    // μ_k = r_c · q_k²   (Eq. 13)
    auto mu_  = r_cutoff * q.pow(2);
    // σ_k = (1/7)(1 + r_c · q_k)   (Eq. 14)
    auto sig_ = (1.0 / 7.0) * (1.0 + r_cutoff * q);

    mu    = register_buffer("mu",    mu_);
    sigma = register_buffer("sigma", sig_);
}

torch::Tensor DistanceFeatureImpl::forward(torch::Tensor r) {
    // r: [...], mu/sigma: [K]
    // broadcast: [..., 1] vs [K] → [..., K]
    auto r_exp = r.unsqueeze(-1);
    auto diff  = r_exp - mu;                        // [..., K]
    auto gauss = torch::exp(-diff.pow(2) / sigma.pow(2));
    return r_exp.pow(2) * gauss;                    // [..., K]  (Eq. 12)
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP helper
// ─────────────────────────────────────────────────────────────────────────────

MLPImpl::MLPImpl(std::vector<int64_t> dims) {
    torch::nn::Sequential s;
    for (size_t i = 0; i + 1 < dims.size(); i++) {
        s->push_back(torch::nn::Linear(dims[i], dims[i + 1]));
        if (i + 2 < dims.size())          // hidden layers get tanh
            s->push_back(torch::nn::Tanh());
    }
    net = register_module("net", s);
}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
    return net->forward(x);
}

// ─────────────────────────────────────────────────────────────────────────────
// SchNetLayer  —  Eq. 11 (one iteration)
// ─────────────────────────────────────────────────────────────────────────────

SchNetLayerImpl::SchNetLayerImpl(int64_t n_up_, int64_t n_down_,
                                  int64_t dim_e, int64_t dim_x_, int64_t dim_z_,
                                  int64_t nw, int64_t nh, int64_t ng)
    : n_up(n_up_), n_down(n_down_), dim_z(dim_z_), dim_x(dim_x_)
{
    auto mk_w = [&](const std::string& nm) {
        std::vector<int64_t> d = {dim_e, dim_x};
        for (int i = 1; i < nw; i++) d.insert(d.begin() + 1, dim_x);
        d.back() = dim_z;
        return register_module(nm, MLP(d));
    };
    auto mk_h = [&](const std::string& nm) {
        std::vector<int64_t> d(nh + 1, dim_x);
        d.back() = dim_z;
        return register_module(nm, MLP(d));
    };
    auto mk_g = [&](const std::string& nm) {
        std::vector<int64_t> d(ng + 1, dim_z);
        d.back() = dim_x;
        return register_module(nm, MLP(d));
    };

    w_same = mk_w("w_same");
    w_opp  = mk_w("w_opp");
    w_nuc  = mk_w("w_nuc");
    h_same = mk_h("h_same");
    h_opp  = mk_h("h_opp");
    g_same = mk_g("g_same");
    g_opp  = mk_g("g_opp");
    g_nuc  = mk_g("g_nuc");
}

torch::Tensor SchNetLayerImpl::forward(torch::Tensor x,
                                        torch::Tensor e_ee,
                                        torch::Tensor e_en) {
    // x:     [N, dim_x]
    // e_ee:  [N, N, dim_e]
    // e_en:  [N, M, dim_e]
    int64_t N = x.size(0);

    // ── Same-spin message (z+) ────────────────────────────────────────────────
    // Electrons interact with others of the same spin group
    // Up electrons: indices [0, n_up), Down electrons: [n_up, N)
    auto aggregate_spin = [&](MLP& w_fn, MLP& h_fn, MLP& g_fn,
                               int64_t start_i, int64_t end_i,
                               int64_t start_j, int64_t end_j) -> torch::Tensor {
        if (end_i <= start_i || end_j <= start_j)
            return torch::zeros({N, dim_x}, x.options());

        auto x_j    = x.slice(0, start_j, end_j);         // [Nj, dim_x]
        auto e_ij   = e_ee.slice(0, start_i, end_i)
                          .slice(1, start_j, end_j);       // [Ni, Nj, dim_e]
        // w: [Ni, Nj, dim_z]
        auto w      = w_fn->forward(e_ij);
        // h: [Nj, dim_z]
        auto h      = h_fn->forward(x_j);
        // message: [Ni, Nj, dim_z] = w ⊙ h  (broadcast)
        auto msg    = w * h.unsqueeze(0);                  // [Ni, Nj, dim_z]
        // aggregate over j: [Ni, dim_z]
        auto z_i    = msg.sum(1);                          // [Ni, dim_z]
        // g: [Ni, dim_x]
        auto update = g_fn->forward(z_i);

        // Embed in full [N, dim_x] tensor
        auto out = torch::zeros({N, dim_x}, x.options());
        out.slice(0, start_i, end_i) = update;
        return out;
    };

    // Same-spin for up-up
    auto dz_same_up = aggregate_spin(w_same, h_same, g_same,
                                      0, n_up, 0, n_up);
    // Same-spin for down-down
    auto dz_same_dn = aggregate_spin(w_same, h_same, g_same,
                                      n_up, N, n_up, N);
    // Opposite-spin up→down
    auto dz_opp_up  = aggregate_spin(w_opp, h_opp, g_opp,
                                      0, n_up, n_up, N);
    // Opposite-spin down→up
    auto dz_opp_dn  = aggregate_spin(w_opp, h_opp, g_opp,
                                      n_up, N, 0, n_up);

    // ── Nucleus message ───────────────────────────────────────────────────────
    // e_en: [N, M, dim_e]  — nuclear embeddings Y_{θ,I} come from the outer model
    // Here we just use w_nuc on e_en (Y is incorporated at PauliNet level)
    auto w_n   = w_nuc->forward(e_en);                // [N, M, dim_z]
    auto z_n   = w_n.sum(1);                          // [N, dim_z]
    auto dz_nuc = g_nuc->forward(z_n);                // [N, dim_x]

    // ── Residual update (Eq. 11) ──────────────────────────────────────────────
    return x + dz_same_up + dz_same_dn
             + dz_opp_up  + dz_opp_dn
             + dz_nuc;
}

// ─────────────────────────────────────────────────────────────────────────────
// ElectronCusp  —  Eq. 9
// γ(r) = -Σ_{i<j} c_{ij} / (1 + |r_i − r_j|)
// c_{ij} = 1/2 (same spin), 1/4 (opposite spin)
// ─────────────────────────────────────────────────────────────────────────────

ElectronCuspImpl::ElectronCuspImpl(int64_t n_up_, int64_t n_down_)
    : n_up(n_up_), n_down(n_down_) {}

torch::Tensor ElectronCuspImpl::forward(torch::Tensor r_ee) {
    // r_ee: [N, N] pairwise distances (strictly lower triangle used)
    int64_t N = r_ee.size(0);
    torch::Tensor gamma = torch::zeros({}, r_ee.options());

    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = i + 1; j < N; j++) {
            // same spin: both up (i,j < n_up) or both down (i,j >= n_up)
            bool same = (i < n_up && j < n_up) ||
                        (i >= n_up && j >= n_up);
            double c = same ? 0.5 : 0.25;
            gamma = gamma - c / (1.0 + r_ee[i][j]);
        }
    }
    return gamma;
}

// ─────────────────────────────────────────────────────────────────────────────
// PauliNetImpl
// ─────────────────────────────────────────────────────────────────────────────

PauliNetImpl::PauliNetImpl(PauliNetConfig c,
                             torch::Tensor npos,
                             torch::Tensor ncharge)
    : cfg(c)
{
    n_elec = cfg.n_up + cfg.n_down;

    nuclei_pos     = register_buffer("nuclei_pos",     npos);
    nuclear_charge = register_buffer("nuclear_charge", ncharge.to(torch::kFloat));

    // Nuclear embeddings Y_{θ,I}: one trainable vector per nucleus (Eq. 11, iv)
    nuclear_emb = register_parameter("nuclear_emb",
        torch::randn({cfg.n_nuclei, cfg.dim_x}) * 0.1);

    // Initial electron features — one per spin type (shared across electrons)
    init_up   = register_parameter("init_up",   torch::randn({cfg.dim_x}) * 0.1);
    init_down = register_parameter("init_down", torch::randn({cfg.dim_x}) * 0.1);

    // Distance featurisation
    dist_feat = register_module("dist_feat",
        DistanceFeature(cfg.n_rbf, cfg.r_cutoff));

    // SchNet layers
    schnet_layers = register_module("schnet_layers", torch::nn::ModuleList());
    for (int64_t l = 0; l < cfg.n_layers; l++) {
        schnet_layers->push_back(SchNetLayer(
            cfg.n_up, cfg.n_down,
            cfg.n_rbf, cfg.dim_x, cfg.dim_z,
            cfg.n_w, cfg.n_h, cfg.n_g));
    }

    // Jastrow η_θ: [dim_x → ... → 1]  (Eq. 5)
    {
        std::vector<int64_t> dims(cfg.n_eta + 1, cfg.dim_x);
        dims.back() = 1;
        jastrow = register_module("jastrow", MLP(dims));
    }

    // Backflow κ_θ: [dim_x → ... → n_det·n_orb_up/dn]  (Eq. 5)
    // Each electron i gets a multiplicative correction to each orbital
    {
        std::vector<int64_t> du(cfg.n_kappa + 1, cfg.dim_x);
        du.back() = cfg.n_det * cfg.n_up;
        backflow_up = register_module("backflow_up", MLP(du));

        std::vector<int64_t> dd(cfg.n_kappa + 1, cfg.dim_x);
        dd.back() = cfg.n_det * cfg.n_down;
        backflow_down = register_module("backflow_down", MLP(dd));
    }

    // Electron cusp
    cusp = register_module("cusp", ElectronCusp(cfg.n_up, cfg.n_down));

    // HF baseline orbitals: simple learned linear maps from 3D position
    // In a full implementation these would be initialised from a PySCF HF run.
    // Here we use a learnable linear layer as a trainable-from-random baseline.
    hf_up   = register_module("hf_up",
        torch::nn::Linear(3, cfg.n_det * cfg.n_up));
    hf_down = register_module("hf_down",
        torch::nn::Linear(3, cfg.n_det * cfg.n_down));

    // Determinant coefficients (Eq. 1)
    det_coeffs = register_parameter("det_coeffs",
        torch::ones({cfg.n_det}) / cfg.n_det);
}

// ── Helper: pairwise distances ────────────────────────────────────────────────

std::pair<torch::Tensor, torch::Tensor>
PauliNetImpl::compute_distances(torch::Tensor r) {
    // r: [N, 3]
    // r_ee: [N, N]  electron–electron distances
    auto diff_ee = r.unsqueeze(1) - r.unsqueeze(0);           // [N, N, 3]
    auto r_ee    = diff_ee.norm(2, /*dim=*/-1) + 1e-10;       // [N, N]

    // r_en: [N, M]  electron–nucleus distances
    auto npos    = nuclei_pos.to(r.device());
    auto diff_en = r.unsqueeze(1) - npos.unsqueeze(0);        // [N, M, 3]
    auto r_en    = diff_en.norm(2, /*dim=*/-1) + 1e-10;       // [N, M]

    return {r_ee, r_en};
}

// ── SchNet electron features x^(L) ───────────────────────────────────────────

torch::Tensor PauliNetImpl::electron_features(torch::Tensor r,
                                               torch::Tensor r_ee,
                                               torch::Tensor r_en) {
    // Initialise x: [N, dim_x]
    torch::Tensor x = torch::zeros({n_elec, cfg.dim_x}, r.options());
    if (cfg.n_up > 0)
        x.slice(0, 0, cfg.n_up) = init_up.unsqueeze(0).expand({cfg.n_up, -1});
    if (cfg.n_down > 0)
        x.slice(0, cfg.n_up, n_elec) = init_down.unsqueeze(0).expand({cfg.n_down, -1});

    // Distance features
    auto e_ee = dist_feat(r_ee);    // [N, N, dim_e]
    auto e_en = dist_feat(r_en);    // [N, M, dim_e]

    // Add nuclear embedding contribution to e_en (Eq. 11, iv: Y_{θ,I})
    // w_nuc maps [N, M, dim_e] → [N, M, dim_z]; Y is added as a bias per nucleus
    // We approximate by projecting Y into dim_e space and adding to e_en
    // (full implementation would add Y inside SchNetLayer; this keeps interfaces clean)
    // Nuclear embeddings Y_{θ,I} (Eq. 11, point iv): per-nucleus trainable vectors.
    // In the full SchNet formulation they modulate the nucleus messages.
    // Here we incorporate them by adding a learned bias to x at each electron,
    // summed over nuclei weighted by the inverse nuclear distance (simple proxy).
    // This preserves the interface of SchNetLayer (which expects [N,M,dim_e] e_en)
    // while still making nuclear_emb trainable and physically motivated.
    {
        torch::NoGradGuard ng_guard;  // nuclear_emb gradient handled separately
        (void)ng_guard;               // just scoping; gradient still flows
    }
    auto Y = nuclear_emb.to(r.device());     // [M, dim_x]
    // Weight Y by 1/r_en to emphasise close nuclei: [N, M, 1] × [M, dim_x]
    auto w_en = (1.0 / r_en).unsqueeze(-1);              // [N, M, 1]
    auto y_contrib = (w_en * Y.unsqueeze(0)).sum(1);     // [N, dim_x]
    // Add to initial x (before SchNet layers)
    x = x + y_contrib;

    // e_en passed unchanged to SchNetLayer
    auto e_en_aug = e_en;  // [N, M, dim_e]

    // Run interaction layers (Eq. 11)
    for (int64_t l = 0; l < cfg.n_layers; l++) {
        auto layer = schnet_layers->at<SchNetLayerImpl>(l);
        x = layer.forward(x, e_ee, e_en_aug);
    }
    return x;
}

// ── Slater matrices with backflow (Eq. 1, φ̃) ─────────────────────────────────

std::pair<torch::Tensor, torch::Tensor>
PauliNetImpl::slater_matrices(torch::Tensor r, torch::Tensor x_final) {
    // HF baseline orbitals φ_μ(r_i)
    // hf_up:   [3 → n_det·n_up],  hf_down: [3 → n_det·n_down]
    // For electron i (spin-up), evaluate orbitals at r_i: [n_det·n_up]

    // Up electrons: rows = electrons [0..n_up), cols = orbitals [0..n_up)
    torch::Tensor M_up, M_down;

    if (cfg.n_up > 0) {
        auto r_up   = r.slice(0, 0, cfg.n_up);               // [n_up, 3]
        auto phi_up = hf_up(r_up)
                         .view({cfg.n_up, cfg.n_det, cfg.n_up}); // [n_up, n_det, n_up]

        // Backflow correction f_i (Eq. 5): κ_θ(x_i^(L)) → [n_up, n_det·n_up]
        auto x_up = x_final.slice(0, 0, cfg.n_up);           // [n_up, dim_x]
        auto f_up = backflow_up(x_up)
                       .view({cfg.n_up, cfg.n_det, cfg.n_up}); // [n_up, n_det, n_up]

        // φ̃ = φ · f  (element-wise; Eq. 1)
        auto phi_tilde = phi_up * f_up;    // [n_up, n_det, n_up]
        // Slater matrix: rows=electrons, cols=orbitals  →  [n_det, n_up, n_up]
        M_up = phi_tilde.permute({1, 0, 2});
    } else {
        M_up = torch::ones({cfg.n_det, 1, 1}, r.options());
    }

    if (cfg.n_down > 0) {
        auto r_dn   = r.slice(0, cfg.n_up, n_elec);          // [n_down, 3]
        auto phi_dn = hf_down(r_dn)
                         .view({cfg.n_down, cfg.n_det, cfg.n_down});
        auto x_dn = x_final.slice(0, cfg.n_up, n_elec);
        auto f_dn = backflow_down(x_dn)
                       .view({cfg.n_down, cfg.n_det, cfg.n_down});
        auto phi_tilde = phi_dn * f_dn;
        M_down = phi_tilde.permute({1, 0, 2});
    } else {
        M_down = torch::ones({cfg.n_det, 1, 1}, r.options());
    }

    return {M_up, M_down};  // [n_det, n_up, n_up], [n_det, n_down, n_down]
}

// ── log|Ψ| and sign (numerically stable multi-determinant) ───────────────────

torch::Tensor PauliNetImpl::log_wavefunction(torch::Tensor slater_up,
                                              torch::Tensor slater_down,
                                              torch::Tensor jastrow_val,
                                              torch::Tensor cusp_val) {
    // log|det| for each determinant (numerically stable via slogdet)
    // slater_up:   [n_det, n_up, n_up]
    // slater_down: [n_det, n_down, n_down]
    auto sd_up   = torch::slogdet(slater_up);
    auto sd_down = torch::slogdet(slater_down);
    auto sign_up   = std::get<0>(sd_up);    auto logdet_up   = std::get<1>(sd_up);
    auto sign_down = std::get<0>(sd_down);  auto logdet_down = std::get<1>(sd_down);

    // log|c_p| + log|det↑_p| + log|det↓_p|  for each p
    auto log_c      = torch::log(torch::abs(det_coeffs) + 1e-30);  // [n_det]
    auto log_terms  = log_c + logdet_up + logdet_down;              // [n_det]
    auto signs      = torch::sign(det_coeffs) * sign_up * sign_down; // [n_det]

    // log|Σ c_p det_p| via log-sum-exp trick
    auto log_max    = log_terms.max();
    auto sum_scaled = (signs * torch::exp(log_terms - log_max)).sum();
    auto log_abs_psi = log_max + torch::log(torch::abs(sum_scaled) + 1e-30);

    // Add Jastrow + cusp exponent (Eq. 1: overall factor e^{J+γ})
    log_abs_psi = log_abs_psi + jastrow_val.squeeze() + cusp_val;

    return log_abs_psi;
}

// ── Forward: {log|ψ|, sign} ──────────────────────────────────────────────────

std::pair<torch::Tensor, torch::Tensor>
PauliNetImpl::forward(torch::Tensor r) {
    // r: [N, 3]
    auto _dist = compute_distances(r); auto r_ee = _dist.first; auto r_en = _dist.second;

    // Electron features
    auto x_final = electron_features(r, r_ee, r_en);

    // Jastrow factor (Eq. 5): J = η_θ(Σ_i x_i^(L))
    auto j_val = jastrow(x_final.sum(0));   // scalar

    // Cusp factor (Eq. 9)
    auto c_val = cusp(r_ee);                // scalar

    // Slater matrices with backflow
    auto _sl = slater_matrices(r, x_final); auto M_up = _sl.first; auto M_down = _sl.second;

    // Multi-det wave function
    auto log_psi = log_wavefunction(M_up, M_down, j_val, c_val);

    // Sign (not propagated through gradient for VMC; returned separately)
    auto sdu = torch::slogdet(M_up);
    auto sdd = torch::slogdet(M_down);
    auto su  = std::get<0>(sdu);
    auto sd2 = std::get<0>(sdd);
    auto sign = (torch::sign(det_coeffs) * su * sd2).sum().sign();

    return std::make_pair(log_psi, sign);
}

// ── Potential energy V(r) ─────────────────────────────────────────────────────

torch::Tensor PauliNetImpl::potential_energy(torch::Tensor r) {
    auto _dist = compute_distances(r); auto r_ee = _dist.first; auto r_en = _dist.second;
    auto Z = nuclear_charge.to(r.device());

    // Electron–nucleus attraction: -Σ_i Σ_I Z_I / r_{iI}
    auto v_en = -(Z.unsqueeze(0) / r_en).sum();   // [N,M] → scalar

    // Electron–electron repulsion: +Σ_{i<j} 1/r_{ij}
    int64_t N = r_ee.size(0);
    // Use upper triangle mask
    auto mask = torch::ones({N, N}, torch::kBool).triu(1);
    auto v_ee = (1.0 / r_ee.masked_select(mask)).sum();

    // Nucleus–nucleus repulsion: +Σ_{I<J} Z_I Z_J / R_{IJ}
    torch::Tensor v_nn = torch::zeros({}, r.options());
    if (cfg.n_nuclei > 1) {
        auto npos = nuclei_pos.to(r.device());
        auto diff_nn = npos.unsqueeze(1) - npos.unsqueeze(0);  // [M,M,3]
        auto r_nn    = diff_nn.norm(2, -1) + 1e-10;            // [M,M]
        auto mask_nn = torch::ones({cfg.n_nuclei, cfg.n_nuclei}, torch::kBool).triu(1);
        auto ZZ = Z.unsqueeze(1) * Z.unsqueeze(0);             // [M,M]
        v_nn = (ZZ.masked_select(mask_nn) / r_nn.masked_select(mask_nn)).sum();
    }

    return v_en + v_ee + v_nn;
}

// ── Local energy E_loc = -½∇²ψ/ψ + V  (Eq. 7) ────────────────────────────────

torch::Tensor PauliNetImpl::local_energy(torch::Tensor r) {
    // Always compute local energy with autograd enabled, regardless of outer context.
    // This is required for kinetic energy (needs ∇²log|ψ|).
    torch::AutoGradMode grad_mode(true);

    // Detach r from any prior graph and re-attach as a fresh leaf
    r = r.detach().requires_grad_(true);

    auto _fw = forward(r); auto log_psi = _fw.first;

    // ∇_r log|ψ|  — gradient w.r.t. all 3N coordinates  [N, 3]
    auto grad_log = torch::autograd::grad({log_psi}, {r},
                                          {torch::ones_like(log_psi)},
                                          /*retain_graph=*/true,
                                          /*create_graph=*/true)[0];

    // Laplacian via diagonal Hessian of log|ψ|
    // T = -½ (∇²log|ψ| + |∇log|ψ||²)
    int64_t n3 = n_elec * 3;
    auto g_flat = grad_log.reshape({n3});

    torch::Tensor laplacian = torch::zeros({}, r.options());
    for (int64_t k = 0; k < n3; k++) {
        auto g2 = torch::autograd::grad({g_flat[k]}, {r},
                                         {torch::ones_like(g_flat[k])},
                                         /*retain_graph=*/(k < n3 - 1),
                                         /*create_graph=*/false)[0];
        laplacian = laplacian + g2.reshape({n3})[k];
    }

    auto kinetic  = -0.5 * (laplacian + g_flat.pow(2).sum());
    auto potential = potential_energy(r.detach());

    return (kinetic + potential).detach();
}

// ── VMC energy estimate ───────────────────────────────────────────────────────

torch::Tensor PauliNetImpl::vmc_energy(torch::Tensor r_batch) {
    // r_batch: [B, N, 3]
    int64_t B = r_batch.size(0);
    auto elocs = torch::zeros({B}, r_batch.options());
    for (int64_t b = 0; b < B; b++) {
        elocs[b] = local_energy(r_batch[b]);
    }
    return elocs.mean();
}

// ── VMC gradient step (Eq. 8) ────────────────────────────────────────────────

double PauliNetImpl::vmc_step(torch::Tensor r_batch) {
    // r_batch: [B, N, 3]
    int64_t B = r_batch.size(0);

    // Compute local energies (no grad needed here for E_loc)
    std::vector<float> eloc_vals(B);
    for (int64_t b = 0; b < B; b++)
        eloc_vals[b] = local_energy(r_batch[b]).item<float>();

    auto eloc_t = torch::tensor(eloc_vals, r_batch.options());
    float E_mean = eloc_t.mean().item<float>();

    // Clip local energies (Methods: 5× median absolute deviance)
    float median  = eloc_t.median().item<float>();
    float mad     = (eloc_t - median).abs().median().item<float>();
    float clip_lo = median - cfg.clip_window * mad;
    float clip_hi = median + cfg.clip_window * mad;
    auto  eloc_clipped = torch::clamp(eloc_t, clip_lo, clip_hi);

    // ∇L = 2·E[(E_loc − Ē) · ∇log|ψ|]  (Eq. 8)
    // Accumulate gradient manually
    for (auto& p : parameters()) {
        if (p.grad().defined()) p.grad().zero_();
    }

    for (int64_t b = 0; b < B; b++) {
        auto r_b = r_batch[b].detach().requires_grad_(false);
        auto _fwb = forward(r_b); auto log_psi = _fwb.first;
        // ∇_θ log|ψ| via backward
        log_psi.backward();
        float weight = 2.0f * (eloc_clipped[b].item<float>() - E_mean) / B;
        for (auto& p : parameters()) {
            if (p.grad().defined())
                p.grad().mul_(weight);
        }
    }

    return E_mean;
}

// ─────────────────────────────────────────────────────────────────────────────
// MCMCSampler — Metropolis–Hastings for |ψ|²
// ─────────────────────────────────────────────────────────────────────────────

MCMCSampler::MCMCSampler(int64_t nw, int64_t ne,
                          double ss, double ta,
                          torch::Tensor init)
    : n_walkers(nw), n_elec(ne), step_size(ss), target_acceptance(ta)
{
    walkers  = init.clone();  // [W, N, 3]
    log_psi2 = torch::full({n_walkers}, -1e30f);
}

double MCMCSampler::step(PauliNet& model, int64_t n_steps) {
    int64_t n_accepted = 0;
    int64_t n_total    = 0;

    // Initialise cached log|ψ|² if needed
    {
        torch::NoGradGuard ng;
        for (int64_t w = 0; w < n_walkers; w++) {
            if (log_psi2[w].item<float>() < -1e29f) {
                auto _fww = model->forward(walkers[w]); auto lp = _fww.first;
                log_psi2[w]  = 2.0f * lp.detach();
            }
        }
    }

    for (int64_t s = 0; s < n_steps; s++) {
        torch::NoGradGuard ng;
        // Propose r' = r + N(0, step_size²)
        auto r_prop = walkers + torch::randn_like(walkers) * step_size;

        for (int64_t w = 0; w < n_walkers; w++) {
            auto _fwp = model->forward(r_prop[w]); auto lp_new = _fwp.first;
            float log_psi2_new = 2.0f * lp_new.item<float>();
            float log_accept   = log_psi2_new - log_psi2[w].item<float>();

            float u = std::log(((float)std::rand() / RAND_MAX) + 1e-30f);
            if (u < log_accept) {
                walkers[w]  = r_prop[w];
                log_psi2[w] = log_psi2_new;
                n_accepted++;
            }
            n_total++;
        }
    }

    double acceptance = static_cast<double>(n_accepted) / n_total;
    // Adapt step size toward target acceptance (Methods)
    if (acceptance > target_acceptance)
        step_size *= 1.05;
    else
        step_size /= 1.05;

    return acceptance;
}

torch::Tensor MCMCSampler::sample_batch(int64_t batch_size) const {
    // Random subset of walkers
    auto idx = torch::randperm(n_walkers).slice(0, 0, batch_size);
    return walkers.index_select(0, idx);
}

// ─────────────────────────────────────────────────────────────────────────────
// VMCTrainer
// ─────────────────────────────────────────────────────────────────────────────

VMCTrainer::VMCTrainer(PauliNetConfig c,
                        torch::Tensor npos,
                        torch::Tensor ncharge)
    : cfg(c),
      model(PauliNet(c, npos, ncharge)),
      sampler(c.n_walkers,
              c.n_up + c.n_down,
              c.mcmc_step_size,
              c.target_acceptance,
              // Initialise walkers near nuclei
              torch::randn({c.n_walkers, c.n_up + c.n_down, 3}) * 0.5),
      optimizer(model->parameters(),
                torch::optim::AdamWOptions(c.lr).weight_decay(1e-5))
{}

std::pair<double, double> VMCTrainer::train_step() {
    // 1. MCMC decorrelation
    sampler.step(model, cfg.n_decorrelate);

    // 2. Sample batch
    auto r_batch = sampler.sample_batch(cfg.batch_size);

    // 3. Compute local energies for variance estimate
    int64_t B = r_batch.size(0);
    std::vector<float> elocs(B);
    {
        torch::NoGradGuard ng;
        for (int64_t b = 0; b < B; b++)
            elocs[b] = model->local_energy(r_batch[b]).item<float>();
    }
    auto eloc_t = torch::tensor(elocs);
    double E_mean = eloc_t.mean().item<double>();
    double E_var  = eloc_t.var().item<double>();

    // 4. VMC gradient (Eq. 8) and optimizer step
    optimizer.zero_grad();
    model->vmc_step(r_batch);
    optimizer.step();

    return {E_mean, E_var};
}

void VMCTrainer::train(int64_t n_steps, bool verbose) {
    // Burn-in MCMC
    sampler.step(model, cfg.n_discard);

    for (int64_t t = 0; t < n_steps; t++) {
        // Periodic resampling (Table 2: every 100 steps)
        if (t % cfg.resample_period == 0 && t > 0)
            sampler.step(model, cfg.n_discard);

        auto _ts = train_step(); auto E = _ts.first; auto var = _ts.second;

        // LR decay (Table 2: decay period t₀=200)
        if ((t + 1) % cfg.lr_decay_period == 0) {
            for (auto& pg : optimizer.param_groups()) {
                auto& opts = static_cast<torch::optim::AdamWOptions&>(pg.options());
                opts.lr(opts.lr() * 0.5);
            }
        }

        if (verbose && (t % 100 == 0 || t < 10)) {
            std::cout << "step " << t
                      << "  E=" << E << " Eh"
                      << "  var=" << var << "\n";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helpers
// ─────────────────────────────────────────────────────────────────────────────

PauliNet make_paulinet_hydrogen() {
    auto cfg = PauliNetConfig::hydrogen();
    auto npos    = torch::zeros({1, 3});
    auto ncharge = torch::tensor({1.0f});
    return PauliNet(cfg, npos, ncharge);
}

PauliNet make_paulinet_h2(double d) {
    auto cfg = PauliNetConfig::h2();
    // Place two protons symmetrically along x-axis (equilibrium d=1.401 a₀)
    auto npos = torch::tensor({{-(float)d/2, 0.f, 0.f},
                                { (float)d/2, 0.f, 0.f}});
    auto ncharge = torch::tensor({1.0f, 1.0f});
    return PauliNet(cfg, npos, ncharge);
}

PauliNet make_paulinet_helium() {
    PauliNetConfig cfg;
    cfg.n_up = 1; cfg.n_down = 1; cfg.n_nuclei = 1; cfg.n_det = 1;
    auto npos    = torch::zeros({1, 3});
    auto ncharge = torch::tensor({2.0f});
    return PauliNet(cfg, npos, ncharge);
}

}  // namespace quantum
}  // namespace models
}  // namespace dm
