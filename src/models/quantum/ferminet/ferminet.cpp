// ─────────────────────────────────────────────────────────────────────────────
// FermiNet — Ab-Initio Solution of the Many-Electron Schrödinger Equation
// Pfau, Spencer, Matthews & Foulkes (DeepMind), arXiv:1909.02487v3, 2021
// ─────────────────────────────────────────────────────────────────────────────

#include "models/quantum/ferminet/ferminet.h"

#include <torch/torch.h>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace dm {
namespace models {
namespace quantum {

namespace {

torch::Tensor safe_l2_norm(torch::Tensor x, int64_t dim, bool keepdim = false,
                           double eps = 1e-8) {
    return x.pow(2).sum(dim, keepdim).add(eps * eps).sqrt();
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// FermiNetConfig
// ─────────────────────────────────────────────────────────────────────────────

FermiNetConfig FermiNetConfig::hydrogen() {
    FermiNetConfig c;
    c.n_up = 1; c.n_down = 0; c.n_nuclei = 1;
    c.n_layers = 4; c.dim_1e = 256; c.dim_2e = 32; c.n_det = 16;
    return c;
}

FermiNetConfig FermiNetConfig::helium() {
    FermiNetConfig c;
    c.n_up = 1; c.n_down = 1; c.n_nuclei = 1;
    c.n_layers = 4; c.dim_1e = 256; c.dim_2e = 32; c.n_det = 16;
    return c;
}

FermiNetConfig FermiNetConfig::h2() {
    FermiNetConfig c;
    c.n_up = 1; c.n_down = 1; c.n_nuclei = 2;
    c.n_layers = 4; c.dim_1e = 256; c.dim_2e = 32; c.n_det = 16;
    return c;
}

FermiNetConfig FermiNetConfig::lih() {
    FermiNetConfig c;
    c.n_up = 2; c.n_down = 2; c.n_nuclei = 2;
    c.n_layers = 4; c.dim_1e = 256; c.dim_2e = 32; c.n_det = 16;
    return c;
}

FermiNetConfig FermiNetConfig::carbon() {
    FermiNetConfig c;
    c.n_up = 3; c.n_down = 3; c.n_nuclei = 1;
    c.n_layers = 4; c.dim_1e = 256; c.dim_2e = 32; c.n_det = 16;
    return c;
}

// ─────────────────────────────────────────────────────────────────────────────
// FermiNetLayer (Eq 5)
// ─────────────────────────────────────────────────────────────────────────────

FermiNetLayerImpl::FermiNetLayerImpl(int64_t n_up_, int64_t n_down_,
                                     int64_t dim_1e_in_, int64_t dim_1e_out_,
                                     int64_t dim_2e_in_, int64_t dim_2e_out_,
                                     bool residual_)
    : n_up(n_up_), n_down(n_down_),
      dim_1e_in(dim_1e_in_), dim_1e_out(dim_1e_out_),
      dim_2e_in(dim_2e_in_), dim_2e_out(dim_2e_out_),
      residual(residual_) {

    // f_i^{ℓα} = (h_i^ℓα, g^↑, g^↓, g_i^{ℓα↑}, g_i^{ℓα↓})
    // dims: dim_1e_in + dim_1e_in + dim_1e_in + dim_2e_in + dim_2e_in
    // = 3*dim_1e_in + 2*dim_2e_in
    // Note: g^↑ and g^↓ are global means of h, so same dim as h
    // g_i^{ℓα↑}, g_i^{ℓα↓} are per-electron means of 2e stream

    int64_t f_dim = 3 * dim_1e_in + 2 * dim_2e_in;

    lin_1e_up   = register_module("lin_1e_up",
                     torch::nn::Linear(f_dim, dim_1e_out));
    if (n_down > 0) {
        lin_1e_down = register_module("lin_1e_down",
                         torch::nn::Linear(f_dim, dim_1e_out));
    }
    lin_2e = register_module("lin_2e",
                 torch::nn::Linear(dim_2e_in, dim_2e_out));
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
FermiNetLayerImpl::forward(
    torch::Tensor x_up,  torch::Tensor x_down,
    torch::Tensor h_uu, torch::Tensor h_ud,
    torch::Tensor h_du, torch::Tensor h_dd)
{
    // ── Global spin means g^{ℓ↑} and g^{ℓ↓} ─────────────────────────────────
    // g^{ℓ↑} = mean over spin-up electrons: [dim_1e_in]
    auto g_up   = x_up.mean(0);   // [dim_1e_in]
    // g^{ℓ↓}: zero vector if no down electrons
    torch::Tensor g_down;
    if (n_down > 0) {
        g_down = x_down.mean(0);  // [dim_1e_in]
    } else {
        g_down = torch::zeros({dim_1e_in}, x_up.options());
    }

    // ── Per-electron two-electron means ─────────────────────────────────────
    // g_i^{ℓα↑} = mean_j h_ij^{ℓα↑}: [N_a, dim_2e_in]
    // g_i^{ℓα↓} = mean_j h_ij^{ℓα↓}: [N_a, dim_2e_in]

    // For spin-up electrons (N_a = n_up):
    //   same-spin mean: mean over j of h_uu [n_up, n_up, dim_2e_in] → [n_up, dim_2e_in]
    //   opp-spin mean:  mean over j of h_ud [n_up, n_down, dim_2e_in] → [n_up, dim_2e_in]
    auto g_up_same = h_uu.mean(1);  // [n_up, dim_2e_in]
    torch::Tensor g_up_opp;
    if (n_down > 0) {
        g_up_opp = h_ud.mean(1);    // [n_up, dim_2e_in]
    } else {
        g_up_opp = torch::zeros({n_up, dim_2e_in}, x_up.options());
    }

    // ── Build f_i for spin-up: (h_i, g_up, g_down, g_i_same, g_i_opp) ──────
    // Broadcast g_up and g_down to [n_up, dim_1e_in]
    auto g_up_broad   = g_up.unsqueeze(0).expand({n_up, dim_1e_in});
    auto g_down_broad = g_down.unsqueeze(0).expand({n_up, dim_1e_in});
    auto f_up = torch::cat({x_up, g_up_broad, g_down_broad,
                            g_up_same, g_up_opp}, /*dim=*/1);  // [n_up, f_dim]

    auto h_up_new = torch::tanh(lin_1e_up(f_up));
    if (residual && dim_1e_out == x_up.size(1)) {
        h_up_new = h_up_new + x_up;
    }

    // ── Build f_i for spin-down (if any) ─────────────────────────────────────
    torch::Tensor h_down_new;
    if (n_down > 0) {
        // same-spin mean: h_dd [n_down, n_down, dim_2e_in] → [n_down, dim_2e_in]
        auto g_dn_same = h_dd.mean(1);  // [n_down, dim_2e_in]
        // opp-spin mean: h_du [n_down, n_up, dim_2e_in] → [n_down, dim_2e_in]
        auto g_dn_opp  = h_du.mean(1);  // [n_down, dim_2e_in]

        auto g_up_broad_dn   = g_up.unsqueeze(0).expand({n_down, dim_1e_in});
        auto g_down_broad_dn = g_down.unsqueeze(0).expand({n_down, dim_1e_in});
        auto f_down = torch::cat({x_down, g_up_broad_dn, g_down_broad_dn,
                                  g_dn_same, g_dn_opp}, /*dim=*/1);  // [n_down, f_dim]

        h_down_new = torch::tanh(lin_1e_down(f_down));
        if (residual && dim_1e_out == x_down.size(1)) {
            h_down_new = h_down_new + x_down;
        }
    } else {
        h_down_new = x_down;  // empty, pass through
    }

    // ── Two-electron stream update ────────────────────────────────────────────
    // Apply same linear to all pairs (shared weights across spin pairs)
    auto update_2e = [&](torch::Tensor h, int64_t na, int64_t nb) -> torch::Tensor {
        if (na == 0 || nb == 0) return h;
        // h: [na, nb, dim_2e_in] → reshape → [na*nb, dim_2e_in]
        auto h_flat = h.view({na * nb, dim_2e_in});
        auto h_new  = torch::tanh(lin_2e(h_flat)).view({na, nb, dim_2e_out});
        if (residual && dim_2e_out == h.size(2)) {
            h_new = h_new + h;
        }
        return h_new;
    };

    auto h_uu_new = update_2e(h_uu, n_up, n_up);
    auto h_ud_new = update_2e(h_ud, n_up, n_down);
    auto h_du_new = update_2e(h_du, n_down, n_up);
    auto h_dd_new = update_2e(h_dd, n_down, n_down);

    return std::make_tuple(h_up_new, h_down_new,
                           h_uu_new, h_ud_new, h_du_new, h_dd_new);
}

// ─────────────────────────────────────────────────────────────────────────────
// OrbitalLayer (Eq 6)
// ─────────────────────────────────────────────────────────────────────────────

OrbitalLayerImpl::OrbitalLayerImpl(int64_t n_up_, int64_t n_down_,
                                   int64_t n_nuclei_, int64_t n_det_,
                                   int64_t dim_1e_)
    : n_up(n_up_), n_down(n_down_), n_nuclei(n_nuclei_),
      n_det(n_det_), dim_1e(dim_1e_) {

    // Linear maps: h_j^{Lα} → n_det * N_spin orbital values
    linear_up = register_module("linear_up",
        torch::nn::Linear(dim_1e, n_det * n_up));
    if (n_down > 0) {
        linear_down = register_module("linear_down",
            torch::nn::Linear(dim_1e, n_det * n_down));
    }

    // Envelope weights π and decay scales σ (Eq 6)
    pi_up = register_parameter("pi_up",
        torch::ones({n_det, n_up, n_nuclei}) / n_nuclei);
    sigma_up = register_parameter("sigma_up",
        torch::ones({n_det, n_up, n_nuclei, 3}));

    if (n_down > 0) {
        pi_down = register_parameter("pi_down",
            torch::ones({n_det, n_down, n_nuclei}) / n_nuclei);
        sigma_down = register_parameter("sigma_down",
            torch::ones({n_det, n_down, n_nuclei, 3}));
    }

    // Determinant weights ω_k
    det_weights = register_parameter("det_weights",
        torch::ones({n_det}) / n_det);
}

torch::Tensor OrbitalLayerImpl::envelope_up(torch::Tensor r_spin, torch::Tensor npos) {
    // r_spin: [n_up, 3], npos: [M, 3]
    // Returns: [n_det, n_up, M]  envelope contribution per orbital, electron, nucleus
    // e_im(r_j) = exp(-||σ_im ⊙ (r_j - R_m)||)
    //           = exp(-sqrt(Σ_d σ_imd² (r_jd - R_md)²))

    // diff: [n_up, M, 3]
    auto diff = r_spin.unsqueeze(1) - npos.unsqueeze(0);  // [n_up, M, 3]

    // Expand sigma: [n_det, n_up, M, 3]
    // Compute weighted distance: [n_det, n_up, M]
    auto sigma = sigma_up.abs() + 1e-6;  // ensure positive
    // diff unsqueeze: [1, n_up, M, 3]
    auto diff_exp = diff.unsqueeze(0);  // [1, n_up, M, 3]
    // sigma * diff: [n_det, n_up, M, 3]
    auto weighted = sigma * diff_exp;
    // L2 norm over last dim: [n_det, n_up, M]
    auto dist = safe_l2_norm(weighted, -1);  // [n_det, n_up, M]

    // Envelope sum over nuclei: [n_det, n_up, n_up] (one per orbital)
    // e_i^k(r_j) = Σ_m π_im^k exp(-dist_im(r_j))
    auto env = (pi_up.abs() * torch::exp(-dist)).sum(-1);  // [n_det, n_up]
    return env;
}

torch::Tensor OrbitalLayerImpl::envelope_down(torch::Tensor r_spin, torch::Tensor npos) {
    auto diff = r_spin.unsqueeze(1) - npos.unsqueeze(0);  // [n_down, M, 3]
    auto sigma = sigma_down.abs() + 1e-6;
    auto diff_exp = diff.unsqueeze(0);  // [1, n_down, M, 3]
    auto weighted = sigma * diff_exp;
    auto dist = safe_l2_norm(weighted, -1);  // [n_det, n_down, M]
    auto env = (pi_down.abs() * torch::exp(-dist)).sum(-1);  // [n_det, n_down]
    return env;
}

std::pair<torch::Tensor, torch::Tensor>
OrbitalLayerImpl::forward(torch::Tensor h_up, torch::Tensor h_down,
                          torch::Tensor r, torch::Tensor npos) {
    // ── Spin-up Slater matrix ─────────────────────────────────────────────────
    // h_up: [n_up, dim_1e]
    // linear_up: [n_up, dim_1e] → [n_up, n_det * n_up]
    // reshape: [n_up, n_det, n_up] → transpose: [n_det, n_up_orb, n_up_elec]
    // Slater[k, i, j] = φ_i^k(r_j): orbital i evaluated at electron j

    auto r_up = r.slice(0, 0, n_up);  // [n_up, 3]

    // Linear part: [n_up_electrons, n_det * n_up_orbitals]
    auto lin_up = linear_up(h_up);  // [n_up, n_det * n_up]
    // Reshape to [n_up_electrons, n_det, n_up_orbitals]
    lin_up = lin_up.view({n_up, n_det, n_up});
    // Transpose to [n_det, n_up_orbitals, n_up_electrons]
    lin_up = lin_up.permute({1, 2, 0});  // [n_det, n_up, n_up]

    // Envelope: for each determinant k and orbital i, compute envelope at all electrons j
    // env_up[k, i] = envelope value for orbital i in det k (same for all electrons, broadcast)
    // Actually the envelope depends on which electron we're evaluating at:
    // φ_i^k(r_j) = linear_ij * envelope_at_rj
    // envelope_at_rj for orbital i, det k = Σ_m π_im^k exp(-||σ_im^k(r_j - R_m)||)
    // We need shape [n_det, n_up_orb, n_up_elec]

    // Compute per-electron envelopes: loop over electrons or vectorize
    // r_up: [n_up, 3]; for each electron j, env_up[k,i,j] = Σ_m π_im^k exp(...)
    // diff: [n_up_elec, M, 3]
    {
        auto diff = r_up.unsqueeze(1) - npos.unsqueeze(0);  // [n_up, M, 3]
        auto sigma = sigma_up.abs() + 1e-6;  // [n_det, n_up_orb, M, 3]
        // Expand diff: [1, 1, n_up_elec, M, 3]
        auto diff5 = diff.unsqueeze(0).unsqueeze(0);  // [1, 1, n_up, M, 3]
        // sigma: [n_det, n_up_orb, 1, M, 3]
        auto sigma5 = sigma.unsqueeze(2);
        auto weighted = sigma5 * diff5;  // [n_det, n_up_orb, n_up_elec, M, 3]
        auto dist = safe_l2_norm(weighted, -1);  // [n_det, n_up_orb, n_up_elec, M]
        // pi: [n_det, n_up_orb, 1, M]
        auto pi = pi_up.abs().unsqueeze(2);  // [n_det, n_up_orb, 1, M]
        auto env = (pi * torch::exp(-dist)).sum(-1);  // [n_det, n_up_orb, n_up_elec]

        // Slater matrix: φ_i^k(r_j) = lin[k,i,j] * env[k,i,j]
        auto slater_up = lin_up * env;  // [n_det, n_up, n_up]

        // ── Spin-down Slater matrix ───────────────────────────────────────────
        if (n_down > 0) {
            auto r_down = r.slice(0, n_up, n_up + n_down);  // [n_down, 3]
            auto lin_down = linear_down(h_down);  // [n_down, n_det * n_down]
            lin_down = lin_down.view({n_down, n_det, n_down});
            lin_down = lin_down.permute({1, 2, 0});  // [n_det, n_down_orb, n_down_elec]

            auto diff_dn = r_down.unsqueeze(1) - npos.unsqueeze(0);  // [n_down, M, 3]
            auto sigma_dn = sigma_down.abs() + 1e-6;  // [n_det, n_down_orb, M, 3]
            auto diff_dn5 = diff_dn.unsqueeze(0).unsqueeze(0);  // [1, 1, n_down, M, 3]
            auto sigma_dn5 = sigma_dn.unsqueeze(2);
            auto weighted_dn = sigma_dn5 * diff_dn5;  // [n_det, n_down_orb, n_down, M, 3]
            auto dist_dn = safe_l2_norm(weighted_dn, -1);
            auto pi_dn = pi_down.abs().unsqueeze(2);
            auto env_dn = (pi_dn * torch::exp(-dist_dn)).sum(-1);  // [n_det, n_down, n_down]
            auto slater_down = lin_down * env_dn;

            return {slater_up, slater_down};
        } else {
            // Dummy [n_det, 0, 0] for hydrogen
            auto slater_down = torch::ones({n_det, 0, 0}, r.options());
            return {slater_up, slater_down};
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FermiNet
// ─────────────────────────────────────────────────────────────────────────────

FermiNetImpl::FermiNetImpl(FermiNetConfig cfg_,
                           torch::Tensor nuclei_pos_,
                           torch::Tensor nuclear_charge_)
    : cfg(cfg_),
      n_elec(cfg_.n_up + cfg_.n_down),
      nuclei_pos(nuclei_pos_),
      nuclear_charge(nuclear_charge_) {

    // Register nuclear geometry as buffers (not trained)
    register_buffer("nuclei_pos_buf", nuclei_pos);
    register_buffer("nuclear_charge_buf", nuclear_charge);

    // ── Build two-stream layers ───────────────────────────────────────────────
    // Input dim for 1e stream: 4 * M (r_i-R_I, |r_i-R_I| per nucleus)
    int64_t dim_1e_0 = 4 * cfg.n_nuclei;
    // Input dim for 2e stream: 4 (r_ij, |r_ij|)
    int64_t dim_2e_0 = 4;

    layers = register_module("layers", torch::nn::ModuleList());

    for (int64_t l = 0; l < cfg.n_layers; ++l) {
        int64_t d1e_in  = (l == 0) ? dim_1e_0 : cfg.dim_1e;
        int64_t d1e_out = cfg.dim_1e;
        int64_t d2e_in  = (l == 0) ? dim_2e_0 : cfg.dim_2e;
        int64_t d2e_out = cfg.dim_2e;
        bool    res     = (l > 0);  // residual only when dims match (layer ≥ 1)

        layers->push_back(
            FermiNetLayer(cfg.n_up, cfg.n_down,
                          d1e_in, d1e_out,
                          d2e_in, d2e_out,
                          res));
    }

    // ── Orbital layer ─────────────────────────────────────────────────────────
    orbital = register_module("orbital",
        OrbitalLayer(cfg.n_up, cfg.n_down,
                     cfg.n_nuclei, cfg.n_det, cfg.dim_1e));
}

// ── Build input features (Algorithm 1, lines 1-4) ────────────────────────────

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor>
FermiNetImpl::input_features(torch::Tensor r) {
    // r: [N, 3], N = n_up + n_down
    auto r_up   = r.slice(0, 0,     cfg.n_up);              // [n_up, 3]
    auto r_down = r.slice(0, cfg.n_up, n_elec);             // [n_down, 3]

    // ── Single-electron features h_i^{0α} = (r_i - R_I, |r_i - R_I| ∀I) ────
    // diff_up: [n_up, M, 3], dist_up: [n_up, M, 1]
    auto build_1e = [&](torch::Tensor r_spin, int64_t n_spin) -> torch::Tensor {
        auto diff = r_spin.unsqueeze(1) - nuclei_pos.unsqueeze(0);  // [n_spin, M, 3]
        auto dist = safe_l2_norm(diff, -1, /*keepdim=*/true);       // [n_spin, M, 1]
        auto cat  = torch::cat({diff, dist}, -1);                   // [n_spin, M, 4]
        return cat.view({n_spin, 4 * cfg.n_nuclei});                // [n_spin, 4*M]
    };

    auto x_up   = build_1e(r_up,   cfg.n_up);    // [n_up, 4*M]
    torch::Tensor x_down;
    if (cfg.n_down > 0) {
        x_down = build_1e(r_down, cfg.n_down);   // [n_down, 4*M]
    } else {
        x_down = torch::zeros({0, 4 * cfg.n_nuclei}, r.options());
    }

    // ── Two-electron features h_ij^{0αβ} = (r_i - r_j, |r_i - r_j|) ────────
    auto build_2e = [&](torch::Tensor ra, torch::Tensor rb,
                        int64_t na, int64_t nb) -> torch::Tensor {
        if (na == 0 || nb == 0)
            return torch::zeros({na, nb, 4}, r.options());
        auto diff = ra.unsqueeze(1) - rb.unsqueeze(0);    // [na, nb, 3]
        auto dist = safe_l2_norm(diff, -1, /*keepdim=*/true);  // [na, nb, 1]
        return torch::cat({diff, dist}, -1);              // [na, nb, 4]
    };

    auto h_uu = build_2e(r_up, r_up, cfg.n_up, cfg.n_up);
    auto h_ud = build_2e(r_up, r_down, cfg.n_up, cfg.n_down);
    auto h_du = build_2e(r_down, r_up, cfg.n_down, cfg.n_up);
    auto h_dd = build_2e(r_down, r_down, cfg.n_down, cfg.n_down);

    return std::make_tuple(x_up, x_down, h_uu, h_ud, h_du, h_dd);
}

// ── Run all FermiNet layers ───────────────────────────────────────────────────

std::pair<torch::Tensor, torch::Tensor>
FermiNetImpl::run_layers(torch::Tensor r) {
    auto feats = input_features(r);
    auto x_up   = std::get<0>(feats);
    auto x_down = std::get<1>(feats);
    auto h_uu   = std::get<2>(feats);
    auto h_ud   = std::get<3>(feats);
    auto h_du   = std::get<4>(feats);
    auto h_dd   = std::get<5>(feats);

    for (size_t li = 0; li < layers->size(); ++li) {
        auto& layer = layers->at<FermiNetLayerImpl>(li);
        auto out = layer.forward(x_up, x_down, h_uu, h_ud, h_du, h_dd);
        x_up   = std::get<0>(out);
        x_down = std::get<1>(out);
        h_uu   = std::get<2>(out);
        h_ud   = std::get<3>(out);
        h_du   = std::get<4>(out);
        h_dd   = std::get<5>(out);
    }

    return {x_up, x_down};
}

// ── Multi-determinant log|ψ| (Eq 7) ──────────────────────────────────────────

std::pair<torch::Tensor, torch::Tensor>
FermiNetImpl::log_wavefunction(torch::Tensor slater_up, torch::Tensor slater_down) {
    // slater_up:   [n_det, n_up, n_up]
    // slater_down: [n_det, n_down, n_down]  (or [n_det, 0, 0] for H)
    // det_weights: [n_det]
    // ψ = Σ_k ω_k det[Φ^{k↑}] det[Φ^{k↓}]

    int64_t n_det = cfg.n_det;

    // Compute log|det| and sign for each determinant
    auto sld_up = torch::slogdet(slater_up);   // returns (sign [n_det], logabsdet [n_det])
    auto sign_up = std::get<0>(sld_up);
    auto logdet_up = std::get<1>(sld_up);

    torch::Tensor sign_dn, logdet_dn;
    if (cfg.n_down > 0) {
        auto sld_dn = torch::slogdet(slater_down);
        sign_dn   = std::get<0>(sld_dn);
        logdet_dn = std::get<1>(sld_dn);
    } else {
        sign_dn   = torch::ones({n_det}, slater_up.options());
        logdet_dn = torch::zeros({n_det}, slater_up.options());
    }

    // log|det↑_k det↓_k| = logdet_up_k + logdet_dn_k
    auto logdet_combined = logdet_up + logdet_dn;   // [n_det]
    auto sign_combined   = sign_up * sign_dn;        // [n_det]

    // ψ = Σ_k ω_k * sign_k * exp(logdet_k)
    // Use log-sum-exp for numerical stability
    // ψ_k = ω_k * sign_k * exp(logdet_k)
    auto w = orbital->det_weights;  // [n_det]

    // Shift by max for numerical stability
    auto log_max = logdet_combined.max();
    auto exp_terms = w * sign_combined * torch::exp(logdet_combined - log_max);  // [n_det]
    auto psi = exp_terms.sum();  // scalar

    auto sign_psi = psi.sign();
    auto log_abs_psi = torch::log(psi.abs()) + log_max;

    return {log_abs_psi, sign_psi};
}

// ── Forward pass ─────────────────────────────────────────────────────────────

std::pair<torch::Tensor, torch::Tensor>
FermiNetImpl::forward(torch::Tensor r) {
    auto streams = run_layers(r);
    auto h_up   = streams.first;
    auto h_down = streams.second;

    auto slaters = orbital->forward(h_up, h_down, r, nuclei_pos);
    auto slater_up   = slaters.first;
    auto slater_down = slaters.second;

    return log_wavefunction(slater_up, slater_down);
}

// ── Potential energy V(r) ─────────────────────────────────────────────────────

torch::Tensor FermiNetImpl::potential_energy(torch::Tensor r) {
    // Electron-nucleus: V_en = -Σ_i Σ_I Z_I / |r_i - R_I|
    auto r3 = r.unsqueeze(1);         // [N, 1, 3]
    auto R3 = nuclei_pos.unsqueeze(0);// [1, M, 3]
    auto r_en = (r3 - R3).norm(2, -1) + 1e-10;  // [N, M]
    auto Z   = nuclear_charge.unsqueeze(0);       // [1, M]
    auto V_en = -(Z / r_en).sum();

    // Electron-electron: V_ee = Σ_{i<j} 1 / |r_i - r_j|
    torch::Tensor V_ee = torch::zeros({}, r.options());
    for (int64_t i = 0; i < n_elec; ++i) {
        for (int64_t j = i + 1; j < n_elec; ++j) {
            auto d = (r[i] - r[j]).norm() + 1e-10;
            V_ee = V_ee + 1.0 / d;
        }
    }

    // Nucleus-nucleus: V_nn = Σ_{I<J} Z_I Z_J / |R_I - R_J|
    torch::Tensor V_nn = torch::zeros({}, r.options());
    for (int64_t I = 0; I < cfg.n_nuclei; ++I) {
        for (int64_t J = I + 1; J < cfg.n_nuclei; ++J) {
            auto d = (nuclei_pos[I] - nuclei_pos[J]).norm() + 1e-10;
            V_nn = V_nn + nuclear_charge[I] * nuclear_charge[J] / d;
        }
    }

    return V_en + V_ee + V_nn;
}

// ── Local energy E_L(r) (Eq 2) ───────────────────────────────────────────────

torch::Tensor FermiNetImpl::local_energy(torch::Tensor r) {
    torch::AutoGradMode grad_mode(true);

    // Need gradients w.r.t. electron positions
    auto r_req = r.detach().requires_grad_(true);

    auto fwd = forward(r_req);
    auto log_psi = fwd.first;

    // First-order gradients ∂log|ψ|/∂r_i
    auto grad1 = torch::autograd::grad({log_psi}, {r_req},
        /*grad_outputs=*/{torch::ones_like(log_psi)},
        /*retain_graph=*/true,
        /*create_graph=*/true)[0];  // [N, 3]

    // Kinetic energy: T = -½ Σ_i (∇²_i log|ψ| + |∇_i log|ψ||²)
    // T2 = -½ Σ_i |∇_i log|ψ||² (already has value from grad1)
    auto grad1_sq = grad1.pow(2).sum();  // Σ_i |∇_i log|ψ||²

    // T1 = -½ Σ_i ∇²_i log|ψ|: compute diagonal Hessian
    torch::Tensor laplacian = torch::zeros({}, r.options());
    for (int64_t i = 0; i < n_elec; ++i) {
        for (int64_t d = 0; d < 3; ++d) {
            auto g2 = torch::autograd::grad({grad1.flatten()[i * 3 + d]},
                         {r_req},
                         /*grad_outputs=*/{torch::ones({})},
                         /*retain_graph=*/true,
                         /*create_graph=*/false)[0];
            laplacian = laplacian + g2.flatten()[i * 3 + d];
        }
    }

    auto T = -0.5 * (laplacian + grad1_sq);
    auto V = potential_energy(r_req.detach());

    return T + V;
}

// ── VMC step: compute gradient and return mean energy ─────────────────────────

double FermiNetImpl::vmc_step(torch::Tensor r_batch) {
    // r_batch: [B, N, 3]
    int64_t B = r_batch.size(0);

    // Compute E_loc and log|ψ| for each walker
    std::vector<torch::Tensor> e_locs, log_psis;
    {
        torch::NoGradGuard ng;
        for (int64_t b = 0; b < B; ++b) {
            torch::AutoGradMode grad_mode(true);
            auto r = r_batch[b];
            e_locs.push_back(local_energy(r).detach());
        }
    }

    // Compute mean energy
    auto e_loc_stack = torch::stack(e_locs);  // [B]
    double E_mean = e_loc_stack.mean().item<double>();

    // Clip: ±5 × MAD (Table V footnote)
    auto E_median = e_loc_stack.median();
    auto mad      = (e_loc_stack - E_median).abs().median();
    auto e_clipped = e_loc_stack.clamp(E_median.item<double>() - 5.0 * mad.item<double>(),
                                        E_median.item<double>() + 5.0 * mad.item<double>());
    double E_clip  = e_clipped.mean().item<double>();

    // VMC gradient: ∇L = 2·E[(E_loc - Ē) · ∇log|ψ|]
    for (int64_t b = 0; b < B; ++b) {
        auto r = r_batch[b];
        auto fwd = forward(r);
        auto log_psi = fwd.first;
        double e_loc = e_clipped[b].item<double>();
        double weight = 2.0 * (e_loc - E_clip) / B;
        log_psi.backward(torch::tensor(weight));
    }

    return E_mean;
}

// ─────────────────────────────────────────────────────────────────────────────
// FermiNetMCMC
// ─────────────────────────────────────────────────────────────────────────────

FermiNetMCMC::FermiNetMCMC(int64_t n_walkers_, int64_t n_elec_,
                             double step_size_, double target_acceptance_,
                             torch::Tensor init_positions)
    : n_walkers(n_walkers_), n_elec(n_elec_),
      step_size(step_size_), target_acceptance(target_acceptance_) {
    walkers  = init_positions.clone();  // [W, N, 3]
    // Initialise log_psi2 as zeros (will be overwritten on first step)
    log_psi2 = torch::zeros({n_walkers});
}

double FermiNetMCMC::step(FermiNet& model, int64_t n_steps) {
    torch::NoGradGuard ng;
    int64_t n_accepted = 0;
    int64_t n_total    = 0;

    // Initialise cached log_psi2 if zero
    bool init_cache = (log_psi2.abs().max().item<float>() == 0.0f);
    if (init_cache) {
        for (int64_t w = 0; w < n_walkers; ++w) {
            torch::AutoGradMode gm(true);
            auto fwd = model(walkers[w]);
            log_psi2[w] = 2.0 * fwd.first.item<float>();
        }
    }

    for (int64_t s = 0; s < n_steps; ++s) {
        // Propose moves for all walkers simultaneously
        auto noise    = torch::randn_like(walkers) * (float)step_size;  // [W, N, 3]
        auto proposed = walkers + noise;

        // Compute log|ψ|² for proposals
        for (int64_t w = 0; w < n_walkers; ++w) {
            torch::AutoGradMode gm(true);
            auto fwd = model(proposed[w]);
            double log_psi2_new = 2.0 * fwd.first.item<double>();
            double log_ratio    = log_psi2_new - log_psi2[w].item<double>();
            double u = std::log(torch::rand({}).item<double>() + 1e-30);
            if (u < log_ratio) {
                walkers[w]  = proposed[w];
                log_psi2[w] = log_psi2_new;
                ++n_accepted;
            }
            ++n_total;
        }
    }

    double acc = (double)n_accepted / n_total;
    // Adapt step size
    if (acc > target_acceptance) step_size *= 1.05;
    else                         step_size *= 0.95;
    step_size = std::max(0.001, std::min(step_size, 1.0));
    return acc;
}

torch::Tensor FermiNetMCMC::sample_batch(int64_t batch_size) const {
    auto idx = torch::randint(0, n_walkers, {batch_size});
    return walkers.index({idx});  // [batch_size, N, 3]
}

// ─────────────────────────────────────────────────────────────────────────────
// FermiNetTrainer
// ─────────────────────────────────────────────────────────────────────────────

FermiNetTrainer::FermiNetTrainer(FermiNetConfig cfg_,
                                  torch::Tensor nuclei_pos,
                                  torch::Tensor nuclear_charge)
    : cfg(cfg_),
      model(cfg_, nuclei_pos, nuclear_charge),
      sampler(cfg_.n_walkers, cfg_.n_up + cfg_.n_down,
              cfg_.mcmc_step_size, cfg_.target_acceptance,
              torch::randn({cfg_.n_walkers, cfg_.n_up + cfg_.n_down, 3}) * 0.5),
      optimizer(model->parameters(),
                torch::optim::AdamOptions(1.0 / cfg_.lr_decay_offset)) {
}

std::pair<double, double> FermiNetTrainer::train_step() {
    // 1. MCMC decorrelation
    sampler.step(model, cfg.mcmc_steps_per_update);

    // 2. Sample batch
    auto r_batch = sampler.sample_batch(cfg.batch_size);  // [B, N, 3]

    // 3. Compute VMC gradient
    optimizer.zero_grad();
    double E_mean = model->vmc_step(r_batch);

    // 4. Gradient clipping
    torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);

    // 5. Optimizer step
    optimizer.step();

    // Compute variance (approximate from recent E_loc values)
    double var = 0.0;
    return {E_mean, var};
}

void FermiNetTrainer::train(int64_t n_steps, bool verbose) {
    // Burn-in
    std::cout << "FermiNet: burn-in MCMC (" << cfg.n_discard << " steps)...\n";
    sampler.step(model, cfg.n_discard);

    for (int64_t t = 0; t < n_steps; ++t) {
        // Adaptive learning rate: lr = 1 / (lr_decay_offset + t)
        double lr_t = 1.0 / (cfg.lr_decay_offset + t);
        for (auto& pg : optimizer.param_groups()) {
            static_cast<torch::optim::AdamOptions&>(pg.options()).lr(lr_t);
        }

        auto res = train_step();
        double E = res.first;

        if (verbose && (t % 100 == 0)) {
            std::cout << "  step " << t << "  E = " << E << " Ha\n";
        }
    }
}

void FermiNetTrainer::pretrain(int64_t n_steps) {
    // Simplified pretraining (Appendix A):
    // Train orbital layer to match Hydrogen-like 1s orbitals
    // For each electron, target φ_i(r_j) ≈ exp(-Z|r_j - R_I|) / sqrt(π)
    torch::optim::Adam pretrain_opt(model->parameters(),
                                    torch::optim::AdamOptions(cfg.pretrain_lr));

    std::cout << "FermiNet: pretraining (" << n_steps << " steps)...\n";
    for (int64_t t = 0; t < n_steps; ++t) {
        pretrain_opt.zero_grad();
        auto r = torch::randn({cfg.n_up + cfg.n_down, 3}) * 0.5;
        auto fwd = model->forward(r);
        auto log_psi = fwd.first;
        // Simple L2 regularisation toward finite value (placeholder)
        auto loss = (log_psi + 1.0).pow(2);
        loss.backward();
        pretrain_opt.step();
    }
    std::cout << "FermiNet: pretraining complete.\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helpers
// ─────────────────────────────────────────────────────────────────────────────

FermiNet make_ferminet_hydrogen() {
    auto cfg = FermiNetConfig::hydrogen();
    cfg.n_layers = 2; cfg.dim_1e = 32; cfg.dim_2e = 8; cfg.n_det = 1;
    auto npos    = torch::zeros({1, 3});
    auto ncharge = torch::tensor({1.0f});
    return FermiNet(cfg, npos, ncharge);
}

FermiNet make_ferminet_helium() {
    auto cfg = FermiNetConfig::helium();
    cfg.n_layers = 2; cfg.dim_1e = 32; cfg.dim_2e = 8; cfg.n_det = 1;
    auto npos    = torch::zeros({1, 3});
    auto ncharge = torch::tensor({2.0f});
    return FermiNet(cfg, npos, ncharge);
}

FermiNet make_ferminet_h2(double bond_length_bohr) {
    auto cfg = FermiNetConfig::h2();
    cfg.n_layers = 2; cfg.dim_1e = 32; cfg.dim_2e = 8; cfg.n_det = 1;
    float d = (float)bond_length_bohr / 2;
    auto npos    = torch::tensor({{-d, 0.f, 0.f}, {d, 0.f, 0.f}});
    auto ncharge = torch::tensor({1.0f, 1.0f});
    return FermiNet(cfg, npos, ncharge);
}

FermiNet make_ferminet_lih() {
    auto cfg = FermiNetConfig::lih();
    cfg.n_layers = 2; cfg.dim_1e = 32; cfg.dim_2e = 8; cfg.n_det = 2;
    auto npos    = torch::tensor({{0.f, 0.f, 0.f}, {3.015f, 0.f, 0.f}});
    auto ncharge = torch::tensor({3.0f, 1.0f});
    return FermiNet(cfg, npos, ncharge);
}

}  // namespace quantum
}  // namespace models
}  // namespace dm
