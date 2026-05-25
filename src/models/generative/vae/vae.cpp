// ─────────────────────────────────────────────────────────────────────────────
// VAE implementation — Kingma & Welling, arXiv:1312.6114v11
// ─────────────────────────────────────────────────────────────────────────────

#include "models/generative/vae/vae.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>

namespace dm {
namespace models {
namespace generative {

// ─────────────────────────────────────────────────────────────────────────────
// VAEEncoder
// h = tanh(W₃ x + b₃)
// μ = W₄ h + b₄
// log σ² = W₅ h + b₅
// ─────────────────────────────────────────────────────────────────────────────
VAEEncoderImpl::VAEEncoderImpl(int64_t input_dim, int64_t hidden_dim,
                                int64_t latent_dim) {
    fc_hidden = register_module("fc_hidden",
                    torch::nn::Linear(input_dim, hidden_dim));
    fc_mu     = register_module("fc_mu",
                    torch::nn::Linear(hidden_dim, latent_dim));
    fc_logvar = register_module("fc_logvar",
                    torch::nn::Linear(hidden_dim, latent_dim));

    // Initialise weights N(0, 0.01) (Section 5)
    for (auto& m : modules(false)) {
        if (auto* lin = m->as<torch::nn::LinearImpl>()) {
            torch::nn::init::normal_(lin->weight, 0.0, 0.01);
            torch::nn::init::zeros_(lin->bias);
        }
    }
}

std::pair<torch::Tensor, torch::Tensor>
VAEEncoderImpl::forward(torch::Tensor x) {
    auto h       = torch::tanh(fc_hidden->forward(x));   // Eq. 12: tanh
    auto mu      = fc_mu->forward(h);
    auto log_var = fc_logvar->forward(h);
    return {mu, log_var};
}

// ─────────────────────────────────────────────────────────────────────────────
// VAEDecoder
// h = tanh(W₁ z + b₁)
// y = sigmoid(W₂ h + b₂)
// ─────────────────────────────────────────────────────────────────────────────
VAEDecoderImpl::VAEDecoderImpl(int64_t latent_dim, int64_t hidden_dim,
                                int64_t output_dim) {
    fc_hidden = register_module("fc_hidden",
                    torch::nn::Linear(latent_dim, hidden_dim));
    fc_out    = register_module("fc_out",
                    torch::nn::Linear(hidden_dim, output_dim));

    for (auto& m : modules(false)) {
        if (auto* lin = m->as<torch::nn::LinearImpl>()) {
            torch::nn::init::normal_(lin->weight, 0.0, 0.01);
            torch::nn::init::zeros_(lin->bias);
        }
    }
}

torch::Tensor VAEDecoderImpl::forward(torch::Tensor z) {
    auto h = torch::tanh(fc_hidden->forward(z));     // Eq. 11: tanh hidden
    return torch::sigmoid(fc_out->forward(h));        // sigmoid output
}

// ─────────────────────────────────────────────────────────────────────────────
// VAE
// ─────────────────────────────────────────────────────────────────────────────
VAEImpl::VAEImpl(int64_t input_dim_, int64_t hidden_dim_, int64_t latent_dim_)
    : input_dim(input_dim_), hidden_dim(hidden_dim_), latent_dim(latent_dim_) {
    encoder = register_module("encoder",
                  VAEEncoder(input_dim_, hidden_dim_, latent_dim_));
    decoder = register_module("decoder",
                  VAEDecoder(latent_dim_, hidden_dim_, input_dim_));
}

// Reparameterisation trick (Section 2.4):  z = μ + σ ⊙ ε,  ε ~ N(0,I)
torch::Tensor VAEImpl::reparameterise(const torch::Tensor& mu,
                                       const torch::Tensor& log_var) {
    if (!is_training()) return mu;   // at inference: use mean
    auto std = torch::exp(0.5 * log_var);           // σ = exp(log σ² / 2)
    auto eps = torch::randn_like(std);              // ε ~ N(0,I)
    return mu + std * eps;                          // z = μ + σ ⊙ ε
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
VAEImpl::forward(torch::Tensor x) {
    auto [mu, log_var] = encoder->forward(x);
    auto z             = reparameterise(mu, log_var);
    auto recon_x       = decoder->forward(z);
    return {recon_x, mu, log_var};
}

torch::Tensor VAEImpl::sample(int64_t n, torch::Device device) {
    auto z = torch::randn({n, latent_dim},
                 torch::TensorOptions().device(device));
    eval();
    torch::NoGradGuard ng;
    return decoder->forward(z);
}

torch::Tensor VAEImpl::encode(torch::Tensor x) {
    eval();
    torch::NoGradGuard ng;
    return encoder->forward(x).first;   // return mean
}

// ─────────────────────────────────────────────────────────────────────────────
// ELBO loss  (Eq. 10 + Appendix B)
//
// -ELBO = BCE(recon_x, x) - KL
//
// KL = -½ Σⱼ (1 + log σⱼ² - μⱼ² - σⱼ²)   [Appendix B, analytical]
//
// Note: sum over pixels and latent dims, mean over batch — gives per-datapoint
// lower bound comparable to the paper's reported values.
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor vae_loss(const torch::Tensor& recon_x,
                       const torch::Tensor& x,
                       const torch::Tensor& mu,
                       const torch::Tensor& log_var) {
    // Reconstruction: binary cross-entropy summed over pixels, mean over batch
    auto bce = torch::nn::functional::binary_cross_entropy(
                   recon_x, x,
                   torch::nn::functional::BinaryCrossEntropyFuncOptions()
                       .reduction(torch::kSum))
               / x.size(0);

    // KL divergence: -½ Σⱼ(1 + log σⱼ² - μⱼ² - σⱼ²), mean over batch
    auto kl = -0.5 * torch::sum(
                  1.0 + log_var - mu.pow(2) - log_var.exp())
              / x.size(0);

    return bce + kl;
}

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────────────────────────────────────
float vae_train_epoch(VAE& model,
                      torch::optim::Adam& optimizer,
                      torch::Device device,
                      const std::vector<torch::Tensor>& batches) {
    model->train();
    double total_loss = 0.0;
    int64_t steps = 0;
    for (auto& batch : batches) {
        auto x = batch.to(device);
        // Flatten if needed: [B, C, H, W] → [B, input_dim]
        if (x.dim() > 2)
            x = x.view({x.size(0), -1});
        // Normalise to [0,1] if byte tensor
        if (x.scalar_type() == torch::kByte)
            x = x.to(torch::kFloat) / 255.0f;

        optimizer.zero_grad();
        auto [recon_x, mu, log_var] = model->forward(x);
        auto loss = vae_loss(recon_x, x, mu, log_var);
        loss.backward();
        optimizer.step();

        total_loss += loss.item<double>();
        ++steps;
    }
    return steps > 0 ? static_cast<float>(total_loss / steps) : 0.f;
}

float vae_evaluate(VAE& model,
                   torch::Device device,
                   const std::vector<torch::Tensor>& batches) {
    model->eval();
    torch::NoGradGuard ng;
    double total_loss = 0.0;
    int64_t steps = 0;
    for (auto& batch : batches) {
        auto x = batch.to(device);
        if (x.dim() > 2) x = x.view({x.size(0), -1});
        if (x.scalar_type() == torch::kByte)
            x = x.to(torch::kFloat) / 255.0f;

        auto [recon_x, mu, log_var] = model->forward(x);
        auto loss = vae_loss(recon_x, x, mu, log_var);
        total_loss += loss.item<double>();
        ++steps;
    }
    return steps > 0 ? static_cast<float>(total_loss / steps) : 0.f;
}

void vae_train(VAE& model,
               const VAETrainConfig& cfg,
               const std::vector<torch::Tensor>& train_batches,
               const std::vector<torch::Tensor>& val_batches,
               const std::string& save_path) {
    model->to(cfg.device);

    // Adam (paper used Adagrad; Adam is the standard modern substitute)
    torch::optim::Adam optimizer(model->parameters(),
        torch::optim::AdamOptions(cfg.lr));

    float best_val = std::numeric_limits<float>::max();

    for (int64_t epoch = 0; epoch < cfg.max_epochs; ++epoch) {
        float train_loss = vae_train_epoch(model, optimizer, cfg.device, train_batches);
        float val_loss   = vae_evaluate(model, cfg.device, val_batches);

        std::cout << "[VAE] epoch " << epoch + 1 << "/" << cfg.max_epochs
                  << "  train_loss=" << train_loss
                  << "  val_loss=" << val_loss << "\n";

        if (val_loss < best_val) {
            best_val = val_loss;
            torch::serialize::OutputArchive archive;
            model->save(archive);
            archive.save_to(save_path);
            std::cout << "[VAE] saved checkpoint → " << save_path << "\n";
        }
    }
}

} // namespace generative
} // namespace models
} // namespace dm
