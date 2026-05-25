#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Variational Auto-Encoder (VAE) — Auto-Encoding Variational Bayes
// Diederik P. Kingma & Max Welling, arXiv:1312.6114v11, 2022
//
// ── Architecture (Section 3, Appendix C) ────────────────────────────────────
//
// Encoder  q_φ(z|x)  — Gaussian MLP (Appendix C.2, Eq. 12):
//   h      = tanh(W₃ x  + b₃)           single hidden layer
//   μ      = W₄ h + b₄                  mean of posterior
//   log σ² = W₅ h + b₅                  log-variance of posterior
//
// Reparameterisation trick (Section 2.4, Eq. 4):
//   z = μ + σ ⊙ ε,   ε ~ N(0, I)
//
// Decoder  p_θ(x|z)  — Bernoulli MLP (Appendix C.1, Eq. 11):
//   h = tanh(W₁ z + b₁)                 single hidden layer
//   y = sigmoid(W₂ h + b₂)              reconstruction probabilities
//
// ── ELBO objective (Section 3, Eq. 10) ──────────────────────────────────────
// L(θ,φ; x) ≃ -KL(q_φ(z|x) || p_θ(z)) + E[log p_θ(x|z)]
//
// KL term (Appendix B, analytical):
//   -D_KL = ½ Σⱼ (1 + log(σⱼ²) - μⱼ² - σⱼ²)
//
// Reconstruction term (Bernoulli, Eq. 11):
//   log p_θ(x|z) = Σᵢ xᵢ log yᵢ + (1-xᵢ) log(1-yᵢ)   [binary cross-entropy]
//
// ── Hyperparameters (Section 5) ─────────────────────────────────────────────
//   hidden_dim = 500 (MNIST), latent_dim = 20
//   minibatch M = 100, samples L = 1 per datapoint
//   Optimizer: Adagrad (paper), Adam also works well in practice
//   Initialisation: N(0, 0.01)
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <string>
#include <vector>

namespace dm {
namespace models {
namespace generative {

// ─────────────────────────────────────────────────────────────────────────────
// Encoder  q_φ(z|x) — Gaussian MLP  (Appendix C.2)
//
// Input:  x [B, input_dim]  (flattened image, values in [0,1])
// Output: μ [B, latent_dim],  log_var [B, latent_dim]
// ─────────────────────────────────────────────────────────────────────────────
struct VAEEncoderImpl : torch::nn::Module {
    torch::nn::Linear fc_hidden{nullptr};   // W₃, b₃  (input_dim → hidden_dim)
    torch::nn::Linear fc_mu{nullptr};       // W₄, b₄  (hidden_dim → latent_dim)
    torch::nn::Linear fc_logvar{nullptr};   // W₅, b₅  (hidden_dim → latent_dim)

    VAEEncoderImpl(int64_t input_dim, int64_t hidden_dim, int64_t latent_dim);

    // Returns {mu, log_var}, each [B, latent_dim]
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(VAEEncoder);

// ─────────────────────────────────────────────────────────────────────────────
// Decoder  p_θ(x|z) — Bernoulli MLP  (Appendix C.1)
//
// Input:  z [B, latent_dim]
// Output: y [B, input_dim]  (reconstruction probabilities, sigmoid output)
// ─────────────────────────────────────────────────────────────────────────────
struct VAEDecoderImpl : torch::nn::Module {
    torch::nn::Linear fc_hidden{nullptr};   // W₁, b₁  (latent_dim → hidden_dim)
    torch::nn::Linear fc_out{nullptr};      // W₂, b₂  (hidden_dim → input_dim)

    VAEDecoderImpl(int64_t latent_dim, int64_t hidden_dim, int64_t output_dim);

    // Returns y [B, output_dim] — pixel probabilities in (0,1)
    torch::Tensor forward(torch::Tensor z);
};
TORCH_MODULE(VAEDecoder);

// ─────────────────────────────────────────────────────────────────────────────
// Full VAE (encoder + reparameterisation + decoder)
// ─────────────────────────────────────────────────────────────────────────────
struct VAEImpl : torch::nn::Module {
    int64_t input_dim;
    int64_t hidden_dim;
    int64_t latent_dim;

    VAEEncoder encoder{nullptr};
    VAEDecoder decoder{nullptr};

    VAEImpl(int64_t input_dim  = 784,   // 28×28 MNIST
            int64_t hidden_dim = 500,   // Section 5: "500 hidden units"
            int64_t latent_dim = 20);   // Section 5: N_z = 20

    // Forward pass.
    // Returns {recon_x, mu, log_var}
    //   recon_x  [B, input_dim] — decoder output (sigmoid)
    //   mu       [B, latent_dim]
    //   log_var  [B, latent_dim]
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    forward(torch::Tensor x);

    // Reparameterisation trick (Section 2.4):  z = μ + σ ⊙ ε
    torch::Tensor reparameterise(const torch::Tensor& mu,
                                 const torch::Tensor& log_var);

    // Sample z ~ p(z) = N(0,I) and decode
    torch::Tensor sample(int64_t n, torch::Device device);

    // Encode x to latent mean (for visualisation / downstream tasks)
    torch::Tensor encode(torch::Tensor x);
};
TORCH_MODULE(VAE);

// ─────────────────────────────────────────────────────────────────────────────
// ELBO loss  (Section 3, Eq. 10 + Appendix B)
//
// loss = -ELBO = reconstruction_loss + kl_loss
//
// reconstruction_loss = -E[log p_θ(x|z)]
//   = binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
//
// kl_loss = -½ Σⱼ (1 + log σⱼ² - μⱼ² - σⱼ²)  summed over latent dims,
//            averaged over batch
//
// Returns scalar loss (to minimise = maximise ELBO)
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor vae_loss(const torch::Tensor& recon_x,
                       const torch::Tensor& x,
                       const torch::Tensor& mu,
                       const torch::Tensor& log_var);

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration  (Section 5)
// ─────────────────────────────────────────────────────────────────────────────
struct VAETrainConfig {
    double   lr          = 1e-3;     // Adam lr (paper used Adagrad; Adam works)
    int64_t  batch_size  = 100;      // Section 5: M = 100
    int64_t  max_epochs  = 100;
    torch::Device device = torch::kCPU;
};

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// batches: vector of {x [B, input_dim] float} — targets are same as inputs
// ─────────────────────────────────────────────────────────────────────────────
float vae_train_epoch(VAE& model,
                      torch::optim::Adam& optimizer,
                      torch::Device device,
                      const std::vector<torch::Tensor>& batches);

// Evaluate average ELBO per datapoint on held-out batches
float vae_evaluate(VAE& model,
                   torch::Device device,
                   const std::vector<torch::Tensor>& batches);

void vae_train(VAE& model,
               const VAETrainConfig& cfg,
               const std::vector<torch::Tensor>& train_batches,
               const std::vector<torch::Tensor>& val_batches,
               const std::string& save_path = "vae_best.pt");

} // namespace generative
} // namespace models
} // namespace dm
