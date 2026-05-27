// ─────────────────────────────────────────────────────────────────────────────
// KAN — Training Binary
// Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025
//
// Usage:
//   kan_train [OPTIONS]
//
// Options:
//   --widths N0,N1,...,NL   Layer widths (default: 2,5,1)
//   --G N                   B-spline grid intervals (default: 5)
//   --k N                   B-spline order (default: 3)
//   --optimizer adam|lbfgs  Optimizer (default: adam)
//   --lr LR                 Learning rate (default: 1e-3)
//   --steps N               Training steps (default: 2000)
//   --batch N               Batch size, 0=full (default: 0)
//   --lambda F              Sparsity regularisation weight (default: 0)
//   --mu1 F                 L1 reg coefficient (default: 1.0)
//   --mu2 F                 Entropy reg coefficient (default: 1.0)
//   --grid-schedule G1,G2.. Grid extension schedule (default: 5)
//   --steps-per-grid N      Steps per grid (default: 200)
//   --task regression|mnist Task (default: regression)
//   --save PATH             Save path (default: kan_best.pt)
//   --device cpu|cuda       Device (default: auto)
//
// Tasks:
//   regression: fit f(x,y) = exp(sin(pi*x) + y^2)  on [-1,1]^2
//   mnist:      flatten-28x28 -> 10-class classification (requires data dir)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/kan/kan.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

using namespace dm::models::nlp;

// ─── CLI helpers ─────────────────────────────────────────────────────────────

static std::string get_arg(const std::vector<std::string>& args,
                           const std::string& key,
                           const std::string& def) {
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == key) return args[i + 1];
    return def;
}

static std::vector<int64_t> parse_ints(const std::string& s) {
    std::vector<int64_t> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) v.push_back(std::stoll(tok));
    return v;
}

// ─── Regression task: f(x,y) = exp(sin(pi*x) + y^2) ─────────────────────────

static std::pair<torch::Tensor, torch::Tensor>
make_regression_batch(int64_t N, torch::Device device) {
    auto xy = torch::rand({N, 2}, device) * 2.0f - 1.0f; // [N,2] in [-1,1]^2
    auto x  = xy.select(1, 0);
    auto y  = xy.select(1, 1);
    auto f  = (torch::sin(x * static_cast<float>(M_PI)) + y.pow(2)).exp()
              .unsqueeze(1); // [N,1]
    return {xy, f};
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    // ── Parse arguments ───────────────────────────────────────────────────
    auto widths_str = get_arg(args, "--widths",       "2,5,1");
    int64_t G_arg   = std::stoll(get_arg(args, "--G",  "5"));
    int64_t k_arg   = std::stoll(get_arg(args, "--k",  "3"));
    std::string opt_name = get_arg(args, "--optimizer","adam");
    double lr       = std::stod (get_arg(args, "--lr",  "0.001"));
    int64_t steps   = std::stoll(get_arg(args, "--steps","2000"));
    int64_t batch   = std::stoll(get_arg(args, "--batch","0"));
    double lambda_  = std::stod (get_arg(args, "--lambda","0.0"));
    double mu1      = std::stod (get_arg(args, "--mu1",  "1.0"));
    double mu2      = std::stod (get_arg(args, "--mu2",  "1.0"));
    auto g_sched_s  = get_arg(args, "--grid-schedule", std::to_string(G_arg));
    int64_t spg     = std::stoll(get_arg(args, "--steps-per-grid","200"));
    std::string task = get_arg(args, "--task",         "regression");
    std::string save = get_arg(args, "--save",         "kan_best.pt");
    std::string dev_str = get_arg(args, "--device",
        torch::cuda::is_available() ? "cuda" : "cpu");

    torch::Device device(dev_str);
    auto widths     = parse_ints(widths_str);
    auto G_schedule = parse_ints(g_sched_s);
    if (G_schedule.empty()) G_schedule.push_back(G_arg);

    std::cout << "KAN training\n"
              << "  widths:    [";
    for (size_t i = 0; i < widths.size(); ++i)
        std::cout << widths[i] << (i + 1 < widths.size() ? "," : "");
    std::cout << "]\n"
              << "  G=" << G_arg << "  k=" << k_arg << "\n"
              << "  optimizer: " << opt_name << "  lr=" << lr << "\n"
              << "  steps:     " << steps << "\n"
              << "  lambda:    " << lambda_ << "\n"
              << "  task:      " << task << "\n"
              << "  device:    " << dev_str << "\n" << std::flush;

    // ── Build model ───────────────────────────────────────────────────────
    KAN model(widths, G_arg, k_arg, -1.0, 1.0, /*update_grid=*/true);
    model->to(device);

    // ── Build optimizer ───────────────────────────────────────────────────
    std::unique_ptr<torch::optim::Optimizer> optimizer;
    if (opt_name == "lbfgs") {
        optimizer = std::make_unique<torch::optim::LBFGS>(
            model->parameters(),
            torch::optim::LBFGSOptions(lr).max_iter(20));
    } else {
        optimizer = std::make_unique<torch::optim::Adam>(
            model->parameters(),
            torch::optim::AdamOptions(lr));
    }

    // ── Dataset (regression demo) ─────────────────────────────────────────
    const int64_t N_train = 1000;
    const int64_t N_test  = 200;
    auto [x_train, y_train] = make_regression_batch(N_train, device);
    auto [x_test,  y_test ] = make_regression_batch(N_test,  device);

    float best_test_loss = std::numeric_limits<float>::infinity();
    int64_t g_idx = 0; // index into G_schedule

    for (int64_t step = 0; step < steps; ++step) {
        // Grid extension schedule
        if (g_idx + 1 < static_cast<int64_t>(G_schedule.size()) &&
            step > 0 && step % spg == 0) {
            ++g_idx;
            int64_t new_G = G_schedule[g_idx];
            std::cout << "  [step " << step << "] extend grid to G=" << new_G << "\n";
            model->extend_grid(new_G, x_train);
            // Rebuild optimizer (parameters changed size)
            if (opt_name == "lbfgs") {
                optimizer = std::make_unique<torch::optim::LBFGS>(
                    model->parameters(),
                    torch::optim::LBFGSOptions(lr).max_iter(20));
            } else {
                optimizer = std::make_unique<torch::optim::Adam>(
                    model->parameters(),
                    torch::optim::AdamOptions(lr));
            }
        }

        model->train();

        // Batch selection
        torch::Tensor xb, yb;
        if (batch <= 0 || batch >= N_train) {
            xb = x_train; yb = y_train;
        } else {
            auto idx = torch::randperm(N_train, device).slice(0, 0, batch);
            xb = x_train.index_select(0, idx);
            yb = y_train.index_select(0, idx);
        }

        float step_loss = 0.0f;

        if (opt_name == "lbfgs") {
            auto closure = [&]() {
                optimizer->zero_grad();
                auto pred = model->forward(xb);
                auto loss = torch::mse_loss(pred, yb);
                if (lambda_ > 0.0)
                    loss = loss + model->sparsity_loss(xb, lambda_, mu1, mu2);
                loss.backward();
                step_loss = loss.template item<float>();
                return loss;
            };
            optimizer->step(closure);
        } else {
            optimizer->zero_grad();
            auto pred = model->forward(xb);
            auto loss = torch::mse_loss(pred, yb);
            if (lambda_ > 0.0)
                loss = loss + model->sparsity_loss(xb, lambda_, mu1, mu2);
            loss.backward();
            optimizer->step();
            step_loss = loss.template item<float>();
        }

        // Evaluate every 200 steps
        if (step % 200 == 0 || step == steps - 1) {
            model->eval();
            torch::NoGradGuard ng;
            auto test_pred = model->forward(x_test);
            float test_loss = torch::mse_loss(test_pred, y_test)
                                  .template item<float>();
            std::cout << "step [" << step << "/" << steps << "]"
                      << "  train=" << step_loss
                      << "  test="  << test_loss
                      << "\n" << std::flush;

            if (test_loss < best_test_loss) {
                best_test_loss = test_loss;
                torch::save(model, save);
            }
        }
    }

    std::cout << "Done. Best test loss: " << best_test_loss
              << "  saved to " << save << "\n";
    return 0;
}
