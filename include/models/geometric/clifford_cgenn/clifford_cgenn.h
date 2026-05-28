#pragma once
// -----------------------------------------------------------------------------
// Clifford Group Equivariant Neural Networks
// Brandstetter, Ruhe & Forre, NeurIPS 2023.
//
// Implements the paper's core Clifford algebra machinery and equivariant layers:
//   - multivectors in the canonical blade basis indexed by bit masks
//   - geometric product for diagonal metrics
//   - grade projections and extended quadratic form
//   - grade-wise linear layer from Eq. 13
//   - fully connected second-order geometric product layer from Eqs. 14-15
//   - grade normalization from Eq. 16 and gated nonlinearity
// -----------------------------------------------------------------------------

#include <torch/torch.h>
#include <cstdint>
#include <vector>

namespace dm {
namespace models {
namespace geometric {

struct CliffordAlgebraConfig {
    int64_t dimension = 3;
    std::vector<double> metric;

    static CliffordAlgebraConfig euclidean(int64_t dimension);
    int64_t n_blades() const;
};

struct CliffordProductTable {
    int64_t dimension = 0;
    int64_t n_blades = 0;
    std::vector<int64_t> result_blade;
    std::vector<double> coefficient;
    std::vector<int64_t> grade;
    std::vector<double> qbar_diag;
};

CliffordProductTable clifford_build_product_table(const CliffordAlgebraConfig& cfg);
torch::Tensor clifford_geometric_product(torch::Tensor a,
                                         torch::Tensor b,
                                         const CliffordProductTable& table);
torch::Tensor clifford_grade_project(torch::Tensor x,
                                     const CliffordProductTable& table,
                                     int64_t grade);
torch::Tensor clifford_qbar(torch::Tensor x,
                            const CliffordProductTable& table,
                            int64_t grade);
torch::Tensor clifford_embed_vector(torch::Tensor vectors,
                                    const CliffordProductTable& table);
torch::Tensor clifford_rotate_2d(torch::Tensor x,
                                 double angle,
                                 const CliffordProductTable& table);

struct CliffordLinearImpl : torch::nn::Module {
    CliffordProductTable table;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    torch::Tensor weight;       // [out,in,n_grades]
    torch::Tensor scalar_bias;  // [out], grade-0 invariant bias

    CliffordLinearImpl(const CliffordProductTable& table,
                       int64_t in_channels,
                       int64_t out_channels,
                       bool bias = true);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CliffordLinear);

struct CliffordProductLayerImpl : torch::nn::Module {
    CliffordProductTable table;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    CliffordLinear mix{nullptr};
    torch::Tensor coeff;  // [out,in,n_grades,n_grades,n_grades]

    CliffordProductLayerImpl(const CliffordProductTable& table,
                             int64_t channels,
                             bool fully_connected = true);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CliffordProductLayer);

struct CliffordGradeNormImpl : torch::nn::Module {
    CliffordProductTable table;
    torch::Tensor alpha;  // [n_grades]

    explicit CliffordGradeNormImpl(const CliffordProductTable& table);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CliffordGradeNorm);

struct CliffordGatedActivationImpl : torch::nn::Module {
    CliffordProductTable table;

    explicit CliffordGatedActivationImpl(const CliffordProductTable& table);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CliffordGatedActivation);

struct CGENNBlockImpl : torch::nn::Module {
    CliffordLinear linear{nullptr};
    CliffordGradeNorm norm{nullptr};
    CliffordProductLayer product{nullptr};
    CliffordGatedActivation activation{nullptr};

    CGENNBlockImpl(const CliffordProductTable& table,
                   int64_t channels);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CGENNBlock);

struct CGENNImpl : torch::nn::Module {
    CliffordProductTable table;
    CliffordLinear input{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    CliffordLinear output{nullptr};

    CGENNImpl(const CliffordProductTable& table,
              int64_t in_channels,
              int64_t hidden_channels,
              int64_t out_channels,
              int64_t depth = 2);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CGENN);

}  // namespace geometric
}  // namespace models
}  // namespace dm
