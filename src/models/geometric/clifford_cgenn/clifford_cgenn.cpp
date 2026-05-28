// -----------------------------------------------------------------------------
// Clifford Group Equivariant Neural Networks implementation
// -----------------------------------------------------------------------------

#include "models/geometric/clifford_cgenn/clifford_cgenn.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dm {
namespace models {
namespace geometric {

namespace {

int64_t popcount64(int64_t x) {
    return static_cast<int64_t>(__builtin_popcountll(static_cast<unsigned long long>(x)));
}

void validate_table(const CliffordProductTable& table) {
    if (table.dimension <= 0 || table.n_blades <= 0)
        throw std::invalid_argument("Invalid Clifford product table");
}

void validate_multivector(torch::Tensor x, const CliffordProductTable& table) {
    validate_table(table);
    if (!x.defined())
        throw std::invalid_argument("Clifford multivector tensor must be defined");
    if (x.dim() < 1 || x.size(-1) != table.n_blades)
        throw std::invalid_argument("Clifford multivector last dimension must equal n_blades");
}

torch::Tensor grade_mask(const CliffordProductTable& table,
                         int64_t grade,
                         const torch::TensorOptions& options) {
    std::vector<float> data(static_cast<size_t>(table.n_blades), 0.0f);
    for (int64_t blade = 0; blade < table.n_blades; ++blade)
        if (table.grade[static_cast<size_t>(blade)] == grade)
            data[static_cast<size_t>(blade)] = 1.0f;
    return torch::from_blob(data.data(), {table.n_blades}, torch::kFloat32).clone().to(options);
}

}  // namespace

CliffordAlgebraConfig CliffordAlgebraConfig::euclidean(int64_t dimension) {
    CliffordAlgebraConfig cfg;
    cfg.dimension = dimension;
    cfg.metric.assign(static_cast<size_t>(dimension), 1.0);
    return cfg;
}

int64_t CliffordAlgebraConfig::n_blades() const {
    if (dimension <= 0 || dimension > 12)
        throw std::invalid_argument("Clifford dimension must be in 1..12");
    return int64_t{1} << dimension;
}

CliffordProductTable clifford_build_product_table(const CliffordAlgebraConfig& cfg) {
    if (cfg.dimension <= 0 || cfg.dimension > 12)
        throw std::invalid_argument("Clifford dimension must be in 1..12");
    if (!cfg.metric.empty() && static_cast<int64_t>(cfg.metric.size()) != cfg.dimension)
        throw std::invalid_argument("Clifford metric length must match dimension");

    CliffordProductTable table;
    table.dimension = cfg.dimension;
    table.n_blades = int64_t{1} << cfg.dimension;
    table.result_blade.resize(static_cast<size_t>(table.n_blades * table.n_blades));
    table.coefficient.resize(static_cast<size_t>(table.n_blades * table.n_blades));
    table.grade.resize(static_cast<size_t>(table.n_blades));
    table.qbar_diag.resize(static_cast<size_t>(table.n_blades));

    std::vector<double> metric = cfg.metric;
    if (metric.empty())
        metric.assign(static_cast<size_t>(cfg.dimension), 1.0);

    for (int64_t blade = 0; blade < table.n_blades; ++blade)
        table.grade[static_cast<size_t>(blade)] = popcount64(blade);

    for (int64_t a = 0; a < table.n_blades; ++a) {
        for (int64_t b = 0; b < table.n_blades; ++b) {
            double coeff = 1.0;
            for (int64_t i = 0; i < cfg.dimension; ++i) {
                if ((a >> i) & 1) {
                    const int64_t lower_bits = b & ((int64_t{1} << i) - 1);
                    if (popcount64(lower_bits) % 2)
                        coeff = -coeff;
                }
            }
            const int64_t common = a & b;
            for (int64_t i = 0; i < cfg.dimension; ++i)
                if ((common >> i) & 1)
                    coeff *= metric[static_cast<size_t>(i)];
            const int64_t idx = a * table.n_blades + b;
            table.result_blade[static_cast<size_t>(idx)] = a ^ b;
            table.coefficient[static_cast<size_t>(idx)] = coeff;
        }
    }

    for (int64_t blade = 0; blade < table.n_blades; ++blade) {
        const int64_t m = table.grade[static_cast<size_t>(blade)];
        const double reversion = ((m * (m - 1) / 2) % 2) ? -1.0 : 1.0;
        const int64_t idx = blade * table.n_blades + blade;
        table.qbar_diag[static_cast<size_t>(blade)] =
            reversion * table.coefficient[static_cast<size_t>(idx)];
    }
    return table;
}

torch::Tensor clifford_geometric_product(torch::Tensor a,
                                         torch::Tensor b,
                                         const CliffordProductTable& table) {
    validate_multivector(a, table);
    validate_multivector(b, table);
    if (a.sizes().slice(0, a.dim() - 1) != b.sizes().slice(0, b.dim() - 1))
        throw std::invalid_argument("Clifford product operands must share batch shape");

    auto aa = a.contiguous();
    auto bb = b.contiguous();
    auto out = torch::zeros_like(aa);
    for (int64_t i = 0; i < table.n_blades; ++i) {
        auto ai = aa.select(-1, i);
        for (int64_t j = 0; j < table.n_blades; ++j) {
            const int64_t idx = i * table.n_blades + j;
            const double c = table.coefficient[static_cast<size_t>(idx)];
            if (c == 0.0)
                continue;
            const int64_t k = table.result_blade[static_cast<size_t>(idx)];
            out.select(-1, k).add_(ai * bb.select(-1, j) * c);
        }
    }
    return out;
}

torch::Tensor clifford_grade_project(torch::Tensor x,
                                     const CliffordProductTable& table,
                                     int64_t grade) {
    validate_multivector(x, table);
    if (grade < 0 || grade > table.dimension)
        throw std::invalid_argument("Invalid Clifford grade");
    auto mask = grade_mask(table, grade, x.options());
    return x * mask;
}

torch::Tensor clifford_qbar(torch::Tensor x,
                            const CliffordProductTable& table,
                            int64_t grade) {
    auto xg = clifford_grade_project(x, table, grade);
    std::vector<float> diag(static_cast<size_t>(table.n_blades), 0.0f);
    for (int64_t i = 0; i < table.n_blades; ++i)
        if (table.grade[static_cast<size_t>(i)] == grade)
            diag[static_cast<size_t>(i)] = static_cast<float>(table.qbar_diag[static_cast<size_t>(i)]);
    auto d = torch::from_blob(diag.data(), {table.n_blades}, torch::kFloat32).clone().to(x.options());
    return (xg * xg * d).sum(-1);
}

torch::Tensor clifford_embed_vector(torch::Tensor vectors,
                                    const CliffordProductTable& table) {
    validate_table(table);
    if (!vectors.defined() || vectors.dim() < 1 || vectors.size(-1) != table.dimension)
        throw std::invalid_argument("Vector embedding expects last dimension equal to algebra dimension");
    auto shape = vectors.sizes().vec();
    shape.back() = table.n_blades;
    auto out = torch::zeros(shape, vectors.options());
    for (int64_t i = 0; i < table.dimension; ++i)
        out.select(-1, int64_t{1} << i).copy_(vectors.select(-1, i));
    return out;
}

torch::Tensor clifford_rotate_2d(torch::Tensor x,
                                 double angle,
                                 const CliffordProductTable& table) {
    validate_multivector(x, table);
    if (table.dimension != 2)
        throw std::invalid_argument("clifford_rotate_2d requires a 2D algebra");
    auto out = torch::zeros_like(x);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    out.select(-1, 0).copy_(x.select(-1, 0));
    out.select(-1, 1).copy_(c * x.select(-1, 1) + s * x.select(-1, 2));
    out.select(-1, 2).copy_(-s * x.select(-1, 1) + c * x.select(-1, 2));
    out.select(-1, 3).copy_(x.select(-1, 3));
    return out;
}

CliffordLinearImpl::CliffordLinearImpl(const CliffordProductTable& table_,
                                       int64_t in_channels_,
                                       int64_t out_channels_,
                                       bool bias)
    : table(table_), in_channels(in_channels_), out_channels(out_channels_) {
    validate_table(table);
    if (in_channels <= 0 || out_channels <= 0)
        throw std::invalid_argument("CliffordLinear requires positive channels");
    weight = register_parameter("weight",
        0.05 * torch::randn({out_channels, in_channels, table.dimension + 1}));
    if (bias) {
        scalar_bias = register_parameter("scalar_bias", torch::zeros({out_channels}));
    } else {
        scalar_bias = register_buffer("scalar_bias", torch::zeros({out_channels}));
    }
}

torch::Tensor CliffordLinearImpl::forward(torch::Tensor x) {
    validate_multivector(x, table);
    if (x.dim() != 3 || x.size(1) != in_channels)
        throw std::invalid_argument("CliffordLinear expects [B,in_channels,n_blades]");
    auto out = torch::zeros({x.size(0), out_channels, table.n_blades}, x.options());
    auto w = weight.to(x.device()).to(x.dtype());
    for (int64_t g = 0; g <= table.dimension; ++g) {
        auto xg = clifford_grade_project(x, table, g);
        for (int64_t oc = 0; oc < out_channels; ++oc) {
            for (int64_t ic = 0; ic < in_channels; ++ic)
                out.select(1, oc).add_(xg.select(1, ic) * w.index({oc, ic, g}));
        }
    }
    out.select(-1, 0).add_(scalar_bias.to(x.device()).to(x.dtype()).unsqueeze(0));
    return out;
}

CliffordProductLayerImpl::CliffordProductLayerImpl(const CliffordProductTable& table_,
                                                   int64_t channels,
                                                   bool)
    : table(table_), in_channels(channels), out_channels(channels) {
    validate_table(table);
    if (channels <= 0)
        throw std::invalid_argument("CliffordProductLayer requires positive channels");
    mix = register_module("mix", CliffordLinear(table, channels, channels, false));
    coeff = register_parameter("coeff",
        0.02 * torch::randn({channels, channels,
                             table.dimension + 1,
                             table.dimension + 1,
                             table.dimension + 1}));
}

torch::Tensor CliffordProductLayerImpl::forward(torch::Tensor x) {
    validate_multivector(x, table);
    if (x.dim() != 3 || x.size(1) != in_channels)
        throw std::invalid_argument("CliffordProductLayer expects [B,C,n_blades]");
    auto y = mix->forward(x);
    auto out = torch::zeros_like(x);
    auto c = coeff.to(x.device()).to(x.dtype());
    for (int64_t oc = 0; oc < out_channels; ++oc) {
        for (int64_t ic = 0; ic < in_channels; ++ic) {
            for (int64_t gi = 0; gi <= table.dimension; ++gi) {
                auto xi = clifford_grade_project(x.select(1, ic), table, gi);
                for (int64_t gj = 0; gj <= table.dimension; ++gj) {
                    auto yj = clifford_grade_project(y.select(1, ic), table, gj);
                    auto prod = clifford_geometric_product(xi, yj, table);
                    for (int64_t gk = 0; gk <= table.dimension; ++gk) {
                        auto pg = clifford_grade_project(prod, table, gk);
                        out.select(1, oc).add_(pg * c.index({oc, ic, gi, gj, gk}));
                    }
                }
            }
        }
    }
    return out;
}

CliffordGradeNormImpl::CliffordGradeNormImpl(const CliffordProductTable& table_)
    : table(table_) {
    validate_table(table);
    alpha = register_parameter("alpha", torch::zeros({table.dimension + 1}));
}

torch::Tensor CliffordGradeNormImpl::forward(torch::Tensor x) {
    validate_multivector(x, table);
    auto out = torch::zeros_like(x);
    auto a = alpha.to(x.device()).to(x.dtype());
    for (int64_t g = 0; g <= table.dimension; ++g) {
        auto xg = clifford_grade_project(x, table, g);
        auto q = clifford_qbar(x, table, g).abs().unsqueeze(-1);
        auto denom = torch::sigmoid(a.index({g})) * (q - 1.0) + 1.0;
        out.add_(xg / denom.clamp_min(1e-6));
    }
    return out;
}

CliffordGatedActivationImpl::CliffordGatedActivationImpl(const CliffordProductTable& table_)
    : table(table_) {
    validate_table(table);
}

torch::Tensor CliffordGatedActivationImpl::forward(torch::Tensor x) {
    validate_multivector(x, table);
    std::vector<torch::Tensor> parts;
    parts.push_back(torch::relu(clifford_grade_project(x, table, 0)));
    for (int64_t g = 1; g <= table.dimension; ++g) {
        auto xg = clifford_grade_project(x, table, g);
        auto gate = torch::sigmoid(clifford_qbar(x, table, g).unsqueeze(-1));
        parts.push_back(xg * gate);
    }
    return torch::stack(parts, 0).sum(0);
}

CGENNBlockImpl::CGENNBlockImpl(const CliffordProductTable& table,
                               int64_t channels) {
    linear = register_module("linear", CliffordLinear(table, channels, channels));
    norm = register_module("norm", CliffordGradeNorm(table));
    product = register_module("product", CliffordProductLayer(table, channels));
    activation = register_module("activation", CliffordGatedActivation(table));
}

torch::Tensor CGENNBlockImpl::forward(torch::Tensor x) {
    auto y = linear->forward(x);
    y = norm->forward(y);
    y = y + product->forward(y);
    return activation->forward(y);
}

CGENNImpl::CGENNImpl(const CliffordProductTable& table_,
                    int64_t in_channels,
                    int64_t hidden_channels,
                    int64_t out_channels,
                    int64_t depth)
    : table(table_) {
    validate_table(table);
    if (depth < 0)
        throw std::invalid_argument("CGENN depth must be nonnegative");
    input = register_module("input", CliffordLinear(table, in_channels, hidden_channels));
    blocks = register_module("blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < depth; ++i)
        blocks->push_back(CGENNBlock(table, hidden_channels));
    output = register_module("output", CliffordLinear(table, hidden_channels, out_channels));
}

torch::Tensor CGENNImpl::forward(torch::Tensor x) {
    x = input->forward(x);
    for (size_t i = 0; i < blocks->size(); ++i) {
        auto& block = blocks->at<CGENNBlockImpl>(i);
        x = block.forward(x);
    }
    return output->forward(x);
}

}  // namespace geometric
}  // namespace models
}  // namespace dm
