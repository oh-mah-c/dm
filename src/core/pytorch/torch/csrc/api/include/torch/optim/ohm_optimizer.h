#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch::serialize {
class OutputArchive;
class InputArchive;
} // namespace torch::serialize

namespace torch::optim {

struct TORCH_API OhmOptimizerOptions : public OptimizerCloneableOptions<OhmOptimizerOptions> {
  OhmOptimizerOptions(double lr);
  TORCH_ARG(double, lr);
  TORCH_ARG(double, alpha) = 0.1;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const OhmOptimizerOptions& lhs,
      const OhmOptimizerOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

class TORCH_API OhmOptimizer : public Optimizer {
 public:
  explicit OhmOptimizer(
      const std::vector<OptimizerParamGroup>& param_groups,
      OhmOptimizerOptions defaults)
      : Optimizer(param_groups, std::make_unique<OhmOptimizerOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.alpha() >= 0, "Invalid alpha value: ", defaults.alpha());
  }

  explicit OhmOptimizer(std::vector<Tensor> params, OhmOptimizerOptions defaults)
      : OhmOptimizer({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(OhmOptimizer);
  }
};
} // namespace torch::optim
