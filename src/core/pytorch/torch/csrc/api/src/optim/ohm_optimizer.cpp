#include <torch/optim/ohm_optimizer.h>

#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/utils.h>

#include <c10/util/irange.h>

#include <functional>

namespace torch::optim {

OhmOptimizerOptions::OhmOptimizerOptions(double lr) : lr_(lr) {}

bool operator==(const OhmOptimizerOptions& lhs, const OhmOptimizerOptions& rhs) {
  return (lhs.lr() == rhs.lr()) && (lhs.alpha() == rhs.alpha());
}

void OhmOptimizerOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(alpha);
}

void OhmOptimizerOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, alpha);
}

double OhmOptimizerOptions::get_lr() const {
  return lr();
}

void OhmOptimizerOptions::set_lr(const double lr) {
  this->lr(lr);
}

Tensor OhmOptimizer::step(LossClosure closure) {
  NoGradGuard no_grad;
  Tensor loss = {};
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  for (auto& group : param_groups_) {
    auto& options = static_cast<OhmOptimizerOptions&>(group.options());
    double lr = options.lr();
    double alpha = options.alpha();

    for (auto& p : group.params()) {
      if (!p.grad().defined()) {
        continue;
      }
      auto g = p.grad().data();
      auto spatial_grad = g.clone();
      
      for (int64_t d = 0; d < g.dim(); ++d) {
          auto shift_left = torch::roll(g, -1, d);
          auto shift_right = torch::roll(g, 1, d);
          spatial_grad.add_(shift_left.add_(shift_right), alpha);
      }
      
      p.data().add_(spatial_grad, -1.0 * lr);
    }
  }
  return loss;
}

void OhmOptimizer::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}

void OhmOptimizer::load(serialize::InputArchive& archive) {
  serialize(*this, archive);
}

} // namespace torch::optim
