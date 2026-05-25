#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include "torch/csrc/jit/frontend/tracer.h"

#include <torch/library.h>

#include "torch/csrc/autograd/function.h"

#include "ATen/quantized/Quantizer.h"

// @generated from ../tools/autograd/templates/TraceType.cpp

// See the `Tracer` section in `torch/csrc/jit/OVERVIEW.md`.
// NOTE See [Sharded File] comment in VariableType

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>
#else
#include <ATen/ops/_cast_Char_ops.h>
#include <ATen/ops/retains_grad_ops.h>
#include <ATen/ops/_unpack_dual_ops.h>
#include <ATen/ops/_has_same_storage_numel_ops.h>
#include <ATen/ops/align_to_ops.h>
#include <ATen/ops/align_to_ops.h>
#include <ATen/ops/_debug_has_internal_overlap_ops.h>
#include <ATen/ops/sgn_ops.h>
#include <ATen/ops/sgn_ops.h>
#include <ATen/ops/sgn_ops.h>
#include <ATen/ops/real_ops.h>
#include <ATen/ops/_neg_view_ops.h>
#include <ATen/ops/avg_pool1d_ops.h>
#include <ATen/ops/adaptive_avg_pool1d_ops.h>
#include <ATen/ops/_is_all_true_ops.h>
#include <ATen/ops/_test_check_tensor_ops.h>
#include <ATen/ops/allclose_ops.h>
#include <ATen/ops/acosh_ops.h>
#include <ATen/ops/acosh_ops.h>
#include <ATen/ops/acosh_ops.h>
#include <ATen/ops/arctanh_ops.h>
#include <ATen/ops/arctanh_ops.h>
#include <ATen/ops/arctanh_ops.h>
#include <ATen/ops/atleast_3d_ops.h>
#include <ATen/ops/atleast_3d_ops.h>
#include <ATen/ops/_batch_norm_impl_index_ops.h>
#include <ATen/ops/logical_or_ops.h>
#include <ATen/ops/logical_or_ops.h>
#include <ATen/ops/logical_or_ops.h>
#include <ATen/ops/broadcast_tensors_ops.h>
#include <ATen/ops/cat_ops.h>
#include <ATen/ops/cat_ops.h>
#include <ATen/ops/cat_ops.h>
#include <ATen/ops/cat_ops.h>
#include <ATen/ops/convolution_ops.h>
#include <ATen/ops/convolution_backward_overrideable_ops.h>
#include <ATen/ops/conv_transpose1d_ops.h>
#include <ATen/ops/cos_ops.h>
#include <ATen/ops/cos_ops.h>
#include <ATen/ops/cos_ops.h>
#include <ATen/ops/cudnn_affine_grid_generator_ops.h>
#include <ATen/ops/cudnn_batch_norm_backward_ops.h>
#include <ATen/ops/cudnn_grid_sampler_backward_ops.h>
#include <ATen/ops/cumsum_ops.h>
#include <ATen/ops/cumsum_ops.h>
#include <ATen/ops/cumsum_ops.h>
#include <ATen/ops/cumsum_ops.h>
#include <ATen/ops/cumsum_ops.h>
#include <ATen/ops/cumsum_ops.h>
#include <ATen/ops/_ctc_loss_ops.h>
#include <ATen/ops/_ctc_loss_ops.h>
#include <ATen/ops/diagflat_ops.h>
#include <ATen/ops/linalg_diagonal_ops.h>
#include <ATen/ops/embedding_dense_backward_ops.h>
#include <ATen/ops/grid_sampler_2d_backward_ops.h>
#include <ATen/ops/index_copy_ops.h>
#include <ATen/ops/index_copy_ops.h>
#include <ATen/ops/index_copy_ops.h>
#include <ATen/ops/index_copy_ops.h>
#include <ATen/ops/index_copy_ops.h>
#include <ATen/ops/_unsafe_index_put_ops.h>
#include <ATen/ops/kron_ops.h>
#include <ATen/ops/kron_ops.h>
#include <ATen/ops/linear_ops.h>
#include <ATen/ops/linear_ops.h>
#include <ATen/ops/mkldnn_linear_ops.h>
#include <ATen/ops/mkldnn_linear_backward_weights_ops.h>
#include <ATen/ops/_cslt_sparse_mm_ops.h>
#include <ATen/ops/_mixed_dtypes_linear_ops.h>
#include <ATen/ops/fbgemm_linear_quantize_weight_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/linspace_ops.h>
#include <ATen/ops/_log_softmax_ops.h>
#include <ATen/ops/_log_softmax_ops.h>
#include <ATen/ops/max_ops.h>
#include <ATen/ops/max_ops.h>
#include <ATen/ops/max_ops.h>
#include <ATen/ops/max_ops.h>
#include <ATen/ops/max_pool1d_with_indices_ops.h>
#include <ATen/ops/mode_ops.h>
#include <ATen/ops/mode_ops.h>
#include <ATen/ops/mode_ops.h>
#include <ATen/ops/mode_ops.h>
#include <ATen/ops/narrow_ops.h>
#include <ATen/ops/narrow_ops.h>
#include <ATen/ops/pdist_ops.h>
#include <ATen/ops/is_pinned_ops.h>
#include <ATen/ops/_pin_memory_ops.h>
#include <ATen/ops/randn_ops.h>
#include <ATen/ops/randn_ops.h>
#include <ATen/ops/randn_ops.h>
#include <ATen/ops/randn_ops.h>
#include <ATen/ops/randn_ops.h>
#include <ATen/ops/randn_ops.h>
#include <ATen/ops/range_ops.h>
#include <ATen/ops/range_ops.h>
#include <ATen/ops/range_ops.h>
#include <ATen/ops/range_ops.h>
#include <ATen/ops/reciprocal_ops.h>
#include <ATen/ops/reciprocal_ops.h>
#include <ATen/ops/reciprocal_ops.h>
#include <ATen/ops/neg_ops.h>
#include <ATen/ops/neg_ops.h>
#include <ATen/ops/neg_ops.h>
#include <ATen/ops/reshape_as_ops.h>
#include <ATen/ops/gelu_backward_ops.h>
#include <ATen/ops/gelu_backward_ops.h>
#include <ATen/ops/silu_backward_ops.h>
#include <ATen/ops/silu_backward_ops.h>
#include <ATen/ops/sin_ops.h>
#include <ATen/ops/sin_ops.h>
#include <ATen/ops/sin_ops.h>
#include <ATen/ops/sym_is_contiguous_ops.h>
#include <ATen/ops/sym_storage_offset_ops.h>
#include <ATen/ops/diagonal_scatter_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/squeeze_ops.h>
#include <ATen/ops/sspaddmm_ops.h>
#include <ATen/ops/sspaddmm_ops.h>
#include <ATen/ops/stride_ops.h>
#include <ATen/ops/stride_ops.h>
#include <ATen/ops/one_hot_ops.h>
#include <ATen/ops/_nested_get_jagged_dummy_ops.h>
#include <ATen/ops/_unique_ops.h>
#include <ATen/ops/_weight_norm_interface_ops.h>
#include <ATen/ops/_standard_gamma_ops.h>
#include <ATen/ops/_philox_uniform_ops.h>
#include <ATen/ops/_sparse_mm_reduce_impl_backward_ops.h>
#include <ATen/ops/_scaled_grouped_mm_ops.h>
#include <ATen/ops/sparse_csc_tensor_ops.h>
#include <ATen/ops/sparse_csc_tensor_ops.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_ops.h>
#include <ATen/ops/_validate_sparse_csr_tensor_args_ops.h>
#include <ATen/ops/_to_cpu_ops.h>
#include <ATen/ops/copy_sparse_to_sparse_ops.h>
#include <ATen/ops/unbind_ops.h>
#include <ATen/ops/unbind_ops.h>
#include <ATen/ops/fake_quantize_per_channel_affine_ops.h>
#include <ATen/ops/fake_quantize_per_channel_affine_cachemask_ops.h>
#include <ATen/ops/_thnn_differentiable_gru_cell_backward_ops.h>
#include <ATen/ops/rnn_tanh_cell_ops.h>
#include <ATen/ops/quantized_gru_cell_ops.h>
#include <ATen/ops/lift_ops.h>
#include <ATen/ops/eq_ops.h>
#include <ATen/ops/eq_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/or_ops.h>
#include <ATen/ops/or_ops.h>
#include <ATen/ops/or_ops.h>
#include <ATen/ops/or_ops.h>
#include <ATen/ops/lshift_ops.h>
#include <ATen/ops/lshift_ops.h>
#include <ATen/ops/lshift_ops.h>
#include <ATen/ops/lshift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/rshift_ops.h>
#include <ATen/ops/rshift_ops.h>
#include <ATen/ops/rshift_ops.h>
#include <ATen/ops/rshift_ops.h>
#include <ATen/ops/trace_backward_ops.h>
#include <ATen/ops/eq_ops.h>
#include <ATen/ops/eq_ops.h>
#include <ATen/ops/eq_ops.h>
#include <ATen/ops/eq_ops.h>
#include <ATen/ops/le_ops.h>
#include <ATen/ops/le_ops.h>
#include <ATen/ops/le_ops.h>
#include <ATen/ops/le_ops.h>
#include <ATen/ops/le_ops.h>
#include <ATen/ops/le_ops.h>
#include <ATen/ops/masked_select_backward_ops.h>
#include <ATen/ops/nonzero_ops.h>
#include <ATen/ops/nonzero_ops.h>
#include <ATen/ops/nonzero_numpy_ops.h>
#include <ATen/ops/cholesky_ops.h>
#include <ATen/ops/cholesky_ops.h>
#include <ATen/ops/lu_solve_ops.h>
#include <ATen/ops/lu_solve_ops.h>
#include <ATen/ops/lgamma_ops.h>
#include <ATen/ops/lgamma_ops.h>
#include <ATen/ops/lgamma_ops.h>
#include <ATen/ops/igamma_ops.h>
#include <ATen/ops/igamma_ops.h>
#include <ATen/ops/igamma_ops.h>
#include <ATen/ops/max_ops.h>
#include <ATen/ops/max_ops.h>
#include <ATen/ops/max_ops.h>
#include <ATen/ops/max_ops.h>
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale_ops.h>
#include <ATen/ops/_foreach_minimum_ops.h>
#include <ATen/ops/_foreach_minimum_ops.h>
#include <ATen/ops/_foreach_minimum_ops.h>
#include <ATen/ops/_foreach_minimum_ops.h>
#include <ATen/ops/_foreach_minimum_ops.h>
#include <ATen/ops/_foreach_minimum_ops.h>
#include <ATen/ops/_foreach_frac_ops.h>
#include <ATen/ops/_foreach_frac_ops.h>
#include <ATen/ops/_foreach_trunc_ops.h>
#include <ATen/ops/_foreach_trunc_ops.h>
#include <ATen/ops/mse_loss_backward_ops.h>
#include <ATen/ops/mse_loss_backward_ops.h>
#include <ATen/ops/multilabel_margin_loss_backward_ops.h>
#include <ATen/ops/multilabel_margin_loss_backward_ops.h>
#include <ATen/ops/elu_backward_ops.h>
#include <ATen/ops/elu_backward_ops.h>
#include <ATen/ops/hardsigmoid_backward_ops.h>
#include <ATen/ops/hardsigmoid_backward_ops.h>
#include <ATen/ops/rrelu_with_noise_backward_ops.h>
#include <ATen/ops/softplus_backward_ops.h>
#include <ATen/ops/softplus_backward_ops.h>
#include <ATen/ops/max_pool2d_with_indices_ops.h>
#include <ATen/ops/max_pool2d_with_indices_ops.h>
#include <ATen/ops/reflection_pad1d_ops.h>
#include <ATen/ops/reflection_pad1d_ops.h>
#include <ATen/ops/_pad_enum_ops.h>
#include <ATen/ops/_upsample_lanczos2d_aa_ops.h>
#include <ATen/ops/upsample_nearest3d_ops.h>
#include <ATen/ops/_upsample_lanczos2d_aa_ops.h>
#include <ATen/ops/_upsample_lanczos2d_aa_ops.h>
#include <ATen/ops/upsample_nearest3d_ops.h>
#include <ATen/ops/upsample_nearest3d_ops.h>
#include <ATen/ops/sigmoid_backward_ops.h>
#include <ATen/ops/sigmoid_backward_ops.h>
#include <ATen/ops/_slow_conv2d_forward_ops.h>
#include <ATen/ops/_slow_conv2d_forward_ops.h>
#include <ATen/ops/special_entr_ops.h>
#include <ATen/ops/special_entr_ops.h>
#include <ATen/ops/special_ndtri_ops.h>
#include <ATen/ops/special_ndtri_ops.h>
#include <ATen/ops/special_erfc_ops.h>
#include <ATen/ops/special_erfc_ops.h>
#include <ATen/ops/special_gammainc_ops.h>
#include <ATen/ops/special_gammainc_ops.h>
#include <ATen/ops/linalg_lu_ops.h>
#include <ATen/ops/linalg_lu_ops.h>
#include <ATen/ops/linalg_vecdot_ops.h>
#include <ATen/ops/linalg_vecdot_ops.h>
#include <ATen/ops/linalg_svdvals_ops.h>
#include <ATen/ops/linalg_svdvals_ops.h>
#include <ATen/ops/_test_serialization_subcmul_ops.h>
#include <ATen/ops/_test_optional_intlist_ops.h>
#include <ATen/ops/_segment_reduce_backward_ops.h>
#include <ATen/ops/expand_copy_ops.h>
#include <ATen/ops/unsqueeze_copy_ops.h>
#include <ATen/ops/crow_indices_copy_ops.h>
#include <ATen/ops/to_padded_tensor_ops.h>
#include <ATen/ops/_flash_attention_forward_ops.h>
#include <ATen/ops/_flash_attention_forward_ops.h>
#include <ATen/ops/special_bessel_j0_ops.h>
#include <ATen/ops/special_bessel_j0_ops.h>
#include <ATen/ops/special_bessel_y0_ops.h>
#include <ATen/ops/special_bessel_y0_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_hermite_polynomial_he_ops.h>
#include <ATen/ops/special_hermite_polynomial_he_ops.h>
#include <ATen/ops/special_hermite_polynomial_he_ops.h>
#include <ATen/ops/special_hermite_polynomial_he_ops.h>
#include <ATen/ops/special_hermite_polynomial_he_ops.h>
#include <ATen/ops/special_hermite_polynomial_he_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v_ops.h>
#include <ATen/ops/special_spherical_bessel_j0_ops.h>
#include <ATen/ops/special_spherical_bessel_j0_ops.h>
#include <ATen/ops/_fused_adagrad_ops.h>
#include <ATen/ops/_fused_adagrad_ops.h>
#include <ATen/ops/avg_pool1d_ops.h>
#include <ATen/ops/adaptive_avg_pool1d_ops.h>
#include <ATen/ops/convolution_ops.h>
#include <ATen/ops/convolution_backward_overrideable_ops.h>
#include <ATen/ops/cudnn_affine_grid_generator_ops.h>
#include <ATen/ops/cudnn_batch_norm_backward_ops.h>
#include <ATen/ops/cudnn_grid_sampler_backward_ops.h>
#include <ATen/ops/_ctc_loss_ops.h>
#include <ATen/ops/_ctc_loss_ops.h>
#include <ATen/ops/embedding_dense_backward_ops.h>
#include <ATen/ops/grid_sampler_2d_backward_ops.h>
#include <ATen/ops/mkldnn_linear_ops.h>
#include <ATen/ops/mkldnn_linear_backward_weights_ops.h>
#include <ATen/ops/_pin_memory_ops.h>
#include <ATen/ops/randn_ops.h>
#include <ATen/ops/randn_ops.h>
#include <ATen/ops/diagonal_scatter_ops.h>
#include <ATen/ops/_unique_ops.h>
#include <ATen/ops/_weight_norm_interface_ops.h>
#include <ATen/ops/_standard_gamma_ops.h>
#include <ATen/ops/_philox_uniform_ops.h>
#include <ATen/ops/_philox_uniform_ops.h>
#include <ATen/ops/copy_sparse_to_sparse_ops.h>
#include <ATen/ops/copy_sparse_to_sparse_ops.h>
#include <ATen/ops/fake_quantize_per_channel_affine_cachemask_ops.h>
#include <ATen/ops/lift_ops.h>
#include <ATen/ops/bitwise_and_ops.h>
#include <ATen/ops/lshift_ops.h>
#include <ATen/ops/lshift_ops.h>
#include <ATen/ops/bitwise_left_shift_ops.h>
#include <ATen/ops/rshift_ops.h>
#include <ATen/ops/rshift_ops.h>
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale_ops.h>
#include <ATen/ops/_amp_foreach_non_finite_check_and_unscale_ops.h>
#include <ATen/ops/_foreach_minimum_ops.h>
#include <ATen/ops/_foreach_minimum_ops.h>
#include <ATen/ops/_foreach_minimum_ops.h>
#include <ATen/ops/_foreach_frac_ops.h>
#include <ATen/ops/_foreach_trunc_ops.h>
#include <ATen/ops/rrelu_with_noise_backward_ops.h>
#include <ATen/ops/_test_optional_intlist_ops.h>
#include <ATen/ops/_segment_reduce_backward_ops.h>
#include <ATen/ops/expand_copy_ops.h>
#include <ATen/ops/unsqueeze_copy_ops.h>
#include <ATen/ops/crow_indices_copy_ops.h>
#include <ATen/ops/to_padded_tensor_ops.h>
#include <ATen/ops/_fused_adagrad_ops.h>
#include <ATen/ops/_fused_adagrad_ops.h>
#include <ATen/ops/_fused_adagrad_ops.h>
#include <ATen/ops/_fused_adagrad_ops.h>
#endif

using namespace at;

namespace torch {

namespace TraceType {

namespace {
at::Tensor _cast_Char(c10::DispatchKeySet ks, const at::Tensor & self, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cast_Char");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cast_Char::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool retains_grad(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::retains_grad::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _unpack_dual(c10::DispatchKeySet ks, const at::Tensor & dual, int64_t level) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_unpack_dual");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "dual", dual);
    jit::tracer::addInputs(node, "level", level);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [primal, tangent] =at::_ops::_unpack_dual::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), dual, level);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, primal);
    jit::tracer::addOutput(node, tangent);
  }
  return std::make_tuple(std::move(primal), std::move(tangent));
}
bool _has_same_storage_numel(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  auto result =at::_ops::_has_same_storage_numel::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  return result;
}
at::Tensor align_to(c10::DispatchKeySet ks, const at::Tensor & self, at::DimnameList names) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::align_to");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "names", names);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::align_to::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, names);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor align_to_ellipsis_idx(c10::DispatchKeySet ks, const at::Tensor & self, at::DimnameList order, int64_t ellipsis_idx) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::align_to");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "order", order);
    jit::tracer::addInputs(node, "ellipsis_idx", ellipsis_idx);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::align_to_ellipsis_idx::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, order, ellipsis_idx);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t _debug_has_internal_overlap(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::_debug_has_internal_overlap::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor sgn(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sgn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sgn::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & sgn_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::sgn");
    } else {
      op_name = c10::Symbol::fromQualString("aten::sgn_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sgn_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sgn_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & sgn_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sgn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sgn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sgn_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor real(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::real");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::real::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _neg_view(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_neg_view");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_neg_view::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor avg_pool1d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::avg_pool1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::avg_pool1d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, ceil_mode, count_include_pad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor adaptive_avg_pool1d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_avg_pool1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::adaptive_avg_pool1d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _is_all_true(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_is_all_true");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_is_all_true::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _test_check_tensor(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_check_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_test_check_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool allclose(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) {
  auto result =at::_ops::allclose::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rtol, atol, equal_nan);
  return result;
}
at::Tensor acosh(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::acosh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::acosh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & acosh_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::acosh");
    } else {
      op_name = c10::Symbol::fromQualString("aten::acosh_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("acosh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::acosh_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & acosh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::acosh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("acosh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::acosh_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor arctanh(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::arctanh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::arctanh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & arctanh_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::arctanh");
    } else {
      op_name = c10::Symbol::fromQualString("aten::arctanh_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arctanh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::arctanh_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & arctanh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::arctanh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arctanh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::arctanh_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor atleast_3d(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atleast_3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::atleast_3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> atleast_3d_Sequence(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atleast_3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::atleast_3d_Sequence::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t> _batch_norm_impl_index(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_batch_norm_impl_index");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3, result4] =at::_ops::_batch_norm_impl_index::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
at::Tensor logical_or(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logical_or");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logical_or::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & logical_or_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::logical_or");
    } else {
      op_name = c10::Symbol::fromQualString("aten::logical_or_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_or_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logical_or_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & logical_or_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logical_or");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_or_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logical_or_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::vector<at::Tensor> broadcast_tensors(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::broadcast_tensors");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::broadcast_tensors::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor cat(c10::DispatchKeySet ks, const at::ITensorListRef & tensors, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cat");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cat::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & cat_out_out(c10::DispatchKeySet ks, const at::ITensorListRef & tensors, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cat");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cat_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cat_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cat_names(c10::DispatchKeySet ks, at::TensorList tensors, at::Dimname dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cat");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cat_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & cat_out_names_out(c10::DispatchKeySet ks, at::TensorList tensors, at::Dimname dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cat");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cat_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cat_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor convolution(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups) {
  auto result =at::_ops::convolution::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, ::std::array<bool,3> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::convolution_backward_overrideable");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [grad_input, grad_weight, grad_bias] =at::_ops::convolution_backward_overrideable::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
    jit::tracer::addOutput(node, grad_weight);
    jit::tracer::addOutput(node, grad_bias);
  }
  return std::make_tuple(std::move(grad_input), std::move(grad_weight), std::move(grad_bias));
}
at::Tensor conv_transpose1d(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymInt groups, c10::SymIntArrayRef dilation) {
  auto result =at::_ops::conv_transpose1d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
at::Tensor cos(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cos");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cos::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & cos_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::cos");
    } else {
      op_name = c10::Symbol::fromQualString("aten::cos_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cos_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cos_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & cos_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cos");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cos_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cos_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cudnn_affine_grid_generator(c10::DispatchKeySet ks, const at::Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_affine_grid_generator");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "theta", theta);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "C", C);
    jit::tracer::addInputs(node, "H", H);
    jit::tracer::addInputs(node, "W", W);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto grid =at::_ops::cudnn_affine_grid_generator::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), theta, N, C, H, W);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grid);
  }
  return grid;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_batch_norm_backward(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, const ::std::optional<at::Tensor> & save_mean, const ::std::optional<at::Tensor> & save_var, double epsilon, const at::Tensor & reserveSpace) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_batch_norm_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_var", save_var);
    jit::tracer::addInputs(node, "epsilon", epsilon);
    jit::tracer::addInputs(node, "reserveSpace", reserveSpace);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::cudnn_batch_norm_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor> cudnn_grid_sampler_backward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grid, const at::Tensor & grad_output) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_grid_sampler_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [grad_self, grad_grid] =at::_ops::cudnn_grid_sampler_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grid, grad_output);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_self);
    jit::tracer::addOutput(node, grad_grid);
  }
  return std::make_tuple(std::move(grad_self), std::move(grad_grid));
}
at::Tensor cumsum(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cumsum::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & cumsum_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::cumsum");
    } else {
      op_name = c10::Symbol::fromQualString("aten::cumsum_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cumsum_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & cumsum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cumsum_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cumsum_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cumsum_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & cumsum__dimname(c10::DispatchKeySet ks, at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::cumsum");
    } else {
      op_name = c10::Symbol::fromQualString("aten::cumsum_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cumsum__dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & cumsum_out_dimname_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cumsum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cumsum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cumsum_dimname_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor> _ctc_loss(c10::DispatchKeySet ks, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool zero_infinity) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_ctc_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_ctc_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> _ctc_loss_Tensor(c10::DispatchKeySet ks, const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, bool zero_infinity) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_ctc_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_ctc_loss_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), log_probs, targets, input_lengths, target_lengths, blank, zero_infinity);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor diagflat(c10::DispatchKeySet ks, const at::Tensor & self, int64_t offset) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::diagflat");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "offset", offset);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::diagflat::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, offset);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor linalg_diagonal(c10::DispatchKeySet ks, const at::Tensor & A, int64_t offset, int64_t dim1, int64_t dim2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_diagonal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_diagonal::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, offset, dim1, dim2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor embedding_dense_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & indices, c10::SymInt num_weights, c10::SymInt padding_idx, bool scale_grad_by_freq) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::embedding_dense_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::embedding_dense_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, indices, num_weights, padding_idx, scale_grad_by_freq);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> grid_sampler_2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::grid_sampler_2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::grid_sampler_2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor & index_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & index_copy_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::index_copy");
    } else {
      op_name = c10::Symbol::fromQualString("aten::index_copy_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_copy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_copy_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor index_copy(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & index_copy__dimname(c10::DispatchKeySet ks, at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::index_copy");
    } else {
      op_name = c10::Symbol::fromQualString("aten::index_copy_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_copy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_copy__dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor index_copy_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_copy_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _unsafe_index_put(c10::DispatchKeySet ks, const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_unsafe_index_put");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_unsafe_index_put::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, values, accumulate);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor kron(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kron");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::kron::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & kron_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kron");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("kron_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::kron_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linear(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linear");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linear::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linear_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linear");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linear_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linear_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor mkldnn_linear(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_linear");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mkldnn_linear::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> mkldnn_linear_backward_weights(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, bool bias_defined) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_linear_backward_weights");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias_defined", bias_defined);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::mkldnn_linear_backward_weights::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input, weight, bias_defined);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor _cslt_sparse_mm(c10::DispatchKeySet ks, const at::Tensor & compressed_A, const at::Tensor & dense_B, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & alpha, ::std::optional<at::ScalarType> out_dtype, bool transpose_result, int64_t alg_id, int64_t split_k, int64_t split_k_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cslt_sparse_mm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "compressed_A", compressed_A);
    jit::tracer::addInputs(node, "dense_B", dense_B);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "out_dtype", out_dtype);
    jit::tracer::addInputs(node, "transpose_result", transpose_result);
    jit::tracer::addInputs(node, "alg_id", alg_id);
    jit::tracer::addInputs(node, "split_k", split_k);
    jit::tracer::addInputs(node, "split_k_mode", split_k_mode);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cslt_sparse_mm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), compressed_A, dense_B, bias, alpha, out_dtype, transpose_result, alg_id, split_k, split_k_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _mixed_dtypes_linear(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & scale, const ::std::optional<at::Tensor> & bias, ::std::optional<c10::string_view> activation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mixed_dtypes_linear");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "activation", activation);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_mixed_dtypes_linear::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, scale, bias, activation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,double,int64_t> fbgemm_linear_quantize_weight(c10::DispatchKeySet ks, const at::Tensor & input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fbgemm_linear_quantize_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::fbgemm_linear_quantize_weight::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
at::Tensor linspace(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, int64_t steps, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linspace::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor linspace_Tensor_Tensor(c10::DispatchKeySet ks, const at::Tensor & start, const at::Tensor & end, int64_t steps, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linspace_Tensor_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor linspace_Tensor_Scalar(c10::DispatchKeySet ks, const at::Tensor & start, const at::Scalar & end, int64_t steps, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linspace_Tensor_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor linspace_Scalar_Tensor(c10::DispatchKeySet ks, const at::Scalar & start, const at::Tensor & end, int64_t steps, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linspace_Scalar_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linspace_out_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, int64_t steps, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linspace_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & linspace_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & start, const at::Tensor & end, int64_t steps, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linspace_Tensor_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & linspace_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & start, const at::Scalar & end, int64_t steps, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linspace_Tensor_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & linspace_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Tensor & end, int64_t steps, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linspace_Scalar_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _log_softmax(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool half_to_float) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_log_softmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "half_to_float", half_to_float);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_log_softmax::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, half_to_float);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _log_softmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_log_softmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "half_to_float", half_to_float);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_log_softmax_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_log_softmax_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, half_to_float, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor> max_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::max_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> max_out_dim_max(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "max", max);
      jit::tracer::addInputs(node, "max_values", max_values);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::max_dim_max::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, max, max_values);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_values);
  }
  return std::forward_as_tuple(max, max_values);
}
::std::tuple<at::Tensor,at::Tensor> max_names_dim(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::max_names_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> max_out_names_dim_max(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "max", max);
      jit::tracer::addInputs(node, "max_values", max_values);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", max);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::max_names_dim_max::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, max, max_values);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, max);
    jit::tracer::addOutput(node, max_values);
  }
  return std::forward_as_tuple(max, max_values);
}
::std::tuple<at::Tensor,at::Tensor> max_pool1d_with_indices(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool1d_with_indices");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::max_pool1d_with_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> mode(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mode");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::mode::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> mode_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mode");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mode_values::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor> mode_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mode");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::mode_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> mode_out_dimname_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mode");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mode_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mode_dimname_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
at::Tensor narrow(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, c10::SymInt start, c10::SymInt length) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::narrow");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "length", length);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::narrow::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, start, length);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor narrow_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & start, c10::SymInt length) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::narrow");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "length", length);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::narrow_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, start, length);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor pdist(c10::DispatchKeySet ks, const at::Tensor & self, double p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::pdist");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::pdist::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool is_pinned(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Device> device) {
  auto result =at::_ops::is_pinned::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, device);
  return result;
}
at::Tensor _pin_memory(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Device> device) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_pin_memory");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "device", device);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_pin_memory::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, device);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randn(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randn::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randn_generator(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randn_generator::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randn_names(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randn_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, names, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randn_generator_with_names(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randn_generator_with_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, names, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & randn_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randn_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randn_out_generator_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randn_generator_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor range_step(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::range");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::range_step::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, step, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor range(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::range");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::range::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & range_out_out_(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::range");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("range_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::range_out_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & range_out_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::range");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("range_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::range_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, step, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor reciprocal(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reciprocal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::reciprocal::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & reciprocal_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::reciprocal");
    } else {
      op_name = c10::Symbol::fromQualString("aten::reciprocal_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reciprocal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::reciprocal_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & reciprocal_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reciprocal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reciprocal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::reciprocal_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor neg(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::neg");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::neg::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & neg_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::neg");
    } else {
      op_name = c10::Symbol::fromQualString("aten::neg_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("neg_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::neg_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & neg_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::neg");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("neg_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::neg_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor reshape_as(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reshape_as");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::reshape_as::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & gelu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gelu_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "approximate", approximate);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gelu_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::gelu_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, approximate, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor gelu_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gelu_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "approximate", approximate);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gelu_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, approximate);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & silu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::silu_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("silu_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::silu_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor silu_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::silu_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::silu_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sin(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sin::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & sin_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::sin");
    } else {
      op_name = c10::Symbol::fromQualString("aten::sin_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sin_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sin_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & sin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sin_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sin_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
c10::SymBool sym_is_contiguous(c10::DispatchKeySet ks, const at::Tensor & self, at::MemoryFormat memory_format) {
  auto result =at::_ops::sym_is_contiguous::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format);
  return result;
}
c10::SymInt sym_storage_offset(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::sym_storage_offset::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor diagonal_scatter(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, int64_t offset, int64_t dim1, int64_t dim2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::diagonal_scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::diagonal_scatter::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, offset, dim1, dim2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor squeeze(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::squeeze");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::squeeze::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor squeeze_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::squeeze");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::squeeze_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor squeeze_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::squeeze");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::squeeze_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor squeeze_dims(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::squeeze");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::squeeze_dims::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & squeeze_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = c10::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::squeeze_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & squeeze__dim(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = c10::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::squeeze__dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & squeeze__dims(c10::DispatchKeySet ks, at::Tensor & self, at::IntArrayRef dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = c10::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::squeeze__dims::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & squeeze__dimname(c10::DispatchKeySet ks, at::Tensor & self, at::Dimname dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::squeeze");
    } else {
      op_name = c10::Symbol::fromQualString("aten::squeeze_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("squeeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::squeeze__dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor sspaddmm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sspaddmm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sspaddmm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat1, mat2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & sspaddmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sspaddmm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sspaddmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sspaddmm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat1, mat2, beta, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
int64_t stride_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto result =at::_ops::stride_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  return result;
}
int64_t stride_Dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim) {
  auto result =at::_ops::stride_Dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  return result;
}
at::Tensor one_hot(c10::DispatchKeySet ks, const at::Tensor & self, int64_t num_classes) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::one_hot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "num_classes", num_classes);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::one_hot::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, num_classes);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _nested_get_jagged_dummy(c10::DispatchKeySet ks, const at::Tensor & any) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_get_jagged_dummy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "any", any);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_get_jagged_dummy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), any);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _unique(c10::DispatchKeySet ks, const at::Tensor & self, bool sorted, bool return_inverse) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_unique");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sorted", sorted);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_unique::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, sorted, return_inverse);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> _weight_norm_interface(c10::DispatchKeySet ks, const at::Tensor & v, const at::Tensor & g, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_weight_norm_interface");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "g", g);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_weight_norm_interface::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), v, g, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor _standard_gamma(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_standard_gamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_standard_gamma::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _philox_uniform_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & key, double low, double high) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::_philox_uniform");
    } else {
      op_name = c10::Symbol::fromQualString("aten::_philox_uniform_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_philox_uniform_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_philox_uniform_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, key, low, high);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
::std::tuple<at::Tensor,at::Tensor> _sparse_mm_reduce_impl_backward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grad_out, const at::Tensor & weight, c10::string_view reduce, const at::Tensor & arg_out, ::std::array<bool,2> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_mm_reduce_impl_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduce", reduce);
    jit::tracer::addInputs(node, "arg_out", arg_out);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_sparse_mm_reduce_impl_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grad_out, weight, reduce, arg_out, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor _scaled_grouped_mm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, const at::Tensor & scale_a, const at::Tensor & scale_b, const ::std::optional<at::Tensor> & offs, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & scale_result, ::std::optional<at::ScalarType> out_dtype, bool use_fast_accum) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_grouped_mm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "scale_a", scale_a);
    jit::tracer::addInputs(node, "scale_b", scale_b);
    jit::tracer::addInputs(node, "offs", offs);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "scale_result", scale_result);
    jit::tracer::addInputs(node, "out_dtype", out_dtype);
    jit::tracer::addInputs(node, "use_fast_accum", use_fast_accum);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_scaled_grouped_mm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat2, scale_a, scale_b, offs, bias, scale_result, out_dtype, use_fast_accum);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sparse_csc_tensor_ccol_row_value_size(c10::DispatchKeySet ks, const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_csc_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "ccol_indices", ccol_indices);
    jit::tracer::addInputs(node, "row_indices", row_indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sparse_csc_tensor_ccol_row_value_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), ccol_indices, row_indices, values, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sparse_csc_tensor_ccol_row_value(c10::DispatchKeySet ks, const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_csc_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "ccol_indices", ccol_indices);
    jit::tracer::addInputs(node, "row_indices", row_indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sparse_csc_tensor_ccol_row_value::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), ccol_indices, row_indices, values, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_coo_tensor_unsafe(c10::DispatchKeySet ks, const at::Tensor & indices, const at::Tensor & values, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<bool> is_coalesced) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_coo_tensor_unsafe");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "is_coalesced", is_coalesced);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_coo_tensor_unsafe::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), indices, values, size, dtype, layout, device, pin_memory, is_coalesced);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _validate_sparse_csr_tensor_args(c10::DispatchKeySet ks, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, ::std::optional<bool> check_pinning) {
  at::_ops::_validate_sparse_csr_tensor_args::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), crow_indices, col_indices, values, size, check_pinning);
}
::std::vector<at::Tensor> _to_cpu(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_cpu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_to_cpu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & copy_sparse_to_sparse_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::copy_sparse_to_sparse");
    } else {
      op_name = c10::Symbol::fromQualString("aten::copy_sparse_to_sparse_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("copy_sparse_to_sparse_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::copy_sparse_to_sparse_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
::std::vector<at::Tensor> unbind_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unbind");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unbind_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> unbind_Dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unbind");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unbind_Dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fake_quantize_per_channel_affine(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fake_quantize_per_channel_affine");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fake_quantize_per_channel_affine::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, axis, quant_min, quant_max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> fake_quantize_per_channel_affine_cachemask(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fake_quantize_per_channel_affine_cachemask");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [output, mask] =at::_ops::fake_quantize_per_channel_affine_cachemask::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, axis, quant_min, quant_max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, mask);
  }
  return std::make_tuple(std::move(output), std::move(mask));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_differentiable_gru_cell_backward(c10::DispatchKeySet ks, const at::Tensor & grad_hy, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & hx, const ::std::optional<at::Tensor> & input_bias, const ::std::optional<at::Tensor> & hidden_bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_thnn_differentiable_gru_cell_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3, result4] =at::_ops::_thnn_differentiable_gru_cell_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_hy, input_gates, hidden_gates, hx, input_bias, hidden_bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, result4);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
at::Tensor rnn_tanh_cell(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) {
  auto result =at::_ops::rnn_tanh_cell::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
at::Tensor quantized_gru_cell(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantized_gru_cell");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "w_ih", w_ih);
    jit::tracer::addInputs(node, "w_hh", w_hh);
    jit::tracer::addInputs(node, "b_ih", b_ih);
    jit::tracer::addInputs(node, "b_hh", b_hh);
    jit::tracer::addInputs(node, "packed_ih", packed_ih);
    jit::tracer::addInputs(node, "packed_hh", packed_hh);
    jit::tracer::addInputs(node, "col_offsets_ih", col_offsets_ih);
    jit::tracer::addInputs(node, "col_offsets_hh", col_offsets_hh);
    jit::tracer::addInputs(node, "scale_ih", scale_ih);
    jit::tracer::addInputs(node, "scale_hh", scale_hh);
    jit::tracer::addInputs(node, "zero_point_ih", zero_point_ih);
    jit::tracer::addInputs(node, "zero_point_hh", zero_point_hh);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::quantized_gru_cell::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor lift(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lift");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::lift::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & eq__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::eq");
    } else {
      op_name = c10::Symbol::fromQualString("aten::eq_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::eq__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & eq__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::eq");
    } else {
      op_name = c10::Symbol::fromQualString("aten::eq_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::eq__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & bitwise_and_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_and_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & bitwise_and_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_and_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor bitwise_and_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bitwise_and_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor bitwise_and_Scalar_Tensor(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bitwise_and_Scalar_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor bitwise_and_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bitwise_and_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & bitwise_and__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::bitwise_and");
    } else {
      op_name = c10::Symbol::fromQualString("aten::bitwise_and_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_and__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & bitwise_and__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::bitwise_and");
    } else {
      op_name = c10::Symbol::fromQualString("aten::bitwise_and_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_and__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor __or___Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__or__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::__or___Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor __or___Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__or__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::__or___Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & __ior___Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__ior__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__ior___Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & __ior___Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__ior__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__ior___Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor __lshift___Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__lshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::__lshift___Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor __lshift___Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__lshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::__lshift___Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & __ilshift___Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__ilshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__ilshift___Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & __ilshift___Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__ilshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__ilshift___Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor bitwise_left_shift_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_left_shift");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bitwise_left_shift_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & bitwise_left_shift__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::bitwise_left_shift");
    } else {
      op_name = c10::Symbol::fromQualString("aten::bitwise_left_shift_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_left_shift_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_left_shift__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & bitwise_left_shift_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_left_shift");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_left_shift_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_left_shift_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor bitwise_left_shift_Tensor_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_left_shift");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bitwise_left_shift_Tensor_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & bitwise_left_shift__Tensor_Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::bitwise_left_shift");
    } else {
      op_name = c10::Symbol::fromQualString("aten::bitwise_left_shift_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_left_shift_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_left_shift__Tensor_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & bitwise_left_shift_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_left_shift");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_left_shift_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_left_shift_Tensor_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor bitwise_left_shift_Scalar_Tensor(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_left_shift");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bitwise_left_shift_Scalar_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor __rshift___Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__rshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::__rshift___Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor __rshift___Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__rshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::__rshift___Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & __irshift___Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__irshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__irshift___Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & __irshift___Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__irshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__irshift___Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor trace_backward(c10::DispatchKeySet ks, const at::Tensor & grad, c10::SymIntArrayRef sizes) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::trace_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "sizes", sizes);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::trace_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, sizes);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & eq_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::eq");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::eq_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor eq_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::eq");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::eq_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & eq_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::eq");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eq_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::eq_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor eq_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::eq");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::eq_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & le_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::le");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::le_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor le_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::le");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::le_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & le_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::le");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::le_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor le_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::le");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::le_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & le__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::le");
    } else {
      op_name = c10::Symbol::fromQualString("aten::le_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::le__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & le__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::le");
    } else {
      op_name = c10::Symbol::fromQualString("aten::le_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("le_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::le__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor masked_select_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & input, const at::Tensor & mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::masked_select_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::masked_select_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, input, mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & nonzero_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nonzero");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nonzero_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nonzero_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor nonzero(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nonzero");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nonzero::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> nonzero_numpy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nonzero_numpy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nonzero_numpy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & cholesky_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool upper, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cholesky");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cholesky_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cholesky_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, upper, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cholesky(c10::DispatchKeySet ks, const at::Tensor & self, bool upper) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cholesky");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cholesky::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, upper);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & lu_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lu_solve");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "LU_data", LU_data);
    jit::tracer::addInputs(node, "LU_pivots", LU_pivots);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lu_solve_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::lu_solve_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, LU_data, LU_pivots, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor lu_solve(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lu_solve");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "LU_data", LU_data);
    jit::tracer::addInputs(node, "LU_pivots", LU_pivots);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::lu_solve::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, LU_data, LU_pivots);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & lgamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lgamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lgamma_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::lgamma_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & lgamma_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::lgamma");
    } else {
      op_name = c10::Symbol::fromQualString("aten::lgamma_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lgamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::lgamma_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor lgamma(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lgamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::lgamma::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & igamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::igamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("igamma_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::igamma_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor igamma(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::igamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::igamma::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & igamma_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::igamma");
    } else {
      op_name = c10::Symbol::fromQualString("aten::igamma_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("igamma_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::igamma_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor max(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::max::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor max_other(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::max_other::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & max_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::max_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & max_out_unary_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::max_unary_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _amp_foreach_non_finite_check_and_unscale_(c10::DispatchKeySet ks, at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale) {
  at::_ops::_amp_foreach_non_finite_check_and_unscale_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, found_inf, inv_scale);
}
::std::vector<at::Tensor> _foreach_minimum_Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_minimum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scalar", scalar);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_minimum_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_minimum__Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar) {
  at::_ops::_foreach_minimum__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar);
}
::std::vector<at::Tensor> _foreach_minimum_List(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_minimum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_minimum_List::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_minimum__List(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other) {
  at::_ops::_foreach_minimum__List::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
}
::std::vector<at::Tensor> _foreach_minimum_ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_minimum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scalars", scalars);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_minimum_ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_minimum__ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  at::_ops::_foreach_minimum__ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars);
}
::std::vector<at::Tensor> _foreach_frac(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_frac");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_frac::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_frac_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_frac_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_trunc(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_trunc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_trunc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_trunc_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_trunc_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
at::Tensor & mse_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mse_loss_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mse_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mse_loss_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, reduction, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor mse_loss_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mse_loss_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mse_loss_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & multilabel_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::multilabel_margin_loss_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "is_target", is_target);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multilabel_margin_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::multilabel_margin_loss_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, reduction, is_target, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor multilabel_margin_loss_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::multilabel_margin_loss_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "is_target", is_target);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::multilabel_margin_loss_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, reduction, is_target);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & elu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::elu_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    jit::tracer::addInputs(node, "is_result", is_result);
    jit::tracer::addInputs(node, "self_or_result", self_or_result);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("elu_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::elu_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor elu_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::elu_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    jit::tracer::addInputs(node, "is_result", is_result);
    jit::tracer::addInputs(node, "self_or_result", self_or_result);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::elu_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, alpha, scale, input_scale, is_result, self_or_result);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hardsigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardsigmoid_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardsigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardsigmoid_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor hardsigmoid_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardsigmoid_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hardsigmoid_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor rrelu_with_noise_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, bool self_is_result) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rrelu_with_noise_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "self_is_result", self_is_result);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::rrelu_with_noise_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, noise, lower, upper, training, self_is_result);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & softplus_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::softplus_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softplus_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::softplus_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, beta, threshold, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor softplus_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::softplus_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::softplus_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, beta, threshold);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> max_pool2d_with_indices_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool2d_with_indices");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool2d_with_indices_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::max_pool2d_with_indices_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(out, indices);
}
::std::tuple<at::Tensor,at::Tensor> max_pool2d_with_indices(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool2d_with_indices");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::max_pool2d_with_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor & reflection_pad1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reflection_pad1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::reflection_pad1d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor reflection_pad1d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reflection_pad1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::reflection_pad1d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _pad_enum(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef pad, int64_t mode, ::std::optional<double> value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_pad_enum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pad", pad);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_pad_enum::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, pad, mode, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _upsample_lanczos2d_aa_vec(c10::DispatchKeySet ks, const at::Tensor & input, at::OptionalSymIntArrayRef output_size, bool align_corners, ::std::optional<at::ArrayRef<double>> scale_factors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_lanczos2d_aa");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scale_factors", scale_factors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_upsample_lanczos2d_aa_vec::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, output_size, align_corners, scale_factors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor upsample_nearest3d_vec(c10::DispatchKeySet ks, const at::Tensor & input, at::OptionalSymIntArrayRef output_size, ::std::optional<at::ArrayRef<double>> scale_factors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_nearest3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scale_factors", scale_factors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::upsample_nearest3d_vec::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, output_size, scale_factors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _upsample_lanczos2d_aa_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_lanczos2d_aa");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_upsample_lanczos2d_aa_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_upsample_lanczos2d_aa_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, align_corners, scales_h, scales_w, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _upsample_lanczos2d_aa(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_lanczos2d_aa");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_upsample_lanczos2d_aa::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, align_corners, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & upsample_nearest3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_nearest3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales_d", scales_d);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::upsample_nearest3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, scales_d, scales_h, scales_w, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor upsample_nearest3d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_nearest3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales_d", scales_d);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::upsample_nearest3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, scales_d, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & sigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sigmoid_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sigmoid_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor sigmoid_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sigmoid_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sigmoid_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _slow_conv2d_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & output) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_slow_conv2d_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "output", output);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_slow_conv2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_slow_conv2d_forward_output::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, output);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
at::Tensor _slow_conv2d_forward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_slow_conv2d_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_slow_conv2d_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_entr(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_entr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_entr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_entr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_entr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_entr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_entr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_ndtri(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_ndtri");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_ndtri::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_ndtri_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_ndtri");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_ndtri_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_ndtri_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_erfc(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_erfc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_erfc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_erfc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_erfc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_erfc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_erfc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_gammainc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_gammainc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_gammainc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_gammainc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_gammainc(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_gammainc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_gammainc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> linalg_lu(c10::DispatchKeySet ks, const at::Tensor & A, bool pivot) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_lu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "pivot", pivot);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [P, L, U] =at::_ops::linalg_lu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, pivot);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, P);
    jit::tracer::addOutput(node, L);
    jit::tracer::addOutput(node, U);
  }
  return std::make_tuple(std::move(P), std::move(L), std::move(U));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> linalg_lu_out_out(c10::DispatchKeySet ks, const at::Tensor & A, bool pivot, at::Tensor & P, at::Tensor & L, at::Tensor & U) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_lu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "pivot", pivot);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "P", P);
      jit::tracer::addInputs(node, "L", L);
      jit::tracer::addInputs(node, "U", U);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_lu_out", P);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_lu_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, pivot, P, L, U);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, P);
    jit::tracer::addOutput(node, L);
    jit::tracer::addOutput(node, U);
  }
  return std::forward_as_tuple(P, L, U);
}
at::Tensor linalg_vecdot(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & y, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_vecdot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "y", y);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_vecdot::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, y, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_vecdot_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & y, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_vecdot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "y", y);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_vecdot_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_vecdot_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, y, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linalg_svdvals(c10::DispatchKeySet ks, const at::Tensor & A, ::std::optional<c10::string_view> driver) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_svdvals");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "driver", driver);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_svdvals::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, driver);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_svdvals_out_out(c10::DispatchKeySet ks, const at::Tensor & A, ::std::optional<c10::string_view> driver, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_svdvals");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "driver", driver);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_svdvals_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_svdvals_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, driver, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _test_serialization_subcmul(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_serialization_subcmul");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_test_serialization_subcmul::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _test_optional_intlist(c10::DispatchKeySet ks, const at::Tensor & values, at::OptionalIntArrayRef addends) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_optional_intlist");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "addends", addends);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_test_optional_intlist::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), values, addends);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _segment_reduce_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & output, const at::Tensor & data, c10::string_view reduce, const ::std::optional<at::Tensor> & lengths, const ::std::optional<at::Tensor> & offsets, int64_t axis, const ::std::optional<at::Scalar> & initial) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_segment_reduce_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "reduce", reduce);
    jit::tracer::addInputs(node, "lengths", lengths);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "initial", initial);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_segment_reduce_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, output, data, reduce, lengths, offsets, axis, initial);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor expand_copy(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, bool implicit) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::expand_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "implicit", implicit);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::expand_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, implicit);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor unsqueeze_copy(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unsqueeze_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unsqueeze_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor crow_indices_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::crow_indices_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::crow_indices_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor to_padded_tensor(c10::DispatchKeySet ks, const at::Tensor & self, double padding, at::OptionalSymIntArrayRef output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::to_padded_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::to_padded_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _flash_attention_forward(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & cum_seq_q, const ::std::optional<at::Tensor> & cum_seq_k, c10::SymInt max_q, c10::SymInt max_k, double dropout_p, bool is_causal, bool return_debug_mask, ::std::optional<double> scale, ::std::optional<c10::SymInt> window_size_left, ::std::optional<c10::SymInt> window_size_right, const ::std::optional<at::Tensor> & seqused_k, const ::std::optional<at::Tensor> & alibi_slopes, const ::std::optional<at::Tensor> & block_table, ::std::optional<int64_t> num_splits) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_flash_attention_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "cum_seq_q", cum_seq_q);
    jit::tracer::addInputs(node, "cum_seq_k", cum_seq_k);
    jit::tracer::addInputs(node, "max_q", max_q);
    jit::tracer::addInputs(node, "max_k", max_k);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "return_debug_mask", return_debug_mask);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "window_size_left", window_size_left);
    jit::tracer::addInputs(node, "window_size_right", window_size_right);
    jit::tracer::addInputs(node, "seqused_k", seqused_k);
    jit::tracer::addInputs(node, "alibi_slopes", alibi_slopes);
    jit::tracer::addInputs(node, "block_table", block_table);
    jit::tracer::addInputs(node, "num_splits", num_splits);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [output, softmax_logsumexp, rng_state, unused, debug_attn_mask] =at::_ops::_flash_attention_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, return_debug_mask, scale, window_size_left, window_size_right, seqused_k, alibi_slopes, block_table, num_splits);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, softmax_logsumexp);
    jit::tracer::addOutput(node, rng_state);
    jit::tracer::addOutput(node, unused);
    jit::tracer::addOutput(node, debug_attn_mask);
  }
  return std::make_tuple(std::move(output), std::move(softmax_logsumexp), std::move(rng_state), std::move(unused), std::move(debug_attn_mask));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _flash_attention_forward_quantized(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & cum_seq_q, const ::std::optional<at::Tensor> & cum_seq_k, c10::SymInt max_q, c10::SymInt max_k, double dropout_p, bool is_causal, bool return_debug_mask, const ::std::optional<at::Tensor> & q_descale, const ::std::optional<at::Tensor> & k_descale, const ::std::optional<at::Tensor> & v_descale, ::std::optional<double> scale, ::std::optional<c10::SymInt> window_size_left, ::std::optional<c10::SymInt> window_size_right, const ::std::optional<at::Tensor> & seqused_k, const ::std::optional<at::Tensor> & alibi_slopes) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_flash_attention_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "cum_seq_q", cum_seq_q);
    jit::tracer::addInputs(node, "cum_seq_k", cum_seq_k);
    jit::tracer::addInputs(node, "max_q", max_q);
    jit::tracer::addInputs(node, "max_k", max_k);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "return_debug_mask", return_debug_mask);
    jit::tracer::addInputs(node, "q_descale", q_descale);
    jit::tracer::addInputs(node, "k_descale", k_descale);
    jit::tracer::addInputs(node, "v_descale", v_descale);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "window_size_left", window_size_left);
    jit::tracer::addInputs(node, "window_size_right", window_size_right);
    jit::tracer::addInputs(node, "seqused_k", seqused_k);
    jit::tracer::addInputs(node, "alibi_slopes", alibi_slopes);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [output, softmax_logsumexp, rng_state, unused, debug_attn_mask] =at::_ops::_flash_attention_forward_quantized::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, return_debug_mask, q_descale, k_descale, v_descale, scale, window_size_left, window_size_right, seqused_k, alibi_slopes);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, softmax_logsumexp);
    jit::tracer::addOutput(node, rng_state);
    jit::tracer::addOutput(node, unused);
    jit::tracer::addOutput(node, debug_attn_mask);
  }
  return std::make_tuple(std::move(output), std::move(softmax_logsumexp), std::move(rng_state), std::move(unused), std::move(debug_attn_mask));
}
at::Tensor special_bessel_j0(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_bessel_j0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_bessel_j0::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_bessel_j0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_bessel_j0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_bessel_j0_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_bessel_j0_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_bessel_y0(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_bessel_y0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_bessel_y0::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_bessel_y0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_bessel_y0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_bessel_y0_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_bessel_y0_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_chebyshev_polynomial_u(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_chebyshev_polynomial_u::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_chebyshev_polynomial_u_x_scalar(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_chebyshev_polynomial_u_x_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_chebyshev_polynomial_u_n_scalar(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_chebyshev_polynomial_u_n_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_chebyshev_polynomial_u_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_chebyshev_polynomial_u_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_chebyshev_polynomial_u_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_chebyshev_polynomial_u_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_chebyshev_polynomial_u_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_chebyshev_polynomial_u_x_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_chebyshev_polynomial_u_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_chebyshev_polynomial_u_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_chebyshev_polynomial_u_n_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_hermite_polynomial_he(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_he");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_hermite_polynomial_he::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_hermite_polynomial_he_x_scalar(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_he");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_hermite_polynomial_he_x_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_hermite_polynomial_he_n_scalar(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_he");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_hermite_polynomial_he_n_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_hermite_polynomial_he_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_he");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_hermite_polynomial_he_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_hermite_polynomial_he_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_hermite_polynomial_he_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_he");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_hermite_polynomial_he_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_hermite_polynomial_he_x_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_hermite_polynomial_he_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_hermite_polynomial_he");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_hermite_polynomial_he_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_hermite_polynomial_he_n_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_shifted_chebyshev_polynomial_v(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_v");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_v::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_shifted_chebyshev_polynomial_v_x_scalar(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_v");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_v_x_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_shifted_chebyshev_polynomial_v_n_scalar(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_v");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_v_n_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_shifted_chebyshev_polynomial_v_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_v");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_v_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_v_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_v_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_v");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_v_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_v_x_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_v_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_v");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_v_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_v_n_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_spherical_bessel_j0(c10::DispatchKeySet ks, const at::Tensor & x) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_spherical_bessel_j0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_spherical_bessel_j0::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_spherical_bessel_j0_out_out(c10::DispatchKeySet ks, const at::Tensor & x, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_spherical_bessel_j0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_spherical_bessel_j0_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_spherical_bessel_j0_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _fused_adagrad_(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList state_sums, at::TensorList state_steps, double lr, double lr_decay, double weight_decay, double eps, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  at::_ops::_fused_adagrad_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, state_sums, state_steps, lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf);
}
void _fused_adagrad__tensor_lr(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList state_sums, at::TensorList state_steps, const at::Tensor & lr, double lr_decay, double weight_decay, double eps, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  at::_ops::_fused_adagrad__tensor_lr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, state_sums, state_steps, lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf);
}
at::Tensor & avg_pool1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::avg_pool1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::avg_pool1d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, ceil_mode, count_include_pad, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & adaptive_avg_pool1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_avg_pool1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_avg_pool1d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::adaptive_avg_pool1d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, at::Tensor & out) {
  at::_ops::convolution_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> convolution_backward_overrideable_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::convolution_backward_overrideable");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "output_mask", output_mask);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("convolution_backward_overrideable_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::convolution_backward_overrideable_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & cudnn_affine_grid_generator_out_out(c10::DispatchKeySet ks, const at::Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_affine_grid_generator");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "theta", theta);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "C", C);
    jit::tracer::addInputs(node, "H", H);
    jit::tracer::addInputs(node, "W", W);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cudnn_affine_grid_generator_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cudnn_affine_grid_generator_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), theta, N, C, H, W, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> cudnn_batch_norm_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, const ::std::optional<at::Tensor> & save_mean, const ::std::optional<at::Tensor> & save_var, double epsilon, const at::Tensor & reserveSpace, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_batch_norm_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_var", save_var);
    jit::tracer::addInputs(node, "epsilon", epsilon);
    jit::tracer::addInputs(node, "reserveSpace", reserveSpace);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cudnn_batch_norm_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cudnn_batch_norm_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon, reserveSpace, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &> cudnn_grid_sampler_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grid, const at::Tensor & grad_output, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_grid_sampler_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "grad_output", grad_output);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cudnn_grid_sampler_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cudnn_grid_sampler_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grid, grad_output, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> _ctc_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank, bool zero_infinity, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_ctc_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_ctc_loss_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_ctc_loss_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), log_probs, targets, input_lengths, target_lengths, blank, zero_infinity, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> _ctc_loss_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, int64_t blank, bool zero_infinity, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_ctc_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_ctc_loss_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_ctc_loss_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), log_probs, targets, input_lengths, target_lengths, blank, zero_infinity, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & embedding_dense_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & indices, c10::SymInt num_weights, c10::SymInt padding_idx, bool scale_grad_by_freq, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::embedding_dense_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("embedding_dense_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::embedding_dense_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> grid_sampler_2d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::grid_sampler_2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "output_mask", output_mask);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("grid_sampler_2d_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::grid_sampler_2d_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & mkldnn_linear_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_linear");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mkldnn_linear_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mkldnn_linear_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> mkldnn_linear_backward_weights_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, bool bias_defined, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_linear_backward_weights");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias_defined", bias_defined);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mkldnn_linear_backward_weights_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mkldnn_linear_backward_weights_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input, weight, bias_defined, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & _pin_memory_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Device> device, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_pin_memory");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "device", device);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_pin_memory_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_pin_memory_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, device, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randn_out_names_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randn_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, names, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randn_out_generator_with_names_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "names", names);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randn_generator_with_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, names, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & diagonal_scatter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, int64_t offset, int64_t dim1, int64_t dim2, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::diagonal_scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("diagonal_scatter_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::diagonal_scatter_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, offset, dim1, dim2, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> _unique_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool sorted, bool return_inverse, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_unique");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sorted", sorted);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_unique_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_unique_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, sorted, return_inverse, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> _weight_norm_interface_out_out(c10::DispatchKeySet ks, const at::Tensor & v, const at::Tensor & g, int64_t dim, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_weight_norm_interface");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "g", g);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_weight_norm_interface_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_weight_norm_interface_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), v, g, dim, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & _standard_gamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_standard_gamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_standard_gamma_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_standard_gamma_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _philox_uniform_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & key, double low, double high, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_philox_uniform");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_philox_uniform_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_philox_uniform_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, key, low, high, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _philox_uniform(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & key, double low, double high) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_philox_uniform");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_philox_uniform::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, key, low, high);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & copy_sparse_to_sparse_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, bool non_blocking, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::copy_sparse_to_sparse");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("copy_sparse_to_sparse_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::copy_sparse_to_sparse_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, non_blocking, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor copy_sparse_to_sparse(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::copy_sparse_to_sparse");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::copy_sparse_to_sparse::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> fake_quantize_per_channel_affine_cachemask_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fake_quantize_per_channel_affine_cachemask");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fake_quantize_per_channel_affine_cachemask_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fake_quantize_per_channel_affine_cachemask_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, axis, quant_min, quant_max, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & lift_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lift");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lift_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::lift_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & bitwise_and_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_and");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_and_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_and_Scalar_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & __lshift___out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__lshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__lshift___Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & __lshift___out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__lshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__lshift___Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & bitwise_left_shift_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_left_shift");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_left_shift_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_left_shift_Scalar_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & __rshift___out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__rshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__rshift___Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & __rshift___out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__rshift__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__rshift___Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _amp_foreach_non_finite_check_and_unscale_out_out(c10::DispatchKeySet ks, at::TensorList self, at::Tensor & found_inf, const at::Tensor & inv_scale, at::TensorList out) {
  at::_ops::_amp_foreach_non_finite_check_and_unscale_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, found_inf, inv_scale, out);
}
::std::tuple<::std::vector<at::Tensor>,at::Tensor> _amp_foreach_non_finite_check_and_unscale(c10::DispatchKeySet ks, at::TensorList self, const at::Tensor & found_inf, const at::Tensor & inv_scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_amp_foreach_non_finite_check_and_unscale");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    jit::tracer::addInputs(node, "inv_scale", inv_scale);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [self_out, found_inf_out] =at::_ops::_amp_foreach_non_finite_check_and_unscale::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, found_inf, inv_scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self_out);
    jit::tracer::addOutput(node, found_inf_out);
  }
  return std::make_tuple(std::move(self_out), std::move(found_inf_out));
}
void _foreach_minimum_out_Scalar_out(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar, at::TensorList out) {
  at::_ops::_foreach_minimum_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar, out);
}
void _foreach_minimum_out_List_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other, at::TensorList out) {
  at::_ops::_foreach_minimum_List_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
}
void _foreach_minimum_out_ScalarList_out(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars, at::TensorList out) {
  at::_ops::_foreach_minimum_ScalarList_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars, out);
}
void _foreach_frac_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_frac_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_trunc_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_trunc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
at::Tensor & rrelu_with_noise_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, bool self_is_result, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rrelu_with_noise_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "self_is_result", self_is_result);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_with_noise_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rrelu_with_noise_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, noise, lower, upper, training, self_is_result, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _test_optional_intlist_out_out(c10::DispatchKeySet ks, const at::Tensor & values, at::OptionalIntArrayRef addends, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_optional_intlist");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "addends", addends);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_test_optional_intlist_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_test_optional_intlist_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), values, addends, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _segment_reduce_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & output, const at::Tensor & data, c10::string_view reduce, const ::std::optional<at::Tensor> & lengths, const ::std::optional<at::Tensor> & offsets, int64_t axis, const ::std::optional<at::Scalar> & initial, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_segment_reduce_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "reduce", reduce);
    jit::tracer::addInputs(node, "lengths", lengths);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "initial", initial);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_segment_reduce_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_segment_reduce_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, output, data, reduce, lengths, offsets, axis, initial, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & expand_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, bool implicit, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::expand_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "implicit", implicit);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("expand_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::expand_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, implicit, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & unsqueeze_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unsqueeze_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("unsqueeze_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::unsqueeze_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & crow_indices_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::crow_indices_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("crow_indices_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::crow_indices_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & to_padded_tensor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double padding, at::OptionalSymIntArrayRef output_size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::to_padded_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_size", output_size);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("to_padded_tensor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::to_padded_tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, output_size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _fused_adagrad_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList state_sums, at::TensorList state_steps, double lr, double lr_decay, double weight_decay, double eps, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf, at::TensorList out) {
  at::_ops::_fused_adagrad_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, state_sums, state_steps, lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf, out);
}
::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>> _fused_adagrad(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList state_sums, at::TensorList state_steps, double lr, double lr_decay, double weight_decay, double eps, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fused_adagrad");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grads", grads);
    jit::tracer::addInputs(node, "state_sums", state_sums);
    jit::tracer::addInputs(node, "state_steps", state_steps);
    jit::tracer::addInputs(node, "lr", lr);
    jit::tracer::addInputs(node, "lr_decay", lr_decay);
    jit::tracer::addInputs(node, "weight_decay", weight_decay);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "maximize", maximize);
    jit::tracer::addInputs(node, "grad_scale", grad_scale);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [self_out, grads_out, state_sums_out, state_steps_out] =at::_ops::_fused_adagrad::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, state_sums, state_steps, lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self_out);
    jit::tracer::addOutput(node, grads_out);
    jit::tracer::addOutput(node, state_sums_out);
    jit::tracer::addOutput(node, state_steps_out);
  }
  return std::make_tuple(std::move(self_out), std::move(grads_out), std::move(state_sums_out), std::move(state_steps_out));
}
void _fused_adagrad_out_tensor_lr_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList state_sums, at::TensorList state_steps, const at::Tensor & lr, double lr_decay, double weight_decay, double eps, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf, at::TensorList out) {
  at::_ops::_fused_adagrad_tensor_lr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, state_sums, state_steps, lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf, out);
}
::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>> _fused_adagrad_tensor_lr(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList state_sums, at::TensorList state_steps, const at::Tensor & lr, double lr_decay, double weight_decay, double eps, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fused_adagrad");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grads", grads);
    jit::tracer::addInputs(node, "state_sums", state_sums);
    jit::tracer::addInputs(node, "state_steps", state_steps);
    jit::tracer::addInputs(node, "lr", lr);
    jit::tracer::addInputs(node, "lr_decay", lr_decay);
    jit::tracer::addInputs(node, "weight_decay", weight_decay);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "maximize", maximize);
    jit::tracer::addInputs(node, "grad_scale", grad_scale);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [self_out, grads_out, state_sums_out] =at::_ops::_fused_adagrad_tensor_lr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, state_sums, state_steps, lr, lr_decay, weight_decay, eps, maximize, grad_scale, found_inf);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self_out);
    jit::tracer::addOutput(node, grads_out);
    jit::tracer::addOutput(node, state_sums_out);
  }
  return std::make_tuple(std::move(self_out), std::move(grads_out), std::move(state_sums_out));
}
}  // namespace
}  // namespace TraceType

namespace {

TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  m.impl("_cast_Char",
         TORCH_FN(TraceType::_cast_Char)
  );
  m.impl("retains_grad",
         TORCH_FN(TraceType::retains_grad)
  );
  m.impl("_unpack_dual",
         TORCH_FN(TraceType::_unpack_dual)
  );
  m.impl("_has_same_storage_numel",
         TORCH_FN(TraceType::_has_same_storage_numel)
  );
  m.impl("align_to",
         TORCH_FN(TraceType::align_to)
  );
  m.impl("align_to.ellipsis_idx",
         TORCH_FN(TraceType::align_to_ellipsis_idx)
  );
  m.impl("_debug_has_internal_overlap",
         TORCH_FN(TraceType::_debug_has_internal_overlap)
  );
  m.impl("sgn",
         TORCH_FN(TraceType::sgn)
  );
  m.impl("sgn_",
         TORCH_FN(TraceType::sgn_)
  );
  m.impl("sgn.out",
         TORCH_FN(TraceType::sgn_out_out)
  );
  m.impl("real",
         TORCH_FN(TraceType::real)
  );
  m.impl("_neg_view",
         TORCH_FN(TraceType::_neg_view)
  );
  m.impl("avg_pool1d",
         TORCH_FN(TraceType::avg_pool1d)
  );
  m.impl("adaptive_avg_pool1d",
         TORCH_FN(TraceType::adaptive_avg_pool1d)
  );
  m.impl("_is_all_true",
         TORCH_FN(TraceType::_is_all_true)
  );
  m.impl("_test_check_tensor",
         TORCH_FN(TraceType::_test_check_tensor)
  );
  m.impl("allclose",
         TORCH_FN(TraceType::allclose)
  );
  m.impl("acosh",
         TORCH_FN(TraceType::acosh)
  );
  m.impl("acosh_",
         TORCH_FN(TraceType::acosh_)
  );
  m.impl("acosh.out",
         TORCH_FN(TraceType::acosh_out_out)
  );
  m.impl("arctanh",
         TORCH_FN(TraceType::arctanh)
  );
  m.impl("arctanh_",
         TORCH_FN(TraceType::arctanh_)
  );
  m.impl("arctanh.out",
         TORCH_FN(TraceType::arctanh_out_out)
  );
  m.impl("atleast_3d",
         TORCH_FN(TraceType::atleast_3d)
  );
  m.impl("atleast_3d.Sequence",
         TORCH_FN(TraceType::atleast_3d_Sequence)
  );
  m.impl("_batch_norm_impl_index",
         TORCH_FN(TraceType::_batch_norm_impl_index)
  );
  m.impl("logical_or",
         TORCH_FN(TraceType::logical_or)
  );
  m.impl("logical_or_",
         TORCH_FN(TraceType::logical_or_)
  );
  m.impl("logical_or.out",
         TORCH_FN(TraceType::logical_or_out_out)
  );
  m.impl("broadcast_tensors",
         TORCH_FN(TraceType::broadcast_tensors)
  );
  m.impl("cat",
         TORCH_FN(TraceType::cat)
  );
  m.impl("cat.out",
         TORCH_FN(TraceType::cat_out_out)
  );
  m.impl("cat.names",
         TORCH_FN(TraceType::cat_names)
  );
  m.impl("cat.names_out",
         TORCH_FN(TraceType::cat_out_names_out)
  );
  m.impl("convolution",
         TORCH_FN(TraceType::convolution)
  );
  m.impl("convolution_backward_overrideable",
         TORCH_FN(TraceType::convolution_backward_overrideable)
  );
  m.impl("conv_transpose1d",
         TORCH_FN(TraceType::conv_transpose1d)
  );
  m.impl("cos",
         TORCH_FN(TraceType::cos)
  );
  m.impl("cos_",
         TORCH_FN(TraceType::cos_)
  );
  m.impl("cos.out",
         TORCH_FN(TraceType::cos_out_out)
  );
  m.impl("cudnn_affine_grid_generator",
         TORCH_FN(TraceType::cudnn_affine_grid_generator)
  );
  m.impl("cudnn_batch_norm_backward",
         TORCH_FN(TraceType::cudnn_batch_norm_backward)
  );
  m.impl("cudnn_grid_sampler_backward",
         TORCH_FN(TraceType::cudnn_grid_sampler_backward)
  );
  m.impl("cumsum",
         TORCH_FN(TraceType::cumsum)
  );
  m.impl("cumsum_",
         TORCH_FN(TraceType::cumsum_)
  );
  m.impl("cumsum.out",
         TORCH_FN(TraceType::cumsum_out_out)
  );
  m.impl("cumsum.dimname",
         TORCH_FN(TraceType::cumsum_dimname)
  );
  m.impl("cumsum_.dimname",
         TORCH_FN(TraceType::cumsum__dimname)
  );
  m.impl("cumsum.dimname_out",
         TORCH_FN(TraceType::cumsum_out_dimname_out)
  );
  m.impl("_ctc_loss",
         TORCH_FN(TraceType::_ctc_loss)
  );
  m.impl("_ctc_loss.Tensor",
         TORCH_FN(TraceType::_ctc_loss_Tensor)
  );
  m.impl("diagflat",
         TORCH_FN(TraceType::diagflat)
  );
  m.impl("linalg_diagonal",
         TORCH_FN(TraceType::linalg_diagonal)
  );
  m.impl("embedding_dense_backward",
         TORCH_FN(TraceType::embedding_dense_backward)
  );
  m.impl("grid_sampler_2d_backward",
         TORCH_FN(TraceType::grid_sampler_2d_backward)
  );
  m.impl("index_copy.out",
         TORCH_FN(TraceType::index_copy_out_out)
  );
  m.impl("index_copy_",
         TORCH_FN(TraceType::index_copy_)
  );
  m.impl("index_copy",
         TORCH_FN(TraceType::index_copy)
  );
  m.impl("index_copy_.dimname",
         TORCH_FN(TraceType::index_copy__dimname)
  );
  m.impl("index_copy.dimname",
         TORCH_FN(TraceType::index_copy_dimname)
  );
  m.impl("_unsafe_index_put",
         TORCH_FN(TraceType::_unsafe_index_put)
  );
  m.impl("kron",
         TORCH_FN(TraceType::kron)
  );
  m.impl("kron.out",
         TORCH_FN(TraceType::kron_out_out)
  );
  m.impl("linear",
         TORCH_FN(TraceType::linear)
  );
  m.impl("linear.out",
         TORCH_FN(TraceType::linear_out_out)
  );
  m.impl("mkldnn_linear",
         TORCH_FN(TraceType::mkldnn_linear)
  );
  m.impl("mkldnn_linear_backward_weights",
         TORCH_FN(TraceType::mkldnn_linear_backward_weights)
  );
  m.impl("_cslt_sparse_mm",
         TORCH_FN(TraceType::_cslt_sparse_mm)
  );
  m.impl("_mixed_dtypes_linear",
         TORCH_FN(TraceType::_mixed_dtypes_linear)
  );
  m.impl("fbgemm_linear_quantize_weight",
         TORCH_FN(TraceType::fbgemm_linear_quantize_weight)
  );
  m.impl("linspace",
         TORCH_FN(TraceType::linspace)
  );
  m.impl("linspace.Tensor_Tensor",
         TORCH_FN(TraceType::linspace_Tensor_Tensor)
  );
  m.impl("linspace.Tensor_Scalar",
         TORCH_FN(TraceType::linspace_Tensor_Scalar)
  );
  m.impl("linspace.Scalar_Tensor",
         TORCH_FN(TraceType::linspace_Scalar_Tensor)
  );
  m.impl("linspace.out",
         TORCH_FN(TraceType::linspace_out_out)
  );
  m.impl("linspace.Tensor_Tensor_out",
         TORCH_FN(TraceType::linspace_out_Tensor_Tensor_out)
  );
  m.impl("linspace.Tensor_Scalar_out",
         TORCH_FN(TraceType::linspace_out_Tensor_Scalar_out)
  );
  m.impl("linspace.Scalar_Tensor_out",
         TORCH_FN(TraceType::linspace_out_Scalar_Tensor_out)
  );
  m.impl("_log_softmax",
         TORCH_FN(TraceType::_log_softmax)
  );
  m.impl("_log_softmax.out",
         TORCH_FN(TraceType::_log_softmax_out_out)
  );
  m.impl("max.dim",
         TORCH_FN(TraceType::max_dim)
  );
  m.impl("max.dim_max",
         TORCH_FN(TraceType::max_out_dim_max)
  );
  m.impl("max.names_dim",
         TORCH_FN(TraceType::max_names_dim)
  );
  m.impl("max.names_dim_max",
         TORCH_FN(TraceType::max_out_names_dim_max)
  );
  m.impl("max_pool1d_with_indices",
         TORCH_FN(TraceType::max_pool1d_with_indices)
  );
  m.impl("mode",
         TORCH_FN(TraceType::mode)
  );
  m.impl("mode.values",
         TORCH_FN(TraceType::mode_out_values)
  );
  m.impl("mode.dimname",
         TORCH_FN(TraceType::mode_dimname)
  );
  m.impl("mode.dimname_out",
         TORCH_FN(TraceType::mode_out_dimname_out)
  );
  m.impl("narrow",
         TORCH_FN(TraceType::narrow)
  );
  m.impl("narrow.Tensor",
         TORCH_FN(TraceType::narrow_Tensor)
  );
  m.impl("pdist",
         TORCH_FN(TraceType::pdist)
  );
  m.impl("is_pinned",
         TORCH_FN(TraceType::is_pinned)
  );
  m.impl("_pin_memory",
         TORCH_FN(TraceType::_pin_memory)
  );
  m.impl("randn",
         TORCH_FN(TraceType::randn)
  );
  m.impl("randn.generator",
         TORCH_FN(TraceType::randn_generator)
  );
  m.impl("randn.names",
         TORCH_FN(TraceType::randn_names)
  );
  m.impl("randn.generator_with_names",
         TORCH_FN(TraceType::randn_generator_with_names)
  );
  m.impl("randn.out",
         TORCH_FN(TraceType::randn_out_out)
  );
  m.impl("randn.generator_out",
         TORCH_FN(TraceType::randn_out_generator_out)
  );
  m.impl("range.step",
         TORCH_FN(TraceType::range_step)
  );
  m.impl("range",
         TORCH_FN(TraceType::range)
  );
  m.impl("range.out_",
         TORCH_FN(TraceType::range_out_out_)
  );
  m.impl("range.out",
         TORCH_FN(TraceType::range_out_out)
  );
  m.impl("reciprocal",
         TORCH_FN(TraceType::reciprocal)
  );
  m.impl("reciprocal_",
         TORCH_FN(TraceType::reciprocal_)
  );
  m.impl("reciprocal.out",
         TORCH_FN(TraceType::reciprocal_out_out)
  );
  m.impl("neg",
         TORCH_FN(TraceType::neg)
  );
  m.impl("neg_",
         TORCH_FN(TraceType::neg_)
  );
  m.impl("neg.out",
         TORCH_FN(TraceType::neg_out_out)
  );
  m.impl("reshape_as",
         TORCH_FN(TraceType::reshape_as)
  );
  m.impl("gelu_backward.grad_input",
         TORCH_FN(TraceType::gelu_backward_out_grad_input)
  );
  m.impl("gelu_backward",
         TORCH_FN(TraceType::gelu_backward)
  );
  m.impl("silu_backward.grad_input",
         TORCH_FN(TraceType::silu_backward_out_grad_input)
  );
  m.impl("silu_backward",
         TORCH_FN(TraceType::silu_backward)
  );
  m.impl("sin",
         TORCH_FN(TraceType::sin)
  );
  m.impl("sin_",
         TORCH_FN(TraceType::sin_)
  );
  m.impl("sin.out",
         TORCH_FN(TraceType::sin_out_out)
  );
  m.impl("sym_is_contiguous",
         TORCH_FN(TraceType::sym_is_contiguous)
  );
  m.impl("sym_storage_offset",
         TORCH_FN(TraceType::sym_storage_offset)
  );
  m.impl("diagonal_scatter",
         TORCH_FN(TraceType::diagonal_scatter)
  );
  m.impl("squeeze",
         TORCH_FN(TraceType::squeeze)
  );
  m.impl("squeeze.dim",
         TORCH_FN(TraceType::squeeze_dim)
  );
  m.impl("squeeze.dimname",
         TORCH_FN(TraceType::squeeze_dimname)
  );
  m.impl("squeeze.dims",
         TORCH_FN(TraceType::squeeze_dims)
  );
  m.impl("squeeze_",
         TORCH_FN(TraceType::squeeze_)
  );
  m.impl("squeeze_.dim",
         TORCH_FN(TraceType::squeeze__dim)
  );
  m.impl("squeeze_.dims",
         TORCH_FN(TraceType::squeeze__dims)
  );
  m.impl("squeeze_.dimname",
         TORCH_FN(TraceType::squeeze__dimname)
  );
  m.impl("sspaddmm",
         TORCH_FN(TraceType::sspaddmm)
  );
  m.impl("sspaddmm.out",
         TORCH_FN(TraceType::sspaddmm_out_out)
  );
  m.impl("stride.int",
         TORCH_FN(TraceType::stride_int)
  );
  m.impl("stride.Dimname",
         TORCH_FN(TraceType::stride_Dimname)
  );
  m.impl("one_hot",
         TORCH_FN(TraceType::one_hot)
  );
  m.impl("_nested_get_jagged_dummy",
         TORCH_FN(TraceType::_nested_get_jagged_dummy)
  );
  m.impl("_unique",
         TORCH_FN(TraceType::_unique)
  );
  m.impl("_weight_norm_interface",
         TORCH_FN(TraceType::_weight_norm_interface)
  );
  m.impl("_standard_gamma",
         TORCH_FN(TraceType::_standard_gamma)
  );
  m.impl("_philox_uniform_",
         TORCH_FN(TraceType::_philox_uniform_)
  );
  m.impl("_sparse_mm_reduce_impl_backward",
         TORCH_FN(TraceType::_sparse_mm_reduce_impl_backward)
  );
  m.impl("_scaled_grouped_mm",
         TORCH_FN(TraceType::_scaled_grouped_mm)
  );
  m.impl("sparse_csc_tensor.ccol_row_value_size",
         TORCH_FN(TraceType::sparse_csc_tensor_ccol_row_value_size)
  );
  m.impl("sparse_csc_tensor.ccol_row_value",
         TORCH_FN(TraceType::sparse_csc_tensor_ccol_row_value)
  );
  m.impl("_sparse_coo_tensor_unsafe",
         TORCH_FN(TraceType::_sparse_coo_tensor_unsafe)
  );
  m.impl("_validate_sparse_csr_tensor_args",
         TORCH_FN(TraceType::_validate_sparse_csr_tensor_args)
  );
  m.impl("_to_cpu",
         TORCH_FN(TraceType::_to_cpu)
  );
  m.impl("copy_sparse_to_sparse_",
         TORCH_FN(TraceType::copy_sparse_to_sparse_)
  );
  m.impl("unbind.int",
         TORCH_FN(TraceType::unbind_int)
  );
  m.impl("unbind.Dimname",
         TORCH_FN(TraceType::unbind_Dimname)
  );
  m.impl("fake_quantize_per_channel_affine",
         TORCH_FN(TraceType::fake_quantize_per_channel_affine)
  );
  m.impl("fake_quantize_per_channel_affine_cachemask",
         TORCH_FN(TraceType::fake_quantize_per_channel_affine_cachemask)
  );
  m.impl("_thnn_differentiable_gru_cell_backward",
         TORCH_FN(TraceType::_thnn_differentiable_gru_cell_backward)
  );
  m.impl("rnn_tanh_cell",
         TORCH_FN(TraceType::rnn_tanh_cell)
  );
  m.impl("quantized_gru_cell",
         TORCH_FN(TraceType::quantized_gru_cell)
  );
  m.impl("lift",
         TORCH_FN(TraceType::lift)
  );
  m.impl("eq_.Scalar",
         TORCH_FN(TraceType::eq__Scalar)
  );
  m.impl("eq_.Tensor",
         TORCH_FN(TraceType::eq__Tensor)
  );
  m.impl("bitwise_and.Tensor_out",
         TORCH_FN(TraceType::bitwise_and_out_Tensor_out)
  );
  m.impl("bitwise_and.Scalar_out",
         TORCH_FN(TraceType::bitwise_and_out_Scalar_out)
  );
  m.impl("bitwise_and.Scalar",
         TORCH_FN(TraceType::bitwise_and_Scalar)
  );
  m.impl("bitwise_and.Scalar_Tensor",
         TORCH_FN(TraceType::bitwise_and_Scalar_Tensor)
  );
  m.impl("bitwise_and.Tensor",
         TORCH_FN(TraceType::bitwise_and_Tensor)
  );
  m.impl("bitwise_and_.Scalar",
         TORCH_FN(TraceType::bitwise_and__Scalar)
  );
  m.impl("bitwise_and_.Tensor",
         TORCH_FN(TraceType::bitwise_and__Tensor)
  );
  m.impl("__or__.Scalar",
         TORCH_FN(TraceType::__or___Scalar)
  );
  m.impl("__or__.Tensor",
         TORCH_FN(TraceType::__or___Tensor)
  );
  m.impl("__ior__.Scalar",
         TORCH_FN(TraceType::__ior___Scalar)
  );
  m.impl("__ior__.Tensor",
         TORCH_FN(TraceType::__ior___Tensor)
  );
  m.impl("__lshift__.Scalar",
         TORCH_FN(TraceType::__lshift___Scalar)
  );
  m.impl("__lshift__.Tensor",
         TORCH_FN(TraceType::__lshift___Tensor)
  );
  m.impl("__ilshift__.Scalar",
         TORCH_FN(TraceType::__ilshift___Scalar)
  );
  m.impl("__ilshift__.Tensor",
         TORCH_FN(TraceType::__ilshift___Tensor)
  );
  m.impl("bitwise_left_shift.Tensor",
         TORCH_FN(TraceType::bitwise_left_shift_Tensor)
  );
  m.impl("bitwise_left_shift_.Tensor",
         TORCH_FN(TraceType::bitwise_left_shift__Tensor)
  );
  m.impl("bitwise_left_shift.Tensor_out",
         TORCH_FN(TraceType::bitwise_left_shift_out_Tensor_out)
  );
  m.impl("bitwise_left_shift.Tensor_Scalar",
         TORCH_FN(TraceType::bitwise_left_shift_Tensor_Scalar)
  );
  m.impl("bitwise_left_shift_.Tensor_Scalar",
         TORCH_FN(TraceType::bitwise_left_shift__Tensor_Scalar)
  );
  m.impl("bitwise_left_shift.Tensor_Scalar_out",
         TORCH_FN(TraceType::bitwise_left_shift_out_Tensor_Scalar_out)
  );
  m.impl("bitwise_left_shift.Scalar_Tensor",
         TORCH_FN(TraceType::bitwise_left_shift_Scalar_Tensor)
  );
  m.impl("__rshift__.Scalar",
         TORCH_FN(TraceType::__rshift___Scalar)
  );
  m.impl("__rshift__.Tensor",
         TORCH_FN(TraceType::__rshift___Tensor)
  );
  m.impl("__irshift__.Scalar",
         TORCH_FN(TraceType::__irshift___Scalar)
  );
  m.impl("__irshift__.Tensor",
         TORCH_FN(TraceType::__irshift___Tensor)
  );
  m.impl("trace_backward",
         TORCH_FN(TraceType::trace_backward)
  );
  m.impl("eq.Scalar_out",
         TORCH_FN(TraceType::eq_out_Scalar_out)
  );
  m.impl("eq.Scalar",
         TORCH_FN(TraceType::eq_Scalar)
  );
  m.impl("eq.Tensor_out",
         TORCH_FN(TraceType::eq_out_Tensor_out)
  );
  m.impl("eq.Tensor",
         TORCH_FN(TraceType::eq_Tensor)
  );
  m.impl("le.Scalar_out",
         TORCH_FN(TraceType::le_out_Scalar_out)
  );
  m.impl("le.Scalar",
         TORCH_FN(TraceType::le_Scalar)
  );
  m.impl("le.Tensor_out",
         TORCH_FN(TraceType::le_out_Tensor_out)
  );
  m.impl("le.Tensor",
         TORCH_FN(TraceType::le_Tensor)
  );
  m.impl("le_.Scalar",
         TORCH_FN(TraceType::le__Scalar)
  );
  m.impl("le_.Tensor",
         TORCH_FN(TraceType::le__Tensor)
  );
  m.impl("masked_select_backward",
         TORCH_FN(TraceType::masked_select_backward)
  );
  m.impl("nonzero.out",
         TORCH_FN(TraceType::nonzero_out_out)
  );
  m.impl("nonzero",
         TORCH_FN(TraceType::nonzero)
  );
  m.impl("nonzero_numpy",
         TORCH_FN(TraceType::nonzero_numpy)
  );
  m.impl("cholesky.out",
         TORCH_FN(TraceType::cholesky_out_out)
  );
  m.impl("cholesky",
         TORCH_FN(TraceType::cholesky)
  );
  m.impl("lu_solve.out",
         TORCH_FN(TraceType::lu_solve_out_out)
  );
  m.impl("lu_solve",
         TORCH_FN(TraceType::lu_solve)
  );
  m.impl("lgamma.out",
         TORCH_FN(TraceType::lgamma_out_out)
  );
  m.impl("lgamma_",
         TORCH_FN(TraceType::lgamma_)
  );
  m.impl("lgamma",
         TORCH_FN(TraceType::lgamma)
  );
  m.impl("igamma.out",
         TORCH_FN(TraceType::igamma_out_out)
  );
  m.impl("igamma",
         TORCH_FN(TraceType::igamma)
  );
  m.impl("igamma_",
         TORCH_FN(TraceType::igamma_)
  );
  m.impl("max",
         TORCH_FN(TraceType::max)
  );
  m.impl("max.other",
         TORCH_FN(TraceType::max_other)
  );
  m.impl("max.out",
         TORCH_FN(TraceType::max_out_out)
  );
  m.impl("max.unary_out",
         TORCH_FN(TraceType::max_out_unary_out)
  );
  m.impl("_amp_foreach_non_finite_check_and_unscale_",
         TORCH_FN(TraceType::_amp_foreach_non_finite_check_and_unscale_)
  );
  m.impl("_foreach_minimum.Scalar",
         TORCH_FN(TraceType::_foreach_minimum_Scalar)
  );
  m.impl("_foreach_minimum_.Scalar",
         TORCH_FN(TraceType::_foreach_minimum__Scalar)
  );
  m.impl("_foreach_minimum.List",
         TORCH_FN(TraceType::_foreach_minimum_List)
  );
  m.impl("_foreach_minimum_.List",
         TORCH_FN(TraceType::_foreach_minimum__List)
  );
  m.impl("_foreach_minimum.ScalarList",
         TORCH_FN(TraceType::_foreach_minimum_ScalarList)
  );
  m.impl("_foreach_minimum_.ScalarList",
         TORCH_FN(TraceType::_foreach_minimum__ScalarList)
  );
  m.impl("_foreach_frac",
         TORCH_FN(TraceType::_foreach_frac)
  );
  m.impl("_foreach_frac_",
         TORCH_FN(TraceType::_foreach_frac_)
  );
  m.impl("_foreach_trunc",
         TORCH_FN(TraceType::_foreach_trunc)
  );
  m.impl("_foreach_trunc_",
         TORCH_FN(TraceType::_foreach_trunc_)
  );
  m.impl("mse_loss_backward.grad_input",
         TORCH_FN(TraceType::mse_loss_backward_out_grad_input)
  );
  m.impl("mse_loss_backward",
         TORCH_FN(TraceType::mse_loss_backward)
  );
  m.impl("multilabel_margin_loss_backward.grad_input",
         TORCH_FN(TraceType::multilabel_margin_loss_backward_out_grad_input)
  );
  m.impl("multilabel_margin_loss_backward",
         TORCH_FN(TraceType::multilabel_margin_loss_backward)
  );
  m.impl("elu_backward.grad_input",
         TORCH_FN(TraceType::elu_backward_out_grad_input)
  );
  m.impl("elu_backward",
         TORCH_FN(TraceType::elu_backward)
  );
  m.impl("hardsigmoid_backward.grad_input",
         TORCH_FN(TraceType::hardsigmoid_backward_out_grad_input)
  );
  m.impl("hardsigmoid_backward",
         TORCH_FN(TraceType::hardsigmoid_backward)
  );
  m.impl("rrelu_with_noise_backward",
         TORCH_FN(TraceType::rrelu_with_noise_backward)
  );
  m.impl("softplus_backward.grad_input",
         TORCH_FN(TraceType::softplus_backward_out_grad_input)
  );
  m.impl("softplus_backward",
         TORCH_FN(TraceType::softplus_backward)
  );
  m.impl("max_pool2d_with_indices.out",
         TORCH_FN(TraceType::max_pool2d_with_indices_out_out)
  );
  m.impl("max_pool2d_with_indices",
         TORCH_FN(TraceType::max_pool2d_with_indices)
  );
  m.impl("reflection_pad1d.out",
         TORCH_FN(TraceType::reflection_pad1d_out_out)
  );
  m.impl("reflection_pad1d",
         TORCH_FN(TraceType::reflection_pad1d)
  );
  m.impl("_pad_enum",
         TORCH_FN(TraceType::_pad_enum)
  );
  m.impl("_upsample_lanczos2d_aa.vec",
         TORCH_FN(TraceType::_upsample_lanczos2d_aa_vec)
  );
  m.impl("upsample_nearest3d.vec",
         TORCH_FN(TraceType::upsample_nearest3d_vec)
  );
  m.impl("_upsample_lanczos2d_aa.out",
         TORCH_FN(TraceType::_upsample_lanczos2d_aa_out_out)
  );
  m.impl("_upsample_lanczos2d_aa",
         TORCH_FN(TraceType::_upsample_lanczos2d_aa)
  );
  m.impl("upsample_nearest3d.out",
         TORCH_FN(TraceType::upsample_nearest3d_out_out)
  );
  m.impl("upsample_nearest3d",
         TORCH_FN(TraceType::upsample_nearest3d)
  );
  m.impl("sigmoid_backward.grad_input",
         TORCH_FN(TraceType::sigmoid_backward_out_grad_input)
  );
  m.impl("sigmoid_backward",
         TORCH_FN(TraceType::sigmoid_backward)
  );
  m.impl("_slow_conv2d_forward.output",
         TORCH_FN(TraceType::_slow_conv2d_forward_out_output)
  );
  m.impl("_slow_conv2d_forward",
         TORCH_FN(TraceType::_slow_conv2d_forward)
  );
  m.impl("special_entr",
         TORCH_FN(TraceType::special_entr)
  );
  m.impl("special_entr.out",
         TORCH_FN(TraceType::special_entr_out_out)
  );
  m.impl("special_ndtri",
         TORCH_FN(TraceType::special_ndtri)
  );
  m.impl("special_ndtri.out",
         TORCH_FN(TraceType::special_ndtri_out_out)
  );
  m.impl("special_erfc",
         TORCH_FN(TraceType::special_erfc)
  );
  m.impl("special_erfc.out",
         TORCH_FN(TraceType::special_erfc_out_out)
  );
  m.impl("special_gammainc.out",
         TORCH_FN(TraceType::special_gammainc_out_out)
  );
  m.impl("special_gammainc",
         TORCH_FN(TraceType::special_gammainc)
  );
  m.impl("linalg_lu",
         TORCH_FN(TraceType::linalg_lu)
  );
  m.impl("linalg_lu.out",
         TORCH_FN(TraceType::linalg_lu_out_out)
  );
  m.impl("linalg_vecdot",
         TORCH_FN(TraceType::linalg_vecdot)
  );
  m.impl("linalg_vecdot.out",
         TORCH_FN(TraceType::linalg_vecdot_out_out)
  );
  m.impl("linalg_svdvals",
         TORCH_FN(TraceType::linalg_svdvals)
  );
  m.impl("linalg_svdvals.out",
         TORCH_FN(TraceType::linalg_svdvals_out_out)
  );
  m.impl("_test_serialization_subcmul",
         TORCH_FN(TraceType::_test_serialization_subcmul)
  );
  m.impl("_test_optional_intlist",
         TORCH_FN(TraceType::_test_optional_intlist)
  );
  m.impl("_segment_reduce_backward",
         TORCH_FN(TraceType::_segment_reduce_backward)
  );
  m.impl("expand_copy",
         TORCH_FN(TraceType::expand_copy)
  );
  m.impl("unsqueeze_copy",
         TORCH_FN(TraceType::unsqueeze_copy)
  );
  m.impl("crow_indices_copy",
         TORCH_FN(TraceType::crow_indices_copy)
  );
  m.impl("to_padded_tensor",
         TORCH_FN(TraceType::to_padded_tensor)
  );
  m.impl("_flash_attention_forward",
         TORCH_FN(TraceType::_flash_attention_forward)
  );
  m.impl("_flash_attention_forward.quantized",
         TORCH_FN(TraceType::_flash_attention_forward_quantized)
  );
  m.impl("special_bessel_j0",
         TORCH_FN(TraceType::special_bessel_j0)
  );
  m.impl("special_bessel_j0.out",
         TORCH_FN(TraceType::special_bessel_j0_out_out)
  );
  m.impl("special_bessel_y0",
         TORCH_FN(TraceType::special_bessel_y0)
  );
  m.impl("special_bessel_y0.out",
         TORCH_FN(TraceType::special_bessel_y0_out_out)
  );
  m.impl("special_chebyshev_polynomial_u",
         TORCH_FN(TraceType::special_chebyshev_polynomial_u)
  );
  m.impl("special_chebyshev_polynomial_u.x_scalar",
         TORCH_FN(TraceType::special_chebyshev_polynomial_u_x_scalar)
  );
  m.impl("special_chebyshev_polynomial_u.n_scalar",
         TORCH_FN(TraceType::special_chebyshev_polynomial_u_n_scalar)
  );
  m.impl("special_chebyshev_polynomial_u.out",
         TORCH_FN(TraceType::special_chebyshev_polynomial_u_out_out)
  );
  m.impl("special_chebyshev_polynomial_u.x_scalar_out",
         TORCH_FN(TraceType::special_chebyshev_polynomial_u_out_x_scalar_out)
  );
  m.impl("special_chebyshev_polynomial_u.n_scalar_out",
         TORCH_FN(TraceType::special_chebyshev_polynomial_u_out_n_scalar_out)
  );
  m.impl("special_hermite_polynomial_he",
         TORCH_FN(TraceType::special_hermite_polynomial_he)
  );
  m.impl("special_hermite_polynomial_he.x_scalar",
         TORCH_FN(TraceType::special_hermite_polynomial_he_x_scalar)
  );
  m.impl("special_hermite_polynomial_he.n_scalar",
         TORCH_FN(TraceType::special_hermite_polynomial_he_n_scalar)
  );
  m.impl("special_hermite_polynomial_he.out",
         TORCH_FN(TraceType::special_hermite_polynomial_he_out_out)
  );
  m.impl("special_hermite_polynomial_he.x_scalar_out",
         TORCH_FN(TraceType::special_hermite_polynomial_he_out_x_scalar_out)
  );
  m.impl("special_hermite_polynomial_he.n_scalar_out",
         TORCH_FN(TraceType::special_hermite_polynomial_he_out_n_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_v",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_v)
  );
  m.impl("special_shifted_chebyshev_polynomial_v.x_scalar",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_v_x_scalar)
  );
  m.impl("special_shifted_chebyshev_polynomial_v.n_scalar",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_v_n_scalar)
  );
  m.impl("special_shifted_chebyshev_polynomial_v.out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_v_out_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_v.x_scalar_out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_v_out_x_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_v.n_scalar_out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_v_out_n_scalar_out)
  );
  m.impl("special_spherical_bessel_j0",
         TORCH_FN(TraceType::special_spherical_bessel_j0)
  );
  m.impl("special_spherical_bessel_j0.out",
         TORCH_FN(TraceType::special_spherical_bessel_j0_out_out)
  );
  m.impl("_fused_adagrad_",
         TORCH_FN(TraceType::_fused_adagrad_)
  );
  m.impl("_fused_adagrad_.tensor_lr",
         TORCH_FN(TraceType::_fused_adagrad__tensor_lr)
  );
  m.impl("avg_pool1d.out",
         TORCH_FN(TraceType::avg_pool1d_out_out)
  );
  m.impl("adaptive_avg_pool1d.out",
         TORCH_FN(TraceType::adaptive_avg_pool1d_out_out)
  );
  m.impl("convolution.out",
         TORCH_FN(TraceType::convolution_out_out)
  );
  m.impl("convolution_backward_overrideable.out",
         TORCH_FN(TraceType::convolution_backward_overrideable_out_out)
  );
  m.impl("cudnn_affine_grid_generator.out",
         TORCH_FN(TraceType::cudnn_affine_grid_generator_out_out)
  );
  m.impl("cudnn_batch_norm_backward.out",
         TORCH_FN(TraceType::cudnn_batch_norm_backward_out_out)
  );
  m.impl("cudnn_grid_sampler_backward.out",
         TORCH_FN(TraceType::cudnn_grid_sampler_backward_out_out)
  );
  m.impl("_ctc_loss.out",
         TORCH_FN(TraceType::_ctc_loss_out_out)
  );
  m.impl("_ctc_loss.Tensor_out",
         TORCH_FN(TraceType::_ctc_loss_out_Tensor_out)
  );
  m.impl("embedding_dense_backward.out",
         TORCH_FN(TraceType::embedding_dense_backward_out_out)
  );
  m.impl("grid_sampler_2d_backward.out",
         TORCH_FN(TraceType::grid_sampler_2d_backward_out_out)
  );
  m.impl("mkldnn_linear.out",
         TORCH_FN(TraceType::mkldnn_linear_out_out)
  );
  m.impl("mkldnn_linear_backward_weights.out",
         TORCH_FN(TraceType::mkldnn_linear_backward_weights_out_out)
  );
  m.impl("_pin_memory.out",
         TORCH_FN(TraceType::_pin_memory_out_out)
  );
  m.impl("randn.names_out",
         TORCH_FN(TraceType::randn_out_names_out)
  );
  m.impl("randn.generator_with_names_out",
         TORCH_FN(TraceType::randn_out_generator_with_names_out)
  );
  m.impl("diagonal_scatter.out",
         TORCH_FN(TraceType::diagonal_scatter_out_out)
  );
  m.impl("_unique.out",
         TORCH_FN(TraceType::_unique_out_out)
  );
  m.impl("_weight_norm_interface.out",
         TORCH_FN(TraceType::_weight_norm_interface_out_out)
  );
  m.impl("_standard_gamma.out",
         TORCH_FN(TraceType::_standard_gamma_out_out)
  );
  m.impl("_philox_uniform.out",
         TORCH_FN(TraceType::_philox_uniform_out_out)
  );
  m.impl("_philox_uniform",
         TORCH_FN(TraceType::_philox_uniform)
  );
  m.impl("copy_sparse_to_sparse.out",
         TORCH_FN(TraceType::copy_sparse_to_sparse_out_out)
  );
  m.impl("copy_sparse_to_sparse",
         TORCH_FN(TraceType::copy_sparse_to_sparse)
  );
  m.impl("fake_quantize_per_channel_affine_cachemask.out",
         TORCH_FN(TraceType::fake_quantize_per_channel_affine_cachemask_out_out)
  );
  m.impl("lift.out",
         TORCH_FN(TraceType::lift_out_out)
  );
  m.impl("bitwise_and.Scalar_Tensor_out",
         TORCH_FN(TraceType::bitwise_and_out_Scalar_Tensor_out)
  );
  m.impl("__lshift__.Scalar_out",
         TORCH_FN(TraceType::__lshift___out_Scalar_out)
  );
  m.impl("__lshift__.Tensor_out",
         TORCH_FN(TraceType::__lshift___out_Tensor_out)
  );
  m.impl("bitwise_left_shift.Scalar_Tensor_out",
         TORCH_FN(TraceType::bitwise_left_shift_out_Scalar_Tensor_out)
  );
  m.impl("__rshift__.Scalar_out",
         TORCH_FN(TraceType::__rshift___out_Scalar_out)
  );
  m.impl("__rshift__.Tensor_out",
         TORCH_FN(TraceType::__rshift___out_Tensor_out)
  );
  m.impl("_amp_foreach_non_finite_check_and_unscale.out",
         TORCH_FN(TraceType::_amp_foreach_non_finite_check_and_unscale_out_out)
  );
  m.impl("_amp_foreach_non_finite_check_and_unscale",
         TORCH_FN(TraceType::_amp_foreach_non_finite_check_and_unscale)
  );
  m.impl("_foreach_minimum.Scalar_out",
         TORCH_FN(TraceType::_foreach_minimum_out_Scalar_out)
  );
  m.impl("_foreach_minimum.List_out",
         TORCH_FN(TraceType::_foreach_minimum_out_List_out)
  );
  m.impl("_foreach_minimum.ScalarList_out",
         TORCH_FN(TraceType::_foreach_minimum_out_ScalarList_out)
  );
  m.impl("_foreach_frac.out",
         TORCH_FN(TraceType::_foreach_frac_out_out)
  );
  m.impl("_foreach_trunc.out",
         TORCH_FN(TraceType::_foreach_trunc_out_out)
  );
  m.impl("rrelu_with_noise_backward.out",
         TORCH_FN(TraceType::rrelu_with_noise_backward_out_out)
  );
  m.impl("_test_optional_intlist.out",
         TORCH_FN(TraceType::_test_optional_intlist_out_out)
  );
  m.impl("_segment_reduce_backward.out",
         TORCH_FN(TraceType::_segment_reduce_backward_out_out)
  );
  m.impl("expand_copy.out",
         TORCH_FN(TraceType::expand_copy_out_out)
  );
  m.impl("unsqueeze_copy.out",
         TORCH_FN(TraceType::unsqueeze_copy_out_out)
  );
  m.impl("crow_indices_copy.out",
         TORCH_FN(TraceType::crow_indices_copy_out_out)
  );
  m.impl("to_padded_tensor.out",
         TORCH_FN(TraceType::to_padded_tensor_out_out)
  );
  m.impl("_fused_adagrad.out",
         TORCH_FN(TraceType::_fused_adagrad_out_out)
  );
  m.impl("_fused_adagrad",
         TORCH_FN(TraceType::_fused_adagrad)
  );
  m.impl("_fused_adagrad.tensor_lr_out",
         TORCH_FN(TraceType::_fused_adagrad_out_tensor_lr_out)
  );
  m.impl("_fused_adagrad.tensor_lr",
         TORCH_FN(TraceType::_fused_adagrad_tensor_lr)
  );;
}

}  // namespace

} // namespace torch
