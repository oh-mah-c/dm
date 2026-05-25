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
#include <ATen/ops/_cast_Short_ops.h>
#include <ATen/ops/absolute_ops.h>
#include <ATen/ops/absolute_ops.h>
#include <ATen/ops/absolute_ops.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/all_ops.h>
#include <ATen/ops/all_ops.h>
#include <ATen/ops/all_ops.h>
#include <ATen/ops/all_ops.h>
#include <ATen/ops/all_ops.h>
#include <ATen/ops/all_ops.h>
#include <ATen/ops/atleast_2d_ops.h>
#include <ATen/ops/atleast_2d_ops.h>
#include <ATen/ops/binary_cross_entropy_with_logits_ops.h>
#include <ATen/ops/_lazy_clone_ops.h>
#include <ATen/ops/logical_and_ops.h>
#include <ATen/ops/logical_and_ops.h>
#include <ATen/ops/logical_and_ops.h>
#include <ATen/ops/block_diag_ops.h>
#include <ATen/ops/chunk_ops.h>
#include <ATen/ops/tensor_split_ops.h>
#include <ATen/ops/tensor_split_ops.h>
#include <ATen/ops/tensor_split_ops.h>
#include <ATen/ops/clamp_ops.h>
#include <ATen/ops/clamp_ops.h>
#include <ATen/ops/clamp_ops.h>
#include <ATen/ops/clamp_ops.h>
#include <ATen/ops/clamp_ops.h>
#include <ATen/ops/clamp_ops.h>
#include <ATen/ops/cudnn_is_acceptable_ops.h>
#include <ATen/ops/cov_ops.h>
#include <ATen/ops/diagonal_backward_ops.h>
#include <ATen/ops/dot_ops.h>
#include <ATen/ops/dot_ops.h>
#include <ATen/ops/embedding_sparse_backward_ops.h>
#include <ATen/ops/_embedding_bag_per_sample_weights_backward_ops.h>
#include <ATen/ops/new_empty_strided_ops.h>
#include <ATen/ops/new_full_ops.h>
#include <ATen/ops/empty_quantized_ops.h>
#include <ATen/ops/empty_strided_ops.h>
#include <ATen/ops/exp_ops.h>
#include <ATen/ops/exp_ops.h>
#include <ATen/ops/exp_ops.h>
#include <ATen/ops/frac_ops.h>
#include <ATen/ops/frac_ops.h>
#include <ATen/ops/frac_ops.h>
#include <ATen/ops/from_file_ops.h>
#include <ATen/ops/_unsafe_masked_index_ops.h>
#include <ATen/ops/_is_zerotensor_ops.h>
#include <ATen/ops/is_signed_ops.h>
#include <ATen/ops/native_layer_norm_backward_ops.h>
#include <ATen/ops/ldexp_ops.h>
#include <ATen/ops/ldexp_ops.h>
#include <ATen/ops/ldexp_ops.h>
#include <ATen/ops/log2_ops.h>
#include <ATen/ops/log2_ops.h>
#include <ATen/ops/log2_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/logspace_ops.h>
#include <ATen/ops/matrix_power_ops.h>
#include <ATen/ops/matrix_power_ops.h>
#include <ATen/ops/mps_convolution_backward_ops.h>
#include <ATen/ops/mkldnn_rnn_layer_ops.h>
#include <ATen/ops/miopen_rnn_ops.h>
#include <ATen/ops/_weight_int4pack_mm_with_scales_and_zeros_ops.h>
#include <ATen/ops/_convert_weight_to_int4pack_for_cpu_ops.h>
#include <ATen/ops/_sparse_mm_ops.h>
#include <ATen/ops/_sparse_mm_ops.h>
#include <ATen/ops/_sparse_sparse_matmul_ops.h>
#include <ATen/ops/_native_batch_norm_legit_ops.h>
#include <ATen/ops/_native_batch_norm_legit_ops.h>
#include <ATen/ops/_native_batch_norm_legit_ops.h>
#include <ATen/ops/_native_batch_norm_legit_ops.h>
#include <ATen/ops/_nnpack_available_ops.h>
#include <ATen/ops/_euclidean_dist_ops.h>
#include <ATen/ops/_cdist_backward_ops.h>
#include <ATen/ops/native_channel_shuffle_ops.h>
#include <ATen/ops/rad2deg_ops.h>
#include <ATen/ops/rad2deg_ops.h>
#include <ATen/ops/rad2deg_ops.h>
#include <ATen/ops/scalar_tensor_ops.h>
#include <ATen/ops/rand_ops.h>
#include <ATen/ops/rand_ops.h>
#include <ATen/ops/rand_ops.h>
#include <ATen/ops/rand_ops.h>
#include <ATen/ops/rand_ops.h>
#include <ATen/ops/rand_ops.h>
#include <ATen/ops/repeat_ops.h>
#include <ATen/ops/_mkldnn_reshape_ops.h>
#include <ATen/ops/rsqrt_ops.h>
#include <ATen/ops/rsqrt_ops.h>
#include <ATen/ops/rsqrt_ops.h>
#include <ATen/ops/sym_size_ops.h>
#include <ATen/ops/istft_ops.h>
#include <ATen/ops/nansum_ops.h>
#include <ATen/ops/nansum_ops.h>
#include <ATen/ops/flipud_ops.h>
#include <ATen/ops/trapz_ops.h>
#include <ATen/ops/trapz_ops.h>
#include <ATen/ops/_nested_get_ragged_idx_ops.h>
#include <ATen/ops/_nested_get_min_seqlen_ops.h>
#include <ATen/ops/trunc_ops.h>
#include <ATen/ops/trunc_ops.h>
#include <ATen/ops/trunc_ops.h>
#include <ATen/ops/var_mean_ops.h>
#include <ATen/ops/var_mean_ops.h>
#include <ATen/ops/var_mean_ops.h>
#include <ATen/ops/var_mean_ops.h>
#include <ATen/ops/var_mean_ops.h>
#include <ATen/ops/norm_except_dim_ops.h>
#include <ATen/ops/native_norm_ops.h>
#include <ATen/ops/native_norm_ops.h>
#include <ATen/ops/_sparse_sum_backward_ops.h>
#include <ATen/ops/_sparse_softmax_ops.h>
#include <ATen/ops/_sparse_softmax_ops.h>
#include <ATen/ops/_sparse_softmax_ops.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_ops.h>
#include <ATen/ops/is_coalesced_ops.h>
#include <ATen/ops/indices_ops.h>
#include <ATen/ops/col_indices_ops.h>
#include <ATen/ops/hspmm_ops.h>
#include <ATen/ops/hspmm_ops.h>
#include <ATen/ops/_to_sparse_semi_structured_ops.h>
#include <ATen/ops/cartesian_prod_ops.h>
#include <ATen/ops/promote_types_ops.h>
#include <ATen/ops/_local_scalar_dense_ops.h>
#include <ATen/ops/rnn_relu_ops.h>
#include <ATen/ops/rnn_relu_ops.h>
#include <ATen/ops/gru_cell_ops.h>
#include <ATen/ops/quantized_lstm_cell_ops.h>
#include <ATen/ops/scatter_reduce_ops.h>
#include <ATen/ops/scatter_reduce_ops.h>
#include <ATen/ops/scatter_reduce_ops.h>
#include <ATen/ops/masked_select_ops.h>
#include <ATen/ops/masked_select_ops.h>
#include <ATen/ops/histogramdd_ops.h>
#include <ATen/ops/histogramdd_ops.h>
#include <ATen/ops/histogramdd_ops.h>
#include <ATen/ops/hypot_ops.h>
#include <ATen/ops/hypot_ops.h>
#include <ATen/ops/hypot_ops.h>
#include <ATen/ops/igammac_ops.h>
#include <ATen/ops/igammac_ops.h>
#include <ATen/ops/igammac_ops.h>
#include <ATen/ops/sort_ops.h>
#include <ATen/ops/sort_ops.h>
#include <ATen/ops/sort_ops.h>
#include <ATen/ops/sort_ops.h>
#include <ATen/ops/sort_ops.h>
#include <ATen/ops/sort_ops.h>
#include <ATen/ops/sort_ops.h>
#include <ATen/ops/sort_ops.h>
#include <ATen/ops/all_ops.h>
#include <ATen/ops/all_ops.h>
#include <ATen/ops/_amp_update_scale_ops.h>
#include <ATen/ops/_foreach_addcdiv_ops.h>
#include <ATen/ops/_foreach_addcdiv_ops.h>
#include <ATen/ops/_foreach_addcdiv_ops.h>
#include <ATen/ops/_foreach_addcdiv_ops.h>
#include <ATen/ops/_foreach_addcdiv_ops.h>
#include <ATen/ops/_foreach_addcdiv_ops.h>
#include <ATen/ops/_foreach_log_ops.h>
#include <ATen/ops/_foreach_log_ops.h>
#include <ATen/ops/_foreach_log1p_ops.h>
#include <ATen/ops/_foreach_log1p_ops.h>
#include <ATen/ops/_foreach_pow_ops.h>
#include <ATen/ops/_foreach_pow_ops.h>
#include <ATen/ops/_foreach_pow_ops.h>
#include <ATen/ops/_foreach_pow_ops.h>
#include <ATen/ops/_foreach_pow_ops.h>
#include <ATen/ops/_foreach_pow_ops.h>
#include <ATen/ops/_foreach_pow_ops.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_ops.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr_ops.h>
#include <ATen/ops/multi_margin_loss_ops.h>
#include <ATen/ops/multi_margin_loss_ops.h>
#include <ATen/ops/multilabel_margin_loss_ops.h>
#include <ATen/ops/multilabel_margin_loss_ops.h>
#include <ATen/ops/multilabel_margin_loss_forward_ops.h>
#include <ATen/ops/multilabel_margin_loss_forward_ops.h>
#include <ATen/ops/nll_loss_forward_ops.h>
#include <ATen/ops/nll_loss_forward_ops.h>
#include <ATen/ops/hardswish_ops.h>
#include <ATen/ops/hardswish_ops.h>
#include <ATen/ops/hardswish_ops.h>
#include <ATen/ops/softshrink_backward_ops.h>
#include <ATen/ops/softshrink_backward_ops.h>
#include <ATen/ops/_adaptive_avg_pool2d_backward_ops.h>
#include <ATen/ops/avg_pool2d_ops.h>
#include <ATen/ops/avg_pool2d_ops.h>
#include <ATen/ops/fractional_max_pool2d_backward_ops.h>
#include <ATen/ops/fractional_max_pool2d_backward_ops.h>
#include <ATen/ops/max_pool3d_with_indices_backward_ops.h>
#include <ATen/ops/max_pool3d_with_indices_backward_ops.h>
#include <ATen/ops/reflection_pad2d_backward_ops.h>
#include <ATen/ops/reflection_pad2d_backward_ops.h>
#include <ATen/ops/_upsample_nearest_exact3d_ops.h>
#include <ATen/ops/upsample_bilinear2d_backward_ops.h>
#include <ATen/ops/upsample_bilinear2d_backward_ops.h>
#include <ATen/ops/_upsample_lanczos2d_aa_backward_ops.h>
#include <ATen/ops/_upsample_lanczos2d_aa_backward_ops.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_ops.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_ops.h>
#include <ATen/ops/_upsample_nearest_exact3d_ops.h>
#include <ATen/ops/_upsample_nearest_exact3d_ops.h>
#include <ATen/ops/slow_conv_dilated3d_ops.h>
#include <ATen/ops/isinf_ops.h>
#include <ATen/ops/special_digamma_ops.h>
#include <ATen/ops/special_digamma_ops.h>
#include <ATen/ops/fft_hfft_ops.h>
#include <ATen/ops/fft_hfft_ops.h>
#include <ATen/ops/fft_ihfft2_ops.h>
#include <ATen/ops/fft_ihfft2_ops.h>
#include <ATen/ops/fft_irfftn_ops.h>
#include <ATen/ops/fft_irfftn_ops.h>
#include <ATen/ops/linalg_eigh_ops.h>
#include <ATen/ops/linalg_eigh_ops.h>
#include <ATen/ops/linalg_householder_product_ops.h>
#include <ATen/ops/linalg_householder_product_ops.h>
#include <ATen/ops/linalg_matrix_norm_ops.h>
#include <ATen/ops/linalg_matrix_norm_ops.h>
#include <ATen/ops/linalg_matrix_norm_ops.h>
#include <ATen/ops/linalg_matrix_norm_ops.h>
#include <ATen/ops/_nested_tensor_from_tensor_list_ops.h>
#include <ATen/ops/transpose_copy_ops.h>
#include <ATen/ops/unfold_copy_ops.h>
#include <ATen/ops/scaled_dot_product_attention_ops.h>
#include <ATen/ops/_scaled_dot_product_cudnn_attention_ops.h>
#include <ATen/ops/special_bessel_y1_ops.h>
#include <ATen/ops/special_bessel_y1_ops.h>
#include <ATen/ops/special_legendre_polynomial_p_ops.h>
#include <ATen/ops/special_legendre_polynomial_p_ops.h>
#include <ATen/ops/special_legendre_polynomial_p_ops.h>
#include <ATen/ops/special_legendre_polynomial_p_ops.h>
#include <ATen/ops/special_legendre_polynomial_p_ops.h>
#include <ATen/ops/special_legendre_polynomial_p_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u_ops.h>
#include <ATen/ops/_fused_adam_ops.h>
#include <ATen/ops/_fused_adam_ops.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/binary_cross_entropy_with_logits_ops.h>
#include <ATen/ops/block_diag_ops.h>
#include <ATen/ops/diagonal_backward_ops.h>
#include <ATen/ops/_embedding_bag_per_sample_weights_backward_ops.h>
#include <ATen/ops/new_empty_strided_ops.h>
#include <ATen/ops/new_full_ops.h>
#include <ATen/ops/empty_quantized_ops.h>
#include <ATen/ops/empty_strided_ops.h>
#include <ATen/ops/from_file_ops.h>
#include <ATen/ops/native_layer_norm_backward_ops.h>
#include <ATen/ops/mps_convolution_backward_ops.h>
#include <ATen/ops/mkldnn_rnn_layer_ops.h>
#include <ATen/ops/miopen_rnn_ops.h>
#include <ATen/ops/_sparse_sparse_matmul_ops.h>
#include <ATen/ops/_native_batch_norm_legit_ops.h>
#include <ATen/ops/_euclidean_dist_ops.h>
#include <ATen/ops/_cdist_backward_ops.h>
#include <ATen/ops/scalar_tensor_ops.h>
#include <ATen/ops/rand_ops.h>
#include <ATen/ops/rand_ops.h>
#include <ATen/ops/repeat_ops.h>
#include <ATen/ops/_mkldnn_reshape_ops.h>
#include <ATen/ops/var_mean_ops.h>
#include <ATen/ops/native_norm_ops.h>
#include <ATen/ops/native_norm_ops.h>
#include <ATen/ops/_sparse_sum_backward_ops.h>
#include <ATen/ops/_sparse_softmax_ops.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors_ops.h>
#include <ATen/ops/_amp_update_scale_ops.h>
#include <ATen/ops/_amp_update_scale_ops.h>
#include <ATen/ops/_foreach_addcdiv_ops.h>
#include <ATen/ops/_foreach_addcdiv_ops.h>
#include <ATen/ops/_foreach_addcdiv_ops.h>
#include <ATen/ops/_foreach_log_ops.h>
#include <ATen/ops/_foreach_log1p_ops.h>
#include <ATen/ops/_foreach_pow_ops.h>
#include <ATen/ops/_foreach_pow_ops.h>
#include <ATen/ops/_foreach_pow_ops.h>
#include <ATen/ops/_adaptive_avg_pool2d_backward_ops.h>
#include <ATen/ops/slow_conv_dilated3d_ops.h>
#include <ATen/ops/isinf_ops.h>
#include <ATen/ops/_nested_tensor_from_tensor_list_ops.h>
#include <ATen/ops/transpose_copy_ops.h>
#include <ATen/ops/unfold_copy_ops.h>
#include <ATen/ops/_fused_adam_ops.h>
#include <ATen/ops/_fused_adam_ops.h>
#include <ATen/ops/_fused_adam_ops.h>
#include <ATen/ops/_fused_adam_ops.h>
#endif

using namespace at;

namespace torch {

namespace TraceType {

namespace {
at::Tensor _cast_Short(c10::DispatchKeySet ks, const at::Tensor & self, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cast_Short");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cast_Short::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor absolute(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::absolute");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::absolute::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & absolute_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::absolute");
    } else {
      op_name = c10::Symbol::fromQualString("aten::absolute_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("absolute_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::absolute_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & absolute_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::absolute");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("absolute_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::absolute_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor add_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::add");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::add_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & add__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::add");
    } else {
      op_name = c10::Symbol::fromQualString("aten::add_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::add__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & add_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::add");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("add_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::add_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor add_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::add");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::add_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & add__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::add");
    } else {
      op_name = c10::Symbol::fromQualString("aten::add_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::add__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor all_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::all");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::all_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor all_dims(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::all");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::all_dims::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & all_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::all");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("all_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::all_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & all_out_dims_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::all");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("all_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::all_dims_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor all_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::all");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::all_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & all_out_dimname_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::all");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("all_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::all_dimname_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor atleast_2d(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atleast_2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::atleast_2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> atleast_2d_Sequence(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atleast_2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::atleast_2d_Sequence::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor binary_cross_entropy_with_logits(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & pos_weight, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::binary_cross_entropy_with_logits");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "pos_weight", pos_weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::binary_cross_entropy_with_logits::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, pos_weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _lazy_clone(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_lazy_clone");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_lazy_clone::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor logical_and(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logical_and");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logical_and::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & logical_and_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::logical_and");
    } else {
      op_name = c10::Symbol::fromQualString("aten::logical_and_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_and_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logical_and_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & logical_and_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logical_and");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_and_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logical_and_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor block_diag(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::block_diag");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::block_diag::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> chunk(c10::DispatchKeySet ks, const at::Tensor & self, int64_t chunks, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::chunk");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "chunks", chunks);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::chunk::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, chunks, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> tensor_split_sections(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt sections, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tensor_split");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sections", sections);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::tensor_split_sections::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, sections, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> tensor_split_indices(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef indices, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tensor_split");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::tensor_split_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> tensor_split_tensor_indices_or_sections(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor_indices_or_sections, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tensor_split");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor_indices_or_sections", tensor_indices_or_sections);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::tensor_split_tensor_indices_or_sections::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor_indices_or_sections, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor clamp(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clamp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::clamp::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor clamp_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clamp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::clamp_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & clamp_(c10::DispatchKeySet ks, at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::clamp");
    } else {
      op_name = c10::Symbol::fromQualString("aten::clamp_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clamp_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & clamp__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::clamp");
    } else {
      op_name = c10::Symbol::fromQualString("aten::clamp_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clamp__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & clamp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clamp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clamp_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & clamp_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clamp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clamp_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
bool cudnn_is_acceptable(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::cudnn_is_acceptable::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor cov(c10::DispatchKeySet ks, const at::Tensor & self, int64_t correction, const ::std::optional<at::Tensor> & fweights, const ::std::optional<at::Tensor> & aweights) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cov");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "correction", correction);
    jit::tracer::addInputs(node, "fweights", fweights);
    jit::tracer::addInputs(node, "aweights", aweights);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cov::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, correction, fweights, aweights);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor diagonal_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::diagonal_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input_sizes", input_sizes);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::diagonal_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input_sizes, offset, dim1, dim2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor dot(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor", tensor);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::dot::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & dot_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor", tensor);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("dot_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::dot_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor embedding_sparse_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::embedding_sparse_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::embedding_sparse_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, indices, num_weights, padding_idx, scale_grad_by_freq);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _embedding_bag_per_sample_weights_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_embedding_bag_per_sample_weights_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_embedding_bag_per_sample_weights_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, weight, indices, offsets, offset2bag, mode, padding_idx);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor new_empty_strided(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::new_empty_strided");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::new_empty_strided::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, stride, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor new_full(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::new_full");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::new_full::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, fill_value, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor empty_quantized(c10::DispatchKeySet ks, at::IntArrayRef size, const at::Tensor & qtensor, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::empty_quantized");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "qtensor", qtensor);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::empty_quantized::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, qtensor, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor empty_strided(c10::DispatchKeySet ks, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::empty_strided");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::empty_strided::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, stride, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor exp(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::exp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::exp::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & exp_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::exp");
    } else {
      op_name = c10::Symbol::fromQualString("aten::exp_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("exp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::exp_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & exp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::exp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("exp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::exp_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor frac(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::frac");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::frac::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & frac_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::frac");
    } else {
      op_name = c10::Symbol::fromQualString("aten::frac_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("frac_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::frac_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & frac_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::frac");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("frac_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::frac_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor from_file(c10::DispatchKeySet ks, c10::string_view filename, ::std::optional<bool> shared, ::std::optional<int64_t> size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::from_file");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "filename", filename);
    jit::tracer::addInputs(node, "shared", shared);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::from_file::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), filename, shared, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _unsafe_masked_index(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, const c10::List<::std::optional<at::Tensor>> & indices, const at::Scalar & fill) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_unsafe_masked_index");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "fill", fill);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_unsafe_masked_index::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, indices, fill);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool _is_zerotensor(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::_is_zerotensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
bool is_signed(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::is_signed::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_layer_norm_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "rstd", rstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::native_layer_norm_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor ldexp_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ldexp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ldexp_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ldexp_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::ldexp");
    } else {
      op_name = c10::Symbol::fromQualString("aten::ldexp_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ldexp_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ldexp_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & ldexp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ldexp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ldexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ldexp_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor log2(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::log2::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & log2_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::log2");
    } else {
      op_name = c10::Symbol::fromQualString("aten::log2_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log2_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log2_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & log2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log2_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor logspace(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, int64_t steps, double base, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logspace::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, base, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor logspace_Tensor_Tensor(c10::DispatchKeySet ks, const at::Tensor & start, const at::Tensor & end, int64_t steps, double base, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logspace_Tensor_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, base, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor logspace_Tensor_Scalar(c10::DispatchKeySet ks, const at::Tensor & start, const at::Scalar & end, int64_t steps, double base, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logspace_Tensor_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, base, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor logspace_Scalar_Tensor(c10::DispatchKeySet ks, const at::Scalar & start, const at::Tensor & end, int64_t steps, double base, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logspace_Scalar_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, base, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & logspace_out_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, int64_t steps, double base, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logspace_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, base, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & logspace_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & start, const at::Tensor & end, int64_t steps, double base, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logspace_Tensor_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, base, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & logspace_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & start, const at::Scalar & end, int64_t steps, double base, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logspace_Tensor_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, base, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & logspace_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Tensor & end, int64_t steps, double base, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logspace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "steps", steps);
    jit::tracer::addInputs(node, "base", base);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logspace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logspace_Scalar_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), start, end, steps, base, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor matrix_power(c10::DispatchKeySet ks, const at::Tensor & self, int64_t n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::matrix_power");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::matrix_power::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & matrix_power_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::matrix_power");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("matrix_power_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::matrix_power_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> mps_convolution_backward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, ::std::array<bool,3> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mps_convolution_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::mps_convolution_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grad_output, weight, padding, stride, dilation, groups, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> mkldnn_rnn_layer(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight0, const at::Tensor & weight1, const at::Tensor & weight2, const at::Tensor & weight3, const at::Tensor & hx_, const at::Tensor & cx_, bool reverse, at::IntArrayRef batch_sizes, int64_t mode, int64_t hidden_size, int64_t num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_rnn_layer");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight0", weight0);
    jit::tracer::addInputs(node, "weight1", weight1);
    jit::tracer::addInputs(node, "weight2", weight2);
    jit::tracer::addInputs(node, "weight3", weight3);
    jit::tracer::addInputs(node, "hx_", hx_);
    jit::tracer::addInputs(node, "cx_", cx_);
    jit::tracer::addInputs(node, "reverse", reverse);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::mkldnn_rnn_layer::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight0, weight1, weight2, weight3, hx_, cx_, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> miopen_rnn(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & hx, const ::std::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const ::std::optional<at::Tensor> & dropout_state) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_rnn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "dropout_state", dropout_state);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3, result4] =at::_ops::miopen_rnn::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
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
at::Tensor _weight_int4pack_mm_with_scales_and_zeros(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, int64_t qGroupSize, const at::Tensor & qScale, const at::Tensor & qZeros) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_weight_int4pack_mm_with_scales_and_zeros");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "qGroupSize", qGroupSize);
    jit::tracer::addInputs(node, "qScale", qScale);
    jit::tracer::addInputs(node, "qZeros", qZeros);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_weight_int4pack_mm_with_scales_and_zeros::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat2, qGroupSize, qScale, qZeros);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _convert_weight_to_int4pack_for_cpu(c10::DispatchKeySet ks, const at::Tensor & self, int64_t innerKTiles) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_convert_weight_to_int4pack_for_cpu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "innerKTiles", innerKTiles);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_convert_weight_to_int4pack_for_cpu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, innerKTiles);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_mm(c10::DispatchKeySet ks, const at::Tensor & sparse, const at::Tensor & dense) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_mm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "dense", dense);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_mm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), sparse, dense);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_mm_reduce(c10::DispatchKeySet ks, const at::Tensor & sparse, const at::Tensor & dense, c10::string_view reduce) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_mm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "dense", dense);
    jit::tracer::addInputs(node, "reduce", reduce);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_mm_reduce::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), sparse, dense, reduce);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_sparse_matmul(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_sparse_matmul");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_sparse_matmul::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & running_mean, at::Tensor & running_var, bool training, double momentum, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::_native_batch_norm_legit");
    } else {
      op_name = c10::Symbol::fromQualString("aten::_native_batch_norm_legit");
    }
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
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::_native_batch_norm_legit::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, momentum, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _native_batch_norm_legit_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & running_mean, at::Tensor & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_native_batch_norm_legit");
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

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
      jit::tracer::addInputs(node, "save_mean", save_mean);
      jit::tracer::addInputs(node, "save_invstd", save_invstd);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_native_batch_norm_legit_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_native_batch_norm_legit_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, save_mean);
    jit::tracer::addOutput(node, save_invstd);
  }
  return std::forward_as_tuple(out, save_mean, save_invstd);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit_no_stats(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, bool training, double momentum, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_native_batch_norm_legit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::_native_batch_norm_legit_no_stats::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, training, momentum, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _native_batch_norm_legit_out_no_stats_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_native_batch_norm_legit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
      jit::tracer::addInputs(node, "save_mean", save_mean);
      jit::tracer::addInputs(node, "save_invstd", save_invstd);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_native_batch_norm_legit_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_native_batch_norm_legit_no_stats_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, training, momentum, eps, out, save_mean, save_invstd);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, save_mean);
    jit::tracer::addOutput(node, save_invstd);
  }
  return std::forward_as_tuple(out, save_mean, save_invstd);
}
bool _nnpack_available(c10::DispatchKeySet ks) {
  auto result =at::_ops::_nnpack_available::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer));
  return result;
}
at::Tensor _euclidean_dist(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_euclidean_dist");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_euclidean_dist::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x1, x2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _cdist_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & x1, const at::Tensor & x2, double p, const at::Tensor & cdist) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cdist_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "cdist", cdist);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cdist_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, x1, x2, p, cdist);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor native_channel_shuffle(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt groups) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_channel_shuffle");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::native_channel_shuffle::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, groups);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor rad2deg(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rad2deg");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::rad2deg::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & rad2deg_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::rad2deg");
    } else {
      op_name = c10::Symbol::fromQualString("aten::rad2deg_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rad2deg_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rad2deg_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & rad2deg_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rad2deg");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rad2deg_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rad2deg_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor scalar_tensor(c10::DispatchKeySet ks, const at::Scalar & s, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scalar_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::scalar_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), s, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor rand_names(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand");
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
  auto result =at::_ops::rand_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, names, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor rand_generator_with_names(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand");
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
  auto result =at::_ops::rand_generator_with_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, names, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor rand(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand");
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
  auto result =at::_ops::rand::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor rand_generator(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand");
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
  auto result =at::_ops::rand_generator::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & rand_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rand_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & rand_out_generator_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rand_generator_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor repeat(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef repeats) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::repeat");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "repeats", repeats);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::repeat::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, repeats);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _mkldnn_reshape(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef shape) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mkldnn_reshape");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "shape", shape);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_mkldnn_reshape::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, shape);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor rsqrt(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rsqrt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::rsqrt::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & rsqrt_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::rsqrt");
    } else {
      op_name = c10::Symbol::fromQualString("aten::rsqrt_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rsqrt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rsqrt_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & rsqrt_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rsqrt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rsqrt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rsqrt_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
c10::SymInt sym_size_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto result =at::_ops::sym_size_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  return result;
}
at::Tensor istft(c10::DispatchKeySet ks, const at::Tensor & self, int64_t n_fft, ::std::optional<int64_t> hop_length, ::std::optional<int64_t> win_length, const ::std::optional<at::Tensor> & window, bool center, bool normalized, ::std::optional<bool> onesided, ::std::optional<int64_t> length, bool return_complex) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::istft");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n_fft", n_fft);
    jit::tracer::addInputs(node, "hop_length", hop_length);
    jit::tracer::addInputs(node, "win_length", win_length);
    jit::tracer::addInputs(node, "window", window);
    jit::tracer::addInputs(node, "center", center);
    jit::tracer::addInputs(node, "normalized", normalized);
    jit::tracer::addInputs(node, "onesided", onesided);
    jit::tracer::addInputs(node, "length", length);
    jit::tracer::addInputs(node, "return_complex", return_complex);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::istft::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n_fft, hop_length, win_length, window, center, normalized, onesided, length, return_complex);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor nansum(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nansum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nansum::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & nansum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nansum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nansum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nansum_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor flipud(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::flipud");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::flipud::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor trapz_x(c10::DispatchKeySet ks, const at::Tensor & y, const at::Tensor & x, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::trapz");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "y", y);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::trapz_x::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), y, x, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor trapz_dx(c10::DispatchKeySet ks, const at::Tensor & y, double dx, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::trapz");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "y", y);
    jit::tracer::addInputs(node, "dx", dx);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::trapz_dx::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), y, dx, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t _nested_get_ragged_idx(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::_nested_get_ragged_idx::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor _nested_get_min_seqlen(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_get_min_seqlen");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_get_min_seqlen::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor trunc(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::trunc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::trunc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & trunc_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::trunc");
    } else {
      op_name = c10::Symbol::fromQualString("aten::trunc_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("trunc_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::trunc_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & trunc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::trunc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("trunc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::trunc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor> var_mean(c10::DispatchKeySet ks, const at::Tensor & self, bool unbiased) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::var_mean::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, unbiased);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> var_mean_dim(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::var_mean_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, unbiased, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> var_mean_correction(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "correction", correction);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::var_mean_correction::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, correction, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> var_mean_names_dim(c10::DispatchKeySet ks, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::var_mean_names_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, unbiased, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> var_mean_correction_names(c10::DispatchKeySet ks, const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "correction", correction);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::var_mean_correction_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, correction, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor norm_except_dim(c10::DispatchKeySet ks, const at::Tensor & v, int64_t pow, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm_except_dim");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "pow", pow);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::norm_except_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), v, pow, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor native_norm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::native_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor native_norm_ScalarOpt_dim_dtype(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::native_norm_ScalarOpt_dim_dtype::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_sum_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & self, at::IntArrayRef dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_sum_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_sum_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_softmax_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_softmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_softmax_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_softmax_Dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_softmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_softmax_Dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_softmax(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool half_to_float) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_softmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "half_to_float", half_to_float);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_softmax::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, half_to_float);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_coo_tensor_with_dims_and_tensors(c10::DispatchKeySet ks, int64_t sparse_dim, int64_t dense_dim, c10::SymIntArrayRef size, const at::Tensor & indices, const at::Tensor & values, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<bool> is_coalesced) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_coo_tensor_with_dims_and_tensors");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "is_coalesced", is_coalesced);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_coo_tensor_with_dims_and_tensors::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), sparse_dim, dense_dim, size, indices, values, dtype, layout, device, pin_memory, is_coalesced);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool is_coalesced(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::is_coalesced::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::indices");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor col_indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::col_indices");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::col_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hspmm_out_out(c10::DispatchKeySet ks, const at::Tensor & mat1, const at::Tensor & mat2, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hspmm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hspmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hspmm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mat1, mat2, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor hspmm(c10::DispatchKeySet ks, const at::Tensor & mat1, const at::Tensor & mat2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hspmm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hspmm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mat1, mat2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _to_sparse_semi_structured(c10::DispatchKeySet ks, const at::Tensor & dense) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_semi_structured");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "dense", dense);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_to_sparse_semi_structured::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), dense);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor cartesian_prod(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cartesian_prod");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cartesian_prod::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::ScalarType promote_types(c10::DispatchKeySet ks, at::ScalarType type1, at::ScalarType type2) {
  auto result =at::_ops::promote_types::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), type1, type2);
  return result;
}
at::Scalar _local_scalar_dense(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::_local_scalar_dense::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
::std::tuple<at::Tensor,at::Tensor> rnn_relu_input(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rnn_relu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::rnn_relu_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> rnn_relu_data(c10::DispatchKeySet ks, const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rnn_relu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "params", params);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::rnn_relu_data::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor gru_cell(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) {
  auto result =at::_ops::gru_cell::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
::std::tuple<at::Tensor,at::Tensor> quantized_lstm_cell(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantized_lstm_cell");
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
  auto [result0, result1] =at::_ops::quantized_lstm_cell::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor scatter_reduce_two(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter_reduce");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "reduce", reduce);
    jit::tracer::addInputs(node, "include_self", include_self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::scatter_reduce_two::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src, reduce, include_self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & scatter_reduce__two(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::scatter_reduce");
    } else {
      op_name = c10::Symbol::fromQualString("aten::scatter_reduce_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "reduce", reduce);
    jit::tracer::addInputs(node, "include_self", include_self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_reduce_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter_reduce__two::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src, reduce, include_self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & scatter_reduce_out_two_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, bool include_self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter_reduce");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "reduce", reduce);
    jit::tracer::addInputs(node, "include_self", include_self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_reduce_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter_reduce_two_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src, reduce, include_self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & masked_select_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::masked_select");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_select_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::masked_select_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor masked_select(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::masked_select");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::masked_select::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,::std::vector<at::Tensor>> histogramdd(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::histogramdd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "range", range);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "density", density);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [hist, bin_edges] =at::_ops::histogramdd::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, range, weight, density);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, hist);
    jit::tracer::addOutput(node, bin_edges);
  }
  return std::make_tuple(std::move(hist), std::move(bin_edges));
}
::std::tuple<at::Tensor,::std::vector<at::Tensor>> histogramdd_int_bins(c10::DispatchKeySet ks, const at::Tensor & self, int64_t bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::histogramdd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "range", range);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "density", density);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [hist, bin_edges] =at::_ops::histogramdd_int_bins::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, range, weight, density);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, hist);
    jit::tracer::addOutput(node, bin_edges);
  }
  return std::make_tuple(std::move(hist), std::move(bin_edges));
}
::std::tuple<at::Tensor,::std::vector<at::Tensor>> histogramdd_TensorList_bins(c10::DispatchKeySet ks, const at::Tensor & self, at::TensorList bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::histogramdd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "range", range);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "density", density);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [hist, bin_edges] =at::_ops::histogramdd_TensorList_bins::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, range, weight, density);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, hist);
    jit::tracer::addOutput(node, bin_edges);
  }
  return std::make_tuple(std::move(hist), std::move(bin_edges));
}
at::Tensor & hypot_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hypot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hypot_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hypot_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor hypot(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hypot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hypot::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hypot_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::hypot");
    } else {
      op_name = c10::Symbol::fromQualString("aten::hypot_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hypot_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hypot_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & igammac_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::igammac");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("igammac_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::igammac_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor igammac(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::igammac");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::igammac::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & igammac_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::igammac");
    } else {
      op_name = c10::Symbol::fromQualString("aten::igammac_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("igammac_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::igammac_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sort_values::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, descending, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out_values_stable(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stable", stable);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sort_values_stable::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, stable, dim, descending, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor> sort(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::sort::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor,at::Tensor> sort_stable(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<bool> stable, int64_t dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stable", stable);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::sort_stable::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, stable, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out_dimname_values(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sort_dimname_values::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, descending, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out_dimname_values_stable(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<bool> stable, at::Dimname dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stable", stable);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sort_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sort_dimname_values_stable::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, stable, dim, descending, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor> sort_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::sort_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor,at::Tensor> sort_dimname_stable(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<bool> stable, at::Dimname dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stable", stable);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::sort_dimname_stable::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, stable, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
at::Tensor all(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::all");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::all::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & all_out_all_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::all");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("all_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::all_all_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _amp_update_scale_(c10::DispatchKeySet ks, at::Tensor & self, at::Tensor & growth_tracker, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::_amp_update_scale");
    } else {
      op_name = c10::Symbol::fromQualString("aten::_amp_update_scale_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "growth_tracker", growth_tracker);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    jit::tracer::addInputs(node, "scale_growth_factor", scale_growth_factor);
    jit::tracer::addInputs(node, "scale_backoff_factor", scale_backoff_factor);
    jit::tracer::addInputs(node, "growth_interval", growth_interval);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_amp_update_scale_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_amp_update_scale_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
::std::vector<at::Tensor> _foreach_addcdiv_Scalar(c10::DispatchKeySet ks, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_addcdiv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_addcdiv_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> _foreach_addcdiv_ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_addcdiv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "scalars", scalars);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_addcdiv_ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, scalars);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> _foreach_addcdiv_Tensor(c10::DispatchKeySet ks, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_addcdiv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "scalars", scalars);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_addcdiv_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, scalars);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_addcdiv__Scalar(c10::DispatchKeySet ks, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
  at::_ops::_foreach_addcdiv__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, value);
}
void _foreach_addcdiv__ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
  at::_ops::_foreach_addcdiv__ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, scalars);
}
void _foreach_addcdiv__Tensor(c10::DispatchKeySet ks, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars) {
  at::_ops::_foreach_addcdiv__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, scalars);
}
::std::vector<at::Tensor> _foreach_log(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_log");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_log::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_log_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_log_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_log1p(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_log1p");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_log1p::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_log1p_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_log1p_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_pow_List(c10::DispatchKeySet ks, at::TensorList self, at::TensorList exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_pow");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_pow_List::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> _foreach_pow_Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_pow");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_pow_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> _foreach_pow_ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_pow");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_pow_ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> _foreach_pow_ScalarAndTensor(c10::DispatchKeySet ks, const at::Scalar & self, at::TensorList exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_pow");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_pow_ScalarAndTensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_pow__List(c10::DispatchKeySet ks, at::TensorList self, at::TensorList exponent) {
  at::_ops::_foreach_pow__List::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
}
void _foreach_pow__Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & exponent) {
  at::_ops::_foreach_pow__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
}
void _foreach_pow__ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> exponent) {
  at::_ops::_foreach_pow__ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
}
at::Tensor _convert_indices_from_coo_to_csr(c10::DispatchKeySet ks, const at::Tensor & self, int64_t size, bool out_int32) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_convert_indices_from_coo_to_csr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_convert_indices_from_coo_to_csr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, out_int32);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _convert_indices_from_coo_to_csr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t size, bool out_int32, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_convert_indices_from_coo_to_csr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "out_int32", out_int32);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_convert_indices_from_coo_to_csr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_convert_indices_from_coo_to_csr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, out_int32, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & multi_margin_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const ::std::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::multi_margin_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multi_margin_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::multi_margin_loss_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, p, margin, weight, reduction, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor multi_margin_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const ::std::optional<at::Tensor> & weight, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::multi_margin_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::multi_margin_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, p, margin, weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & multilabel_margin_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::multilabel_margin_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multilabel_margin_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::multilabel_margin_loss_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor multilabel_margin_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::multilabel_margin_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::multilabel_margin_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> multilabel_margin_loss_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & output, at::Tensor & is_target) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::multilabel_margin_loss_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "output", output);
      jit::tracer::addInputs(node, "is_target", is_target);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multilabel_margin_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::multilabel_margin_loss_forward_output::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction, output, is_target);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, is_target);
  }
  return std::forward_as_tuple(output, is_target);
}
::std::tuple<at::Tensor,at::Tensor> multilabel_margin_loss_forward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::multilabel_margin_loss_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [output, is_target] =at::_ops::multilabel_margin_loss_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, is_target);
  }
  return std::make_tuple(std::move(output), std::move(is_target));
}
::std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, at::Tensor & output, at::Tensor & total_weight) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nll_loss_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "output", output);
      jit::tracer::addInputs(node, "total_weight", total_weight);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nll_loss_forward_output::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, reduction, ignore_index, output, total_weight);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  return std::forward_as_tuple(output, total_weight);
}
::std::tuple<at::Tensor,at::Tensor> nll_loss_forward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nll_loss_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "ignore_index", ignore_index);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [output, total_weight] =at::_ops::nll_loss_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, reduction, ignore_index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
at::Tensor & hardswish_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardswish");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardswish_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardswish_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor hardswish(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardswish");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hardswish::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hardswish_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::hardswish");
    } else {
      op_name = c10::Symbol::fromQualString("aten::hardswish_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardswish_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardswish_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & softshrink_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::softshrink_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softshrink_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::softshrink_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, lambd, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor softshrink_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::softshrink_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::softshrink_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, lambd);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _adaptive_avg_pool2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_adaptive_avg_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_adaptive_avg_pool2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & avg_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::avg_pool2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::avg_pool2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor avg_pool2d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::avg_pool2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::avg_pool2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fractional_max_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fractional_max_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "indices", indices);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fractional_max_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fractional_max_pool2d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, output_size, indices, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor fractional_max_pool2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fractional_max_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fractional_max_pool2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, output_size, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & max_pool3d_with_indices_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool3d_with_indices_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool3d_with_indices_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::max_pool3d_with_indices_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor max_pool3d_with_indices_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool3d_with_indices_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::max_pool3d_with_indices_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & reflection_pad2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reflection_pad2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::reflection_pad2d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, padding, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor reflection_pad2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reflection_pad2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::reflection_pad2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _upsample_nearest_exact3d_vec(c10::DispatchKeySet ks, const at::Tensor & input, at::OptionalSymIntArrayRef output_size, ::std::optional<at::ArrayRef<double>> scale_factors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scale_factors", scale_factors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_upsample_nearest_exact3d_vec::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, output_size, scale_factors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & upsample_bilinear2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_bilinear2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_bilinear2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::upsample_bilinear2d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor upsample_bilinear2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_bilinear2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::upsample_bilinear2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _upsample_lanczos2d_aa_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_lanczos2d_aa_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_upsample_lanczos2d_aa_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_upsample_lanczos2d_aa_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor _upsample_lanczos2d_aa_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_lanczos2d_aa_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_upsample_lanczos2d_aa_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _upsample_nearest_exact1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact1d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales", scales);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_upsample_nearest_exact1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_upsample_nearest_exact1d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, scales, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor _upsample_nearest_exact1d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact1d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_upsample_nearest_exact1d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, scales);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _upsample_nearest_exact3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact3d");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_upsample_nearest_exact3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_upsample_nearest_exact3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, scales_d, scales_h, scales_w, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _upsample_nearest_exact3d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_d, ::std::optional<double> scales_h, ::std::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact3d");
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
  auto result =at::_ops::_upsample_nearest_exact3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, scales_d, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor slow_conv_dilated3d(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slow_conv_dilated3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::slow_conv_dilated3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor isinf(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isinf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::isinf::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_digamma(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_digamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_digamma::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_digamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_digamma");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_digamma_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_digamma_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_hfft(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_hfft");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_hfft::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_hfft_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_hfft");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_hfft_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_hfft_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_ihfft2(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ihfft2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_ihfft2::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_ihfft2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ihfft2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_ihfft2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_ihfft2_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_irfftn(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_irfftn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_irfftn::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_irfftn_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_irfftn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_irfftn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_irfftn_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor> linalg_eigh(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view UPLO) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_eigh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "UPLO", UPLO);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [eigenvalues, eigenvectors] =at::_ops::linalg_eigh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, UPLO);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, eigenvalues);
    jit::tracer::addOutput(node, eigenvectors);
  }
  return std::make_tuple(std::move(eigenvalues), std::move(eigenvectors));
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_eigh_out_eigvals(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view UPLO, at::Tensor & eigvals, at::Tensor & eigvecs) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_eigh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "UPLO", UPLO);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "eigvals", eigvals);
      jit::tracer::addInputs(node, "eigvecs", eigvecs);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_eigh_out", eigvals);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_eigh_eigvals::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, UPLO, eigvals, eigvecs);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, eigvals);
    jit::tracer::addOutput(node, eigvecs);
  }
  return std::forward_as_tuple(eigvals, eigvecs);
}
at::Tensor linalg_householder_product(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & tau) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_householder_product");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "tau", tau);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_householder_product::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, tau);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_householder_product_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & tau, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_householder_product");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "tau", tau);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_householder_product_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_householder_product_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, tau, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linalg_matrix_norm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_matrix_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "ord", ord);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_matrix_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, ord, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_matrix_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_matrix_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "ord", ord);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_matrix_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_matrix_norm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, ord, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linalg_matrix_norm_str_ord(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_matrix_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "ord", ord);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_matrix_norm_str_ord::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, ord, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_matrix_norm_out_str_ord_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_matrix_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "ord", ord);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_matrix_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_matrix_norm_str_ord_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, ord, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _nested_tensor_from_tensor_list(c10::DispatchKeySet ks, at::TensorList list, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_tensor_from_tensor_list");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "list", list);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_tensor_from_tensor_list::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), list, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor transpose_copy_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim0, int64_t dim1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::transpose_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::transpose_copy_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim0, dim1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor unfold_copy(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unfold_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dimension", dimension);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unfold_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dimension, size, step);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor scaled_dot_product_attention(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, ::std::optional<double> scale, bool enable_gqa) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scaled_dot_product_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "attn_mask", attn_mask);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "enable_gqa", enable_gqa);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::scaled_dot_product_attention::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::SymInt,c10::SymInt,at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_cudnn_attention(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_bias, bool compute_log_sumexp, double dropout_p, bool is_causal, bool return_debug_mask, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_dot_product_cudnn_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "attn_bias", attn_bias);
    jit::tracer::addInputs(node, "compute_log_sumexp", compute_log_sumexp);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "return_debug_mask", return_debug_mask);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask] =at::_ops::_scaled_dot_product_cudnn_attention::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal, return_debug_mask, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, logsumexp);
    jit::tracer::addOutput(node, cum_seq_q);
    jit::tracer::addOutput(node, cum_seq_k);
    jit::tracer::addOutput(node, max_q);
    jit::tracer::addOutput(node, max_k);
    jit::tracer::addOutput(node, philox_seed);
    jit::tracer::addOutput(node, philox_offset);
    jit::tracer::addOutput(node, debug_attn_mask);
  }
  return std::make_tuple(std::move(output), std::move(logsumexp), std::move(cum_seq_q), std::move(cum_seq_k), std::move(max_q), std::move(max_k), std::move(philox_seed), std::move(philox_offset), std::move(debug_attn_mask));
}
at::Tensor special_bessel_y1(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_bessel_y1");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_bessel_y1::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_bessel_y1_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_bessel_y1");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_bessel_y1_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_bessel_y1_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_legendre_polynomial_p(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_legendre_polynomial_p");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_legendre_polynomial_p::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_legendre_polynomial_p_x_scalar(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_legendre_polynomial_p");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_legendre_polynomial_p_x_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_legendre_polynomial_p_n_scalar(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_legendre_polynomial_p");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_legendre_polynomial_p_n_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_legendre_polynomial_p_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_legendre_polynomial_p");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_legendre_polynomial_p_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_legendre_polynomial_p_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_legendre_polynomial_p_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_legendre_polynomial_p");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_legendre_polynomial_p_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_legendre_polynomial_p_x_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_legendre_polynomial_p_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_legendre_polynomial_p");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_legendre_polynomial_p_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_legendre_polynomial_p_n_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_shifted_chebyshev_polynomial_u(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_u::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_shifted_chebyshev_polynomial_u_x_scalar(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_u_x_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_shifted_chebyshev_polynomial_u_n_scalar(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_u_n_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_shifted_chebyshev_polynomial_u_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_u_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_u_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_u_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_u_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_u_x_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_u_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_u");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_u_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_u_n_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _fused_adam_(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  at::_ops::_fused_adam_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
}
void _fused_adam__tensor_lr(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, const at::Tensor & lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  at::_ops::_fused_adam__tensor_lr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
}
at::Tensor & add_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::add");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("add_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::add_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & binary_cross_entropy_with_logits_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & pos_weight, int64_t reduction, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::binary_cross_entropy_with_logits");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "pos_weight", pos_weight);
    jit::tracer::addInputs(node, "reduction", reduction);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("binary_cross_entropy_with_logits_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::binary_cross_entropy_with_logits_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, pos_weight, reduction, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & block_diag_out_out(c10::DispatchKeySet ks, at::TensorList tensors, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::block_diag");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("block_diag_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::block_diag_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & diagonal_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef input_sizes, int64_t offset, int64_t dim1, int64_t dim2, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::diagonal_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input_sizes", input_sizes);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("diagonal_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::diagonal_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input_sizes, offset, dim1, dim2, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _embedding_bag_per_sample_weights_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, int64_t mode, int64_t padding_idx, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_embedding_bag_per_sample_weights_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_embedding_bag_per_sample_weights_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_embedding_bag_per_sample_weights_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, weight, indices, offsets, offset2bag, mode, padding_idx, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & new_empty_strided_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::new_empty_strided");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("new_empty_strided_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::new_empty_strided_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, stride, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & new_full_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::new_full");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("new_full_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::new_full_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, fill_value, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & empty_quantized_out_out(c10::DispatchKeySet ks, at::IntArrayRef size, const at::Tensor & qtensor, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::empty_quantized");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "qtensor", qtensor);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("empty_quantized_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::empty_quantized_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, qtensor, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & empty_strided_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::empty_strided");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("empty_strided_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::empty_strided_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, stride, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & from_file_out_out(c10::DispatchKeySet ks, c10::string_view filename, ::std::optional<bool> shared, ::std::optional<int64_t> size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::from_file");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "filename", filename);
    jit::tracer::addInputs(node, "shared", shared);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("from_file_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::from_file_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), filename, shared, size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_layer_norm_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const at::Tensor & mean, const at::Tensor & rstd, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_layer_norm_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "rstd", rstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "output_mask", output_mask);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_layer_norm_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::native_layer_norm_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> mps_convolution_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mps_convolution_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "output_mask", output_mask);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mps_convolution_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mps_convolution_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grad_output, weight, padding, stride, dilation, groups, output_mask, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> mkldnn_rnn_layer_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight0, const at::Tensor & weight1, const at::Tensor & weight2, const at::Tensor & weight3, const at::Tensor & hx_, const at::Tensor & cx_, bool reverse, at::IntArrayRef batch_sizes, int64_t mode, int64_t hidden_size, int64_t num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_rnn_layer");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight0", weight0);
    jit::tracer::addInputs(node, "weight1", weight1);
    jit::tracer::addInputs(node, "weight2", weight2);
    jit::tracer::addInputs(node, "weight3", weight3);
    jit::tracer::addInputs(node, "hx_", hx_);
    jit::tracer::addInputs(node, "cx_", cx_);
    jit::tracer::addInputs(node, "reverse", reverse);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "has_biases", has_biases);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "train", train);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
      jit::tracer::addInputs(node, "out3", out3);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mkldnn_rnn_layer_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mkldnn_rnn_layer_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight0, weight1, weight2, weight3, hx_, cx_, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train, out0, out1, out2, out3);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
    jit::tracer::addOutput(node, out3);
  }
  return std::forward_as_tuple(out0, out1, out2, out3);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> miopen_rnn_out_out(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & hx, const ::std::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const ::std::optional<at::Tensor> & dropout_state, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3, at::Tensor & out4) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_rnn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "dropout_state", dropout_state);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
      jit::tracer::addInputs(node, "out3", out3);
      jit::tracer::addInputs(node, "out4", out4);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("miopen_rnn_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::miopen_rnn_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, weight_stride0, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, out0, out1, out2, out3, out4);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
    jit::tracer::addOutput(node, out3);
    jit::tracer::addOutput(node, out4);
  }
  return std::forward_as_tuple(out0, out1, out2, out3, out4);
}
at::Tensor & _sparse_sparse_matmul_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_sparse_matmul");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_sparse_matmul_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_sparse_sparse_matmul_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _native_batch_norm_legit_functional(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & running_mean, const at::Tensor & running_var, bool training, double momentum, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_native_batch_norm_legit_functional");
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
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, running_mean_out, running_var_out] =at::_ops::_native_batch_norm_legit_functional::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, momentum, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, running_mean_out);
    jit::tracer::addOutput(node, running_var_out);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(running_mean_out), std::move(running_var_out));
}
at::Tensor & _euclidean_dist_out_out(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_euclidean_dist");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_euclidean_dist_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_euclidean_dist_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x1, x2, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _cdist_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & x1, const at::Tensor & x2, double p, const at::Tensor & cdist, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cdist_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "cdist", cdist);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cdist_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_cdist_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, x1, x2, p, cdist, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & scalar_tensor_out_out(c10::DispatchKeySet ks, const at::Scalar & s, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scalar_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "s", s);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scalar_tensor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scalar_tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), s, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & rand_out_names_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rand_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, names, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & rand_out_generator_with_names_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rand_generator_with_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, generator, names, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & repeat_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef repeats, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::repeat");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "repeats", repeats);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("repeat_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::repeat_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, repeats, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _mkldnn_reshape_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef shape, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mkldnn_reshape");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "shape", shape);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mkldnn_reshape_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_mkldnn_reshape_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, shape, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> var_mean_out_correction_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var_mean");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "correction", correction);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("var_mean_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::var_mean_correction_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, correction, keepdim, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & native_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & p, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::native_norm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & native_norm_out_ScalarOpt_dim_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::native_norm_ScalarOpt_dim_dtype_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _sparse_sum_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & self, at::IntArrayRef dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_sum_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_sum_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_sparse_sum_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, self, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _sparse_softmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_softmax");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_softmax_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_sparse_softmax_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, half_to_float, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _sparse_coo_tensor_with_dims_and_tensors_out_out(c10::DispatchKeySet ks, int64_t sparse_dim, int64_t dense_dim, c10::SymIntArrayRef size, const at::Tensor & indices, const at::Tensor & values, ::std::optional<bool> is_coalesced, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_coo_tensor_with_dims_and_tensors");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "is_coalesced", is_coalesced);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_coo_tensor_with_dims_and_tensors_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_sparse_coo_tensor_with_dims_and_tensors_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), sparse_dim, dense_dim, size, indices, values, is_coalesced, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _amp_update_scale_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & growth_tracker, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_amp_update_scale");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "growth_tracker", growth_tracker);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    jit::tracer::addInputs(node, "scale_growth_factor", scale_growth_factor);
    jit::tracer::addInputs(node, "scale_backoff_factor", scale_backoff_factor);
    jit::tracer::addInputs(node, "growth_interval", growth_interval);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_amp_update_scale_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_amp_update_scale_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor> _amp_update_scale(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & growth_tracker, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_amp_update_scale");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "growth_tracker", growth_tracker);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    jit::tracer::addInputs(node, "scale_growth_factor", scale_growth_factor);
    jit::tracer::addInputs(node, "scale_backoff_factor", scale_backoff_factor);
    jit::tracer::addInputs(node, "growth_interval", growth_interval);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, growth_tracker_out] =at::_ops::_amp_update_scale::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, growth_tracker_out);
  }
  return std::make_tuple(std::move(result0), std::move(growth_tracker_out));
}
void _foreach_addcdiv_out_Scalar_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value, at::TensorList out) {
  at::_ops::_foreach_addcdiv_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, value, out);
}
void _foreach_addcdiv_out_ScalarList_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars, at::TensorList out) {
  at::_ops::_foreach_addcdiv_ScalarList_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, scalars, out);
}
void _foreach_addcdiv_out_Tensor_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars, at::TensorList out) {
  at::_ops::_foreach_addcdiv_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, scalars, out);
}
void _foreach_log_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_log_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_log1p_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_log1p_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_pow_out_List_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList exponent, at::TensorList out) {
  at::_ops::_foreach_pow_List_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent, out);
}
void _foreach_pow_out_Scalar_out(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & exponent, at::TensorList out) {
  at::_ops::_foreach_pow_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent, out);
}
void _foreach_pow_out_ScalarList_out(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> exponent, at::TensorList out) {
  at::_ops::_foreach_pow_ScalarList_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent, out);
}
at::Tensor & _adaptive_avg_pool2d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_adaptive_avg_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_adaptive_avg_pool2d_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_adaptive_avg_pool2d_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & slow_conv_dilated3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slow_conv_dilated3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("slow_conv_dilated3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::slow_conv_dilated3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, dilation, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & isinf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isinf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("isinf_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::isinf_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _nested_tensor_from_tensor_list_out_out(c10::DispatchKeySet ks, at::TensorList list, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_tensor_from_tensor_list");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "list", list);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_nested_tensor_from_tensor_list_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_nested_tensor_from_tensor_list_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), list, dtype, layout, device, pin_memory, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & transpose_copy_out_int_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim0, int64_t dim1, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::transpose_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("transpose_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::transpose_copy_int_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim0, dim1, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & unfold_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dimension, int64_t size, int64_t step, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unfold_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dimension", dimension);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("unfold_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::unfold_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dimension, size, step, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _fused_adam_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf, at::TensorList out) {
  at::_ops::_fused_adam_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, out);
}
::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>> _fused_adam(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, double lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fused_adam");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grads", grads);
    jit::tracer::addInputs(node, "exp_avgs", exp_avgs);
    jit::tracer::addInputs(node, "exp_avg_sqs", exp_avg_sqs);
    jit::tracer::addInputs(node, "max_exp_avg_sqs", max_exp_avg_sqs);
    jit::tracer::addInputs(node, "state_steps", state_steps);
    jit::tracer::addInputs(node, "lr", lr);
    jit::tracer::addInputs(node, "beta1", beta1);
    jit::tracer::addInputs(node, "beta2", beta2);
    jit::tracer::addInputs(node, "weight_decay", weight_decay);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "amsgrad", amsgrad);
    jit::tracer::addInputs(node, "maximize", maximize);
    jit::tracer::addInputs(node, "grad_scale", grad_scale);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [self_out, grads_out, exp_avgs_out, exp_avg_sqs_out, max_exp_avg_sqs_out] =at::_ops::_fused_adam::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self_out);
    jit::tracer::addOutput(node, grads_out);
    jit::tracer::addOutput(node, exp_avgs_out);
    jit::tracer::addOutput(node, exp_avg_sqs_out);
    jit::tracer::addOutput(node, max_exp_avg_sqs_out);
  }
  return std::make_tuple(std::move(self_out), std::move(grads_out), std::move(exp_avgs_out), std::move(exp_avg_sqs_out), std::move(max_exp_avg_sqs_out));
}
void _fused_adam_out_tensor_lr_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, const at::Tensor & lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf, at::TensorList out) {
  at::_ops::_fused_adam_tensor_lr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf, out);
}
::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>> _fused_adam_tensor_lr(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs, at::TensorList max_exp_avg_sqs, at::TensorList state_steps, const at::Tensor & lr, double beta1, double beta2, double weight_decay, double eps, bool amsgrad, bool maximize, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fused_adam");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grads", grads);
    jit::tracer::addInputs(node, "exp_avgs", exp_avgs);
    jit::tracer::addInputs(node, "exp_avg_sqs", exp_avg_sqs);
    jit::tracer::addInputs(node, "max_exp_avg_sqs", max_exp_avg_sqs);
    jit::tracer::addInputs(node, "state_steps", state_steps);
    jit::tracer::addInputs(node, "lr", lr);
    jit::tracer::addInputs(node, "beta1", beta1);
    jit::tracer::addInputs(node, "beta2", beta2);
    jit::tracer::addInputs(node, "weight_decay", weight_decay);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "amsgrad", amsgrad);
    jit::tracer::addInputs(node, "maximize", maximize);
    jit::tracer::addInputs(node, "grad_scale", grad_scale);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [self_out, grads_out, exp_avgs_out, exp_avg_sqs_out, max_exp_avg_sqs_out] =at::_ops::_fused_adam_tensor_lr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize, grad_scale, found_inf);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self_out);
    jit::tracer::addOutput(node, grads_out);
    jit::tracer::addOutput(node, exp_avgs_out);
    jit::tracer::addOutput(node, exp_avg_sqs_out);
    jit::tracer::addOutput(node, max_exp_avg_sqs_out);
  }
  return std::make_tuple(std::move(self_out), std::move(grads_out), std::move(exp_avgs_out), std::move(exp_avg_sqs_out), std::move(max_exp_avg_sqs_out));
}
}  // namespace
}  // namespace TraceType

namespace {

TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  m.impl("_cast_Short",
         TORCH_FN(TraceType::_cast_Short)
  );
  m.impl("absolute",
         TORCH_FN(TraceType::absolute)
  );
  m.impl("absolute_",
         TORCH_FN(TraceType::absolute_)
  );
  m.impl("absolute.out",
         TORCH_FN(TraceType::absolute_out_out)
  );
  m.impl("add.Tensor",
         TORCH_FN(TraceType::add_Tensor)
  );
  m.impl("add_.Tensor",
         TORCH_FN(TraceType::add__Tensor)
  );
  m.impl("add.out",
         TORCH_FN(TraceType::add_out_out)
  );
  m.impl("add.Scalar",
         TORCH_FN(TraceType::add_Scalar)
  );
  m.impl("add_.Scalar",
         TORCH_FN(TraceType::add__Scalar)
  );
  m.impl("all.dim",
         TORCH_FN(TraceType::all_dim)
  );
  m.impl("all.dims",
         TORCH_FN(TraceType::all_dims)
  );
  m.impl("all.out",
         TORCH_FN(TraceType::all_out_out)
  );
  m.impl("all.dims_out",
         TORCH_FN(TraceType::all_out_dims_out)
  );
  m.impl("all.dimname",
         TORCH_FN(TraceType::all_dimname)
  );
  m.impl("all.dimname_out",
         TORCH_FN(TraceType::all_out_dimname_out)
  );
  m.impl("atleast_2d",
         TORCH_FN(TraceType::atleast_2d)
  );
  m.impl("atleast_2d.Sequence",
         TORCH_FN(TraceType::atleast_2d_Sequence)
  );
  m.impl("binary_cross_entropy_with_logits",
         TORCH_FN(TraceType::binary_cross_entropy_with_logits)
  );
  m.impl("_lazy_clone",
         TORCH_FN(TraceType::_lazy_clone)
  );
  m.impl("logical_and",
         TORCH_FN(TraceType::logical_and)
  );
  m.impl("logical_and_",
         TORCH_FN(TraceType::logical_and_)
  );
  m.impl("logical_and.out",
         TORCH_FN(TraceType::logical_and_out_out)
  );
  m.impl("block_diag",
         TORCH_FN(TraceType::block_diag)
  );
  m.impl("chunk",
         TORCH_FN(TraceType::chunk)
  );
  m.impl("tensor_split.sections",
         TORCH_FN(TraceType::tensor_split_sections)
  );
  m.impl("tensor_split.indices",
         TORCH_FN(TraceType::tensor_split_indices)
  );
  m.impl("tensor_split.tensor_indices_or_sections",
         TORCH_FN(TraceType::tensor_split_tensor_indices_or_sections)
  );
  m.impl("clamp",
         TORCH_FN(TraceType::clamp)
  );
  m.impl("clamp.Tensor",
         TORCH_FN(TraceType::clamp_Tensor)
  );
  m.impl("clamp_",
         TORCH_FN(TraceType::clamp_)
  );
  m.impl("clamp_.Tensor",
         TORCH_FN(TraceType::clamp__Tensor)
  );
  m.impl("clamp.out",
         TORCH_FN(TraceType::clamp_out_out)
  );
  m.impl("clamp.Tensor_out",
         TORCH_FN(TraceType::clamp_out_Tensor_out)
  );
  m.impl("cudnn_is_acceptable",
         TORCH_FN(TraceType::cudnn_is_acceptable)
  );
  m.impl("cov",
         TORCH_FN(TraceType::cov)
  );
  m.impl("diagonal_backward",
         TORCH_FN(TraceType::diagonal_backward)
  );
  m.impl("dot",
         TORCH_FN(TraceType::dot)
  );
  m.impl("dot.out",
         TORCH_FN(TraceType::dot_out_out)
  );
  m.impl("embedding_sparse_backward",
         TORCH_FN(TraceType::embedding_sparse_backward)
  );
  m.impl("_embedding_bag_per_sample_weights_backward",
         TORCH_FN(TraceType::_embedding_bag_per_sample_weights_backward)
  );
  m.impl("new_empty_strided",
         TORCH_FN(TraceType::new_empty_strided)
  );
  m.impl("new_full",
         TORCH_FN(TraceType::new_full)
  );
  m.impl("empty_quantized",
         TORCH_FN(TraceType::empty_quantized)
  );
  m.impl("empty_strided",
         TORCH_FN(TraceType::empty_strided)
  );
  m.impl("exp",
         TORCH_FN(TraceType::exp)
  );
  m.impl("exp_",
         TORCH_FN(TraceType::exp_)
  );
  m.impl("exp.out",
         TORCH_FN(TraceType::exp_out_out)
  );
  m.impl("frac",
         TORCH_FN(TraceType::frac)
  );
  m.impl("frac_",
         TORCH_FN(TraceType::frac_)
  );
  m.impl("frac.out",
         TORCH_FN(TraceType::frac_out_out)
  );
  m.impl("from_file",
         TORCH_FN(TraceType::from_file)
  );
  m.impl("_unsafe_masked_index",
         TORCH_FN(TraceType::_unsafe_masked_index)
  );
  m.impl("_is_zerotensor",
         TORCH_FN(TraceType::_is_zerotensor)
  );
  m.impl("is_signed",
         TORCH_FN(TraceType::is_signed)
  );
  m.impl("native_layer_norm_backward",
         TORCH_FN(TraceType::native_layer_norm_backward)
  );
  m.impl("ldexp.Tensor",
         TORCH_FN(TraceType::ldexp_Tensor)
  );
  m.impl("ldexp_",
         TORCH_FN(TraceType::ldexp_)
  );
  m.impl("ldexp.out",
         TORCH_FN(TraceType::ldexp_out_out)
  );
  m.impl("log2",
         TORCH_FN(TraceType::log2)
  );
  m.impl("log2_",
         TORCH_FN(TraceType::log2_)
  );
  m.impl("log2.out",
         TORCH_FN(TraceType::log2_out_out)
  );
  m.impl("logspace",
         TORCH_FN(TraceType::logspace)
  );
  m.impl("logspace.Tensor_Tensor",
         TORCH_FN(TraceType::logspace_Tensor_Tensor)
  );
  m.impl("logspace.Tensor_Scalar",
         TORCH_FN(TraceType::logspace_Tensor_Scalar)
  );
  m.impl("logspace.Scalar_Tensor",
         TORCH_FN(TraceType::logspace_Scalar_Tensor)
  );
  m.impl("logspace.out",
         TORCH_FN(TraceType::logspace_out_out)
  );
  m.impl("logspace.Tensor_Tensor_out",
         TORCH_FN(TraceType::logspace_out_Tensor_Tensor_out)
  );
  m.impl("logspace.Tensor_Scalar_out",
         TORCH_FN(TraceType::logspace_out_Tensor_Scalar_out)
  );
  m.impl("logspace.Scalar_Tensor_out",
         TORCH_FN(TraceType::logspace_out_Scalar_Tensor_out)
  );
  m.impl("matrix_power",
         TORCH_FN(TraceType::matrix_power)
  );
  m.impl("matrix_power.out",
         TORCH_FN(TraceType::matrix_power_out_out)
  );
  m.impl("mps_convolution_backward",
         TORCH_FN(TraceType::mps_convolution_backward)
  );
  m.impl("mkldnn_rnn_layer",
         TORCH_FN(TraceType::mkldnn_rnn_layer)
  );
  m.impl("miopen_rnn",
         TORCH_FN(TraceType::miopen_rnn)
  );
  m.impl("_weight_int4pack_mm_with_scales_and_zeros",
         TORCH_FN(TraceType::_weight_int4pack_mm_with_scales_and_zeros)
  );
  m.impl("_convert_weight_to_int4pack_for_cpu",
         TORCH_FN(TraceType::_convert_weight_to_int4pack_for_cpu)
  );
  m.impl("_sparse_mm",
         TORCH_FN(TraceType::_sparse_mm)
  );
  m.impl("_sparse_mm.reduce",
         TORCH_FN(TraceType::_sparse_mm_reduce)
  );
  m.impl("_sparse_sparse_matmul",
         TORCH_FN(TraceType::_sparse_sparse_matmul)
  );
  m.impl("_native_batch_norm_legit",
         TORCH_FN(TraceType::_native_batch_norm_legit)
  );
  m.impl("_native_batch_norm_legit.out",
         TORCH_FN(TraceType::_native_batch_norm_legit_out_out)
  );
  m.impl("_native_batch_norm_legit.no_stats",
         TORCH_FN(TraceType::_native_batch_norm_legit_no_stats)
  );
  m.impl("_native_batch_norm_legit.no_stats_out",
         TORCH_FN(TraceType::_native_batch_norm_legit_out_no_stats_out)
  );
  m.impl("_nnpack_available",
         TORCH_FN(TraceType::_nnpack_available)
  );
  m.impl("_euclidean_dist",
         TORCH_FN(TraceType::_euclidean_dist)
  );
  m.impl("_cdist_backward",
         TORCH_FN(TraceType::_cdist_backward)
  );
  m.impl("native_channel_shuffle",
         TORCH_FN(TraceType::native_channel_shuffle)
  );
  m.impl("rad2deg",
         TORCH_FN(TraceType::rad2deg)
  );
  m.impl("rad2deg_",
         TORCH_FN(TraceType::rad2deg_)
  );
  m.impl("rad2deg.out",
         TORCH_FN(TraceType::rad2deg_out_out)
  );
  m.impl("scalar_tensor",
         TORCH_FN(TraceType::scalar_tensor)
  );
  m.impl("rand.names",
         TORCH_FN(TraceType::rand_names)
  );
  m.impl("rand.generator_with_names",
         TORCH_FN(TraceType::rand_generator_with_names)
  );
  m.impl("rand",
         TORCH_FN(TraceType::rand)
  );
  m.impl("rand.generator",
         TORCH_FN(TraceType::rand_generator)
  );
  m.impl("rand.out",
         TORCH_FN(TraceType::rand_out_out)
  );
  m.impl("rand.generator_out",
         TORCH_FN(TraceType::rand_out_generator_out)
  );
  m.impl("repeat",
         TORCH_FN(TraceType::repeat)
  );
  m.impl("_mkldnn_reshape",
         TORCH_FN(TraceType::_mkldnn_reshape)
  );
  m.impl("rsqrt",
         TORCH_FN(TraceType::rsqrt)
  );
  m.impl("rsqrt_",
         TORCH_FN(TraceType::rsqrt_)
  );
  m.impl("rsqrt.out",
         TORCH_FN(TraceType::rsqrt_out_out)
  );
  m.impl("sym_size.int",
         TORCH_FN(TraceType::sym_size_int)
  );
  m.impl("istft",
         TORCH_FN(TraceType::istft)
  );
  m.impl("nansum",
         TORCH_FN(TraceType::nansum)
  );
  m.impl("nansum.out",
         TORCH_FN(TraceType::nansum_out_out)
  );
  m.impl("flipud",
         TORCH_FN(TraceType::flipud)
  );
  m.impl("trapz.x",
         TORCH_FN(TraceType::trapz_x)
  );
  m.impl("trapz.dx",
         TORCH_FN(TraceType::trapz_dx)
  );
  m.impl("_nested_get_ragged_idx",
         TORCH_FN(TraceType::_nested_get_ragged_idx)
  );
  m.impl("_nested_get_min_seqlen",
         TORCH_FN(TraceType::_nested_get_min_seqlen)
  );
  m.impl("trunc",
         TORCH_FN(TraceType::trunc)
  );
  m.impl("trunc_",
         TORCH_FN(TraceType::trunc_)
  );
  m.impl("trunc.out",
         TORCH_FN(TraceType::trunc_out_out)
  );
  m.impl("var_mean",
         TORCH_FN(TraceType::var_mean)
  );
  m.impl("var_mean.dim",
         TORCH_FN(TraceType::var_mean_dim)
  );
  m.impl("var_mean.correction",
         TORCH_FN(TraceType::var_mean_correction)
  );
  m.impl("var_mean.names_dim",
         TORCH_FN(TraceType::var_mean_names_dim)
  );
  m.impl("var_mean.correction_names",
         TORCH_FN(TraceType::var_mean_correction_names)
  );
  m.impl("norm_except_dim",
         TORCH_FN(TraceType::norm_except_dim)
  );
  m.impl("native_norm",
         TORCH_FN(TraceType::native_norm)
  );
  m.impl("native_norm.ScalarOpt_dim_dtype",
         TORCH_FN(TraceType::native_norm_ScalarOpt_dim_dtype)
  );
  m.impl("_sparse_sum_backward",
         TORCH_FN(TraceType::_sparse_sum_backward)
  );
  m.impl("_sparse_softmax.int",
         TORCH_FN(TraceType::_sparse_softmax_int)
  );
  m.impl("_sparse_softmax.Dimname",
         TORCH_FN(TraceType::_sparse_softmax_Dimname)
  );
  m.impl("_sparse_softmax",
         TORCH_FN(TraceType::_sparse_softmax)
  );
  m.impl("_sparse_coo_tensor_with_dims_and_tensors",
         TORCH_FN(TraceType::_sparse_coo_tensor_with_dims_and_tensors)
  );
  m.impl("is_coalesced",
         TORCH_FN(TraceType::is_coalesced)
  );
  m.impl("indices",
         TORCH_FN(TraceType::indices)
  );
  m.impl("col_indices",
         TORCH_FN(TraceType::col_indices)
  );
  m.impl("hspmm.out",
         TORCH_FN(TraceType::hspmm_out_out)
  );
  m.impl("hspmm",
         TORCH_FN(TraceType::hspmm)
  );
  m.impl("_to_sparse_semi_structured",
         TORCH_FN(TraceType::_to_sparse_semi_structured)
  );
  m.impl("cartesian_prod",
         TORCH_FN(TraceType::cartesian_prod)
  );
  m.impl("promote_types",
         TORCH_FN(TraceType::promote_types)
  );
  m.impl("_local_scalar_dense",
         TORCH_FN(TraceType::_local_scalar_dense)
  );
  m.impl("rnn_relu.input",
         TORCH_FN(TraceType::rnn_relu_input)
  );
  m.impl("rnn_relu.data",
         TORCH_FN(TraceType::rnn_relu_data)
  );
  m.impl("gru_cell",
         TORCH_FN(TraceType::gru_cell)
  );
  m.impl("quantized_lstm_cell",
         TORCH_FN(TraceType::quantized_lstm_cell)
  );
  m.impl("scatter_reduce.two",
         TORCH_FN(TraceType::scatter_reduce_two)
  );
  m.impl("scatter_reduce_.two",
         TORCH_FN(TraceType::scatter_reduce__two)
  );
  m.impl("scatter_reduce.two_out",
         TORCH_FN(TraceType::scatter_reduce_out_two_out)
  );
  m.impl("masked_select.out",
         TORCH_FN(TraceType::masked_select_out_out)
  );
  m.impl("masked_select",
         TORCH_FN(TraceType::masked_select)
  );
  m.impl("histogramdd",
         TORCH_FN(TraceType::histogramdd)
  );
  m.impl("histogramdd.int_bins",
         TORCH_FN(TraceType::histogramdd_int_bins)
  );
  m.impl("histogramdd.TensorList_bins",
         TORCH_FN(TraceType::histogramdd_TensorList_bins)
  );
  m.impl("hypot.out",
         TORCH_FN(TraceType::hypot_out_out)
  );
  m.impl("hypot",
         TORCH_FN(TraceType::hypot)
  );
  m.impl("hypot_",
         TORCH_FN(TraceType::hypot_)
  );
  m.impl("igammac.out",
         TORCH_FN(TraceType::igammac_out_out)
  );
  m.impl("igammac",
         TORCH_FN(TraceType::igammac)
  );
  m.impl("igammac_",
         TORCH_FN(TraceType::igammac_)
  );
  m.impl("sort.values",
         TORCH_FN(TraceType::sort_out_values)
  );
  m.impl("sort.values_stable",
         TORCH_FN(TraceType::sort_out_values_stable)
  );
  m.impl("sort",
         TORCH_FN(TraceType::sort)
  );
  m.impl("sort.stable",
         TORCH_FN(TraceType::sort_stable)
  );
  m.impl("sort.dimname_values",
         TORCH_FN(TraceType::sort_out_dimname_values)
  );
  m.impl("sort.dimname_values_stable",
         TORCH_FN(TraceType::sort_out_dimname_values_stable)
  );
  m.impl("sort.dimname",
         TORCH_FN(TraceType::sort_dimname)
  );
  m.impl("sort.dimname_stable",
         TORCH_FN(TraceType::sort_dimname_stable)
  );
  m.impl("all",
         TORCH_FN(TraceType::all)
  );
  m.impl("all.all_out",
         TORCH_FN(TraceType::all_out_all_out)
  );
  m.impl("_amp_update_scale_",
         TORCH_FN(TraceType::_amp_update_scale_)
  );
  m.impl("_foreach_addcdiv.Scalar",
         TORCH_FN(TraceType::_foreach_addcdiv_Scalar)
  );
  m.impl("_foreach_addcdiv.ScalarList",
         TORCH_FN(TraceType::_foreach_addcdiv_ScalarList)
  );
  m.impl("_foreach_addcdiv.Tensor",
         TORCH_FN(TraceType::_foreach_addcdiv_Tensor)
  );
  m.impl("_foreach_addcdiv_.Scalar",
         TORCH_FN(TraceType::_foreach_addcdiv__Scalar)
  );
  m.impl("_foreach_addcdiv_.ScalarList",
         TORCH_FN(TraceType::_foreach_addcdiv__ScalarList)
  );
  m.impl("_foreach_addcdiv_.Tensor",
         TORCH_FN(TraceType::_foreach_addcdiv__Tensor)
  );
  m.impl("_foreach_log",
         TORCH_FN(TraceType::_foreach_log)
  );
  m.impl("_foreach_log_",
         TORCH_FN(TraceType::_foreach_log_)
  );
  m.impl("_foreach_log1p",
         TORCH_FN(TraceType::_foreach_log1p)
  );
  m.impl("_foreach_log1p_",
         TORCH_FN(TraceType::_foreach_log1p_)
  );
  m.impl("_foreach_pow.List",
         TORCH_FN(TraceType::_foreach_pow_List)
  );
  m.impl("_foreach_pow.Scalar",
         TORCH_FN(TraceType::_foreach_pow_Scalar)
  );
  m.impl("_foreach_pow.ScalarList",
         TORCH_FN(TraceType::_foreach_pow_ScalarList)
  );
  m.impl("_foreach_pow.ScalarAndTensor",
         TORCH_FN(TraceType::_foreach_pow_ScalarAndTensor)
  );
  m.impl("_foreach_pow_.List",
         TORCH_FN(TraceType::_foreach_pow__List)
  );
  m.impl("_foreach_pow_.Scalar",
         TORCH_FN(TraceType::_foreach_pow__Scalar)
  );
  m.impl("_foreach_pow_.ScalarList",
         TORCH_FN(TraceType::_foreach_pow__ScalarList)
  );
  m.impl("_convert_indices_from_coo_to_csr",
         TORCH_FN(TraceType::_convert_indices_from_coo_to_csr)
  );
  m.impl("_convert_indices_from_coo_to_csr.out",
         TORCH_FN(TraceType::_convert_indices_from_coo_to_csr_out_out)
  );
  m.impl("multi_margin_loss.out",
         TORCH_FN(TraceType::multi_margin_loss_out_out)
  );
  m.impl("multi_margin_loss",
         TORCH_FN(TraceType::multi_margin_loss)
  );
  m.impl("multilabel_margin_loss.out",
         TORCH_FN(TraceType::multilabel_margin_loss_out_out)
  );
  m.impl("multilabel_margin_loss",
         TORCH_FN(TraceType::multilabel_margin_loss)
  );
  m.impl("multilabel_margin_loss_forward.output",
         TORCH_FN(TraceType::multilabel_margin_loss_forward_out_output)
  );
  m.impl("multilabel_margin_loss_forward",
         TORCH_FN(TraceType::multilabel_margin_loss_forward)
  );
  m.impl("nll_loss_forward.output",
         TORCH_FN(TraceType::nll_loss_forward_out_output)
  );
  m.impl("nll_loss_forward",
         TORCH_FN(TraceType::nll_loss_forward)
  );
  m.impl("hardswish.out",
         TORCH_FN(TraceType::hardswish_out_out)
  );
  m.impl("hardswish",
         TORCH_FN(TraceType::hardswish)
  );
  m.impl("hardswish_",
         TORCH_FN(TraceType::hardswish_)
  );
  m.impl("softshrink_backward.grad_input",
         TORCH_FN(TraceType::softshrink_backward_out_grad_input)
  );
  m.impl("softshrink_backward",
         TORCH_FN(TraceType::softshrink_backward)
  );
  m.impl("_adaptive_avg_pool2d_backward",
         TORCH_FN(TraceType::_adaptive_avg_pool2d_backward)
  );
  m.impl("avg_pool2d.out",
         TORCH_FN(TraceType::avg_pool2d_out_out)
  );
  m.impl("avg_pool2d",
         TORCH_FN(TraceType::avg_pool2d)
  );
  m.impl("fractional_max_pool2d_backward.grad_input",
         TORCH_FN(TraceType::fractional_max_pool2d_backward_out_grad_input)
  );
  m.impl("fractional_max_pool2d_backward",
         TORCH_FN(TraceType::fractional_max_pool2d_backward)
  );
  m.impl("max_pool3d_with_indices_backward.grad_input",
         TORCH_FN(TraceType::max_pool3d_with_indices_backward_out_grad_input)
  );
  m.impl("max_pool3d_with_indices_backward",
         TORCH_FN(TraceType::max_pool3d_with_indices_backward)
  );
  m.impl("reflection_pad2d_backward.grad_input",
         TORCH_FN(TraceType::reflection_pad2d_backward_out_grad_input)
  );
  m.impl("reflection_pad2d_backward",
         TORCH_FN(TraceType::reflection_pad2d_backward)
  );
  m.impl("_upsample_nearest_exact3d.vec",
         TORCH_FN(TraceType::_upsample_nearest_exact3d_vec)
  );
  m.impl("upsample_bilinear2d_backward.grad_input",
         TORCH_FN(TraceType::upsample_bilinear2d_backward_out_grad_input)
  );
  m.impl("upsample_bilinear2d_backward",
         TORCH_FN(TraceType::upsample_bilinear2d_backward)
  );
  m.impl("_upsample_lanczos2d_aa_backward.grad_input",
         TORCH_FN(TraceType::_upsample_lanczos2d_aa_backward_out_grad_input)
  );
  m.impl("_upsample_lanczos2d_aa_backward",
         TORCH_FN(TraceType::_upsample_lanczos2d_aa_backward)
  );
  m.impl("_upsample_nearest_exact1d_backward.grad_input",
         TORCH_FN(TraceType::_upsample_nearest_exact1d_backward_out_grad_input)
  );
  m.impl("_upsample_nearest_exact1d_backward",
         TORCH_FN(TraceType::_upsample_nearest_exact1d_backward)
  );
  m.impl("_upsample_nearest_exact3d.out",
         TORCH_FN(TraceType::_upsample_nearest_exact3d_out_out)
  );
  m.impl("_upsample_nearest_exact3d",
         TORCH_FN(TraceType::_upsample_nearest_exact3d)
  );
  m.impl("slow_conv_dilated3d",
         TORCH_FN(TraceType::slow_conv_dilated3d)
  );
  m.impl("isinf",
         TORCH_FN(TraceType::isinf)
  );
  m.impl("special_digamma",
         TORCH_FN(TraceType::special_digamma)
  );
  m.impl("special_digamma.out",
         TORCH_FN(TraceType::special_digamma_out_out)
  );
  m.impl("fft_hfft",
         TORCH_FN(TraceType::fft_hfft)
  );
  m.impl("fft_hfft.out",
         TORCH_FN(TraceType::fft_hfft_out_out)
  );
  m.impl("fft_ihfft2",
         TORCH_FN(TraceType::fft_ihfft2)
  );
  m.impl("fft_ihfft2.out",
         TORCH_FN(TraceType::fft_ihfft2_out_out)
  );
  m.impl("fft_irfftn",
         TORCH_FN(TraceType::fft_irfftn)
  );
  m.impl("fft_irfftn.out",
         TORCH_FN(TraceType::fft_irfftn_out_out)
  );
  m.impl("linalg_eigh",
         TORCH_FN(TraceType::linalg_eigh)
  );
  m.impl("linalg_eigh.eigvals",
         TORCH_FN(TraceType::linalg_eigh_out_eigvals)
  );
  m.impl("linalg_householder_product",
         TORCH_FN(TraceType::linalg_householder_product)
  );
  m.impl("linalg_householder_product.out",
         TORCH_FN(TraceType::linalg_householder_product_out_out)
  );
  m.impl("linalg_matrix_norm",
         TORCH_FN(TraceType::linalg_matrix_norm)
  );
  m.impl("linalg_matrix_norm.out",
         TORCH_FN(TraceType::linalg_matrix_norm_out_out)
  );
  m.impl("linalg_matrix_norm.str_ord",
         TORCH_FN(TraceType::linalg_matrix_norm_str_ord)
  );
  m.impl("linalg_matrix_norm.str_ord_out",
         TORCH_FN(TraceType::linalg_matrix_norm_out_str_ord_out)
  );
  m.impl("_nested_tensor_from_tensor_list",
         TORCH_FN(TraceType::_nested_tensor_from_tensor_list)
  );
  m.impl("transpose_copy.int",
         TORCH_FN(TraceType::transpose_copy_int)
  );
  m.impl("unfold_copy",
         TORCH_FN(TraceType::unfold_copy)
  );
  m.impl("scaled_dot_product_attention",
         TORCH_FN(TraceType::scaled_dot_product_attention)
  );
  m.impl("_scaled_dot_product_cudnn_attention",
         TORCH_FN(TraceType::_scaled_dot_product_cudnn_attention)
  );
  m.impl("special_bessel_y1",
         TORCH_FN(TraceType::special_bessel_y1)
  );
  m.impl("special_bessel_y1.out",
         TORCH_FN(TraceType::special_bessel_y1_out_out)
  );
  m.impl("special_legendre_polynomial_p",
         TORCH_FN(TraceType::special_legendre_polynomial_p)
  );
  m.impl("special_legendre_polynomial_p.x_scalar",
         TORCH_FN(TraceType::special_legendre_polynomial_p_x_scalar)
  );
  m.impl("special_legendre_polynomial_p.n_scalar",
         TORCH_FN(TraceType::special_legendre_polynomial_p_n_scalar)
  );
  m.impl("special_legendre_polynomial_p.out",
         TORCH_FN(TraceType::special_legendre_polynomial_p_out_out)
  );
  m.impl("special_legendre_polynomial_p.x_scalar_out",
         TORCH_FN(TraceType::special_legendre_polynomial_p_out_x_scalar_out)
  );
  m.impl("special_legendre_polynomial_p.n_scalar_out",
         TORCH_FN(TraceType::special_legendre_polynomial_p_out_n_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_u",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_u)
  );
  m.impl("special_shifted_chebyshev_polynomial_u.x_scalar",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_u_x_scalar)
  );
  m.impl("special_shifted_chebyshev_polynomial_u.n_scalar",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_u_n_scalar)
  );
  m.impl("special_shifted_chebyshev_polynomial_u.out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_u_out_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_u.x_scalar_out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_u_out_x_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_u.n_scalar_out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_u_out_n_scalar_out)
  );
  m.impl("_fused_adam_",
         TORCH_FN(TraceType::_fused_adam_)
  );
  m.impl("_fused_adam_.tensor_lr",
         TORCH_FN(TraceType::_fused_adam__tensor_lr)
  );
  m.impl("add.Scalar_out",
         TORCH_FN(TraceType::add_out_Scalar_out)
  );
  m.impl("binary_cross_entropy_with_logits.out",
         TORCH_FN(TraceType::binary_cross_entropy_with_logits_out_out)
  );
  m.impl("block_diag.out",
         TORCH_FN(TraceType::block_diag_out_out)
  );
  m.impl("diagonal_backward.out",
         TORCH_FN(TraceType::diagonal_backward_out_out)
  );
  m.impl("_embedding_bag_per_sample_weights_backward.out",
         TORCH_FN(TraceType::_embedding_bag_per_sample_weights_backward_out_out)
  );
  m.impl("new_empty_strided.out",
         TORCH_FN(TraceType::new_empty_strided_out_out)
  );
  m.impl("new_full.out",
         TORCH_FN(TraceType::new_full_out_out)
  );
  m.impl("empty_quantized.out",
         TORCH_FN(TraceType::empty_quantized_out_out)
  );
  m.impl("empty_strided.out",
         TORCH_FN(TraceType::empty_strided_out_out)
  );
  m.impl("from_file.out",
         TORCH_FN(TraceType::from_file_out_out)
  );
  m.impl("native_layer_norm_backward.out",
         TORCH_FN(TraceType::native_layer_norm_backward_out_out)
  );
  m.impl("mps_convolution_backward.out",
         TORCH_FN(TraceType::mps_convolution_backward_out_out)
  );
  m.impl("mkldnn_rnn_layer.out",
         TORCH_FN(TraceType::mkldnn_rnn_layer_out_out)
  );
  m.impl("miopen_rnn.out",
         TORCH_FN(TraceType::miopen_rnn_out_out)
  );
  m.impl("_sparse_sparse_matmul.out",
         TORCH_FN(TraceType::_sparse_sparse_matmul_out_out)
  );
  m.impl("_native_batch_norm_legit_functional",
         TORCH_FN(TraceType::_native_batch_norm_legit_functional)
  );
  m.impl("_euclidean_dist.out",
         TORCH_FN(TraceType::_euclidean_dist_out_out)
  );
  m.impl("_cdist_backward.out",
         TORCH_FN(TraceType::_cdist_backward_out_out)
  );
  m.impl("scalar_tensor.out",
         TORCH_FN(TraceType::scalar_tensor_out_out)
  );
  m.impl("rand.names_out",
         TORCH_FN(TraceType::rand_out_names_out)
  );
  m.impl("rand.generator_with_names_out",
         TORCH_FN(TraceType::rand_out_generator_with_names_out)
  );
  m.impl("repeat.out",
         TORCH_FN(TraceType::repeat_out_out)
  );
  m.impl("_mkldnn_reshape.out",
         TORCH_FN(TraceType::_mkldnn_reshape_out_out)
  );
  m.impl("var_mean.correction_out",
         TORCH_FN(TraceType::var_mean_out_correction_out)
  );
  m.impl("native_norm.out",
         TORCH_FN(TraceType::native_norm_out_out)
  );
  m.impl("native_norm.ScalarOpt_dim_dtype_out",
         TORCH_FN(TraceType::native_norm_out_ScalarOpt_dim_dtype_out)
  );
  m.impl("_sparse_sum_backward.out",
         TORCH_FN(TraceType::_sparse_sum_backward_out_out)
  );
  m.impl("_sparse_softmax.out",
         TORCH_FN(TraceType::_sparse_softmax_out_out)
  );
  m.impl("_sparse_coo_tensor_with_dims_and_tensors.out",
         TORCH_FN(TraceType::_sparse_coo_tensor_with_dims_and_tensors_out_out)
  );
  m.impl("_amp_update_scale.out",
         TORCH_FN(TraceType::_amp_update_scale_out_out)
  );
  m.impl("_amp_update_scale",
         TORCH_FN(TraceType::_amp_update_scale)
  );
  m.impl("_foreach_addcdiv.Scalar_out",
         TORCH_FN(TraceType::_foreach_addcdiv_out_Scalar_out)
  );
  m.impl("_foreach_addcdiv.ScalarList_out",
         TORCH_FN(TraceType::_foreach_addcdiv_out_ScalarList_out)
  );
  m.impl("_foreach_addcdiv.Tensor_out",
         TORCH_FN(TraceType::_foreach_addcdiv_out_Tensor_out)
  );
  m.impl("_foreach_log.out",
         TORCH_FN(TraceType::_foreach_log_out_out)
  );
  m.impl("_foreach_log1p.out",
         TORCH_FN(TraceType::_foreach_log1p_out_out)
  );
  m.impl("_foreach_pow.List_out",
         TORCH_FN(TraceType::_foreach_pow_out_List_out)
  );
  m.impl("_foreach_pow.Scalar_out",
         TORCH_FN(TraceType::_foreach_pow_out_Scalar_out)
  );
  m.impl("_foreach_pow.ScalarList_out",
         TORCH_FN(TraceType::_foreach_pow_out_ScalarList_out)
  );
  m.impl("_adaptive_avg_pool2d_backward.out",
         TORCH_FN(TraceType::_adaptive_avg_pool2d_backward_out_out)
  );
  m.impl("slow_conv_dilated3d.out",
         TORCH_FN(TraceType::slow_conv_dilated3d_out_out)
  );
  m.impl("isinf.out",
         TORCH_FN(TraceType::isinf_out_out)
  );
  m.impl("_nested_tensor_from_tensor_list.out",
         TORCH_FN(TraceType::_nested_tensor_from_tensor_list_out_out)
  );
  m.impl("transpose_copy.int_out",
         TORCH_FN(TraceType::transpose_copy_out_int_out)
  );
  m.impl("unfold_copy.out",
         TORCH_FN(TraceType::unfold_copy_out_out)
  );
  m.impl("_fused_adam.out",
         TORCH_FN(TraceType::_fused_adam_out_out)
  );
  m.impl("_fused_adam",
         TORCH_FN(TraceType::_fused_adam)
  );
  m.impl("_fused_adam.tensor_lr_out",
         TORCH_FN(TraceType::_fused_adam_out_tensor_lr_out)
  );
  m.impl("_fused_adam.tensor_lr",
         TORCH_FN(TraceType::_fused_adam_tensor_lr)
  );;
}

}  // namespace

} // namespace torch
