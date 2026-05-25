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
#include <ATen/ops/align_tensors_ops.h>
#include <ATen/ops/_functional_assert_async_ops.h>
#include <ATen/ops/_masked_scale_ops.h>
#include <ATen/ops/_sobol_engine_draw_ops.h>
#include <ATen/ops/alpha_dropout_ops.h>
#include <ATen/ops/alpha_dropout_ops.h>
#include <ATen/ops/view_as_complex_ops.h>
#include <ATen/ops/chalf_ops.h>
#include <ATen/ops/any_ops.h>
#include <ATen/ops/any_ops.h>
#include <ATen/ops/any_ops.h>
#include <ATen/ops/any_ops.h>
#include <ATen/ops/any_ops.h>
#include <ATen/ops/any_ops.h>
#include <ATen/ops/atleast_1d_ops.h>
#include <ATen/ops/atleast_1d_ops.h>
#include <ATen/ops/copysign_ops.h>
#include <ATen/ops/copysign_ops.h>
#include <ATen/ops/copysign_ops.h>
#include <ATen/ops/copysign_ops.h>
#include <ATen/ops/copysign_ops.h>
#include <ATen/ops/copysign_ops.h>
#include <ATen/ops/constant_pad_nd_ops.h>
#include <ATen/ops/convolution_overrideable_ops.h>
#include <ATen/ops/cudnn_batch_norm_ops.h>
#include <ATen/ops/cudnn_batch_norm_ops.h>
#include <ATen/ops/mps_convolution_transpose_backward_ops.h>
#include <ATen/ops/fill_diagonal_ops.h>
#include <ATen/ops/embedding_ops.h>
#include <ATen/ops/_embedding_bag_dense_backward_ops.h>
#include <ATen/ops/erfc_ops.h>
#include <ATen/ops/erfc_ops.h>
#include <ATen/ops/erfc_ops.h>
#include <ATen/ops/_grid_sampler_2d_cpu_fallback_backward_ops.h>
#include <ATen/ops/_cufft_set_plan_cache_max_size_ops.h>
#include <ATen/ops/index_put_ops.h>
#include <ATen/ops/index_put_ops.h>
#include <ATen/ops/instance_norm_ops.h>
#include <ATen/ops/isclose_ops.h>
#include <ATen/ops/is_floating_point_ops.h>
#include <ATen/ops/is_complex_ops.h>
#include <ATen/ops/_cslt_sparse_mm_search_ops.h>
#include <ATen/ops/_sparse_semi_structured_addmm_ops.h>
#include <ATen/ops/fbgemm_pack_gemm_matrix_fp16_ops.h>
#include <ATen/ops/_wrapped_quantized_linear_prepacked_ops.h>
#include <ATen/ops/matmul_backward_ops.h>
#include <ATen/ops/mkldnn_max_pool2d_backward_ops.h>
#include <ATen/ops/max_pool3d_ops.h>
#include <ATen/ops/median_ops.h>
#include <ATen/ops/median_ops.h>
#include <ATen/ops/median_ops.h>
#include <ATen/ops/median_ops.h>
#include <ATen/ops/median_ops.h>
#include <ATen/ops/nanmedian_ops.h>
#include <ATen/ops/nanmedian_ops.h>
#include <ATen/ops/nanmedian_ops.h>
#include <ATen/ops/nanmedian_ops.h>
#include <ATen/ops/nanmedian_ops.h>
#include <ATen/ops/miopen_rnn_backward_ops.h>
#include <ATen/ops/_convert_weight_to_int4pack_ops.h>
#include <ATen/ops/multiply_ops.h>
#include <ATen/ops/multiply_ops.h>
#include <ATen/ops/multiply_ops.h>
#include <ATen/ops/multiply_ops.h>
#include <ATen/ops/multiply_ops.h>
#include <ATen/ops/batch_norm_elemt_ops.h>
#include <ATen/ops/batch_norm_elemt_ops.h>
#include <ATen/ops/cdist_ops.h>
#include <ATen/ops/adjoint_ops.h>
#include <ATen/ops/channel_shuffle_ops.h>
#include <ATen/ops/poisson_nll_loss_ops.h>
#include <ATen/ops/randperm_ops.h>
#include <ATen/ops/randperm_ops.h>
#include <ATen/ops/randperm_ops.h>
#include <ATen/ops/randperm_ops.h>
#include <ATen/ops/negative_ops.h>
#include <ATen/ops/negative_ops.h>
#include <ATen/ops/negative_ops.h>
#include <ATen/ops/infinitely_differentiable_gelu_backward_ops.h>
#include <ATen/ops/hardshrink_backward_ops.h>
#include <ATen/ops/hardshrink_backward_ops.h>
#include <ATen/ops/slice_inverse_ops.h>
#include <ATen/ops/smm_ops.h>
#include <ATen/ops/unsafe_split_with_sizes_ops.h>
#include <ATen/ops/_chunk_cat_ops.h>
#include <ATen/ops/_chunk_cat_ops.h>
#include <ATen/ops/dstack_ops.h>
#include <ATen/ops/dstack_ops.h>
#include <ATen/ops/tan_ops.h>
#include <ATen/ops/tan_ops.h>
#include <ATen/ops/tan_ops.h>
#include <ATen/ops/_nested_get_values_ops.h>
#include <ATen/ops/unsqueeze_ops.h>
#include <ATen/ops/unsqueeze_ops.h>
#include <ATen/ops/_efficientzerotensor_ops.h>
#include <ATen/ops/poisson_ops.h>
#include <ATen/ops/sub_ops.h>
#include <ATen/ops/sub_ops.h>
#include <ATen/ops/sub_ops.h>
#include <ATen/ops/sub_ops.h>
#include <ATen/ops/sub_ops.h>
#include <ATen/ops/subtract_ops.h>
#include <ATen/ops/subtract_ops.h>
#include <ATen/ops/subtract_ops.h>
#include <ATen/ops/subtract_ops.h>
#include <ATen/ops/subtract_ops.h>
#include <ATen/ops/_addmm_activation_ops.h>
#include <ATen/ops/_addmm_activation_ops.h>
#include <ATen/ops/sparse_compressed_tensor_ops.h>
#include <ATen/ops/sparse_bsr_tensor_ops.h>
#include <ATen/ops/sparse_compressed_tensor_ops.h>
#include <ATen/ops/sparse_bsr_tensor_ops.h>
#include <ATen/ops/_validate_sparse_compressed_tensor_args_ops.h>
#include <ATen/ops/to_dense_ops.h>
#include <ATen/ops/_dimI_ops.h>
#include <ATen/ops/_nnz_ops.h>
#include <ATen/ops/ccol_indices_ops.h>
#include <ATen/ops/to_sparse_csr_ops.h>
#include <ATen/ops/_to_sparse_bsr_ops.h>
#include <ATen/ops/q_per_channel_axis_ops.h>
#include <ATen/ops/_make_per_channel_quantized_tensor_ops.h>
#include <ATen/ops/fake_quantize_per_tensor_affine_cachemask_backward_ops.h>
#include <ATen/ops/fake_quantize_per_channel_affine_cachemask_backward_ops.h>
#include <ATen/ops/result_type_ops.h>
#include <ATen/ops/result_type_ops.h>
#include <ATen/ops/result_type_ops.h>
#include <ATen/ops/result_type_ops.h>
#include <ATen/ops/lstm_cell_ops.h>
#include <ATen/ops/masked_scatter_ops.h>
#include <ATen/ops/masked_scatter_ops.h>
#include <ATen/ops/_masked_softmax_backward_ops.h>
#include <ATen/ops/index_add_ops.h>
#include <ATen/ops/index_add_ops.h>
#include <ATen/ops/index_add_ops.h>
#include <ATen/ops/index_add_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/triu_indices_ops.h>
#include <ATen/ops/greater_equal_ops.h>
#include <ATen/ops/greater_equal_ops.h>
#include <ATen/ops/greater_equal_ops.h>
#include <ATen/ops/greater_equal_ops.h>
#include <ATen/ops/greater_equal_ops.h>
#include <ATen/ops/greater_equal_ops.h>
#include <ATen/ops/take_ops.h>
#include <ATen/ops/take_ops.h>
#include <ATen/ops/index_select_backward_ops.h>
#include <ATen/ops/argwhere_ops.h>
#include <ATen/ops/geqrf_ops.h>
#include <ATen/ops/geqrf_ops.h>
#include <ATen/ops/orgqr_ops.h>
#include <ATen/ops/orgqr_ops.h>
#include <ATen/ops/erfinv_ops.h>
#include <ATen/ops/erfinv_ops.h>
#include <ATen/ops/erfinv_ops.h>
#include <ATen/ops/signbit_ops.h>
#include <ATen/ops/signbit_ops.h>
#include <ATen/ops/fmod_ops.h>
#include <ATen/ops/fmod_ops.h>
#include <ATen/ops/fmod_ops.h>
#include <ATen/ops/fmod_ops.h>
#include <ATen/ops/fmod_ops.h>
#include <ATen/ops/fmod_ops.h>
#include <ATen/ops/any_ops.h>
#include <ATen/ops/any_ops.h>
#include <ATen/ops/_foreach_clamp_max_ops.h>
#include <ATen/ops/_foreach_clamp_max_ops.h>
#include <ATen/ops/_foreach_clamp_max_ops.h>
#include <ATen/ops/_foreach_clamp_max_ops.h>
#include <ATen/ops/_foreach_clamp_max_ops.h>
#include <ATen/ops/_foreach_clamp_max_ops.h>
#include <ATen/ops/_foreach_abs_ops.h>
#include <ATen/ops/_foreach_abs_ops.h>
#include <ATen/ops/_foreach_expm1_ops.h>
#include <ATen/ops/_foreach_expm1_ops.h>
#include <ATen/ops/_foreach_log10_ops.h>
#include <ATen/ops/_foreach_log10_ops.h>
#include <ATen/ops/_foreach_max_ops.h>
#include <ATen/ops/_foreach_sign_ops.h>
#include <ATen/ops/_foreach_sign_ops.h>
#include <ATen/ops/_foreach_tan_ops.h>
#include <ATen/ops/_foreach_tan_ops.h>
#include <ATen/ops/_foreach_copy_ops.h>
#include <ATen/ops/_foreach_copy_ops.h>
#include <ATen/ops/searchsorted_ops.h>
#include <ATen/ops/searchsorted_ops.h>
#include <ATen/ops/searchsorted_ops.h>
#include <ATen/ops/searchsorted_ops.h>
#include <ATen/ops/smooth_l1_loss_ops.h>
#include <ATen/ops/smooth_l1_loss_ops.h>
#include <ATen/ops/elu_ops.h>
#include <ATen/ops/elu_ops.h>
#include <ATen/ops/elu_ops.h>
#include <ATen/ops/leaky_relu_backward_ops.h>
#include <ATen/ops/leaky_relu_backward_ops.h>
#include <ATen/ops/mkldnn_adaptive_avg_pool2d_ops.h>
#include <ATen/ops/mkldnn_adaptive_avg_pool2d_ops.h>
#include <ATen/ops/_adaptive_avg_pool2d_ops.h>
#include <ATen/ops/avg_pool3d_ops.h>
#include <ATen/ops/avg_pool3d_ops.h>
#include <ATen/ops/max_pool2d_with_indices_backward_ops.h>
#include <ATen/ops/max_pool2d_with_indices_backward_ops.h>
#include <ATen/ops/reflection_pad2d_ops.h>
#include <ATen/ops/reflection_pad2d_ops.h>
#include <ATen/ops/upsample_linear1d_backward_ops.h>
#include <ATen/ops/upsample_linear1d_backward_ops.h>
#include <ATen/ops/upsample_nearest1d_backward_ops.h>
#include <ATen/ops/upsample_nearest1d_backward_ops.h>
#include <ATen/ops/slow_conv_transpose3d_ops.h>
#include <ATen/ops/slow_conv_transpose3d_ops.h>
#include <ATen/ops/_add_batch_dim_ops.h>
#include <ATen/ops/special_psi_ops.h>
#include <ATen/ops/special_psi_ops.h>
#include <ATen/ops/special_i0e_ops.h>
#include <ATen/ops/special_i0e_ops.h>
#include <ATen/ops/special_logit_ops.h>
#include <ATen/ops/special_logit_ops.h>
#include <ATen/ops/fft_fft2_ops.h>
#include <ATen/ops/fft_fft2_ops.h>
#include <ATen/ops/linalg_lu_solve_ops.h>
#include <ATen/ops/linalg_lu_solve_ops.h>
#include <ATen/ops/linalg_inv_ops.h>
#include <ATen/ops/linalg_inv_ops.h>
#include <ATen/ops/outer_ops.h>
#include <ATen/ops/outer_ops.h>
#include <ATen/ops/_linalg_svd_ops.h>
#include <ATen/ops/_linalg_svd_ops.h>
#include <ATen/ops/linalg_qr_ops.h>
#include <ATen/ops/linalg_qr_ops.h>
#include <ATen/ops/_test_warn_in_autograd_ops.h>
#include <ATen/ops/select_copy_ops.h>
#include <ATen/ops/split_with_sizes_copy_ops.h>
#include <ATen/ops/col_indices_copy_ops.h>
#include <ATen/ops/unbind_copy_ops.h>
#include <ATen/ops/unbind_copy_ops.h>
#include <ATen/ops/split_with_sizes_copy_ops.h>
#include <ATen/ops/alias_copy_ops.h>
#include <ATen/ops/_jagged_to_padded_dense_forward_ops.h>
#include <ATen/ops/_scaled_dot_product_fused_attention_overrideable_ops.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_backward_ops.h>
#include <ATen/ops/_cudnn_attention_forward_ops.h>
#include <ATen/ops/_triton_scaled_dot_attention_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_scaled_modified_bessel_k1_ops.h>
#include <ATen/ops/special_scaled_modified_bessel_k1_ops.h>
#include <ATen/ops/_masked_scale_ops.h>
#include <ATen/ops/constant_pad_nd_ops.h>
#include <ATen/ops/convolution_overrideable_ops.h>
#include <ATen/ops/mps_convolution_transpose_backward_ops.h>
#include <ATen/ops/embedding_ops.h>
#include <ATen/ops/_embedding_bag_dense_backward_ops.h>
#include <ATen/ops/resize_ops.h>
#include <ATen/ops/resize_ops.h>
#include <ATen/ops/index_put_ops.h>
#include <ATen/ops/matmul_backward_ops.h>
#include <ATen/ops/mkldnn_max_pool2d_backward_ops.h>
#include <ATen/ops/median_ops.h>
#include <ATen/ops/nanmedian_ops.h>
#include <ATen/ops/miopen_rnn_backward_ops.h>
#include <ATen/ops/channel_shuffle_ops.h>
#include <ATen/ops/unsafe_split_with_sizes_ops.h>
#include <ATen/ops/_efficientzerotensor_ops.h>
#include <ATen/ops/poisson_ops.h>
#include <ATen/ops/sub_ops.h>
#include <ATen/ops/_to_sparse_bsr_ops.h>
#include <ATen/ops/_make_per_channel_quantized_tensor_ops.h>
#include <ATen/ops/masked_scatter_ops.h>
#include <ATen/ops/_masked_softmax_backward_ops.h>
#include <ATen/ops/bitwise_or_ops.h>
#include <ATen/ops/triu_indices_ops.h>
#include <ATen/ops/_foreach_clamp_max_ops.h>
#include <ATen/ops/_foreach_clamp_max_ops.h>
#include <ATen/ops/_foreach_clamp_max_ops.h>
#include <ATen/ops/_foreach_abs_ops.h>
#include <ATen/ops/_foreach_expm1_ops.h>
#include <ATen/ops/_foreach_log10_ops.h>
#include <ATen/ops/_foreach_max_ops.h>
#include <ATen/ops/_foreach_sign_ops.h>
#include <ATen/ops/_foreach_tan_ops.h>
#include <ATen/ops/_foreach_copy_ops.h>
#include <ATen/ops/_adaptive_avg_pool2d_ops.h>
#include <ATen/ops/_test_warn_in_autograd_ops.h>
#include <ATen/ops/select_copy_ops.h>
#include <ATen/ops/col_indices_copy_ops.h>
#include <ATen/ops/alias_copy_ops.h>
#include <ATen/ops/_triton_scaled_dot_attention_ops.h>
#endif

using namespace at;

namespace torch {

namespace TraceType {

namespace {
::std::vector<at::Tensor> align_tensors(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::align_tensors");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::align_tensors::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _functional_assert_async_msg(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view assert_msg, const at::Tensor & dep_token) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_functional_assert_async");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "assert_msg", assert_msg);
    jit::tracer::addInputs(node, "dep_token", dep_token);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_functional_assert_async_msg::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, assert_msg, dep_token);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _masked_scale(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, double scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_masked_scale");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_masked_scale::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _sobol_engine_draw(c10::DispatchKeySet ks, const at::Tensor & quasi, int64_t n, const at::Tensor & sobolstate, int64_t dimension, int64_t num_generated, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sobol_engine_draw");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "quasi", quasi);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "sobolstate", sobolstate);
    jit::tracer::addInputs(node, "dimension", dimension);
    jit::tracer::addInputs(node, "num_generated", num_generated);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_sobol_engine_draw::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), quasi, n, sobolstate, dimension, num_generated, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor alpha_dropout(c10::DispatchKeySet ks, const at::Tensor & input, double p, bool train) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::alpha_dropout");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::alpha_dropout::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & alpha_dropout_(c10::DispatchKeySet ks, at::Tensor & self, double p, bool train) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::alpha_dropout");
    } else {
      op_name = c10::Symbol::fromQualString("aten::alpha_dropout_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("alpha_dropout_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::alpha_dropout_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor view_as_complex(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::view_as_complex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::view_as_complex::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor chalf(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::chalf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::chalf::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor any_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::any");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::any_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor any_dims(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::any");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::any_dims::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & any_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::any");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("any_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::any_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & any_out_dims_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::any");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("any_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::any_dims_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor any_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::any");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::any_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & any_out_dimname_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::any");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("any_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::any_dimname_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor atleast_1d(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atleast_1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::atleast_1d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> atleast_1d_Sequence(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atleast_1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::atleast_1d_Sequence::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & copysign_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::copysign");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("copysign_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::copysign_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor copysign_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::copysign");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::copysign_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & copysign__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::copysign");
    } else {
      op_name = c10::Symbol::fromQualString("aten::copysign_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("copysign_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::copysign__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor copysign_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::copysign");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::copysign_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & copysign__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::copysign");
    } else {
      op_name = c10::Symbol::fromQualString("aten::copysign_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("copysign_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::copysign__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & copysign_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::copysign");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("copysign_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::copysign_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor constant_pad_nd(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef pad, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::constant_pad_nd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pad", pad);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::constant_pad_nd::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, pad, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor convolution_overrideable(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::convolution_overrideable");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::convolution_overrideable::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> cudnn_batch_norm(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_batch_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "exponential_average_factor", exponential_average_factor);
    jit::tracer::addInputs(node, "epsilon", epsilon);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::cudnn_batch_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> cudnn_batch_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_batch_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "exponential_average_factor", exponential_average_factor);
    jit::tracer::addInputs(node, "epsilon", epsilon);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
      jit::tracer::addInputs(node, "out3", out3);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cudnn_batch_norm_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cudnn_batch_norm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon, out0, out1, out2, out3);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
    jit::tracer::addOutput(node, out3);
  }
  return std::forward_as_tuple(out0, out1, out2, out3);
}
::std::tuple<at::Tensor,at::Tensor> mps_convolution_transpose_backward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, ::std::array<bool,2> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mps_convolution_transpose_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::mps_convolution_transpose_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor & fill_diagonal_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & fill_value, bool wrap) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::fill_diagonal");
    } else {
      op_name = c10::Symbol::fromQualString("aten::fill_diagonal_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "wrap", wrap);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fill_diagonal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fill_diagonal_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, fill_value, wrap);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor embedding(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::embedding");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "sparse", sparse);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::embedding::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight, indices, padding_idx, scale_grad_by_freq, sparse);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _embedding_bag_dense_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, c10::SymInt num_weights, bool scale_grad_by_freq, int64_t mode, const ::std::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_embedding_bag_dense_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "bag_size", bag_size);
    jit::tracer::addInputs(node, "maximum_indices", maximum_indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_embedding_bag_dense_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor erfc(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::erfc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::erfc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & erfc_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::erfc");
    } else {
      op_name = c10::Symbol::fromQualString("aten::erfc_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfc_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::erfc_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & erfc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::erfc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::erfc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor> _grid_sampler_2d_cpu_fallback_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_grid_sampler_2d_cpu_fallback_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_grid_sampler_2d_cpu_fallback_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input, grid, interpolation_mode, padding_mode, align_corners);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
void _cufft_set_plan_cache_max_size(c10::DispatchKeySet ks, at::DeviceIndex device_index, int64_t max_size) {
  at::_ops::_cufft_set_plan_cache_max_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), device_index, max_size);
}
at::Tensor & index_put_(c10::DispatchKeySet ks, at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::index_put");
    } else {
      op_name = c10::Symbol::fromQualString("aten::index_put_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_put_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_put_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, values, accumulate);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor index_put(c10::DispatchKeySet ks, const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_put");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_put::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, values, accumulate);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor instance_norm(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::instance_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "use_input_stats", use_input_stats);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enabled", cudnn_enabled);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::instance_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor isclose(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isclose");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rtol", rtol);
    jit::tracer::addInputs(node, "atol", atol);
    jit::tracer::addInputs(node, "equal_nan", equal_nan);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::isclose::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rtol, atol, equal_nan);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool is_floating_point(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::is_floating_point::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
bool is_complex(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::is_complex::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
int64_t _cslt_sparse_mm_search(c10::DispatchKeySet ks, const at::Tensor & compressed_A, const at::Tensor & dense_B, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & alpha, ::std::optional<at::ScalarType> out_dtype, bool transpose_result) {
  auto result =at::_ops::_cslt_sparse_mm_search::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), compressed_A, dense_B, bias, alpha, out_dtype, transpose_result);
  return result;
}
at::Tensor _sparse_semi_structured_addmm(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & mat1, const at::Tensor & mat1_meta, const at::Tensor & mat2, const at::Scalar & alpha, const at::Scalar & beta, ::std::optional<at::ScalarType> out_dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_semi_structured_addmm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat1_meta", mat1_meta);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "out_dtype", out_dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_semi_structured_addmm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, mat1, mat1_meta, mat2, alpha, beta, out_dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fbgemm_pack_gemm_matrix_fp16(c10::DispatchKeySet ks, const at::Tensor & input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fbgemm_pack_gemm_matrix_fp16");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fbgemm_pack_gemm_matrix_fp16::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _wrapped_quantized_linear_prepacked(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & input_scale, const at::Tensor & input_zero_point, const at::Tensor & packed_weight, const at::Tensor & output_scale, const at::Tensor & output_zero_point, int64_t out_channel) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_wrapped_quantized_linear_prepacked");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    jit::tracer::addInputs(node, "input_zero_point", input_zero_point);
    jit::tracer::addInputs(node, "packed_weight", packed_weight);
    jit::tracer::addInputs(node, "output_scale", output_scale);
    jit::tracer::addInputs(node, "output_zero_point", output_zero_point);
    jit::tracer::addInputs(node, "out_channel", out_channel);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_wrapped_quantized_linear_prepacked::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, input_scale, input_zero_point, packed_weight, output_scale, output_zero_point, out_channel);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> matmul_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & self, const at::Tensor & other, ::std::array<bool,2> mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::matmul_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::matmul_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, self, other, mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor mkldnn_max_pool2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_max_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mkldnn_max_pool2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor max_pool3d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool3d");
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
  auto result =at::_ops::max_pool3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor median(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::median");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::median::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> median_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::median");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::median_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> median_out_dim_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::median");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("median_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::median_dim_values::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor> median_names_dim(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::median");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::median_names_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> median_out_names_dim_values(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::median");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("median_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::median_names_dim_values::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
at::Tensor nanmedian(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nanmedian");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nanmedian::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> nanmedian_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nanmedian");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::nanmedian_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> nanmedian_out_dim_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nanmedian");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("nanmedian_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nanmedian_dim_values::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor> nanmedian_names_dim(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nanmedian");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::nanmedian_names_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> nanmedian_out_names_dim_values(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nanmedian");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("nanmedian_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nanmedian_names_dim_values::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,::std::vector<at::Tensor>> miopen_rnn_backward(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const ::std::optional<at::Tensor> & cx, const at::Tensor & output, const ::std::optional<at::Tensor> & grad_output, const ::std::optional<at::Tensor> & grad_hy, const ::std::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const ::std::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, ::std::array<bool,4> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_rnn_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "weight_buf", weight_buf);
    jit::tracer::addInputs(node, "hx", hx);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "grad_cy", grad_cy);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "dropout_state", dropout_state);
    jit::tracer::addInputs(node, "reserve", reserve);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::miopen_rnn_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
at::Tensor _convert_weight_to_int4pack(c10::DispatchKeySet ks, const at::Tensor & self, int64_t innerKTiles) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_convert_weight_to_int4pack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "innerKTiles", innerKTiles);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_convert_weight_to_int4pack::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, innerKTiles);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor multiply_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::multiply");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::multiply_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & multiply__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::multiply");
    } else {
      op_name = c10::Symbol::fromQualString("aten::multiply_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multiply_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::multiply__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & multiply_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::multiply");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multiply_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::multiply_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor multiply_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::multiply");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::multiply_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & multiply__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::multiply");
    } else {
      op_name = c10::Symbol::fromQualString("aten::multiply_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("multiply_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::multiply__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor batch_norm_elemt(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_elemt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::batch_norm_elemt::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, mean, invstd, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & batch_norm_elemt_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_elemt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "eps", eps);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("batch_norm_elemt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::batch_norm_elemt_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, mean, invstd, eps, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cdist(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, double p, ::std::optional<int64_t> compute_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cdist");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "compute_mode", compute_mode);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cdist::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x1, x2, p, compute_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor adjoint(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adjoint");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::adjoint::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor channel_shuffle(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt groups) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::channel_shuffle");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::channel_shuffle::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, groups);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor poisson_nll_loss(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & target, bool log_input, bool full, double eps, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::poisson_nll_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "log_input", log_input);
    jit::tracer::addInputs(node, "full", full);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::poisson_nll_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, target, log_input, full, eps, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randperm(c10::DispatchKeySet ks, c10::SymInt n, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randperm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randperm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randperm_generator(c10::DispatchKeySet ks, c10::SymInt n, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randperm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randperm_generator::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, generator, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & randperm_out_out(c10::DispatchKeySet ks, c10::SymInt n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randperm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randperm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randperm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randperm_out_generator_out(c10::DispatchKeySet ks, c10::SymInt n, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randperm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("randperm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randperm_generator_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor negative(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::negative");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::negative::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & negative_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::negative");
    } else {
      op_name = c10::Symbol::fromQualString("aten::negative_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("negative_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::negative_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & negative_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::negative");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("negative_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::negative_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor infinitely_differentiable_gelu_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::infinitely_differentiable_gelu_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::infinitely_differentiable_gelu_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hardshrink_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardshrink_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardshrink_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardshrink_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, self, lambd, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor hardshrink_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardshrink_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "lambd", lambd);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hardshrink_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, self, lambd);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor slice_inverse(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slice_inverse");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::slice_inverse::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, dim, start, end, step);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor smm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::smm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::smm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> unsafe_split_with_sizes(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unsafe_split_with_sizes");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "split_sizes", split_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unsafe_split_with_sizes::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, split_sizes, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _chunk_cat(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, int64_t num_chunks) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_chunk_cat");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "num_chunks", num_chunks);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_chunk_cat::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, num_chunks);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _chunk_cat_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, int64_t num_chunks, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_chunk_cat");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "num_chunks", num_chunks);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_chunk_cat_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_chunk_cat_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, num_chunks, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor dstack(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dstack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::dstack::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & dstack_out_out(c10::DispatchKeySet ks, at::TensorList tensors, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dstack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("dstack_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::dstack_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor tan(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::tan::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & tan_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::tan");
    } else {
      op_name = c10::Symbol::fromQualString("aten::tan_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tan_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::tan_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & tan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tan_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::tan_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _nested_get_values(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_get_values");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_get_values::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor unsqueeze(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unsqueeze");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unsqueeze::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & unsqueeze_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::unsqueeze");
    } else {
      op_name = c10::Symbol::fromQualString("aten::unsqueeze_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("unsqueeze_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::unsqueeze_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor _efficientzerotensor(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_efficientzerotensor");
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
  auto result =at::_ops::_efficientzerotensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor poisson(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::poisson");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::poisson::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & sub_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sub");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sub_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor sub_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sub");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sub_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & sub__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::sub");
    } else {
      op_name = c10::Symbol::fromQualString("aten::sub_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sub__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor sub_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sub");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sub_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & sub__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::sub");
    } else {
      op_name = c10::Symbol::fromQualString("aten::sub_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sub__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & subtract_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::subtract");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("subtract_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::subtract_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor subtract_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::subtract");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::subtract_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & subtract__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::subtract");
    } else {
      op_name = c10::Symbol::fromQualString("aten::subtract_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("subtract_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::subtract__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor subtract_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::subtract");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::subtract_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & subtract__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::subtract");
    } else {
      op_name = c10::Symbol::fromQualString("aten::subtract_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("subtract_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::subtract__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & _addmm_activation_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, bool use_gelu, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_addmm_activation");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "use_gelu", use_gelu);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_addmm_activation_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_addmm_activation_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat1, mat2, beta, alpha, use_gelu, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _addmm_activation(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, bool use_gelu) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_addmm_activation");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat1", mat1);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "use_gelu", use_gelu);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_addmm_activation::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat1, mat2, beta, alpha, use_gelu);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sparse_compressed_tensor_comp_plain_value_size(c10::DispatchKeySet ks, const at::Tensor & compressed_indices, const at::Tensor & plain_indices, const at::Tensor & values, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_compressed_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "compressed_indices", compressed_indices);
    jit::tracer::addInputs(node, "plain_indices", plain_indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sparse_compressed_tensor_comp_plain_value_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), compressed_indices, plain_indices, values, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sparse_bsr_tensor_crow_col_value_size(c10::DispatchKeySet ks, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_bsr_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "crow_indices", crow_indices);
    jit::tracer::addInputs(node, "col_indices", col_indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sparse_bsr_tensor_crow_col_value_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), crow_indices, col_indices, values, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sparse_compressed_tensor_comp_plain_value(c10::DispatchKeySet ks, const at::Tensor & compressed_indices, const at::Tensor & plain_indices, const at::Tensor & values, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_compressed_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "compressed_indices", compressed_indices);
    jit::tracer::addInputs(node, "plain_indices", plain_indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sparse_compressed_tensor_comp_plain_value::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), compressed_indices, plain_indices, values, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sparse_bsr_tensor_crow_col_value(c10::DispatchKeySet ks, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_bsr_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "crow_indices", crow_indices);
    jit::tracer::addInputs(node, "col_indices", col_indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sparse_bsr_tensor_crow_col_value::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), crow_indices, col_indices, values, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _validate_sparse_compressed_tensor_args(c10::DispatchKeySet ks, const at::Tensor & compressed_indices, const at::Tensor & plain_indices, const at::Tensor & values, at::IntArrayRef size, at::Layout layout, ::std::optional<bool> check_pinning) {
  at::_ops::_validate_sparse_compressed_tensor_args::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), compressed_indices, plain_indices, values, size, layout, check_pinning);
}
at::Tensor to_dense(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<bool> masked_grad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::to_dense");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "masked_grad", masked_grad);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::to_dense::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, masked_grad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t _dimI(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::_dimI::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
int64_t _nnz(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::_nnz::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor ccol_indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ccol_indices");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ccol_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor to_sparse_csr(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::to_sparse_csr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::to_sparse_csr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _to_sparse_bsr(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_bsr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "blocksize", blocksize);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_to_sparse_bsr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, blocksize, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t q_per_channel_axis(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::q_per_channel_axis::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor _make_per_channel_quantized_tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_make_per_channel_quantized_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_make_per_channel_quantized_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, axis);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fake_quantize_per_tensor_affine_cachemask_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine_cachemask_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fake_quantize_per_tensor_affine_cachemask_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fake_quantize_per_channel_affine_cachemask_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fake_quantize_per_channel_affine_cachemask_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fake_quantize_per_channel_affine_cachemask_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::ScalarType result_type_Tensor(c10::DispatchKeySet ks, const at::Tensor & tensor, const at::Tensor & other) {
  auto result =at::_ops::result_type_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensor, other);
  return result;
}
at::ScalarType result_type_Scalar(c10::DispatchKeySet ks, const at::Tensor & tensor, const at::Scalar & other) {
  auto result =at::_ops::result_type_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensor, other);
  return result;
}
at::ScalarType result_type_Scalar_Tensor(c10::DispatchKeySet ks, const at::Scalar & scalar, const at::Tensor & tensor) {
  auto result =at::_ops::result_type_Scalar_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), scalar, tensor);
  return result;
}
at::ScalarType result_type_Scalar_Scalar(c10::DispatchKeySet ks, const at::Scalar & scalar1, const at::Scalar & scalar2) {
  auto result =at::_ops::result_type_Scalar_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), scalar1, scalar2);
  return result;
}
::std::tuple<at::Tensor,at::Tensor> lstm_cell(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) {
  auto [result0, result1] =at::_ops::lstm_cell::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, w_ih, w_hh, b_ih, b_hh);
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor & masked_scatter_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::masked_scatter");
    } else {
      op_name = c10::Symbol::fromQualString("aten::masked_scatter_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::masked_scatter_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, source);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor masked_scatter(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::masked_scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::masked_scatter::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, source);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _masked_softmax_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & mask, ::std::optional<int64_t> dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_masked_softmax_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_masked_softmax_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output, mask, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & index_add_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_add");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "alpha", alpha);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_add_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_add_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & index_add_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::index_add");
    } else {
      op_name = c10::Symbol::fromQualString("aten::index_add_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_add_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_add_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor index_add(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_add");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_add::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor index_add_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_add");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_add_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, source, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & bitwise_or_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_or");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_or_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_or_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & bitwise_or_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_or");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_or_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_or_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor bitwise_or_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_or");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bitwise_or_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor bitwise_or_Scalar_Tensor(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_or");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bitwise_or_Scalar_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor bitwise_or_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_or");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bitwise_or_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & bitwise_or__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::bitwise_or");
    } else {
      op_name = c10::Symbol::fromQualString("aten::bitwise_or_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_or_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_or__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & bitwise_or__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::bitwise_or");
    } else {
      op_name = c10::Symbol::fromQualString("aten::bitwise_or_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_or_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_or__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor triu_indices(c10::DispatchKeySet ks, int64_t row, int64_t col, int64_t offset, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::triu_indices");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "row", row);
    jit::tracer::addInputs(node, "col", col);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::triu_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), row, col, offset, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & greater_equal_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::greater_equal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("greater_equal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::greater_equal_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor greater_equal_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::greater_equal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::greater_equal_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & greater_equal_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::greater_equal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("greater_equal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::greater_equal_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor greater_equal_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::greater_equal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::greater_equal_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & greater_equal__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::greater_equal");
    } else {
      op_name = c10::Symbol::fromQualString("aten::greater_equal_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("greater_equal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::greater_equal__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & greater_equal__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::greater_equal");
    } else {
      op_name = c10::Symbol::fromQualString("aten::greater_equal_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("greater_equal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::greater_equal__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & take_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & index, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::take");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("take_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::take_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, index, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor take(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::take");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::take::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor index_select_backward(c10::DispatchKeySet ks, const at::Tensor & grad, c10::SymIntArrayRef self_sizes, int64_t dim, const at::Tensor & index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_select_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self_sizes", self_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::index_select_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, self_sizes, dim, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor argwhere(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::argwhere");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::argwhere::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> geqrf_out_a(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & a, at::Tensor & tau) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::geqrf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "a", a);
      jit::tracer::addInputs(node, "tau", tau);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("geqrf_out", a);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::geqrf_a::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, a, tau);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, a);
    jit::tracer::addOutput(node, tau);
  }
  return std::forward_as_tuple(a, tau);
}
::std::tuple<at::Tensor,at::Tensor> geqrf(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::geqrf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [a, tau] =at::_ops::geqrf::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, a);
    jit::tracer::addOutput(node, tau);
  }
  return std::make_tuple(std::move(a), std::move(tau));
}
at::Tensor orgqr(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & input2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::orgqr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::orgqr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, input2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & orgqr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & input2, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::orgqr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "input2", input2);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("orgqr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::orgqr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, input2, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor erfinv(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::erfinv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::erfinv::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & erfinv_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::erfinv");
    } else {
      op_name = c10::Symbol::fromQualString("aten::erfinv_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfinv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::erfinv_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & erfinv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::erfinv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("erfinv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::erfinv_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor signbit(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::signbit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::signbit::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & signbit_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::signbit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("signbit_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::signbit_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & fmod_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fmod");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fmod_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fmod_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fmod");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fmod_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fmod__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::fmod");
    } else {
      op_name = c10::Symbol::fromQualString("aten::fmod_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fmod__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & fmod_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fmod");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fmod_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fmod_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fmod");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fmod_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fmod__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::fmod");
    } else {
      op_name = c10::Symbol::fromQualString("aten::fmod_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmod_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fmod__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor any(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::any");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::any::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & any_out_all_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::any");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("any_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::any_all_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::vector<at::Tensor> _foreach_clamp_max_Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_clamp_max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scalar", scalar);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_clamp_max_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_clamp_max__Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar) {
  at::_ops::_foreach_clamp_max__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar);
}
::std::vector<at::Tensor> _foreach_clamp_max_List(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_clamp_max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_clamp_max_List::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_clamp_max__List(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other) {
  at::_ops::_foreach_clamp_max__List::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
}
::std::vector<at::Tensor> _foreach_clamp_max_ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_clamp_max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scalars", scalars);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_clamp_max_ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_clamp_max__ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  at::_ops::_foreach_clamp_max__ScalarList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars);
}
::std::vector<at::Tensor> _foreach_abs(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_abs");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_abs::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_abs_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_abs_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_expm1(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_expm1");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_expm1::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_expm1_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_expm1_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_log10(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_log10");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_log10::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_log10_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_log10_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_max(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_max::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> _foreach_sign(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_sign");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_sign::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_sign_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_sign_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_tan(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_tan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_tan::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_tan_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_tan_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
void _foreach_copy_(c10::DispatchKeySet ks, at::TensorList self, at::TensorList src, bool non_blocking) {
  at::_ops::_foreach_copy_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, non_blocking);
}
::std::vector<at::Tensor> _foreach_copy(c10::DispatchKeySet ks, at::TensorList self, at::TensorList src, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto self_out =at::_ops::_foreach_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self_out);
  }
  return self_out;
}
at::Tensor searchsorted_Tensor(c10::DispatchKeySet ks, const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, ::std::optional<c10::string_view> side, const ::std::optional<at::Tensor> & sorter) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::searchsorted");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sorted_sequence", sorted_sequence);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    jit::tracer::addInputs(node, "side", side);
    jit::tracer::addInputs(node, "sorter", sorter);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::searchsorted_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), sorted_sequence, self, out_int32, right, side, sorter);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & searchsorted_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, ::std::optional<c10::string_view> side, const ::std::optional<at::Tensor> & sorter, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::searchsorted");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sorted_sequence", sorted_sequence);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    jit::tracer::addInputs(node, "side", side);
    jit::tracer::addInputs(node, "sorter", sorter);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("searchsorted_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::searchsorted_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), sorted_sequence, self, out_int32, right, side, sorter, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor searchsorted_Scalar(c10::DispatchKeySet ks, const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right, ::std::optional<c10::string_view> side, const ::std::optional<at::Tensor> & sorter) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::searchsorted");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sorted_sequence", sorted_sequence);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    jit::tracer::addInputs(node, "side", side);
    jit::tracer::addInputs(node, "sorter", sorter);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::searchsorted_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), sorted_sequence, self, out_int32, right, side, sorter);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & searchsorted_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right, ::std::optional<c10::string_view> side, const ::std::optional<at::Tensor> & sorter, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::searchsorted");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "sorted_sequence", sorted_sequence);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    jit::tracer::addInputs(node, "side", side);
    jit::tracer::addInputs(node, "sorter", sorter);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("searchsorted_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::searchsorted_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), sorted_sequence, self, out_int32, right, side, sorter, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & smooth_l1_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::smooth_l1_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "beta", beta);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("smooth_l1_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::smooth_l1_loss_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction, beta, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor smooth_l1_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::smooth_l1_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "beta", beta);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::smooth_l1_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction, beta);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & elu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::elu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("elu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::elu_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, alpha, scale, input_scale, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor elu(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::elu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::elu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, alpha, scale, input_scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & elu_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::elu");
    } else {
      op_name = c10::Symbol::fromQualString("aten::elu_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "input_scale", input_scale);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("elu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::elu_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, alpha, scale, input_scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & leaky_relu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::leaky_relu_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    jit::tracer::addInputs(node, "self_is_result", self_is_result);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("leaky_relu_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::leaky_relu_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, negative_slope, self_is_result, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor leaky_relu_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::leaky_relu_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    jit::tracer::addInputs(node, "self_is_result", self_is_result);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::leaky_relu_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, negative_slope, self_is_result);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor mkldnn_adaptive_avg_pool2d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_adaptive_avg_pool2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mkldnn_adaptive_avg_pool2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & mkldnn_adaptive_avg_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_adaptive_avg_pool2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mkldnn_adaptive_avg_pool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mkldnn_adaptive_avg_pool2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _adaptive_avg_pool2d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_adaptive_avg_pool2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_adaptive_avg_pool2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & avg_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::avg_pool3d");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::avg_pool3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor avg_pool3d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::avg_pool3d");
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
  auto result =at::_ops::avg_pool3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & max_pool2d_with_indices_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool2d_with_indices_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool2d_with_indices_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::max_pool2d_with_indices_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor max_pool2d_with_indices_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool2d_with_indices_backward");
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
  auto result =at::_ops::max_pool2d_with_indices_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & reflection_pad2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reflection_pad2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::reflection_pad2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor reflection_pad2d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reflection_pad2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::reflection_pad2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & upsample_linear1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_linear1d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales", scales);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_linear1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::upsample_linear1d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, align_corners, scales, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor upsample_linear1d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_linear1d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::upsample_linear1d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, align_corners, scales);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & upsample_nearest1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_nearest1d_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::upsample_nearest1d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, scales, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor upsample_nearest1d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_nearest1d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales", scales);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::upsample_nearest1d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, scales);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & slow_conv_transpose3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef dilation, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slow_conv_transpose3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("slow_conv_transpose3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::slow_conv_transpose3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor slow_conv_transpose3d(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef dilation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slow_conv_transpose3d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::slow_conv_transpose3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _add_batch_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t batch_dim, int64_t level) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_add_batch_dim");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch_dim", batch_dim);
    jit::tracer::addInputs(node, "level", level);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_add_batch_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, batch_dim, level);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_psi(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_psi");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_psi::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_psi_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_psi");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_psi_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_psi_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_i0e(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_i0e");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_i0e::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_i0e_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_i0e");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_i0e_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_i0e_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_logit(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<double> eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_logit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_logit::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_logit_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<double> eps, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_logit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "eps", eps);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_logit_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_logit_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, eps, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_fft2(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fft2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_fft2::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_fft2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fft2");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_fft2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_fft2_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linalg_lu_solve(c10::DispatchKeySet ks, const at::Tensor & LU, const at::Tensor & pivots, const at::Tensor & B, bool left, bool adjoint) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_lu_solve");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "LU", LU);
    jit::tracer::addInputs(node, "pivots", pivots);
    jit::tracer::addInputs(node, "B", B);
    jit::tracer::addInputs(node, "left", left);
    jit::tracer::addInputs(node, "adjoint", adjoint);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_lu_solve::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), LU, pivots, B, left, adjoint);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_lu_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & LU, const at::Tensor & pivots, const at::Tensor & B, bool left, bool adjoint, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_lu_solve");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "LU", LU);
    jit::tracer::addInputs(node, "pivots", pivots);
    jit::tracer::addInputs(node, "B", B);
    jit::tracer::addInputs(node, "left", left);
    jit::tracer::addInputs(node, "adjoint", adjoint);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_lu_solve_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_lu_solve_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), LU, pivots, B, left, adjoint, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor linalg_inv(c10::DispatchKeySet ks, const at::Tensor & A) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_inv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_inv::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_inv_out_out(c10::DispatchKeySet ks, const at::Tensor & A, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_inv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_inv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_inv_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor outer(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::outer");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec2", vec2);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::outer::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & outer_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec2, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::outer");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec2", vec2);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("outer_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::outer_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec2, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _linalg_svd(c10::DispatchKeySet ks, const at::Tensor & A, bool full_matrices, bool compute_uv, ::std::optional<c10::string_view> driver) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_linalg_svd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "full_matrices", full_matrices);
    jit::tracer::addInputs(node, "compute_uv", compute_uv);
    jit::tracer::addInputs(node, "driver", driver);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [U, S, Vh] =at::_ops::_linalg_svd::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, full_matrices, compute_uv, driver);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, U);
    jit::tracer::addOutput(node, S);
    jit::tracer::addOutput(node, Vh);
  }
  return std::make_tuple(std::move(U), std::move(S), std::move(Vh));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _linalg_svd_out_U(c10::DispatchKeySet ks, const at::Tensor & A, bool full_matrices, bool compute_uv, ::std::optional<c10::string_view> driver, at::Tensor & U, at::Tensor & S, at::Tensor & Vh) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_linalg_svd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "full_matrices", full_matrices);
    jit::tracer::addInputs(node, "compute_uv", compute_uv);
    jit::tracer::addInputs(node, "driver", driver);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "U", U);
      jit::tracer::addInputs(node, "S", S);
      jit::tracer::addInputs(node, "Vh", Vh);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_linalg_svd_out", U);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_linalg_svd_U::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, full_matrices, compute_uv, driver, U, S, Vh);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, U);
    jit::tracer::addOutput(node, S);
    jit::tracer::addOutput(node, Vh);
  }
  return std::forward_as_tuple(U, S, Vh);
}
::std::tuple<at::Tensor,at::Tensor> linalg_qr(c10::DispatchKeySet ks, const at::Tensor & A, c10::string_view mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_qr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "mode", mode);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [Q, R] =at::_ops::linalg_qr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, Q);
    jit::tracer::addOutput(node, R);
  }
  return std::make_tuple(std::move(Q), std::move(R));
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_qr_out_out(c10::DispatchKeySet ks, const at::Tensor & A, c10::string_view mode, at::Tensor & Q, at::Tensor & R) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_qr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "mode", mode);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "Q", Q);
      jit::tracer::addInputs(node, "R", R);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_qr_out", Q);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_qr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, mode, Q, R);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, Q);
    jit::tracer::addOutput(node, R);
  }
  return std::forward_as_tuple(Q, R);
}
at::Tensor _test_warn_in_autograd(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_warn_in_autograd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_test_warn_in_autograd::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor select_copy_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, c10::SymInt index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::select_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::select_copy_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> split_with_sizes_copy(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::split_with_sizes_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "split_sizes", split_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::split_with_sizes_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, split_sizes, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor col_indices_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::col_indices_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::col_indices_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> unbind_copy_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unbind_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unbind_copy_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void unbind_copy_out_int_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::TensorList out) {
  at::_ops::unbind_copy_int_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, out);
}
void split_with_sizes_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim, at::TensorList out) {
  at::_ops::split_with_sizes_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, split_sizes, dim, out);
}
at::Tensor alias_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::alias_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::alias_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _jagged_to_padded_dense_forward(c10::DispatchKeySet ks, const at::Tensor & values, at::TensorList offsets, c10::SymIntArrayRef max_lengths, double padding_value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_jagged_to_padded_dense_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "max_lengths", max_lengths);
    jit::tracer::addInputs(node, "padding_value", padding_value);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_jagged_to_padded_dense_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), values, offsets, max_lengths, padding_value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::SymInt,c10::SymInt,at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_fused_attention_overrideable(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_bias, double dropout_p, bool is_causal, bool return_debug_mask, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_dot_product_fused_attention_overrideable");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "attn_bias", attn_bias);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "return_debug_mask", return_debug_mask);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, philox_seed, philox_offset, debug_attn_mask] =at::_ops::_scaled_dot_product_fused_attention_overrideable::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, attn_bias, dropout_p, is_causal, return_debug_mask, scale);
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
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_flash_attention_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & out, const at::Tensor & logsumexp, const at::Tensor & cum_seq_q, const at::Tensor & cum_seq_k, c10::SymInt max_q, c10::SymInt max_k, double dropout_p, bool is_causal, const at::Tensor & philox_seed, const at::Tensor & philox_offset, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_dot_product_flash_attention_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "out", out);
    jit::tracer::addInputs(node, "logsumexp", logsumexp);
    jit::tracer::addInputs(node, "cum_seq_q", cum_seq_q);
    jit::tracer::addInputs(node, "cum_seq_k", cum_seq_k);
    jit::tracer::addInputs(node, "max_q", max_q);
    jit::tracer::addInputs(node, "max_k", max_k);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "philox_seed", philox_seed);
    jit::tracer::addInputs(node, "philox_offset", philox_offset);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [grad_query, grad_key, grad_value] =at::_ops::_scaled_dot_product_flash_attention_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, query, key, value, out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, philox_seed, philox_offset, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_query);
    jit::tracer::addOutput(node, grad_key);
    jit::tracer::addOutput(node, grad_value);
  }
  return std::make_tuple(std::move(grad_query), std::move(grad_key), std::move(grad_value));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,c10::SymInt,c10::SymInt,at::Tensor,at::Tensor,at::Tensor> _cudnn_attention_forward(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_bias, const ::std::optional<at::Tensor> & cum_seq_q, const ::std::optional<at::Tensor> & cum_seq_k, c10::SymInt max_q, c10::SymInt max_k, bool compute_log_sumexp, double dropout_p, bool is_causal, bool return_debug_mask, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cudnn_attention_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "attn_bias", attn_bias);
    jit::tracer::addInputs(node, "cum_seq_q", cum_seq_q);
    jit::tracer::addInputs(node, "cum_seq_k", cum_seq_k);
    jit::tracer::addInputs(node, "max_q", max_q);
    jit::tracer::addInputs(node, "max_k", max_k);
    jit::tracer::addInputs(node, "compute_log_sumexp", compute_log_sumexp);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "return_debug_mask", return_debug_mask);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [output, logsumexp, cum_seq_q_return, cum_seq_k_return, max_q_return, max_k_return, philox_seed, philox_offset, debug_attn_mask] =at::_ops::_cudnn_attention_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, attn_bias, cum_seq_q, cum_seq_k, max_q, max_k, compute_log_sumexp, dropout_p, is_causal, return_debug_mask, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, logsumexp);
    jit::tracer::addOutput(node, cum_seq_q_return);
    jit::tracer::addOutput(node, cum_seq_k_return);
    jit::tracer::addOutput(node, max_q_return);
    jit::tracer::addOutput(node, max_k_return);
    jit::tracer::addOutput(node, philox_seed);
    jit::tracer::addOutput(node, philox_offset);
    jit::tracer::addOutput(node, debug_attn_mask);
  }
  return std::make_tuple(std::move(output), std::move(logsumexp), std::move(cum_seq_q_return), std::move(cum_seq_k_return), std::move(max_q_return), std::move(max_k_return), std::move(philox_seed), std::move(philox_offset), std::move(debug_attn_mask));
}
at::Tensor _triton_scaled_dot_attention(c10::DispatchKeySet ks, const at::Tensor & q, const at::Tensor & k, const at::Tensor & v, double dropout_p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_triton_scaled_dot_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_triton_scaled_dot_attention::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), q, k, v, dropout_p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_chebyshev_polynomial_t(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_chebyshev_polynomial_t::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_chebyshev_polynomial_t_x_scalar(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_chebyshev_polynomial_t_x_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_chebyshev_polynomial_t_n_scalar(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_chebyshev_polynomial_t_n_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_chebyshev_polynomial_t_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_chebyshev_polynomial_t_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_chebyshev_polynomial_t_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_chebyshev_polynomial_t_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_chebyshev_polynomial_t_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_chebyshev_polynomial_t_x_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_chebyshev_polynomial_t_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_chebyshev_polynomial_t_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_chebyshev_polynomial_t_n_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_scaled_modified_bessel_k1(c10::DispatchKeySet ks, const at::Tensor & x) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_scaled_modified_bessel_k1");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_scaled_modified_bessel_k1::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_scaled_modified_bessel_k1_out_out(c10::DispatchKeySet ks, const at::Tensor & x, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_scaled_modified_bessel_k1");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_scaled_modified_bessel_k1_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_scaled_modified_bessel_k1_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _masked_scale_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, double scale, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_masked_scale");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "scale", scale);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_masked_scale_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_masked_scale_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, scale, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & constant_pad_nd_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef pad, const at::Scalar & value, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::constant_pad_nd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pad", pad);
    jit::tracer::addInputs(node, "value", value);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("constant_pad_nd_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::constant_pad_nd_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, pad, value, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & convolution_overrideable_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::convolution_overrideable");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "transposed", transposed);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "groups", groups);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("convolution_overrideable_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::convolution_overrideable_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> mps_convolution_transpose_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grad_output, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, ::std::array<bool,2> output_mask, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mps_convolution_transpose_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "output_mask", output_mask);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mps_convolution_transpose_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mps_convolution_transpose_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grad_output, weight, padding, output_padding, stride, dilation, groups, output_mask, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & embedding_out_out(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, c10::SymInt padding_idx, bool scale_grad_by_freq, bool sparse, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::embedding");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "sparse", sparse);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("embedding_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::embedding_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight, indices, padding_idx, scale_grad_by_freq, sparse, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _embedding_bag_dense_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, c10::SymInt num_weights, bool scale_grad_by_freq, int64_t mode, const ::std::optional<at::Tensor> & per_sample_weights, int64_t padding_idx, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_embedding_bag_dense_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "bag_size", bag_size);
    jit::tracer::addInputs(node, "maximum_indices", maximum_indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_embedding_bag_dense_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_embedding_bag_dense_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, indices, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, per_sample_weights, padding_idx, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
const at::Tensor & resize_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, ::std::optional<at::MemoryFormat> memory_format, const at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::resize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("resize_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::resize_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor resize(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::resize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::resize::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & index_put_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::index_put");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "accumulate", accumulate);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("index_put_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::index_put_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, values, accumulate, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> matmul_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & self, const at::Tensor & other, ::std::array<bool,2> mask, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::matmul_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "mask", mask);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("matmul_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::matmul_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, self, other, mask, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & mkldnn_max_pool2d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & input, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_max_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mkldnn_max_pool2d_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mkldnn_max_pool2d_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output, input, kernel_size, stride, padding, dilation, ceil_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & median_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::median");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("median_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::median_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & nanmedian_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nanmedian");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nanmedian_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nanmedian_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void miopen_rnn_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & weight_buf, const at::Tensor & hx, const ::std::optional<at::Tensor> & cx, const at::Tensor & output, const ::std::optional<at::Tensor> & grad_output, const ::std::optional<at::Tensor> & grad_hy, const ::std::optional<at::Tensor> & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const ::std::optional<at::Tensor> & dropout_state, const at::Tensor & reserve, ::std::array<bool,4> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::TensorList out3) {
  at::_ops::miopen_rnn_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask, out0, out1, out2, out3);
}
at::Tensor & channel_shuffle_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt groups, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::channel_shuffle");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "groups", groups);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("channel_shuffle_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::channel_shuffle_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, groups, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void unsafe_split_with_sizes_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef split_sizes, int64_t dim, at::TensorList out) {
  at::_ops::unsafe_split_with_sizes_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, split_sizes, dim, out);
}
at::Tensor & _efficientzerotensor_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_efficientzerotensor");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_efficientzerotensor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_efficientzerotensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & poisson_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::poisson");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("poisson_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::poisson_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & sub_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sub");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("sub_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sub_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _to_sparse_bsr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_bsr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "blocksize", blocksize);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_to_sparse_bsr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_to_sparse_bsr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, blocksize, dense_dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _make_per_channel_quantized_tensor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_make_per_channel_quantized_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_make_per_channel_quantized_tensor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_make_per_channel_quantized_tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, axis, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & masked_scatter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::masked_scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "source", source);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_scatter_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::masked_scatter_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, source, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _masked_softmax_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, const at::Tensor & mask, ::std::optional<int64_t> dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_masked_softmax_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output", output);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_masked_softmax_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_masked_softmax_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output, mask, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & bitwise_or_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_or");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_or_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_or_Scalar_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & triu_indices_out_out(c10::DispatchKeySet ks, int64_t row, int64_t col, int64_t offset, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::triu_indices");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "row", row);
    jit::tracer::addInputs(node, "col", col);
    jit::tracer::addInputs(node, "offset", offset);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("triu_indices_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::triu_indices_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), row, col, offset, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _foreach_clamp_max_out_Scalar_out(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar, at::TensorList out) {
  at::_ops::_foreach_clamp_max_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalar, out);
}
void _foreach_clamp_max_out_List_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other, at::TensorList out) {
  at::_ops::_foreach_clamp_max_List_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
}
void _foreach_clamp_max_out_ScalarList_out(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars, at::TensorList out) {
  at::_ops::_foreach_clamp_max_ScalarList_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scalars, out);
}
void _foreach_abs_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_abs_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_expm1_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_expm1_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_log10_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_log10_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_max_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_max_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_sign_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_sign_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_tan_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_tan_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_copy_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList src, bool non_blocking, at::TensorList out) {
  at::_ops::_foreach_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, non_blocking, out);
}
at::Tensor & _adaptive_avg_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_adaptive_avg_pool2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_adaptive_avg_pool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_adaptive_avg_pool2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _test_warn_in_autograd_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_warn_in_autograd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_test_warn_in_autograd_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_test_warn_in_autograd_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & select_copy_out_int_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, c10::SymInt index, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::select_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("select_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::select_copy_int_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & col_indices_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::col_indices_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("col_indices_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::col_indices_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & alias_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::alias_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("alias_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::alias_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _triton_scaled_dot_attention_out_out(c10::DispatchKeySet ks, const at::Tensor & q, const at::Tensor & k, const at::Tensor & v, double dropout_p, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_triton_scaled_dot_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "v", v);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_triton_scaled_dot_attention_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_triton_scaled_dot_attention_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), q, k, v, dropout_p, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
}  // namespace
}  // namespace TraceType

namespace {

TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  m.impl("align_tensors",
         TORCH_FN(TraceType::align_tensors)
  );
  m.impl("_functional_assert_async.msg",
         TORCH_FN(TraceType::_functional_assert_async_msg)
  );
  m.impl("_masked_scale",
         TORCH_FN(TraceType::_masked_scale)
  );
  m.impl("_sobol_engine_draw",
         TORCH_FN(TraceType::_sobol_engine_draw)
  );
  m.impl("alpha_dropout",
         TORCH_FN(TraceType::alpha_dropout)
  );
  m.impl("alpha_dropout_",
         TORCH_FN(TraceType::alpha_dropout_)
  );
  m.impl("view_as_complex",
         TORCH_FN(TraceType::view_as_complex)
  );
  m.impl("chalf",
         TORCH_FN(TraceType::chalf)
  );
  m.impl("any.dim",
         TORCH_FN(TraceType::any_dim)
  );
  m.impl("any.dims",
         TORCH_FN(TraceType::any_dims)
  );
  m.impl("any.out",
         TORCH_FN(TraceType::any_out_out)
  );
  m.impl("any.dims_out",
         TORCH_FN(TraceType::any_out_dims_out)
  );
  m.impl("any.dimname",
         TORCH_FN(TraceType::any_dimname)
  );
  m.impl("any.dimname_out",
         TORCH_FN(TraceType::any_out_dimname_out)
  );
  m.impl("atleast_1d",
         TORCH_FN(TraceType::atleast_1d)
  );
  m.impl("atleast_1d.Sequence",
         TORCH_FN(TraceType::atleast_1d_Sequence)
  );
  m.impl("copysign.out",
         TORCH_FN(TraceType::copysign_out_out)
  );
  m.impl("copysign.Tensor",
         TORCH_FN(TraceType::copysign_Tensor)
  );
  m.impl("copysign_.Tensor",
         TORCH_FN(TraceType::copysign__Tensor)
  );
  m.impl("copysign.Scalar",
         TORCH_FN(TraceType::copysign_Scalar)
  );
  m.impl("copysign_.Scalar",
         TORCH_FN(TraceType::copysign__Scalar)
  );
  m.impl("copysign.Scalar_out",
         TORCH_FN(TraceType::copysign_out_Scalar_out)
  );
  m.impl("constant_pad_nd",
         TORCH_FN(TraceType::constant_pad_nd)
  );
  m.impl("convolution_overrideable",
         TORCH_FN(TraceType::convolution_overrideable)
  );
  m.impl("cudnn_batch_norm",
         TORCH_FN(TraceType::cudnn_batch_norm)
  );
  m.impl("cudnn_batch_norm.out",
         TORCH_FN(TraceType::cudnn_batch_norm_out_out)
  );
  m.impl("mps_convolution_transpose_backward",
         TORCH_FN(TraceType::mps_convolution_transpose_backward)
  );
  m.impl("fill_diagonal_",
         TORCH_FN(TraceType::fill_diagonal_)
  );
  m.impl("embedding",
         TORCH_FN(TraceType::embedding)
  );
  m.impl("_embedding_bag_dense_backward",
         TORCH_FN(TraceType::_embedding_bag_dense_backward)
  );
  m.impl("erfc",
         TORCH_FN(TraceType::erfc)
  );
  m.impl("erfc_",
         TORCH_FN(TraceType::erfc_)
  );
  m.impl("erfc.out",
         TORCH_FN(TraceType::erfc_out_out)
  );
  m.impl("_grid_sampler_2d_cpu_fallback_backward",
         TORCH_FN(TraceType::_grid_sampler_2d_cpu_fallback_backward)
  );
  m.impl("_cufft_set_plan_cache_max_size",
         TORCH_FN(TraceType::_cufft_set_plan_cache_max_size)
  );
  m.impl("index_put_",
         TORCH_FN(TraceType::index_put_)
  );
  m.impl("index_put",
         TORCH_FN(TraceType::index_put)
  );
  m.impl("instance_norm",
         TORCH_FN(TraceType::instance_norm)
  );
  m.impl("isclose",
         TORCH_FN(TraceType::isclose)
  );
  m.impl("is_floating_point",
         TORCH_FN(TraceType::is_floating_point)
  );
  m.impl("is_complex",
         TORCH_FN(TraceType::is_complex)
  );
  m.impl("_cslt_sparse_mm_search",
         TORCH_FN(TraceType::_cslt_sparse_mm_search)
  );
  m.impl("_sparse_semi_structured_addmm",
         TORCH_FN(TraceType::_sparse_semi_structured_addmm)
  );
  m.impl("fbgemm_pack_gemm_matrix_fp16",
         TORCH_FN(TraceType::fbgemm_pack_gemm_matrix_fp16)
  );
  m.impl("_wrapped_quantized_linear_prepacked",
         TORCH_FN(TraceType::_wrapped_quantized_linear_prepacked)
  );
  m.impl("matmul_backward",
         TORCH_FN(TraceType::matmul_backward)
  );
  m.impl("mkldnn_max_pool2d_backward",
         TORCH_FN(TraceType::mkldnn_max_pool2d_backward)
  );
  m.impl("max_pool3d",
         TORCH_FN(TraceType::max_pool3d)
  );
  m.impl("median",
         TORCH_FN(TraceType::median)
  );
  m.impl("median.dim",
         TORCH_FN(TraceType::median_dim)
  );
  m.impl("median.dim_values",
         TORCH_FN(TraceType::median_out_dim_values)
  );
  m.impl("median.names_dim",
         TORCH_FN(TraceType::median_names_dim)
  );
  m.impl("median.names_dim_values",
         TORCH_FN(TraceType::median_out_names_dim_values)
  );
  m.impl("nanmedian",
         TORCH_FN(TraceType::nanmedian)
  );
  m.impl("nanmedian.dim",
         TORCH_FN(TraceType::nanmedian_dim)
  );
  m.impl("nanmedian.dim_values",
         TORCH_FN(TraceType::nanmedian_out_dim_values)
  );
  m.impl("nanmedian.names_dim",
         TORCH_FN(TraceType::nanmedian_names_dim)
  );
  m.impl("nanmedian.names_dim_values",
         TORCH_FN(TraceType::nanmedian_out_names_dim_values)
  );
  m.impl("miopen_rnn_backward",
         TORCH_FN(TraceType::miopen_rnn_backward)
  );
  m.impl("_convert_weight_to_int4pack",
         TORCH_FN(TraceType::_convert_weight_to_int4pack)
  );
  m.impl("multiply.Tensor",
         TORCH_FN(TraceType::multiply_Tensor)
  );
  m.impl("multiply_.Tensor",
         TORCH_FN(TraceType::multiply__Tensor)
  );
  m.impl("multiply.out",
         TORCH_FN(TraceType::multiply_out_out)
  );
  m.impl("multiply.Scalar",
         TORCH_FN(TraceType::multiply_Scalar)
  );
  m.impl("multiply_.Scalar",
         TORCH_FN(TraceType::multiply__Scalar)
  );
  m.impl("batch_norm_elemt",
         TORCH_FN(TraceType::batch_norm_elemt)
  );
  m.impl("batch_norm_elemt.out",
         TORCH_FN(TraceType::batch_norm_elemt_out_out)
  );
  m.impl("cdist",
         TORCH_FN(TraceType::cdist)
  );
  m.impl("adjoint",
         TORCH_FN(TraceType::adjoint)
  );
  m.impl("channel_shuffle",
         TORCH_FN(TraceType::channel_shuffle)
  );
  m.impl("poisson_nll_loss",
         TORCH_FN(TraceType::poisson_nll_loss)
  );
  m.impl("randperm",
         TORCH_FN(TraceType::randperm)
  );
  m.impl("randperm.generator",
         TORCH_FN(TraceType::randperm_generator)
  );
  m.impl("randperm.out",
         TORCH_FN(TraceType::randperm_out_out)
  );
  m.impl("randperm.generator_out",
         TORCH_FN(TraceType::randperm_out_generator_out)
  );
  m.impl("negative",
         TORCH_FN(TraceType::negative)
  );
  m.impl("negative_",
         TORCH_FN(TraceType::negative_)
  );
  m.impl("negative.out",
         TORCH_FN(TraceType::negative_out_out)
  );
  m.impl("infinitely_differentiable_gelu_backward",
         TORCH_FN(TraceType::infinitely_differentiable_gelu_backward)
  );
  m.impl("hardshrink_backward.grad_input",
         TORCH_FN(TraceType::hardshrink_backward_out_grad_input)
  );
  m.impl("hardshrink_backward",
         TORCH_FN(TraceType::hardshrink_backward)
  );
  m.impl("slice_inverse",
         TORCH_FN(TraceType::slice_inverse)
  );
  m.impl("smm",
         TORCH_FN(TraceType::smm)
  );
  m.impl("unsafe_split_with_sizes",
         TORCH_FN(TraceType::unsafe_split_with_sizes)
  );
  m.impl("_chunk_cat",
         TORCH_FN(TraceType::_chunk_cat)
  );
  m.impl("_chunk_cat.out",
         TORCH_FN(TraceType::_chunk_cat_out_out)
  );
  m.impl("dstack",
         TORCH_FN(TraceType::dstack)
  );
  m.impl("dstack.out",
         TORCH_FN(TraceType::dstack_out_out)
  );
  m.impl("tan",
         TORCH_FN(TraceType::tan)
  );
  m.impl("tan_",
         TORCH_FN(TraceType::tan_)
  );
  m.impl("tan.out",
         TORCH_FN(TraceType::tan_out_out)
  );
  m.impl("_nested_get_values",
         TORCH_FN(TraceType::_nested_get_values)
  );
  m.impl("unsqueeze",
         TORCH_FN(TraceType::unsqueeze)
  );
  m.impl("unsqueeze_",
         TORCH_FN(TraceType::unsqueeze_)
  );
  m.impl("_efficientzerotensor",
         TORCH_FN(TraceType::_efficientzerotensor)
  );
  m.impl("poisson",
         TORCH_FN(TraceType::poisson)
  );
  m.impl("sub.out",
         TORCH_FN(TraceType::sub_out_out)
  );
  m.impl("sub.Tensor",
         TORCH_FN(TraceType::sub_Tensor)
  );
  m.impl("sub_.Tensor",
         TORCH_FN(TraceType::sub__Tensor)
  );
  m.impl("sub.Scalar",
         TORCH_FN(TraceType::sub_Scalar)
  );
  m.impl("sub_.Scalar",
         TORCH_FN(TraceType::sub__Scalar)
  );
  m.impl("subtract.out",
         TORCH_FN(TraceType::subtract_out_out)
  );
  m.impl("subtract.Tensor",
         TORCH_FN(TraceType::subtract_Tensor)
  );
  m.impl("subtract_.Tensor",
         TORCH_FN(TraceType::subtract__Tensor)
  );
  m.impl("subtract.Scalar",
         TORCH_FN(TraceType::subtract_Scalar)
  );
  m.impl("subtract_.Scalar",
         TORCH_FN(TraceType::subtract__Scalar)
  );
  m.impl("_addmm_activation.out",
         TORCH_FN(TraceType::_addmm_activation_out_out)
  );
  m.impl("_addmm_activation",
         TORCH_FN(TraceType::_addmm_activation)
  );
  m.impl("sparse_compressed_tensor.comp_plain_value_size",
         TORCH_FN(TraceType::sparse_compressed_tensor_comp_plain_value_size)
  );
  m.impl("sparse_bsr_tensor.crow_col_value_size",
         TORCH_FN(TraceType::sparse_bsr_tensor_crow_col_value_size)
  );
  m.impl("sparse_compressed_tensor.comp_plain_value",
         TORCH_FN(TraceType::sparse_compressed_tensor_comp_plain_value)
  );
  m.impl("sparse_bsr_tensor.crow_col_value",
         TORCH_FN(TraceType::sparse_bsr_tensor_crow_col_value)
  );
  m.impl("_validate_sparse_compressed_tensor_args",
         TORCH_FN(TraceType::_validate_sparse_compressed_tensor_args)
  );
  m.impl("to_dense",
         TORCH_FN(TraceType::to_dense)
  );
  m.impl("_dimI",
         TORCH_FN(TraceType::_dimI)
  );
  m.impl("_nnz",
         TORCH_FN(TraceType::_nnz)
  );
  m.impl("ccol_indices",
         TORCH_FN(TraceType::ccol_indices)
  );
  m.impl("to_sparse_csr",
         TORCH_FN(TraceType::to_sparse_csr)
  );
  m.impl("_to_sparse_bsr",
         TORCH_FN(TraceType::_to_sparse_bsr)
  );
  m.impl("q_per_channel_axis",
         TORCH_FN(TraceType::q_per_channel_axis)
  );
  m.impl("_make_per_channel_quantized_tensor",
         TORCH_FN(TraceType::_make_per_channel_quantized_tensor)
  );
  m.impl("fake_quantize_per_tensor_affine_cachemask_backward",
         TORCH_FN(TraceType::fake_quantize_per_tensor_affine_cachemask_backward)
  );
  m.impl("fake_quantize_per_channel_affine_cachemask_backward",
         TORCH_FN(TraceType::fake_quantize_per_channel_affine_cachemask_backward)
  );
  m.impl("result_type.Tensor",
         TORCH_FN(TraceType::result_type_Tensor)
  );
  m.impl("result_type.Scalar",
         TORCH_FN(TraceType::result_type_Scalar)
  );
  m.impl("result_type.Scalar_Tensor",
         TORCH_FN(TraceType::result_type_Scalar_Tensor)
  );
  m.impl("result_type.Scalar_Scalar",
         TORCH_FN(TraceType::result_type_Scalar_Scalar)
  );
  m.impl("lstm_cell",
         TORCH_FN(TraceType::lstm_cell)
  );
  m.impl("masked_scatter_",
         TORCH_FN(TraceType::masked_scatter_)
  );
  m.impl("masked_scatter",
         TORCH_FN(TraceType::masked_scatter)
  );
  m.impl("_masked_softmax_backward",
         TORCH_FN(TraceType::_masked_softmax_backward)
  );
  m.impl("index_add.out",
         TORCH_FN(TraceType::index_add_out_out)
  );
  m.impl("index_add_",
         TORCH_FN(TraceType::index_add_)
  );
  m.impl("index_add",
         TORCH_FN(TraceType::index_add)
  );
  m.impl("index_add.dimname",
         TORCH_FN(TraceType::index_add_dimname)
  );
  m.impl("bitwise_or.Tensor_out",
         TORCH_FN(TraceType::bitwise_or_out_Tensor_out)
  );
  m.impl("bitwise_or.Scalar_out",
         TORCH_FN(TraceType::bitwise_or_out_Scalar_out)
  );
  m.impl("bitwise_or.Scalar",
         TORCH_FN(TraceType::bitwise_or_Scalar)
  );
  m.impl("bitwise_or.Scalar_Tensor",
         TORCH_FN(TraceType::bitwise_or_Scalar_Tensor)
  );
  m.impl("bitwise_or.Tensor",
         TORCH_FN(TraceType::bitwise_or_Tensor)
  );
  m.impl("bitwise_or_.Scalar",
         TORCH_FN(TraceType::bitwise_or__Scalar)
  );
  m.impl("bitwise_or_.Tensor",
         TORCH_FN(TraceType::bitwise_or__Tensor)
  );
  m.impl("triu_indices",
         TORCH_FN(TraceType::triu_indices)
  );
  m.impl("greater_equal.Scalar_out",
         TORCH_FN(TraceType::greater_equal_out_Scalar_out)
  );
  m.impl("greater_equal.Scalar",
         TORCH_FN(TraceType::greater_equal_Scalar)
  );
  m.impl("greater_equal.Tensor_out",
         TORCH_FN(TraceType::greater_equal_out_Tensor_out)
  );
  m.impl("greater_equal.Tensor",
         TORCH_FN(TraceType::greater_equal_Tensor)
  );
  m.impl("greater_equal_.Scalar",
         TORCH_FN(TraceType::greater_equal__Scalar)
  );
  m.impl("greater_equal_.Tensor",
         TORCH_FN(TraceType::greater_equal__Tensor)
  );
  m.impl("take.out",
         TORCH_FN(TraceType::take_out_out)
  );
  m.impl("take",
         TORCH_FN(TraceType::take)
  );
  m.impl("index_select_backward",
         TORCH_FN(TraceType::index_select_backward)
  );
  m.impl("argwhere",
         TORCH_FN(TraceType::argwhere)
  );
  m.impl("geqrf.a",
         TORCH_FN(TraceType::geqrf_out_a)
  );
  m.impl("geqrf",
         TORCH_FN(TraceType::geqrf)
  );
  m.impl("orgqr",
         TORCH_FN(TraceType::orgqr)
  );
  m.impl("orgqr.out",
         TORCH_FN(TraceType::orgqr_out_out)
  );
  m.impl("erfinv",
         TORCH_FN(TraceType::erfinv)
  );
  m.impl("erfinv_",
         TORCH_FN(TraceType::erfinv_)
  );
  m.impl("erfinv.out",
         TORCH_FN(TraceType::erfinv_out_out)
  );
  m.impl("signbit",
         TORCH_FN(TraceType::signbit)
  );
  m.impl("signbit.out",
         TORCH_FN(TraceType::signbit_out_out)
  );
  m.impl("fmod.Scalar_out",
         TORCH_FN(TraceType::fmod_out_Scalar_out)
  );
  m.impl("fmod.Scalar",
         TORCH_FN(TraceType::fmod_Scalar)
  );
  m.impl("fmod_.Scalar",
         TORCH_FN(TraceType::fmod__Scalar)
  );
  m.impl("fmod.Tensor_out",
         TORCH_FN(TraceType::fmod_out_Tensor_out)
  );
  m.impl("fmod.Tensor",
         TORCH_FN(TraceType::fmod_Tensor)
  );
  m.impl("fmod_.Tensor",
         TORCH_FN(TraceType::fmod__Tensor)
  );
  m.impl("any",
         TORCH_FN(TraceType::any)
  );
  m.impl("any.all_out",
         TORCH_FN(TraceType::any_out_all_out)
  );
  m.impl("_foreach_clamp_max.Scalar",
         TORCH_FN(TraceType::_foreach_clamp_max_Scalar)
  );
  m.impl("_foreach_clamp_max_.Scalar",
         TORCH_FN(TraceType::_foreach_clamp_max__Scalar)
  );
  m.impl("_foreach_clamp_max.List",
         TORCH_FN(TraceType::_foreach_clamp_max_List)
  );
  m.impl("_foreach_clamp_max_.List",
         TORCH_FN(TraceType::_foreach_clamp_max__List)
  );
  m.impl("_foreach_clamp_max.ScalarList",
         TORCH_FN(TraceType::_foreach_clamp_max_ScalarList)
  );
  m.impl("_foreach_clamp_max_.ScalarList",
         TORCH_FN(TraceType::_foreach_clamp_max__ScalarList)
  );
  m.impl("_foreach_abs",
         TORCH_FN(TraceType::_foreach_abs)
  );
  m.impl("_foreach_abs_",
         TORCH_FN(TraceType::_foreach_abs_)
  );
  m.impl("_foreach_expm1",
         TORCH_FN(TraceType::_foreach_expm1)
  );
  m.impl("_foreach_expm1_",
         TORCH_FN(TraceType::_foreach_expm1_)
  );
  m.impl("_foreach_log10",
         TORCH_FN(TraceType::_foreach_log10)
  );
  m.impl("_foreach_log10_",
         TORCH_FN(TraceType::_foreach_log10_)
  );
  m.impl("_foreach_max",
         TORCH_FN(TraceType::_foreach_max)
  );
  m.impl("_foreach_sign",
         TORCH_FN(TraceType::_foreach_sign)
  );
  m.impl("_foreach_sign_",
         TORCH_FN(TraceType::_foreach_sign_)
  );
  m.impl("_foreach_tan",
         TORCH_FN(TraceType::_foreach_tan)
  );
  m.impl("_foreach_tan_",
         TORCH_FN(TraceType::_foreach_tan_)
  );
  m.impl("_foreach_copy_",
         TORCH_FN(TraceType::_foreach_copy_)
  );
  m.impl("_foreach_copy",
         TORCH_FN(TraceType::_foreach_copy)
  );
  m.impl("searchsorted.Tensor",
         TORCH_FN(TraceType::searchsorted_Tensor)
  );
  m.impl("searchsorted.Tensor_out",
         TORCH_FN(TraceType::searchsorted_out_Tensor_out)
  );
  m.impl("searchsorted.Scalar",
         TORCH_FN(TraceType::searchsorted_Scalar)
  );
  m.impl("searchsorted.Scalar_out",
         TORCH_FN(TraceType::searchsorted_out_Scalar_out)
  );
  m.impl("smooth_l1_loss.out",
         TORCH_FN(TraceType::smooth_l1_loss_out_out)
  );
  m.impl("smooth_l1_loss",
         TORCH_FN(TraceType::smooth_l1_loss)
  );
  m.impl("elu.out",
         TORCH_FN(TraceType::elu_out_out)
  );
  m.impl("elu",
         TORCH_FN(TraceType::elu)
  );
  m.impl("elu_",
         TORCH_FN(TraceType::elu_)
  );
  m.impl("leaky_relu_backward.grad_input",
         TORCH_FN(TraceType::leaky_relu_backward_out_grad_input)
  );
  m.impl("leaky_relu_backward",
         TORCH_FN(TraceType::leaky_relu_backward)
  );
  m.impl("mkldnn_adaptive_avg_pool2d",
         TORCH_FN(TraceType::mkldnn_adaptive_avg_pool2d)
  );
  m.impl("mkldnn_adaptive_avg_pool2d.out",
         TORCH_FN(TraceType::mkldnn_adaptive_avg_pool2d_out_out)
  );
  m.impl("_adaptive_avg_pool2d",
         TORCH_FN(TraceType::_adaptive_avg_pool2d)
  );
  m.impl("avg_pool3d.out",
         TORCH_FN(TraceType::avg_pool3d_out_out)
  );
  m.impl("avg_pool3d",
         TORCH_FN(TraceType::avg_pool3d)
  );
  m.impl("max_pool2d_with_indices_backward.grad_input",
         TORCH_FN(TraceType::max_pool2d_with_indices_backward_out_grad_input)
  );
  m.impl("max_pool2d_with_indices_backward",
         TORCH_FN(TraceType::max_pool2d_with_indices_backward)
  );
  m.impl("reflection_pad2d.out",
         TORCH_FN(TraceType::reflection_pad2d_out_out)
  );
  m.impl("reflection_pad2d",
         TORCH_FN(TraceType::reflection_pad2d)
  );
  m.impl("upsample_linear1d_backward.grad_input",
         TORCH_FN(TraceType::upsample_linear1d_backward_out_grad_input)
  );
  m.impl("upsample_linear1d_backward",
         TORCH_FN(TraceType::upsample_linear1d_backward)
  );
  m.impl("upsample_nearest1d_backward.grad_input",
         TORCH_FN(TraceType::upsample_nearest1d_backward_out_grad_input)
  );
  m.impl("upsample_nearest1d_backward",
         TORCH_FN(TraceType::upsample_nearest1d_backward)
  );
  m.impl("slow_conv_transpose3d.out",
         TORCH_FN(TraceType::slow_conv_transpose3d_out_out)
  );
  m.impl("slow_conv_transpose3d",
         TORCH_FN(TraceType::slow_conv_transpose3d)
  );
  m.impl("_add_batch_dim",
         TORCH_FN(TraceType::_add_batch_dim)
  );
  m.impl("special_psi",
         TORCH_FN(TraceType::special_psi)
  );
  m.impl("special_psi.out",
         TORCH_FN(TraceType::special_psi_out_out)
  );
  m.impl("special_i0e",
         TORCH_FN(TraceType::special_i0e)
  );
  m.impl("special_i0e.out",
         TORCH_FN(TraceType::special_i0e_out_out)
  );
  m.impl("special_logit",
         TORCH_FN(TraceType::special_logit)
  );
  m.impl("special_logit.out",
         TORCH_FN(TraceType::special_logit_out_out)
  );
  m.impl("fft_fft2",
         TORCH_FN(TraceType::fft_fft2)
  );
  m.impl("fft_fft2.out",
         TORCH_FN(TraceType::fft_fft2_out_out)
  );
  m.impl("linalg_lu_solve",
         TORCH_FN(TraceType::linalg_lu_solve)
  );
  m.impl("linalg_lu_solve.out",
         TORCH_FN(TraceType::linalg_lu_solve_out_out)
  );
  m.impl("linalg_inv",
         TORCH_FN(TraceType::linalg_inv)
  );
  m.impl("linalg_inv.out",
         TORCH_FN(TraceType::linalg_inv_out_out)
  );
  m.impl("outer",
         TORCH_FN(TraceType::outer)
  );
  m.impl("outer.out",
         TORCH_FN(TraceType::outer_out_out)
  );
  m.impl("_linalg_svd",
         TORCH_FN(TraceType::_linalg_svd)
  );
  m.impl("_linalg_svd.U",
         TORCH_FN(TraceType::_linalg_svd_out_U)
  );
  m.impl("linalg_qr",
         TORCH_FN(TraceType::linalg_qr)
  );
  m.impl("linalg_qr.out",
         TORCH_FN(TraceType::linalg_qr_out_out)
  );
  m.impl("_test_warn_in_autograd",
         TORCH_FN(TraceType::_test_warn_in_autograd)
  );
  m.impl("select_copy.int",
         TORCH_FN(TraceType::select_copy_int)
  );
  m.impl("split_with_sizes_copy",
         TORCH_FN(TraceType::split_with_sizes_copy)
  );
  m.impl("col_indices_copy",
         TORCH_FN(TraceType::col_indices_copy)
  );
  m.impl("unbind_copy.int",
         TORCH_FN(TraceType::unbind_copy_int)
  );
  m.impl("unbind_copy.int_out",
         TORCH_FN(TraceType::unbind_copy_out_int_out)
  );
  m.impl("split_with_sizes_copy.out",
         TORCH_FN(TraceType::split_with_sizes_copy_out_out)
  );
  m.impl("alias_copy",
         TORCH_FN(TraceType::alias_copy)
  );
  m.impl("_jagged_to_padded_dense_forward",
         TORCH_FN(TraceType::_jagged_to_padded_dense_forward)
  );
  m.impl("_scaled_dot_product_fused_attention_overrideable",
         TORCH_FN(TraceType::_scaled_dot_product_fused_attention_overrideable)
  );
  m.impl("_scaled_dot_product_flash_attention_backward",
         TORCH_FN(TraceType::_scaled_dot_product_flash_attention_backward)
  );
  m.impl("_cudnn_attention_forward",
         TORCH_FN(TraceType::_cudnn_attention_forward)
  );
  m.impl("_triton_scaled_dot_attention",
         TORCH_FN(TraceType::_triton_scaled_dot_attention)
  );
  m.impl("special_chebyshev_polynomial_t",
         TORCH_FN(TraceType::special_chebyshev_polynomial_t)
  );
  m.impl("special_chebyshev_polynomial_t.x_scalar",
         TORCH_FN(TraceType::special_chebyshev_polynomial_t_x_scalar)
  );
  m.impl("special_chebyshev_polynomial_t.n_scalar",
         TORCH_FN(TraceType::special_chebyshev_polynomial_t_n_scalar)
  );
  m.impl("special_chebyshev_polynomial_t.out",
         TORCH_FN(TraceType::special_chebyshev_polynomial_t_out_out)
  );
  m.impl("special_chebyshev_polynomial_t.x_scalar_out",
         TORCH_FN(TraceType::special_chebyshev_polynomial_t_out_x_scalar_out)
  );
  m.impl("special_chebyshev_polynomial_t.n_scalar_out",
         TORCH_FN(TraceType::special_chebyshev_polynomial_t_out_n_scalar_out)
  );
  m.impl("special_scaled_modified_bessel_k1",
         TORCH_FN(TraceType::special_scaled_modified_bessel_k1)
  );
  m.impl("special_scaled_modified_bessel_k1.out",
         TORCH_FN(TraceType::special_scaled_modified_bessel_k1_out_out)
  );
  m.impl("_masked_scale.out",
         TORCH_FN(TraceType::_masked_scale_out_out)
  );
  m.impl("constant_pad_nd.out",
         TORCH_FN(TraceType::constant_pad_nd_out_out)
  );
  m.impl("convolution_overrideable.out",
         TORCH_FN(TraceType::convolution_overrideable_out_out)
  );
  m.impl("mps_convolution_transpose_backward.out",
         TORCH_FN(TraceType::mps_convolution_transpose_backward_out_out)
  );
  m.impl("embedding.out",
         TORCH_FN(TraceType::embedding_out_out)
  );
  m.impl("_embedding_bag_dense_backward.out",
         TORCH_FN(TraceType::_embedding_bag_dense_backward_out_out)
  );
  m.impl("resize.out",
         TORCH_FN(TraceType::resize_out_out)
  );
  m.impl("resize",
         TORCH_FN(TraceType::resize)
  );
  m.impl("index_put.out",
         TORCH_FN(TraceType::index_put_out_out)
  );
  m.impl("matmul_backward.out",
         TORCH_FN(TraceType::matmul_backward_out_out)
  );
  m.impl("mkldnn_max_pool2d_backward.out",
         TORCH_FN(TraceType::mkldnn_max_pool2d_backward_out_out)
  );
  m.impl("median.out",
         TORCH_FN(TraceType::median_out_out)
  );
  m.impl("nanmedian.out",
         TORCH_FN(TraceType::nanmedian_out_out)
  );
  m.impl("miopen_rnn_backward.out",
         TORCH_FN(TraceType::miopen_rnn_backward_out_out)
  );
  m.impl("channel_shuffle.out",
         TORCH_FN(TraceType::channel_shuffle_out_out)
  );
  m.impl("unsafe_split_with_sizes.out",
         TORCH_FN(TraceType::unsafe_split_with_sizes_out_out)
  );
  m.impl("_efficientzerotensor.out",
         TORCH_FN(TraceType::_efficientzerotensor_out_out)
  );
  m.impl("poisson.out",
         TORCH_FN(TraceType::poisson_out_out)
  );
  m.impl("sub.Scalar_out",
         TORCH_FN(TraceType::sub_out_Scalar_out)
  );
  m.impl("_to_sparse_bsr.out",
         TORCH_FN(TraceType::_to_sparse_bsr_out_out)
  );
  m.impl("_make_per_channel_quantized_tensor.out",
         TORCH_FN(TraceType::_make_per_channel_quantized_tensor_out_out)
  );
  m.impl("masked_scatter.out",
         TORCH_FN(TraceType::masked_scatter_out_out)
  );
  m.impl("_masked_softmax_backward.out",
         TORCH_FN(TraceType::_masked_softmax_backward_out_out)
  );
  m.impl("bitwise_or.Scalar_Tensor_out",
         TORCH_FN(TraceType::bitwise_or_out_Scalar_Tensor_out)
  );
  m.impl("triu_indices.out",
         TORCH_FN(TraceType::triu_indices_out_out)
  );
  m.impl("_foreach_clamp_max.Scalar_out",
         TORCH_FN(TraceType::_foreach_clamp_max_out_Scalar_out)
  );
  m.impl("_foreach_clamp_max.List_out",
         TORCH_FN(TraceType::_foreach_clamp_max_out_List_out)
  );
  m.impl("_foreach_clamp_max.ScalarList_out",
         TORCH_FN(TraceType::_foreach_clamp_max_out_ScalarList_out)
  );
  m.impl("_foreach_abs.out",
         TORCH_FN(TraceType::_foreach_abs_out_out)
  );
  m.impl("_foreach_expm1.out",
         TORCH_FN(TraceType::_foreach_expm1_out_out)
  );
  m.impl("_foreach_log10.out",
         TORCH_FN(TraceType::_foreach_log10_out_out)
  );
  m.impl("_foreach_max.out",
         TORCH_FN(TraceType::_foreach_max_out_out)
  );
  m.impl("_foreach_sign.out",
         TORCH_FN(TraceType::_foreach_sign_out_out)
  );
  m.impl("_foreach_tan.out",
         TORCH_FN(TraceType::_foreach_tan_out_out)
  );
  m.impl("_foreach_copy.out",
         TORCH_FN(TraceType::_foreach_copy_out_out)
  );
  m.impl("_adaptive_avg_pool2d.out",
         TORCH_FN(TraceType::_adaptive_avg_pool2d_out_out)
  );
  m.impl("_test_warn_in_autograd.out",
         TORCH_FN(TraceType::_test_warn_in_autograd_out_out)
  );
  m.impl("select_copy.int_out",
         TORCH_FN(TraceType::select_copy_out_int_out)
  );
  m.impl("col_indices_copy.out",
         TORCH_FN(TraceType::col_indices_copy_out_out)
  );
  m.impl("alias_copy.out",
         TORCH_FN(TraceType::alias_copy_out_out)
  );
  m.impl("_triton_scaled_dot_attention.out",
         TORCH_FN(TraceType::_triton_scaled_dot_attention_out_out)
  );;
}

}  // namespace

} // namespace torch
