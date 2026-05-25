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
#include <ATen/ops/_assert_async_ops.h>
#include <ATen/ops/_assert_async_ops.h>
#include <ATen/ops/_functional_assert_scalar_ops.h>
#include <ATen/ops/_reshape_from_tensor_ops.h>
#include <ATen/ops/view_as_real_ops.h>
#include <ATen/ops/conj_physical_ops.h>
#include <ATen/ops/conj_physical_ops.h>
#include <ATen/ops/conj_physical_ops.h>
#include <ATen/ops/acos_ops.h>
#include <ATen/ops/acos_ops.h>
#include <ATen/ops/acos_ops.h>
#include <ATen/ops/arccos_ops.h>
#include <ATen/ops/arccos_ops.h>
#include <ATen/ops/arccos_ops.h>
#include <ATen/ops/arccosh_ops.h>
#include <ATen/ops/arccosh_ops.h>
#include <ATen/ops/arccosh_ops.h>
#include <ATen/ops/asin_ops.h>
#include <ATen/ops/asin_ops.h>
#include <ATen/ops/asin_ops.h>
#include <ATen/ops/logical_xor_ops.h>
#include <ATen/ops/logical_xor_ops.h>
#include <ATen/ops/logical_xor_ops.h>
#include <ATen/ops/broadcast_to_ops.h>
#include <ATen/ops/contiguous_ops.h>
#include <ATen/ops/convolution_backward_ops.h>
#include <ATen/ops/_convolution_double_backward_ops.h>
#include <ATen/ops/conv2d_ops.h>
#include <ATen/ops/conv2d_ops.h>
#include <ATen/ops/_copy_from_ops.h>
#include <ATen/ops/corrcoef_ops.h>
#include <ATen/ops/_mps_convolution_transpose_ops.h>
#include <ATen/ops/cummaxmin_backward_ops.h>
#include <ATen/ops/cumprod_backward_ops.h>
#include <ATen/ops/_rowwise_prune_ops.h>
#include <ATen/ops/row_stack_ops.h>
#include <ATen/ops/row_stack_ops.h>
#include <ATen/ops/_embedding_bag_backward_ops.h>
#include <ATen/ops/floor_divide_ops.h>
#include <ATen/ops/floor_divide_ops.h>
#include <ATen/ops/floor_divide_ops.h>
#include <ATen/ops/floor_divide_ops.h>
#include <ATen/ops/floor_divide_ops.h>
#include <ATen/ops/full_ops.h>
#include <ATen/ops/full_ops.h>
#include <ATen/ops/full_ops.h>
#include <ATen/ops/full_like_ops.h>
#include <ATen/ops/grid_sampler_2d_ops.h>
#include <ATen/ops/kaiser_window_ops.h>
#include <ATen/ops/kaiser_window_ops.h>
#include <ATen/ops/kaiser_window_ops.h>
#include <ATen/ops/_fft_c2r_ops.h>
#include <ATen/ops/_fft_c2r_ops.h>
#include <ATen/ops/_unsafe_masked_index_put_accumulate_ops.h>
#include <ATen/ops/is_same_size_ops.h>
#include <ATen/ops/get_device_ops.h>
#include <ATen/ops/is_contiguous_ops.h>
#include <ATen/ops/is_contiguous_ops.h>
#include <ATen/ops/kl_div_ops.h>
#include <ATen/ops/_fused_rms_norm_ops.h>
#include <ATen/ops/_cslt_compress_ops.h>
#include <ATen/ops/margin_ranking_loss_ops.h>
#include <ATen/ops/matmul_ops.h>
#include <ATen/ops/matmul_ops.h>
#include <ATen/ops/matrix_exp_ops.h>
#include <ATen/ops/_compute_linear_combination_ops.h>
#include <ATen/ops/_compute_linear_combination_ops.h>
#include <ATen/ops/max_pool2d_backward_ops.h>
#include <ATen/ops/miopen_batch_norm_ops.h>
#include <ATen/ops/miopen_convolution_transpose_ops.h>
#include <ATen/ops/miopen_convolution_add_relu_ops.h>
#include <ATen/ops/mT_ops.h>
#include <ATen/ops/deg2rad_ops.h>
#include <ATen/ops/deg2rad_ops.h>
#include <ATen/ops/deg2rad_ops.h>
#include <ATen/ops/_reshape_copy_ops.h>
#include <ATen/ops/relu_ops.h>
#include <ATen/ops/relu_ops.h>
#include <ATen/ops/sinc_ops.h>
#include <ATen/ops/sinc_ops.h>
#include <ATen/ops/sinc_ops.h>
#include <ATen/ops/slice_ops.h>
#include <ATen/ops/select_scatter_ops.h>
#include <ATen/ops/prod_ops.h>
#include <ATen/ops/prod_ops.h>
#include <ATen/ops/prod_ops.h>
#include <ATen/ops/prod_ops.h>
#include <ATen/ops/prod_ops.h>
#include <ATen/ops/trapezoid_ops.h>
#include <ATen/ops/trapezoid_ops.h>
#include <ATen/ops/_nested_tensor_from_mask_ops.h>
#include <ATen/ops/_nested_tensor_from_mask_left_aligned_ops.h>
#include <ATen/ops/_nested_tensor_size_ops.h>
#include <ATen/ops/_nested_view_from_buffer_copy_ops.h>
#include <ATen/ops/unique_dim_consecutive_ops.h>
#include <ATen/ops/_unsafe_view_ops.h>
#include <ATen/ops/_batch_norm_no_update_ops.h>
#include <ATen/ops/heaviside_ops.h>
#include <ATen/ops/heaviside_ops.h>
#include <ATen/ops/heaviside_ops.h>
#include <ATen/ops/sparse_coo_tensor_ops.h>
#include <ATen/ops/sparse_coo_tensor_ops.h>
#include <ATen/ops/sparse_coo_tensor_ops.h>
#include <ATen/ops/sparse_resize_and_clear_ops.h>
#include <ATen/ops/sparse_dim_ops.h>
#include <ATen/ops/_to_sparse_csr_ops.h>
#include <ATen/ops/to_sparse_bsr_ops.h>
#include <ATen/ops/mkldnn_reorder_conv3d_weight_ops.h>
#include <ATen/ops/q_scale_ops.h>
#include <ATen/ops/_make_per_tensor_quantized_tensor_ops.h>
#include <ATen/ops/_saturate_weight_to_fp16_ops.h>
#include <ATen/ops/_autocast_to_reduced_precision_ops.h>
#include <ATen/ops/_thnn_fused_lstm_cell_backward_ops.h>
#include <ATen/ops/quantized_rnn_relu_cell_ops.h>
#include <ATen/ops/masked_fill_ops.h>
#include <ATen/ops/masked_fill_ops.h>
#include <ATen/ops/masked_fill_ops.h>
#include <ATen/ops/masked_fill_ops.h>
#include <ATen/ops/diag_ops.h>
#include <ATen/ops/diag_ops.h>
#include <ATen/ops/trace_ops.h>
#include <ATen/ops/svd_ops.h>
#include <ATen/ops/svd_ops.h>
#include <ATen/ops/dist_ops.h>
#include <ATen/ops/_histogramdd_from_bin_cts_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/nanquantile_ops.h>
#include <ATen/ops/nanquantile_ops.h>
#include <ATen/ops/nanquantile_ops.h>
#include <ATen/ops/nanquantile_ops.h>
#include <ATen/ops/renorm_ops.h>
#include <ATen/ops/renorm_ops.h>
#include <ATen/ops/renorm_ops.h>
#include <ATen/ops/unfold_ops.h>
#include <ATen/ops/float_power_ops.h>
#include <ATen/ops/float_power_ops.h>
#include <ATen/ops/float_power_ops.h>
#include <ATen/ops/float_power_ops.h>
#include <ATen/ops/float_power_ops.h>
#include <ATen/ops/float_power_ops.h>
#include <ATen/ops/float_power_ops.h>
#include <ATen/ops/float_power_ops.h>
#include <ATen/ops/_foreach_sinh_ops.h>
#include <ATen/ops/_foreach_sinh_ops.h>
#include <ATen/ops/glu_backward_ops.h>
#include <ATen/ops/glu_backward_ops.h>
#include <ATen/ops/hardtanh_backward_ops.h>
#include <ATen/ops/hardtanh_backward_ops.h>
#include <ATen/ops/softplus_ops.h>
#include <ATen/ops/softplus_ops.h>
#include <ATen/ops/avg_pool3d_backward_ops.h>
#include <ATen/ops/avg_pool3d_backward_ops.h>
#include <ATen/ops/max_pool3d_with_indices_ops.h>
#include <ATen/ops/max_pool3d_with_indices_ops.h>
#include <ATen/ops/_upsample_bilinear2d_aa_ops.h>
#include <ATen/ops/_upsample_bilinear2d_aa_ops.h>
#include <ATen/ops/_upsample_bilinear2d_aa_ops.h>
#include <ATen/ops/upsample_nearest2d_backward_ops.h>
#include <ATen/ops/upsample_nearest2d_backward_ops.h>
#include <ATen/ops/slow_conv3d_forward_ops.h>
#include <ATen/ops/slow_conv3d_forward_ops.h>
#include <ATen/ops/im2col_ops.h>
#include <ATen/ops/im2col_ops.h>
#include <ATen/ops/isneginf_ops.h>
#include <ATen/ops/isneginf_ops.h>
#include <ATen/ops/special_erfcx_ops.h>
#include <ATen/ops/special_erfcx_ops.h>
#include <ATen/ops/special_i1_ops.h>
#include <ATen/ops/special_i1_ops.h>
#include <ATen/ops/special_log_softmax_ops.h>
#include <ATen/ops/special_gammaincc_ops.h>
#include <ATen/ops/special_gammaincc_ops.h>
#include <ATen/ops/special_multigammaln_ops.h>
#include <ATen/ops/special_multigammaln_ops.h>
#include <ATen/ops/fft_fftn_ops.h>
#include <ATen/ops/fft_fftn_ops.h>
#include <ATen/ops/fft_fftshift_ops.h>
#include <ATen/ops/linalg_lu_factor_ops.h>
#include <ATen/ops/linalg_lu_factor_ops.h>
#include <ATen/ops/linalg_det_ops.h>
#include <ATen/ops/linalg_det_ops.h>
#include <ATen/ops/_linalg_slogdet_ops.h>
#include <ATen/ops/_linalg_slogdet_ops.h>
#include <ATen/ops/ger_ops.h>
#include <ATen/ops/ger_ops.h>
#include <ATen/ops/_linalg_solve_ex_ops.h>
#include <ATen/ops/_linalg_solve_ex_ops.h>
#include <ATen/ops/nested_to_padded_tensor_ops.h>
#include <ATen/ops/_test_autograd_multiple_dispatch_view_ops.h>
#include <ATen/ops/diagonal_copy_ops.h>
#include <ATen/ops/permute_copy_ops.h>
#include <ATen/ops/slice_copy_ops.h>
#include <ATen/ops/t_copy_ops.h>
#include <ATen/ops/_nested_from_padded_tensor_ops.h>
#include <ATen/ops/_scaled_dot_product_attention_math_ops.h>
#include <ATen/ops/_cudnn_attention_backward_ops.h>
#include <ATen/ops/_foobar_ops.h>
#include <ATen/ops/convolution_backward_ops.h>
#include <ATen/ops/_copy_from_ops.h>
#include <ATen/ops/_mps_convolution_transpose_ops.h>
#include <ATen/ops/floor_divide_ops.h>
#include <ATen/ops/full_ops.h>
#include <ATen/ops/full_like_ops.h>
#include <ATen/ops/grid_sampler_2d_ops.h>
#include <ATen/ops/kaiser_window_ops.h>
#include <ATen/ops/kaiser_window_ops.h>
#include <ATen/ops/kaiser_window_ops.h>
#include <ATen/ops/max_pool2d_backward_ops.h>
#include <ATen/ops/miopen_batch_norm_ops.h>
#include <ATen/ops/miopen_convolution_transpose_ops.h>
#include <ATen/ops/relu_ops.h>
#include <ATen/ops/select_scatter_ops.h>
#include <ATen/ops/prod_ops.h>
#include <ATen/ops/_nested_tensor_from_mask_ops.h>
#include <ATen/ops/_nested_tensor_size_ops.h>
#include <ATen/ops/_nested_view_from_buffer_copy_ops.h>
#include <ATen/ops/unique_dim_consecutive_ops.h>
#include <ATen/ops/_unsafe_view_ops.h>
#include <ATen/ops/_batch_norm_no_update_ops.h>
#include <ATen/ops/sparse_coo_tensor_ops.h>
#include <ATen/ops/sparse_resize_and_clear_ops.h>
#include <ATen/ops/sparse_resize_and_clear_ops.h>
#include <ATen/ops/_to_sparse_csr_ops.h>
#include <ATen/ops/mkldnn_reorder_conv3d_weight_ops.h>
#include <ATen/ops/_make_per_tensor_quantized_tensor_ops.h>
#include <ATen/ops/masked_fill_ops.h>
#include <ATen/ops/masked_fill_ops.h>
#include <ATen/ops/trace_ops.h>
#include <ATen/ops/dist_ops.h>
#include <ATen/ops/_histogramdd_from_bin_cts_ops.h>
#include <ATen/ops/remainder_ops.h>
#include <ATen/ops/_foreach_sinh_ops.h>
#include <ATen/ops/diagonal_copy_ops.h>
#include <ATen/ops/permute_copy_ops.h>
#include <ATen/ops/slice_copy_ops.h>
#include <ATen/ops/t_copy_ops.h>
#include <ATen/ops/_foobar_ops.h>
#endif

using namespace at;

namespace torch {

namespace TraceType {

namespace {
void _assert_async(c10::DispatchKeySet ks, const at::Tensor & self) {
  at::_ops::_assert_async::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
void _assert_async_msg(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view assert_msg) {
  at::_ops::_assert_async_msg::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, assert_msg);
}
at::Tensor _functional_assert_scalar(c10::DispatchKeySet ks, const at::Scalar & self, c10::string_view assert_msg, const at::Tensor & dep_token) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_functional_assert_scalar");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "assert_msg", assert_msg);
    jit::tracer::addInputs(node, "dep_token", dep_token);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_functional_assert_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, assert_msg, dep_token);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _reshape_from_tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & shape) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_reshape_from_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "shape", shape);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_reshape_from_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, shape);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor view_as_real(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::view_as_real");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::view_as_real::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor conj_physical(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::conj_physical");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::conj_physical::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & conj_physical_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::conj_physical");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("conj_physical_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::conj_physical_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & conj_physical_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::conj_physical");
    } else {
      op_name = c10::Symbol::fromQualString("aten::conj_physical_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("conj_physical_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::conj_physical_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor acos(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::acos");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::acos::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & acos_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::acos");
    } else {
      op_name = c10::Symbol::fromQualString("aten::acos_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("acos_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::acos_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & acos_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::acos");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("acos_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::acos_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor arccos(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::arccos");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::arccos::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & arccos_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::arccos");
    } else {
      op_name = c10::Symbol::fromQualString("aten::arccos_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arccos_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::arccos_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & arccos_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::arccos");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arccos_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::arccos_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor arccosh(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::arccosh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::arccosh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & arccosh_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::arccosh");
    } else {
      op_name = c10::Symbol::fromQualString("aten::arccosh_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arccosh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::arccosh_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & arccosh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::arccosh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("arccosh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::arccosh_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor asin(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::asin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::asin::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & asin_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::asin");
    } else {
      op_name = c10::Symbol::fromQualString("aten::asin_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("asin_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::asin_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & asin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::asin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("asin_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::asin_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor logical_xor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logical_xor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logical_xor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & logical_xor_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::logical_xor");
    } else {
      op_name = c10::Symbol::fromQualString("aten::logical_xor_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_xor_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logical_xor_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & logical_xor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logical_xor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_xor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logical_xor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor broadcast_to(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::broadcast_to");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::broadcast_to::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor contiguous(c10::DispatchKeySet ks, const at::Tensor & self, at::MemoryFormat memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::contiguous");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::contiguous::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalSymIntArrayRef bias_sizes, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, ::std::array<bool,3> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::convolution_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias_sizes", bias_sizes);
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
  auto [result0, result1, result2] =at::_ops::convolution_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _convolution_double_backward(c10::DispatchKeySet ks, const ::std::optional<at::Tensor> & ggI, const ::std::optional<at::Tensor> & ggW, const ::std::optional<at::Tensor> & ggb, const at::Tensor & gO, const at::Tensor & weight, const at::Tensor & self, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, ::std::array<bool,3> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_convolution_double_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "ggI", ggI);
    jit::tracer::addInputs(node, "ggW", ggW);
    jit::tracer::addInputs(node, "ggb", ggb);
    jit::tracer::addInputs(node, "gO", gO);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "self", self);
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
  auto [result0, result1, result2] =at::_ops::_convolution_double_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor conv2d(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups) {
  auto result =at::_ops::conv2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, stride, padding, dilation, groups);
  return result;
}
at::Tensor conv2d_padding(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::string_view padding, c10::SymIntArrayRef dilation, c10::SymInt groups) {
  auto result =at::_ops::conv2d_padding::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, stride, padding, dilation, groups);
  return result;
}
at::Tensor _copy_from(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & dst, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_copy_from");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dst", dst);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_copy_from::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dst, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor corrcoef(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::corrcoef");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::corrcoef::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _mps_convolution_transpose(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mps_convolution_transpose");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_mps_convolution_transpose::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, padding, output_padding, stride, dilation, groups);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor cummaxmin_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & input, const at::Tensor & indices, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cummaxmin_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cummaxmin_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, input, indices, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor cumprod_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & input, int64_t dim, const at::Tensor & output) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cumprod_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cumprod_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, input, dim, output);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _rowwise_prune(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & mask, at::ScalarType compressed_indices_dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_rowwise_prune");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "compressed_indices_dtype", compressed_indices_dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_rowwise_prune::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight, mask, compressed_indices_dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor row_stack(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::row_stack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::row_stack::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & row_stack_out_out(c10::DispatchKeySet ks, at::TensorList tensors, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::row_stack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("row_stack_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::row_stack_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _embedding_bag_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & indices, const at::Tensor & offsets, const at::Tensor & offset2bag, const at::Tensor & bag_size, const at::Tensor & maximum_indices, c10::SymInt num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, int64_t padding_idx) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_embedding_bag_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "offset2bag", offset2bag);
    jit::tracer::addInputs(node, "bag_size", bag_size);
    jit::tracer::addInputs(node, "maximum_indices", maximum_indices);
    jit::tracer::addInputs(node, "num_weights", num_weights);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_embedding_bag_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, indices, offsets, offset2bag, bag_size, maximum_indices, num_weights, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor floor_divide(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::floor_divide");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::floor_divide::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & floor_divide__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::floor_divide");
    } else {
      op_name = c10::Symbol::fromQualString("aten::floor_divide_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_divide_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::floor_divide__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & floor_divide_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::floor_divide");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_divide_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::floor_divide_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor floor_divide_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::floor_divide");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::floor_divide_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & floor_divide__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::floor_divide");
    } else {
      op_name = c10::Symbol::fromQualString("aten::floor_divide_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_divide_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::floor_divide__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor full_names(c10::DispatchKeySet ks, at::IntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::full");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::full_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, fill_value, names, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor full(c10::DispatchKeySet ks, c10::SymIntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::full");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::full::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, fill_value, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & full_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, const at::Scalar & fill_value, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::full");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("full_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::full_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, fill_value, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor full_like(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & fill_value, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::full_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::full_like::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, fill_value, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor grid_sampler_2d(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::grid_sampler_2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::grid_sampler_2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, grid, interpolation_mode, padding_mode, align_corners);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor kaiser_window(c10::DispatchKeySet ks, int64_t window_length, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kaiser_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::kaiser_window::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor kaiser_window_periodic(c10::DispatchKeySet ks, int64_t window_length, bool periodic, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kaiser_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::kaiser_window_periodic::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor kaiser_window_beta(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double beta, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kaiser_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::kaiser_window_beta::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, beta, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _fft_c2r(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, c10::SymInt last_dim_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fft_c2r");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "normalization", normalization);
    jit::tracer::addInputs(node, "last_dim_size", last_dim_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_fft_c2r::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, normalization, last_dim_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _fft_c2r_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, c10::SymInt last_dim_size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fft_c2r");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "normalization", normalization);
    jit::tracer::addInputs(node, "last_dim_size", last_dim_size);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_fft_c2r_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_fft_c2r_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, normalization, last_dim_size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _unsafe_masked_index_put_accumulate(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, const c10::List<::std::optional<at::Tensor>> & indices, const at::Tensor & values) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_unsafe_masked_index_put_accumulate");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "values", values);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_unsafe_masked_index_put_accumulate::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, indices, values);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool is_same_size(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  auto result =at::_ops::is_same_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  return result;
}
int64_t get_device(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::get_device::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
bool is_contiguous(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::is_contiguous::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
bool is_contiguous_memory_format(c10::DispatchKeySet ks, const at::Tensor & self, at::MemoryFormat memory_format) {
  auto result =at::_ops::is_contiguous_memory_format::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format);
  return result;
}
at::Tensor kl_div(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kl_div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    jit::tracer::addInputs(node, "log_target", log_target);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::kl_div::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction, log_target);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _fused_rms_norm(c10::DispatchKeySet ks, const at::Tensor & input, at::IntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, ::std::optional<double> eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fused_rms_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_fused_rms_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, normalized_shape, weight, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor _cslt_compress(c10::DispatchKeySet ks, const at::Tensor & input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cslt_compress");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cslt_compress::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor margin_ranking_loss(c10::DispatchKeySet ks, const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & target, double margin, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::margin_ranking_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input1", input1);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::margin_ranking_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input1, input2, target, margin, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor matmul(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::matmul");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::matmul::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & matmul_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::matmul");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("matmul_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::matmul_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor matrix_exp(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::matrix_exp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::matrix_exp::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _compute_linear_combination(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & coefficients) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_compute_linear_combination");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "coefficients", coefficients);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_compute_linear_combination::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, coefficients);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _compute_linear_combination_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & coefficients, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_compute_linear_combination");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "coefficients", coefficients);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_compute_linear_combination_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_compute_linear_combination_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, coefficients, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor max_pool2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::max_pool2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_batch_norm(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_batch_norm");
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
  auto [result0, result1, result2] =at::_ops::miopen_batch_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor miopen_convolution_transpose(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_convolution_transpose");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::miopen_convolution_transpose::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor miopen_convolution_add_relu(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const at::Tensor & z, const ::std::optional<at::Scalar> & alpha, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_convolution_add_relu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "z", z);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::miopen_convolution_add_relu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, z, alpha, bias, stride, padding, dilation, groups);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor mT(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mT");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mT::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor deg2rad(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::deg2rad");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::deg2rad::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & deg2rad_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::deg2rad");
    } else {
      op_name = c10::Symbol::fromQualString("aten::deg2rad_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("deg2rad_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::deg2rad_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & deg2rad_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::deg2rad");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("deg2rad_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::deg2rad_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _reshape_copy(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_reshape_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_reshape_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor relu(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::relu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::relu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & relu_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::relu");
    } else {
      op_name = c10::Symbol::fromQualString("aten::relu_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("relu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::relu_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor sinc(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sinc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sinc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & sinc_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::sinc");
    } else {
      op_name = c10::Symbol::fromQualString("aten::sinc_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sinc_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sinc_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & sinc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sinc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sinc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sinc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor slice_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slice");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::slice_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, start, end, step);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor select_scatter(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, int64_t dim, c10::SymInt index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::select_scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::select_scatter::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, dim, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor prod(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::prod");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::prod::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor prod_dim_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::prod");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::prod_dim_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & prod_out_int_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::prod");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("prod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::prod_int_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor prod_dim_Dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::prod");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::prod_dim_Dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & prod_out_Dimname_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::prod");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("prod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::prod_Dimname_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor trapezoid_x(c10::DispatchKeySet ks, const at::Tensor & y, const at::Tensor & x, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::trapezoid");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "y", y);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::trapezoid_x::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), y, x, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor trapezoid_dx(c10::DispatchKeySet ks, const at::Tensor & y, const at::Scalar & dx, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::trapezoid");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "y", y);
    jit::tracer::addInputs(node, "dx", dx);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::trapezoid_dx::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), y, dx, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _nested_tensor_from_mask(c10::DispatchKeySet ks, const at::Tensor & t, const at::Tensor & mask, bool mask_check) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_tensor_from_mask");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "t", t);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "mask_check", mask_check);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_tensor_from_mask::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), t, mask, mask_check);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool _nested_tensor_from_mask_left_aligned(c10::DispatchKeySet ks, const at::Tensor & t, const at::Tensor & mask) {
  auto result =at::_ops::_nested_tensor_from_mask_left_aligned::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), t, mask);
  return result;
}
at::Tensor _nested_tensor_size(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_tensor_size");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_tensor_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _nested_view_from_buffer_copy(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & nested_size, const at::Tensor & nested_strides, const at::Tensor & offsets) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_view_from_buffer_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "nested_size", nested_size);
    jit::tracer::addInputs(node, "nested_strides", nested_strides);
    jit::tracer::addInputs(node, "offsets", offsets);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_view_from_buffer_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, nested_size, nested_strides, offsets);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim_consecutive(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool return_inverse, bool return_counts) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unique_dim_consecutive");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::unique_dim_consecutive::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, return_inverse, return_counts);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor _unsafe_view(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_unsafe_view");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_unsafe_view::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _batch_norm_no_update(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_batch_norm_no_update");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::_batch_norm_no_update::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, momentum, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
at::Tensor & heaviside_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & values, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::heaviside");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "values", values);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("heaviside_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::heaviside_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, values, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor heaviside(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & values) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::heaviside");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "values", values);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::heaviside::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, values);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & heaviside_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & values) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::heaviside");
    } else {
      op_name = c10::Symbol::fromQualString("aten::heaviside_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "values", values);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("heaviside_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::heaviside_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, values);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor sparse_coo_tensor_size(c10::DispatchKeySet ks, at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_coo_tensor");
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
  auto result =at::_ops::sparse_coo_tensor_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sparse_coo_tensor_indices(c10::DispatchKeySet ks, const at::Tensor & indices, const at::Tensor & values, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<bool> is_coalesced) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_coo_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
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
  auto result =at::_ops::sparse_coo_tensor_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), indices, values, dtype, layout, device, pin_memory, is_coalesced);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sparse_coo_tensor_indices_size(c10::DispatchKeySet ks, const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<bool> is_coalesced) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_coo_tensor");
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
  auto result =at::_ops::sparse_coo_tensor_indices_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), indices, values, size, dtype, layout, device, pin_memory, is_coalesced);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
const at::Tensor & sparse_resize_and_clear_(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::sparse_resize_and_clear");
    } else {
      op_name = c10::Symbol::fromQualString("aten::sparse_resize_and_clear_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sparse_resize_and_clear_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sparse_resize_and_clear_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, sparse_dim, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
int64_t sparse_dim(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::sparse_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor _to_sparse_csr(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_csr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_to_sparse_csr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor to_sparse_bsr(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::to_sparse_bsr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "blocksize", blocksize);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::to_sparse_bsr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, blocksize, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor mkldnn_reorder_conv3d_weight(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::OptionalSymIntArrayRef input_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_reorder_conv3d_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "input_size", input_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mkldnn_reorder_conv3d_weight::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, stride, dilation, groups, input_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
double q_scale(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::q_scale::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor _make_per_tensor_quantized_tensor(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_make_per_tensor_quantized_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_make_per_tensor_quantized_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _saturate_weight_to_fp16(c10::DispatchKeySet ks, const at::Tensor & weight) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_saturate_weight_to_fp16");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_saturate_weight_to_fp16::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _autocast_to_reduced_precision(c10::DispatchKeySet ks, const at::Tensor & self, bool cuda_enabled, bool cpu_enabled, at::ScalarType cuda_dtype, at::ScalarType cpu_dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_autocast_to_reduced_precision");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "cuda_enabled", cuda_enabled);
    jit::tracer::addInputs(node, "cpu_enabled", cpu_enabled);
    jit::tracer::addInputs(node, "cuda_dtype", cuda_dtype);
    jit::tracer::addInputs(node, "cpu_dtype", cpu_dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_autocast_to_reduced_precision::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_fused_lstm_cell_backward(c10::DispatchKeySet ks, const ::std::optional<at::Tensor> & grad_hy, const ::std::optional<at::Tensor> & grad_cy, const at::Tensor & cx, const at::Tensor & cy, const at::Tensor & workspace, bool has_bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_thnn_fused_lstm_cell_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "grad_cy", grad_cy);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "cy", cy);
    jit::tracer::addInputs(node, "workspace", workspace);
    jit::tracer::addInputs(node, "has_bias", has_bias);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3, result4] =at::_ops::_thnn_fused_lstm_cell_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_hy, grad_cy, cx, cy, workspace, has_bias);
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
at::Tensor quantized_rnn_relu_cell(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const at::Tensor & b_ih, const at::Tensor & b_hh, const at::Tensor & packed_ih, const at::Tensor & packed_hh, const at::Tensor & col_offsets_ih, const at::Tensor & col_offsets_hh, const at::Scalar & scale_ih, const at::Scalar & scale_hh, const at::Scalar & zero_point_ih, const at::Scalar & zero_point_hh) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantized_rnn_relu_cell");
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
  auto result =at::_ops::quantized_rnn_relu_cell::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, w_ih, w_hh, b_ih, b_hh, packed_ih, packed_hh, col_offsets_ih, col_offsets_hh, scale_ih, scale_hh, zero_point_ih, zero_point_hh);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & masked_fill__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::masked_fill");
    } else {
      op_name = c10::Symbol::fromQualString("aten::masked_fill_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::masked_fill__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor masked_fill_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::masked_fill");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::masked_fill_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & masked_fill__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::masked_fill");
    } else {
      op_name = c10::Symbol::fromQualString("aten::masked_fill_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_fill_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::masked_fill__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor masked_fill_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::masked_fill");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::masked_fill_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & diag_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::diag");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("diag_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::diag_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, diagonal, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor diag(c10::DispatchKeySet ks, const at::Tensor & self, int64_t diagonal) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::diag");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::diag::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, diagonal);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor trace(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::trace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::trace::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> svd_out_U(c10::DispatchKeySet ks, const at::Tensor & self, bool some, bool compute_uv, at::Tensor & U, at::Tensor & S, at::Tensor & V) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::svd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    jit::tracer::addInputs(node, "compute_uv", compute_uv);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "U", U);
      jit::tracer::addInputs(node, "S", S);
      jit::tracer::addInputs(node, "V", V);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("svd_out", U);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::svd_U::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, some, compute_uv, U, S, V);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, U);
    jit::tracer::addOutput(node, S);
    jit::tracer::addOutput(node, V);
  }
  return std::forward_as_tuple(U, S, V);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> svd(c10::DispatchKeySet ks, const at::Tensor & self, bool some, bool compute_uv) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::svd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "some", some);
    jit::tracer::addInputs(node, "compute_uv", compute_uv);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [U, S, V] =at::_ops::svd::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, some, compute_uv);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, U);
    jit::tracer::addOutput(node, S);
    jit::tracer::addOutput(node, V);
  }
  return std::make_tuple(std::move(U), std::move(S), std::move(V));
}
at::Tensor dist(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dist");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::dist::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _histogramdd_from_bin_cts(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_histogramdd_from_bin_cts");
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
  auto result =at::_ops::_histogramdd_from_bin_cts::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, range, weight, density);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & remainder_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::remainder");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::remainder_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor remainder_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::remainder");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::remainder_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & remainder__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::remainder");
    } else {
      op_name = c10::Symbol::fromQualString("aten::remainder_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::remainder__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & remainder_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::remainder");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::remainder_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor remainder_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::remainder");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::remainder_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & remainder__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::remainder");
    } else {
      op_name = c10::Symbol::fromQualString("aten::remainder_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::remainder__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor remainder_Scalar_Tensor(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::remainder");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::remainder_Scalar_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor nanquantile(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nanquantile");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "interpolation", interpolation);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nanquantile::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & nanquantile_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nanquantile");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "interpolation", interpolation);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nanquantile_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nanquantile_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor nanquantile_scalar(c10::DispatchKeySet ks, const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nanquantile");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "interpolation", interpolation);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nanquantile_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & nanquantile_out_scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nanquantile");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "q", q);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "interpolation", interpolation);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nanquantile_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nanquantile_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & renorm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::renorm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("renorm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::renorm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, maxnorm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor renorm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::renorm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::renorm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, maxnorm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & renorm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::renorm");
    } else {
      op_name = c10::Symbol::fromQualString("aten::renorm_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "maxnorm", maxnorm);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("renorm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::renorm_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, maxnorm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor unfold(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unfold");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dimension", dimension);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unfold::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dimension, size, step);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & float_power_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::float_power");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("float_power_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::float_power_Tensor_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor float_power_Tensor_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::float_power");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::float_power_Tensor_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & float_power_out_Scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::float_power");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("float_power_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::float_power_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor float_power_Scalar(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::float_power");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::float_power_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & float_power_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::float_power");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("float_power_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::float_power_Tensor_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor float_power_Tensor_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::float_power");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::float_power_Tensor_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & float_power__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::float_power");
    } else {
      op_name = c10::Symbol::fromQualString("aten::float_power_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("float_power_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::float_power__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & float_power__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & exponent) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::float_power");
    } else {
      op_name = c10::Symbol::fromQualString("aten::float_power_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "exponent", exponent);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("float_power_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::float_power__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, exponent);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
::std::vector<at::Tensor> _foreach_sinh(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_sinh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_sinh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_sinh_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_sinh_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
at::Tensor & glu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::glu_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("glu_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::glu_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, dim, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor glu_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::glu_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::glu_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hardtanh_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardtanh_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardtanh_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardtanh_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, min_val, max_val, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor hardtanh_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardtanh_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hardtanh_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, min_val, max_val);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & softplus_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::softplus");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("softplus_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::softplus_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, beta, threshold, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor softplus(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::softplus");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "threshold", threshold);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::softplus::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, beta, threshold);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & avg_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::avg_pool3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "ceil_mode", ceil_mode);
    jit::tracer::addInputs(node, "count_include_pad", count_include_pad);
    jit::tracer::addInputs(node, "divisor_override", divisor_override);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("avg_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::avg_pool3d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor avg_pool3d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, ::std::optional<int64_t> divisor_override) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::avg_pool3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
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
  auto result =at::_ops::avg_pool3d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> max_pool3d_with_indices_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool3d_with_indices");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool3d_with_indices_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::max_pool3d_with_indices_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(out, indices);
}
::std::tuple<at::Tensor,at::Tensor> max_pool3d_with_indices(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool3d_with_indices");
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
  auto [result0, result1] =at::_ops::max_pool3d_with_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor _upsample_bilinear2d_aa_vec(c10::DispatchKeySet ks, const at::Tensor & input, at::OptionalSymIntArrayRef output_size, bool align_corners, ::std::optional<at::ArrayRef<double>> scale_factors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_bilinear2d_aa");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "align_corners", align_corners);
    jit::tracer::addInputs(node, "scale_factors", scale_factors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_upsample_bilinear2d_aa_vec::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, output_size, align_corners, scale_factors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _upsample_bilinear2d_aa_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_bilinear2d_aa");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_upsample_bilinear2d_aa_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_upsample_bilinear2d_aa_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, align_corners, scales_h, scales_w, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _upsample_bilinear2d_aa(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_bilinear2d_aa");
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
  auto result =at::_ops::_upsample_bilinear2d_aa::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, align_corners, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & upsample_nearest2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_nearest2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("upsample_nearest2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::upsample_nearest2d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, scales_h, scales_w, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor upsample_nearest2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::upsample_nearest2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::upsample_nearest2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & slow_conv3d_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & output) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slow_conv3d_forward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("slow_conv3d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::slow_conv3d_forward_output::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, output);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
at::Tensor slow_conv3d_forward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slow_conv3d_forward");
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
  auto result =at::_ops::slow_conv3d_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & im2col_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::im2col");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("im2col_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::im2col_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, dilation, padding, stride, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor im2col(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::im2col");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "kernel_size", kernel_size);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::im2col::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, dilation, padding, stride);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor isneginf(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isneginf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::isneginf::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & isneginf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isneginf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("isneginf_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::isneginf_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_erfcx(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_erfcx");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_erfcx::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_erfcx_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_erfcx");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_erfcx_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_erfcx_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_i1(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_i1");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_i1::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_i1_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_i1");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_i1_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_i1_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_log_softmax(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_log_softmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_log_softmax::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_gammaincc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_gammaincc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_gammaincc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_gammaincc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_gammaincc(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_gammaincc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_gammaincc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_multigammaln(c10::DispatchKeySet ks, const at::Tensor & self, int64_t p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_multigammaln");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_multigammaln::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_multigammaln_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t p, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_multigammaln");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_multigammaln_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_multigammaln_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_fftn(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fftn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_fftn::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_fftn_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fftn");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_fftn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_fftn_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_fftshift(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fftshift");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_fftshift::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> linalg_lu_factor(c10::DispatchKeySet ks, const at::Tensor & A, bool pivot) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_lu_factor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "pivot", pivot);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [LU, pivots] =at::_ops::linalg_lu_factor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, pivot);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, LU);
    jit::tracer::addOutput(node, pivots);
  }
  return std::make_tuple(std::move(LU), std::move(pivots));
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_lu_factor_out_out(c10::DispatchKeySet ks, const at::Tensor & A, bool pivot, at::Tensor & LU, at::Tensor & pivots) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_lu_factor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "pivot", pivot);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "LU", LU);
      jit::tracer::addInputs(node, "pivots", pivots);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_lu_factor_out", LU);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_lu_factor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, pivot, LU, pivots);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, LU);
    jit::tracer::addOutput(node, pivots);
  }
  return std::forward_as_tuple(LU, pivots);
}
at::Tensor linalg_det(c10::DispatchKeySet ks, const at::Tensor & A) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_det");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_det::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_det_out_out(c10::DispatchKeySet ks, const at::Tensor & A, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_det");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_det_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_det_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _linalg_slogdet(c10::DispatchKeySet ks, const at::Tensor & A) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_linalg_slogdet");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [sign, logabsdet, LU, pivots] =at::_ops::_linalg_slogdet::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, sign);
    jit::tracer::addOutput(node, logabsdet);
    jit::tracer::addOutput(node, LU);
    jit::tracer::addOutput(node, pivots);
  }
  return std::make_tuple(std::move(sign), std::move(logabsdet), std::move(LU), std::move(pivots));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _linalg_slogdet_out_sign(c10::DispatchKeySet ks, const at::Tensor & A, at::Tensor & sign, at::Tensor & logabsdet, at::Tensor & LU, at::Tensor & pivots) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_linalg_slogdet");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "sign", sign);
      jit::tracer::addInputs(node, "logabsdet", logabsdet);
      jit::tracer::addInputs(node, "LU", LU);
      jit::tracer::addInputs(node, "pivots", pivots);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_linalg_slogdet_out", sign);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_linalg_slogdet_sign::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, sign, logabsdet, LU, pivots);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, sign);
    jit::tracer::addOutput(node, logabsdet);
    jit::tracer::addOutput(node, LU);
    jit::tracer::addOutput(node, pivots);
  }
  return std::forward_as_tuple(sign, logabsdet, LU, pivots);
}
at::Tensor ger(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ger");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec2", vec2);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ger::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ger_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec2, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ger");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec2", vec2);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ger_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ger_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec2, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _linalg_solve_ex(c10::DispatchKeySet ks, const at::Tensor & A, const at::Tensor & B, bool left, bool check_errors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_linalg_solve_ex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "B", B);
    jit::tracer::addInputs(node, "left", left);
    jit::tracer::addInputs(node, "check_errors", check_errors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result, LU, pivots, info] =at::_ops::_linalg_solve_ex::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, B, left, check_errors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
    jit::tracer::addOutput(node, LU);
    jit::tracer::addOutput(node, pivots);
    jit::tracer::addOutput(node, info);
  }
  return std::make_tuple(std::move(result), std::move(LU), std::move(pivots), std::move(info));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _linalg_solve_ex_out_result(c10::DispatchKeySet ks, const at::Tensor & A, const at::Tensor & B, bool left, bool check_errors, at::Tensor & result, at::Tensor & LU, at::Tensor & pivots, at::Tensor & info) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_linalg_solve_ex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "B", B);
    jit::tracer::addInputs(node, "left", left);
    jit::tracer::addInputs(node, "check_errors", check_errors);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "result", result);
      jit::tracer::addInputs(node, "LU", LU);
      jit::tracer::addInputs(node, "pivots", pivots);
      jit::tracer::addInputs(node, "info", info);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_linalg_solve_ex_out", result);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_linalg_solve_ex_result::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, B, left, check_errors, result, LU, pivots, info);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
    jit::tracer::addOutput(node, LU);
    jit::tracer::addOutput(node, pivots);
    jit::tracer::addOutput(node, info);
  }
  return std::forward_as_tuple(result, LU, pivots, info);
}
at::Tensor nested_to_padded_tensor(c10::DispatchKeySet ks, const at::Tensor & self, double padding, at::OptionalIntArrayRef output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nested_to_padded_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nested_to_padded_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _test_autograd_multiple_dispatch_view(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_autograd_multiple_dispatch_view");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_test_autograd_multiple_dispatch_view::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor diagonal_copy(c10::DispatchKeySet ks, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::diagonal_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::diagonal_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, offset, dim1, dim2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor permute_copy(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dims) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::permute_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::permute_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dims);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor slice_copy_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slice_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::slice_copy_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, start, end, step);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor t_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::t_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::t_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _nested_from_padded_tensor(c10::DispatchKeySet ks, const at::Tensor & padded, const at::Tensor & offsets, const at::Tensor & dummy, int64_t ragged_idx, const ::std::optional<at::Tensor> & min_seqlen, const ::std::optional<at::Tensor> & max_seqlen, ::std::optional<c10::SymInt> sum_S) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_from_padded_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "padded", padded);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "dummy", dummy);
    jit::tracer::addInputs(node, "ragged_idx", ragged_idx);
    jit::tracer::addInputs(node, "min_seqlen", min_seqlen);
    jit::tracer::addInputs(node, "max_seqlen", max_seqlen);
    jit::tracer::addInputs(node, "sum_S", sum_S);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_from_padded_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), padded, offsets, dummy, ragged_idx, min_seqlen, max_seqlen, sum_S);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _scaled_dot_product_attention_math(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, const ::std::optional<at::Tensor> & dropout_mask, ::std::optional<double> scale, bool enable_gqa) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_dot_product_attention_math");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "attn_mask", attn_mask);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "dropout_mask", dropout_mask);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "enable_gqa", enable_gqa);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_scaled_dot_product_attention_math::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, attn_mask, dropout_p, is_causal, dropout_mask, scale, enable_gqa);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _cudnn_attention_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & out, const at::Tensor & logsumexp, const at::Tensor & philox_seed, const at::Tensor & philox_offset, const at::Tensor & attn_bias, const at::Tensor & cum_seq_q, const at::Tensor & cum_seq_k, c10::SymInt max_q, c10::SymInt max_k, double dropout_p, bool is_causal, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cudnn_attention_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "out", out);
    jit::tracer::addInputs(node, "logsumexp", logsumexp);
    jit::tracer::addInputs(node, "philox_seed", philox_seed);
    jit::tracer::addInputs(node, "philox_offset", philox_offset);
    jit::tracer::addInputs(node, "attn_bias", attn_bias);
    jit::tracer::addInputs(node, "cum_seq_q", cum_seq_q);
    jit::tracer::addInputs(node, "cum_seq_k", cum_seq_k);
    jit::tracer::addInputs(node, "max_q", max_q);
    jit::tracer::addInputs(node, "max_k", max_k);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::_cudnn_attention_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, query, key, value, out, logsumexp, philox_seed, philox_offset, attn_bias, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor _foobar(c10::DispatchKeySet ks, const at::Tensor & self, bool arg1, bool arg2, bool arg3) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foobar");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "arg1", arg1);
    jit::tracer::addInputs(node, "arg2", arg2);
    jit::tracer::addInputs(node, "arg3", arg3);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foobar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, arg1, arg2, arg3);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> convolution_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, at::OptionalSymIntArrayRef bias_sizes, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed, c10::SymIntArrayRef output_padding, c10::SymInt groups, ::std::array<bool,3> output_mask, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::convolution_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias_sizes", bias_sizes);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("convolution_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::convolution_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & _copy_from_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & dst, bool non_blocking, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_copy_from");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dst", dst);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_copy_from_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_copy_from_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dst, non_blocking, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _mps_convolution_transpose_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mps_convolution_transpose");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mps_convolution_transpose_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_mps_convolution_transpose_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, padding, output_padding, stride, dilation, groups, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & floor_divide_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::floor_divide");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("floor_divide_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::floor_divide_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & full_out_names_out(c10::DispatchKeySet ks, at::IntArrayRef size, const at::Scalar & fill_value, ::std::optional<at::DimnameList> names, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::full");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("full_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::full_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, fill_value, names, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & full_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & fill_value, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::full_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("full_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::full_like_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, fill_value, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & grid_sampler_2d_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::grid_sampler_2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "grid", grid);
    jit::tracer::addInputs(node, "interpolation_mode", interpolation_mode);
    jit::tracer::addInputs(node, "padding_mode", padding_mode);
    jit::tracer::addInputs(node, "align_corners", align_corners);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("grid_sampler_2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::grid_sampler_2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, grid, interpolation_mode, padding_mode, align_corners, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & kaiser_window_out_out(c10::DispatchKeySet ks, int64_t window_length, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kaiser_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("kaiser_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::kaiser_window_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & kaiser_window_out_periodic_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kaiser_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("kaiser_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::kaiser_window_periodic_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & kaiser_window_out_beta_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double beta, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kaiser_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "beta", beta);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("kaiser_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::kaiser_window_beta_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, beta, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & max_pool2d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::max_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("max_pool2d_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::max_pool2d_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> miopen_batch_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double exponential_average_factor, double epsilon, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_batch_norm");
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
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("miopen_batch_norm_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::miopen_batch_norm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & miopen_convolution_transpose_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_convolution_transpose");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "output_padding", output_padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("miopen_convolution_transpose_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::miopen_convolution_transpose_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & relu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::relu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("relu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::relu_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & select_scatter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & src, int64_t dim, c10::SymInt index, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::select_scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("select_scatter_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::select_scatter_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, src, dim, index, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & prod_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::prod");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("prod_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::prod_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _nested_tensor_from_mask_out_out(c10::DispatchKeySet ks, const at::Tensor & t, const at::Tensor & mask, bool mask_check, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_tensor_from_mask");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "t", t);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "mask_check", mask_check);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_nested_tensor_from_mask_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_nested_tensor_from_mask_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), t, mask, mask_check, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _nested_tensor_size_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_tensor_size");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_nested_tensor_size_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_nested_tensor_size_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _nested_view_from_buffer_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & nested_size, const at::Tensor & nested_strides, const at::Tensor & offsets, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_view_from_buffer_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "nested_size", nested_size);
    jit::tracer::addInputs(node, "nested_strides", nested_strides);
    jit::tracer::addInputs(node, "offsets", offsets);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_nested_view_from_buffer_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_nested_view_from_buffer_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, nested_size, nested_strides, offsets, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> unique_dim_consecutive_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool return_inverse, bool return_counts, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unique_dim_consecutive");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("unique_dim_consecutive_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::unique_dim_consecutive_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, return_inverse, return_counts, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & _unsafe_view_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_unsafe_view");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_unsafe_view_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_unsafe_view_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _batch_norm_no_update_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, double eps, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_batch_norm_no_update");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "eps", eps);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
      jit::tracer::addInputs(node, "out3", out3);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_batch_norm_no_update_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_batch_norm_no_update_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, momentum, eps, out0, out1, out2, out3);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
    jit::tracer::addOutput(node, out3);
  }
  return std::forward_as_tuple(out0, out1, out2, out3);
}
at::Tensor & sparse_coo_tensor_out_size_out(c10::DispatchKeySet ks, at::IntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_coo_tensor");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("sparse_coo_tensor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sparse_coo_tensor_size_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
const at::Tensor & sparse_resize_and_clear_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim, const at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_resize_and_clear");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sparse_resize_and_clear_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sparse_resize_and_clear_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, sparse_dim, dense_dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor sparse_resize_and_clear(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_resize_and_clear");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "sparse_dim", sparse_dim);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sparse_resize_and_clear::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, sparse_dim, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _to_sparse_csr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dense_dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_csr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_to_sparse_csr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_to_sparse_csr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dense_dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & mkldnn_reorder_conv3d_weight_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::OptionalSymIntArrayRef input_size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_reorder_conv3d_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "input_size", input_size);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mkldnn_reorder_conv3d_weight_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mkldnn_reorder_conv3d_weight_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, stride, dilation, groups, input_size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _make_per_tensor_quantized_tensor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_make_per_tensor_quantized_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_make_per_tensor_quantized_tensor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_make_per_tensor_quantized_tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & masked_fill_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, const at::Scalar & value, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::masked_fill");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_fill_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::masked_fill_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, value, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & masked_fill_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, const at::Tensor & value, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::masked_fill");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "value", value);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("masked_fill_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::masked_fill_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, value, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & trace_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::trace");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("trace_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::trace_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & dist_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & p, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dist");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "p", p);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("dist_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::dist_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, p, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _histogramdd_from_bin_cts_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef bins, ::std::optional<at::ArrayRef<double>> range, const ::std::optional<at::Tensor> & weight, bool density, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_histogramdd_from_bin_cts");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "range", range);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "density", density);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_histogramdd_from_bin_cts_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_histogramdd_from_bin_cts_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, range, weight, density, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & remainder_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::remainder");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("remainder_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::remainder_Scalar_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _foreach_sinh_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_sinh_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
at::Tensor & diagonal_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::diagonal_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "offset", offset);
    jit::tracer::addInputs(node, "dim1", dim1);
    jit::tracer::addInputs(node, "dim2", dim2);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("diagonal_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::diagonal_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, offset, dim1, dim2, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & permute_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dims, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::permute_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("permute_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::permute_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dims, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & slice_copy_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<c10::SymInt> start, ::std::optional<c10::SymInt> end, c10::SymInt step, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slice_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "start", start);
    jit::tracer::addInputs(node, "end", end);
    jit::tracer::addInputs(node, "step", step);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("slice_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::slice_copy_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, start, end, step, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & t_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::t_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("t_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::t_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _foobar_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool arg1, bool arg2, bool arg3, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foobar");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "arg1", arg1);
    jit::tracer::addInputs(node, "arg2", arg2);
    jit::tracer::addInputs(node, "arg3", arg3);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_foobar_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_foobar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, arg1, arg2, arg3, out);
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
  m.impl("_assert_async",
         TORCH_FN(TraceType::_assert_async)
  );
  m.impl("_assert_async.msg",
         TORCH_FN(TraceType::_assert_async_msg)
  );
  m.impl("_functional_assert_scalar",
         TORCH_FN(TraceType::_functional_assert_scalar)
  );
  m.impl("_reshape_from_tensor",
         TORCH_FN(TraceType::_reshape_from_tensor)
  );
  m.impl("view_as_real",
         TORCH_FN(TraceType::view_as_real)
  );
  m.impl("conj_physical",
         TORCH_FN(TraceType::conj_physical)
  );
  m.impl("conj_physical.out",
         TORCH_FN(TraceType::conj_physical_out_out)
  );
  m.impl("conj_physical_",
         TORCH_FN(TraceType::conj_physical_)
  );
  m.impl("acos",
         TORCH_FN(TraceType::acos)
  );
  m.impl("acos_",
         TORCH_FN(TraceType::acos_)
  );
  m.impl("acos.out",
         TORCH_FN(TraceType::acos_out_out)
  );
  m.impl("arccos",
         TORCH_FN(TraceType::arccos)
  );
  m.impl("arccos_",
         TORCH_FN(TraceType::arccos_)
  );
  m.impl("arccos.out",
         TORCH_FN(TraceType::arccos_out_out)
  );
  m.impl("arccosh",
         TORCH_FN(TraceType::arccosh)
  );
  m.impl("arccosh_",
         TORCH_FN(TraceType::arccosh_)
  );
  m.impl("arccosh.out",
         TORCH_FN(TraceType::arccosh_out_out)
  );
  m.impl("asin",
         TORCH_FN(TraceType::asin)
  );
  m.impl("asin_",
         TORCH_FN(TraceType::asin_)
  );
  m.impl("asin.out",
         TORCH_FN(TraceType::asin_out_out)
  );
  m.impl("logical_xor",
         TORCH_FN(TraceType::logical_xor)
  );
  m.impl("logical_xor_",
         TORCH_FN(TraceType::logical_xor_)
  );
  m.impl("logical_xor.out",
         TORCH_FN(TraceType::logical_xor_out_out)
  );
  m.impl("broadcast_to",
         TORCH_FN(TraceType::broadcast_to)
  );
  m.impl("contiguous",
         TORCH_FN(TraceType::contiguous)
  );
  m.impl("convolution_backward",
         TORCH_FN(TraceType::convolution_backward)
  );
  m.impl("_convolution_double_backward",
         TORCH_FN(TraceType::_convolution_double_backward)
  );
  m.impl("conv2d",
         TORCH_FN(TraceType::conv2d)
  );
  m.impl("conv2d.padding",
         TORCH_FN(TraceType::conv2d_padding)
  );
  m.impl("_copy_from",
         TORCH_FN(TraceType::_copy_from)
  );
  m.impl("corrcoef",
         TORCH_FN(TraceType::corrcoef)
  );
  m.impl("_mps_convolution_transpose",
         TORCH_FN(TraceType::_mps_convolution_transpose)
  );
  m.impl("cummaxmin_backward",
         TORCH_FN(TraceType::cummaxmin_backward)
  );
  m.impl("cumprod_backward",
         TORCH_FN(TraceType::cumprod_backward)
  );
  m.impl("_rowwise_prune",
         TORCH_FN(TraceType::_rowwise_prune)
  );
  m.impl("row_stack",
         TORCH_FN(TraceType::row_stack)
  );
  m.impl("row_stack.out",
         TORCH_FN(TraceType::row_stack_out_out)
  );
  m.impl("_embedding_bag_backward",
         TORCH_FN(TraceType::_embedding_bag_backward)
  );
  m.impl("floor_divide",
         TORCH_FN(TraceType::floor_divide)
  );
  m.impl("floor_divide_.Tensor",
         TORCH_FN(TraceType::floor_divide__Tensor)
  );
  m.impl("floor_divide.out",
         TORCH_FN(TraceType::floor_divide_out_out)
  );
  m.impl("floor_divide.Scalar",
         TORCH_FN(TraceType::floor_divide_Scalar)
  );
  m.impl("floor_divide_.Scalar",
         TORCH_FN(TraceType::floor_divide__Scalar)
  );
  m.impl("full.names",
         TORCH_FN(TraceType::full_names)
  );
  m.impl("full",
         TORCH_FN(TraceType::full)
  );
  m.impl("full.out",
         TORCH_FN(TraceType::full_out_out)
  );
  m.impl("full_like",
         TORCH_FN(TraceType::full_like)
  );
  m.impl("grid_sampler_2d",
         TORCH_FN(TraceType::grid_sampler_2d)
  );
  m.impl("kaiser_window",
         TORCH_FN(TraceType::kaiser_window)
  );
  m.impl("kaiser_window.periodic",
         TORCH_FN(TraceType::kaiser_window_periodic)
  );
  m.impl("kaiser_window.beta",
         TORCH_FN(TraceType::kaiser_window_beta)
  );
  m.impl("_fft_c2r",
         TORCH_FN(TraceType::_fft_c2r)
  );
  m.impl("_fft_c2r.out",
         TORCH_FN(TraceType::_fft_c2r_out_out)
  );
  m.impl("_unsafe_masked_index_put_accumulate",
         TORCH_FN(TraceType::_unsafe_masked_index_put_accumulate)
  );
  m.impl("is_same_size",
         TORCH_FN(TraceType::is_same_size)
  );
  m.impl("get_device",
         TORCH_FN(TraceType::get_device)
  );
  m.impl("is_contiguous",
         TORCH_FN(TraceType::is_contiguous)
  );
  m.impl("is_contiguous.memory_format",
         TORCH_FN(TraceType::is_contiguous_memory_format)
  );
  m.impl("kl_div",
         TORCH_FN(TraceType::kl_div)
  );
  m.impl("_fused_rms_norm",
         TORCH_FN(TraceType::_fused_rms_norm)
  );
  m.impl("_cslt_compress",
         TORCH_FN(TraceType::_cslt_compress)
  );
  m.impl("margin_ranking_loss",
         TORCH_FN(TraceType::margin_ranking_loss)
  );
  m.impl("matmul",
         TORCH_FN(TraceType::matmul)
  );
  m.impl("matmul.out",
         TORCH_FN(TraceType::matmul_out_out)
  );
  m.impl("matrix_exp",
         TORCH_FN(TraceType::matrix_exp)
  );
  m.impl("_compute_linear_combination",
         TORCH_FN(TraceType::_compute_linear_combination)
  );
  m.impl("_compute_linear_combination.out",
         TORCH_FN(TraceType::_compute_linear_combination_out_out)
  );
  m.impl("max_pool2d_backward",
         TORCH_FN(TraceType::max_pool2d_backward)
  );
  m.impl("miopen_batch_norm",
         TORCH_FN(TraceType::miopen_batch_norm)
  );
  m.impl("miopen_convolution_transpose",
         TORCH_FN(TraceType::miopen_convolution_transpose)
  );
  m.impl("miopen_convolution_add_relu",
         TORCH_FN(TraceType::miopen_convolution_add_relu)
  );
  m.impl("mT",
         TORCH_FN(TraceType::mT)
  );
  m.impl("deg2rad",
         TORCH_FN(TraceType::deg2rad)
  );
  m.impl("deg2rad_",
         TORCH_FN(TraceType::deg2rad_)
  );
  m.impl("deg2rad.out",
         TORCH_FN(TraceType::deg2rad_out_out)
  );
  m.impl("_reshape_copy",
         TORCH_FN(TraceType::_reshape_copy)
  );
  m.impl("relu",
         TORCH_FN(TraceType::relu)
  );
  m.impl("relu_",
         TORCH_FN(TraceType::relu_)
  );
  m.impl("sinc",
         TORCH_FN(TraceType::sinc)
  );
  m.impl("sinc_",
         TORCH_FN(TraceType::sinc_)
  );
  m.impl("sinc.out",
         TORCH_FN(TraceType::sinc_out_out)
  );
  m.impl("slice.Tensor",
         TORCH_FN(TraceType::slice_Tensor)
  );
  m.impl("select_scatter",
         TORCH_FN(TraceType::select_scatter)
  );
  m.impl("prod",
         TORCH_FN(TraceType::prod)
  );
  m.impl("prod.dim_int",
         TORCH_FN(TraceType::prod_dim_int)
  );
  m.impl("prod.int_out",
         TORCH_FN(TraceType::prod_out_int_out)
  );
  m.impl("prod.dim_Dimname",
         TORCH_FN(TraceType::prod_dim_Dimname)
  );
  m.impl("prod.Dimname_out",
         TORCH_FN(TraceType::prod_out_Dimname_out)
  );
  m.impl("trapezoid.x",
         TORCH_FN(TraceType::trapezoid_x)
  );
  m.impl("trapezoid.dx",
         TORCH_FN(TraceType::trapezoid_dx)
  );
  m.impl("_nested_tensor_from_mask",
         TORCH_FN(TraceType::_nested_tensor_from_mask)
  );
  m.impl("_nested_tensor_from_mask_left_aligned",
         TORCH_FN(TraceType::_nested_tensor_from_mask_left_aligned)
  );
  m.impl("_nested_tensor_size",
         TORCH_FN(TraceType::_nested_tensor_size)
  );
  m.impl("_nested_view_from_buffer_copy",
         TORCH_FN(TraceType::_nested_view_from_buffer_copy)
  );
  m.impl("unique_dim_consecutive",
         TORCH_FN(TraceType::unique_dim_consecutive)
  );
  m.impl("_unsafe_view",
         TORCH_FN(TraceType::_unsafe_view)
  );
  m.impl("_batch_norm_no_update",
         TORCH_FN(TraceType::_batch_norm_no_update)
  );
  m.impl("heaviside.out",
         TORCH_FN(TraceType::heaviside_out_out)
  );
  m.impl("heaviside",
         TORCH_FN(TraceType::heaviside)
  );
  m.impl("heaviside_",
         TORCH_FN(TraceType::heaviside_)
  );
  m.impl("sparse_coo_tensor.size",
         TORCH_FN(TraceType::sparse_coo_tensor_size)
  );
  m.impl("sparse_coo_tensor.indices",
         TORCH_FN(TraceType::sparse_coo_tensor_indices)
  );
  m.impl("sparse_coo_tensor.indices_size",
         TORCH_FN(TraceType::sparse_coo_tensor_indices_size)
  );
  m.impl("sparse_resize_and_clear_",
         TORCH_FN(TraceType::sparse_resize_and_clear_)
  );
  m.impl("sparse_dim",
         TORCH_FN(TraceType::sparse_dim)
  );
  m.impl("_to_sparse_csr",
         TORCH_FN(TraceType::_to_sparse_csr)
  );
  m.impl("to_sparse_bsr",
         TORCH_FN(TraceType::to_sparse_bsr)
  );
  m.impl("mkldnn_reorder_conv3d_weight",
         TORCH_FN(TraceType::mkldnn_reorder_conv3d_weight)
  );
  m.impl("q_scale",
         TORCH_FN(TraceType::q_scale)
  );
  m.impl("_make_per_tensor_quantized_tensor",
         TORCH_FN(TraceType::_make_per_tensor_quantized_tensor)
  );
  m.impl("_saturate_weight_to_fp16",
         TORCH_FN(TraceType::_saturate_weight_to_fp16)
  );
  m.impl("_autocast_to_reduced_precision",
         TORCH_FN(TraceType::_autocast_to_reduced_precision)
  );
  m.impl("_thnn_fused_lstm_cell_backward",
         TORCH_FN(TraceType::_thnn_fused_lstm_cell_backward)
  );
  m.impl("quantized_rnn_relu_cell",
         TORCH_FN(TraceType::quantized_rnn_relu_cell)
  );
  m.impl("masked_fill_.Scalar",
         TORCH_FN(TraceType::masked_fill__Scalar)
  );
  m.impl("masked_fill.Scalar",
         TORCH_FN(TraceType::masked_fill_Scalar)
  );
  m.impl("masked_fill_.Tensor",
         TORCH_FN(TraceType::masked_fill__Tensor)
  );
  m.impl("masked_fill.Tensor",
         TORCH_FN(TraceType::masked_fill_Tensor)
  );
  m.impl("diag.out",
         TORCH_FN(TraceType::diag_out_out)
  );
  m.impl("diag",
         TORCH_FN(TraceType::diag)
  );
  m.impl("trace",
         TORCH_FN(TraceType::trace)
  );
  m.impl("svd.U",
         TORCH_FN(TraceType::svd_out_U)
  );
  m.impl("svd",
         TORCH_FN(TraceType::svd)
  );
  m.impl("dist",
         TORCH_FN(TraceType::dist)
  );
  m.impl("_histogramdd_from_bin_cts",
         TORCH_FN(TraceType::_histogramdd_from_bin_cts)
  );
  m.impl("remainder.Scalar_out",
         TORCH_FN(TraceType::remainder_out_Scalar_out)
  );
  m.impl("remainder.Scalar",
         TORCH_FN(TraceType::remainder_Scalar)
  );
  m.impl("remainder_.Scalar",
         TORCH_FN(TraceType::remainder__Scalar)
  );
  m.impl("remainder.Tensor_out",
         TORCH_FN(TraceType::remainder_out_Tensor_out)
  );
  m.impl("remainder.Tensor",
         TORCH_FN(TraceType::remainder_Tensor)
  );
  m.impl("remainder_.Tensor",
         TORCH_FN(TraceType::remainder__Tensor)
  );
  m.impl("remainder.Scalar_Tensor",
         TORCH_FN(TraceType::remainder_Scalar_Tensor)
  );
  m.impl("nanquantile",
         TORCH_FN(TraceType::nanquantile)
  );
  m.impl("nanquantile.out",
         TORCH_FN(TraceType::nanquantile_out_out)
  );
  m.impl("nanquantile.scalar",
         TORCH_FN(TraceType::nanquantile_scalar)
  );
  m.impl("nanquantile.scalar_out",
         TORCH_FN(TraceType::nanquantile_out_scalar_out)
  );
  m.impl("renorm.out",
         TORCH_FN(TraceType::renorm_out_out)
  );
  m.impl("renorm",
         TORCH_FN(TraceType::renorm)
  );
  m.impl("renorm_",
         TORCH_FN(TraceType::renorm_)
  );
  m.impl("unfold",
         TORCH_FN(TraceType::unfold)
  );
  m.impl("float_power.Tensor_Tensor_out",
         TORCH_FN(TraceType::float_power_out_Tensor_Tensor_out)
  );
  m.impl("float_power.Tensor_Tensor",
         TORCH_FN(TraceType::float_power_Tensor_Tensor)
  );
  m.impl("float_power.Scalar_out",
         TORCH_FN(TraceType::float_power_out_Scalar_out)
  );
  m.impl("float_power.Scalar",
         TORCH_FN(TraceType::float_power_Scalar)
  );
  m.impl("float_power.Tensor_Scalar_out",
         TORCH_FN(TraceType::float_power_out_Tensor_Scalar_out)
  );
  m.impl("float_power.Tensor_Scalar",
         TORCH_FN(TraceType::float_power_Tensor_Scalar)
  );
  m.impl("float_power_.Scalar",
         TORCH_FN(TraceType::float_power__Scalar)
  );
  m.impl("float_power_.Tensor",
         TORCH_FN(TraceType::float_power__Tensor)
  );
  m.impl("_foreach_sinh",
         TORCH_FN(TraceType::_foreach_sinh)
  );
  m.impl("_foreach_sinh_",
         TORCH_FN(TraceType::_foreach_sinh_)
  );
  m.impl("glu_backward.grad_input",
         TORCH_FN(TraceType::glu_backward_out_grad_input)
  );
  m.impl("glu_backward",
         TORCH_FN(TraceType::glu_backward)
  );
  m.impl("hardtanh_backward.grad_input",
         TORCH_FN(TraceType::hardtanh_backward_out_grad_input)
  );
  m.impl("hardtanh_backward",
         TORCH_FN(TraceType::hardtanh_backward)
  );
  m.impl("softplus.out",
         TORCH_FN(TraceType::softplus_out_out)
  );
  m.impl("softplus",
         TORCH_FN(TraceType::softplus)
  );
  m.impl("avg_pool3d_backward.grad_input",
         TORCH_FN(TraceType::avg_pool3d_backward_out_grad_input)
  );
  m.impl("avg_pool3d_backward",
         TORCH_FN(TraceType::avg_pool3d_backward)
  );
  m.impl("max_pool3d_with_indices.out",
         TORCH_FN(TraceType::max_pool3d_with_indices_out_out)
  );
  m.impl("max_pool3d_with_indices",
         TORCH_FN(TraceType::max_pool3d_with_indices)
  );
  m.impl("_upsample_bilinear2d_aa.vec",
         TORCH_FN(TraceType::_upsample_bilinear2d_aa_vec)
  );
  m.impl("_upsample_bilinear2d_aa.out",
         TORCH_FN(TraceType::_upsample_bilinear2d_aa_out_out)
  );
  m.impl("_upsample_bilinear2d_aa",
         TORCH_FN(TraceType::_upsample_bilinear2d_aa)
  );
  m.impl("upsample_nearest2d_backward.grad_input",
         TORCH_FN(TraceType::upsample_nearest2d_backward_out_grad_input)
  );
  m.impl("upsample_nearest2d_backward",
         TORCH_FN(TraceType::upsample_nearest2d_backward)
  );
  m.impl("slow_conv3d_forward.output",
         TORCH_FN(TraceType::slow_conv3d_forward_out_output)
  );
  m.impl("slow_conv3d_forward",
         TORCH_FN(TraceType::slow_conv3d_forward)
  );
  m.impl("im2col.out",
         TORCH_FN(TraceType::im2col_out_out)
  );
  m.impl("im2col",
         TORCH_FN(TraceType::im2col)
  );
  m.impl("isneginf",
         TORCH_FN(TraceType::isneginf)
  );
  m.impl("isneginf.out",
         TORCH_FN(TraceType::isneginf_out_out)
  );
  m.impl("special_erfcx",
         TORCH_FN(TraceType::special_erfcx)
  );
  m.impl("special_erfcx.out",
         TORCH_FN(TraceType::special_erfcx_out_out)
  );
  m.impl("special_i1",
         TORCH_FN(TraceType::special_i1)
  );
  m.impl("special_i1.out",
         TORCH_FN(TraceType::special_i1_out_out)
  );
  m.impl("special_log_softmax",
         TORCH_FN(TraceType::special_log_softmax)
  );
  m.impl("special_gammaincc.out",
         TORCH_FN(TraceType::special_gammaincc_out_out)
  );
  m.impl("special_gammaincc",
         TORCH_FN(TraceType::special_gammaincc)
  );
  m.impl("special_multigammaln",
         TORCH_FN(TraceType::special_multigammaln)
  );
  m.impl("special_multigammaln.out",
         TORCH_FN(TraceType::special_multigammaln_out_out)
  );
  m.impl("fft_fftn",
         TORCH_FN(TraceType::fft_fftn)
  );
  m.impl("fft_fftn.out",
         TORCH_FN(TraceType::fft_fftn_out_out)
  );
  m.impl("fft_fftshift",
         TORCH_FN(TraceType::fft_fftshift)
  );
  m.impl("linalg_lu_factor",
         TORCH_FN(TraceType::linalg_lu_factor)
  );
  m.impl("linalg_lu_factor.out",
         TORCH_FN(TraceType::linalg_lu_factor_out_out)
  );
  m.impl("linalg_det",
         TORCH_FN(TraceType::linalg_det)
  );
  m.impl("linalg_det.out",
         TORCH_FN(TraceType::linalg_det_out_out)
  );
  m.impl("_linalg_slogdet",
         TORCH_FN(TraceType::_linalg_slogdet)
  );
  m.impl("_linalg_slogdet.sign",
         TORCH_FN(TraceType::_linalg_slogdet_out_sign)
  );
  m.impl("ger",
         TORCH_FN(TraceType::ger)
  );
  m.impl("ger.out",
         TORCH_FN(TraceType::ger_out_out)
  );
  m.impl("_linalg_solve_ex",
         TORCH_FN(TraceType::_linalg_solve_ex)
  );
  m.impl("_linalg_solve_ex.result",
         TORCH_FN(TraceType::_linalg_solve_ex_out_result)
  );
  m.impl("nested_to_padded_tensor",
         TORCH_FN(TraceType::nested_to_padded_tensor)
  );
  m.impl("_test_autograd_multiple_dispatch_view",
         TORCH_FN(TraceType::_test_autograd_multiple_dispatch_view)
  );
  m.impl("diagonal_copy",
         TORCH_FN(TraceType::diagonal_copy)
  );
  m.impl("permute_copy",
         TORCH_FN(TraceType::permute_copy)
  );
  m.impl("slice_copy.Tensor",
         TORCH_FN(TraceType::slice_copy_Tensor)
  );
  m.impl("t_copy",
         TORCH_FN(TraceType::t_copy)
  );
  m.impl("_nested_from_padded_tensor",
         TORCH_FN(TraceType::_nested_from_padded_tensor)
  );
  m.impl("_scaled_dot_product_attention_math",
         TORCH_FN(TraceType::_scaled_dot_product_attention_math)
  );
  m.impl("_cudnn_attention_backward",
         TORCH_FN(TraceType::_cudnn_attention_backward)
  );
  m.impl("_foobar",
         TORCH_FN(TraceType::_foobar)
  );
  m.impl("convolution_backward.out",
         TORCH_FN(TraceType::convolution_backward_out_out)
  );
  m.impl("_copy_from.out",
         TORCH_FN(TraceType::_copy_from_out_out)
  );
  m.impl("_mps_convolution_transpose.out",
         TORCH_FN(TraceType::_mps_convolution_transpose_out_out)
  );
  m.impl("floor_divide.Scalar_out",
         TORCH_FN(TraceType::floor_divide_out_Scalar_out)
  );
  m.impl("full.names_out",
         TORCH_FN(TraceType::full_out_names_out)
  );
  m.impl("full_like.out",
         TORCH_FN(TraceType::full_like_out_out)
  );
  m.impl("grid_sampler_2d.out",
         TORCH_FN(TraceType::grid_sampler_2d_out_out)
  );
  m.impl("kaiser_window.out",
         TORCH_FN(TraceType::kaiser_window_out_out)
  );
  m.impl("kaiser_window.periodic_out",
         TORCH_FN(TraceType::kaiser_window_out_periodic_out)
  );
  m.impl("kaiser_window.beta_out",
         TORCH_FN(TraceType::kaiser_window_out_beta_out)
  );
  m.impl("max_pool2d_backward.out",
         TORCH_FN(TraceType::max_pool2d_backward_out_out)
  );
  m.impl("miopen_batch_norm.out",
         TORCH_FN(TraceType::miopen_batch_norm_out_out)
  );
  m.impl("miopen_convolution_transpose.out",
         TORCH_FN(TraceType::miopen_convolution_transpose_out_out)
  );
  m.impl("relu.out",
         TORCH_FN(TraceType::relu_out_out)
  );
  m.impl("select_scatter.out",
         TORCH_FN(TraceType::select_scatter_out_out)
  );
  m.impl("prod.out",
         TORCH_FN(TraceType::prod_out_out)
  );
  m.impl("_nested_tensor_from_mask.out",
         TORCH_FN(TraceType::_nested_tensor_from_mask_out_out)
  );
  m.impl("_nested_tensor_size.out",
         TORCH_FN(TraceType::_nested_tensor_size_out_out)
  );
  m.impl("_nested_view_from_buffer_copy.out",
         TORCH_FN(TraceType::_nested_view_from_buffer_copy_out_out)
  );
  m.impl("unique_dim_consecutive.out",
         TORCH_FN(TraceType::unique_dim_consecutive_out_out)
  );
  m.impl("_unsafe_view.out",
         TORCH_FN(TraceType::_unsafe_view_out_out)
  );
  m.impl("_batch_norm_no_update.out",
         TORCH_FN(TraceType::_batch_norm_no_update_out_out)
  );
  m.impl("sparse_coo_tensor.size_out",
         TORCH_FN(TraceType::sparse_coo_tensor_out_size_out)
  );
  m.impl("sparse_resize_and_clear.out",
         TORCH_FN(TraceType::sparse_resize_and_clear_out_out)
  );
  m.impl("sparse_resize_and_clear",
         TORCH_FN(TraceType::sparse_resize_and_clear)
  );
  m.impl("_to_sparse_csr.out",
         TORCH_FN(TraceType::_to_sparse_csr_out_out)
  );
  m.impl("mkldnn_reorder_conv3d_weight.out",
         TORCH_FN(TraceType::mkldnn_reorder_conv3d_weight_out_out)
  );
  m.impl("_make_per_tensor_quantized_tensor.out",
         TORCH_FN(TraceType::_make_per_tensor_quantized_tensor_out_out)
  );
  m.impl("masked_fill.Scalar_out",
         TORCH_FN(TraceType::masked_fill_out_Scalar_out)
  );
  m.impl("masked_fill.Tensor_out",
         TORCH_FN(TraceType::masked_fill_out_Tensor_out)
  );
  m.impl("trace.out",
         TORCH_FN(TraceType::trace_out_out)
  );
  m.impl("dist.out",
         TORCH_FN(TraceType::dist_out_out)
  );
  m.impl("_histogramdd_from_bin_cts.out",
         TORCH_FN(TraceType::_histogramdd_from_bin_cts_out_out)
  );
  m.impl("remainder.Scalar_Tensor_out",
         TORCH_FN(TraceType::remainder_out_Scalar_Tensor_out)
  );
  m.impl("_foreach_sinh.out",
         TORCH_FN(TraceType::_foreach_sinh_out_out)
  );
  m.impl("diagonal_copy.out",
         TORCH_FN(TraceType::diagonal_copy_out_out)
  );
  m.impl("permute_copy.out",
         TORCH_FN(TraceType::permute_copy_out_out)
  );
  m.impl("slice_copy.Tensor_out",
         TORCH_FN(TraceType::slice_copy_out_Tensor_out)
  );
  m.impl("t_copy.out",
         TORCH_FN(TraceType::t_copy_out_out)
  );
  m.impl("_foobar.out",
         TORCH_FN(TraceType::_foobar_out_out)
  );;
}

}  // namespace

} // namespace torch
