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
#include <ATen/ops/_cast_Double_ops.h>
#include <ATen/ops/_cast_Float_ops.h>
#include <ATen/ops/_use_cudnn_rnn_flatten_weight_ops.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight_ops.h>
#include <ATen/ops/abs_ops.h>
#include <ATen/ops/abs_ops.h>
#include <ATen/ops/abs_ops.h>
#include <ATen/ops/imag_ops.h>
#include <ATen/ops/resolve_conj_ops.h>
#include <ATen/ops/adaptive_max_pool1d_ops.h>
#include <ATen/ops/addmv_ops.h>
#include <ATen/ops/addmv_ops.h>
#include <ATen/ops/addmv_ops.h>
#include <ATen/ops/addr_ops.h>
#include <ATen/ops/addr_ops.h>
#include <ATen/ops/addr_ops.h>
#include <ATen/ops/atan_ops.h>
#include <ATen/ops/atan_ops.h>
#include <ATen/ops/atan_ops.h>
#include <ATen/ops/binary_cross_entropy_backward_ops.h>
#include <ATen/ops/binary_cross_entropy_backward_ops.h>
#include <ATen/ops/bitwise_not_ops.h>
#include <ATen/ops/bitwise_not_ops.h>
#include <ATen/ops/bitwise_not_ops.h>
#include <ATen/ops/logical_not_ops.h>
#include <ATen/ops/logical_not_ops.h>
#include <ATen/ops/logical_not_ops.h>
#include <ATen/ops/concatenate_ops.h>
#include <ATen/ops/concatenate_ops.h>
#include <ATen/ops/concatenate_ops.h>
#include <ATen/ops/concatenate_ops.h>
#include <ATen/ops/cosh_ops.h>
#include <ATen/ops/cosh_ops.h>
#include <ATen/ops/cosh_ops.h>
#include <ATen/ops/cudnn_grid_sampler_ops.h>
#include <ATen/ops/_cummin_helper_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/embedding_bag_ops.h>
#include <ATen/ops/embedding_bag_ops.h>
#include <ATen/ops/grid_sampler_3d_ops.h>
#include <ATen/ops/hann_window_ops.h>
#include <ATen/ops/hann_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/_validate_compressed_sparse_indices_ops.h>
#include <ATen/ops/_cufft_get_plan_cache_size_ops.h>
#include <ATen/ops/_cufft_get_plan_cache_max_size_ops.h>
#include <ATen/ops/isnan_ops.h>
#include <ATen/ops/kthvalue_ops.h>
#include <ATen/ops/kthvalue_ops.h>
#include <ATen/ops/kthvalue_ops.h>
#include <ATen/ops/kthvalue_ops.h>
#include <ATen/ops/native_layer_norm_ops.h>
#include <ATen/ops/_fused_rms_norm_backward_ops.h>
#include <ATen/ops/_sparse_semi_structured_apply_ops.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation_ops.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_fp32_activation_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/xlogy_ops.h>
#include <ATen/ops/matrix_exp_backward_ops.h>
#include <ATen/ops/mkldnn_max_pool2d_ops.h>
#include <ATen/ops/_mps_convolution_ops.h>
#include <ATen/ops/batch_norm_stats_ops.h>
#include <ATen/ops/batch_norm_backward_reduce_ops.h>
#include <ATen/ops/_cdist_forward_ops.h>
#include <ATen/ops/cosine_similarity_ops.h>
#include <ATen/ops/movedim_ops.h>
#include <ATen/ops/movedim_ops.h>
#include <ATen/ops/mH_ops.h>
#include <ATen/ops/rand_like_ops.h>
#include <ATen/ops/rand_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/round_ops.h>
#include <ATen/ops/sigmoid_ops.h>
#include <ATen/ops/sigmoid_ops.h>
#include <ATen/ops/sigmoid_ops.h>
#include <ATen/ops/hsplit_ops.h>
#include <ATen/ops/hsplit_ops.h>
#include <ATen/ops/stack_ops.h>
#include <ATen/ops/stack_ops.h>
#include <ATen/ops/hash_tensor_ops.h>
#include <ATen/ops/hash_tensor_ops.h>
#include <ATen/ops/tile_ops.h>
#include <ATen/ops/_mkldnn_transpose_ops.h>
#include <ATen/ops/_mkldnn_transpose_ops.h>
#include <ATen/ops/_nested_from_padded_and_nested_example_ops.h>
#include <ATen/ops/_nested_compute_contiguous_strides_offsets_ops.h>
#include <ATen/ops/unique_dim_ops.h>
#include <ATen/ops/vander_ops.h>
#include <ATen/ops/_philox_key_split_ops.h>
#include <ATen/ops/batch_norm_backward_ops.h>
#include <ATen/ops/clone_ops.h>
#include <ATen/ops/positive_ops.h>
#include <ATen/ops/_scaled_mm_v2_ops.h>
#include <ATen/ops/_scaled_mm_v2_ops.h>
#include <ATen/ops/sparse_csr_tensor_ops.h>
#include <ATen/ops/sparse_csr_tensor_ops.h>
#include <ATen/ops/_sparse_bsc_tensor_unsafe_ops.h>
#include <ATen/ops/dense_dim_ops.h>
#include <ATen/ops/_dimV_ops.h>
#include <ATen/ops/coalesce_ops.h>
#include <ATen/ops/_indices_ops.h>
#include <ATen/ops/_to_sparse_csc_ops.h>
#include <ATen/ops/_to_sparse_bsc_ops.h>
#include <ATen/ops/dequantize_ops.h>
#include <ATen/ops/dequantize_ops.h>
#include <ATen/ops/q_per_channel_zero_points_ops.h>
#include <ATen/ops/fake_quantize_per_tensor_affine_ops.h>
#include <ATen/ops/fake_quantize_per_tensor_affine_ops.h>
#include <ATen/ops/_fake_quantize_learnable_per_channel_affine_ops.h>
#include <ATen/ops/item_ops.h>
#include <ATen/ops/_thnn_fused_lstm_cell_ops.h>
#include <ATen/ops/rnn_tanh_ops.h>
#include <ATen/ops/rnn_tanh_ops.h>
#include <ATen/ops/rnn_relu_cell_ops.h>
#include <ATen/ops/_pad_packed_sequence_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/cauchy_ops.h>
#include <ATen/ops/log_normal_ops.h>
#include <ATen/ops/cross_ops.h>
#include <ATen/ops/cross_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/ne_ops.h>
#include <ATen/ops/_gather_sparse_backward_ops.h>
#include <ATen/ops/linalg_vander_ops.h>
#include <ATen/ops/swapaxes_ops.h>
#include <ATen/ops/swapaxes_ops.h>
#include <ATen/ops/histc_ops.h>
#include <ATen/ops/histc_ops.h>
#include <ATen/ops/quantile_ops.h>
#include <ATen/ops/quantile_ops.h>
#include <ATen/ops/quantile_ops.h>
#include <ATen/ops/quantile_ops.h>
#include <ATen/ops/msort_ops.h>
#include <ATen/ops/msort_ops.h>
#include <ATen/ops/argsort_ops.h>
#include <ATen/ops/argsort_ops.h>
#include <ATen/ops/argsort_ops.h>
#include <ATen/ops/argsort_ops.h>
#include <ATen/ops/unfold_backward_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/_foreach_atan_ops.h>
#include <ATen/ops/_foreach_atan_ops.h>
#include <ATen/ops/_foreach_ceil_ops.h>
#include <ATen/ops/_foreach_ceil_ops.h>
#include <ATen/ops/_foreach_erf_ops.h>
#include <ATen/ops/_foreach_erf_ops.h>
#include <ATen/ops/_foreach_log2_ops.h>
#include <ATen/ops/_foreach_log2_ops.h>
#include <ATen/ops/bucketize_ops.h>
#include <ATen/ops/bucketize_ops.h>
#include <ATen/ops/bucketize_ops.h>
#include <ATen/ops/mse_loss_ops.h>
#include <ATen/ops/mse_loss_ops.h>
#include <ATen/ops/l1_loss_ops.h>
#include <ATen/ops/nll_loss2d_forward_ops.h>
#include <ATen/ops/nll_loss2d_forward_ops.h>
#include <ATen/ops/soft_margin_loss_ops.h>
#include <ATen/ops/soft_margin_loss_ops.h>
#include <ATen/ops/glu_backward_jvp_ops.h>
#include <ATen/ops/hardtanh_ops.h>
#include <ATen/ops/hardtanh_ops.h>
#include <ATen/ops/hardtanh_ops.h>
#include <ATen/ops/hardswish_backward_ops.h>
#include <ATen/ops/leaky_relu_ops.h>
#include <ATen/ops/leaky_relu_ops.h>
#include <ATen/ops/leaky_relu_ops.h>
#include <ATen/ops/log_sigmoid_backward_ops.h>
#include <ATen/ops/log_sigmoid_backward_ops.h>
#include <ATen/ops/_adaptive_avg_pool3d_backward_ops.h>
#include <ATen/ops/adaptive_max_pool2d_backward_ops.h>
#include <ATen/ops/adaptive_max_pool2d_backward_ops.h>
#include <ATen/ops/adaptive_max_pool3d_backward_ops.h>
#include <ATen/ops/adaptive_max_pool3d_backward_ops.h>
#include <ATen/ops/replication_pad2d_ops.h>
#include <ATen/ops/replication_pad2d_ops.h>
#include <ATen/ops/_pad_circular_ops.h>
#include <ATen/ops/_conv_depthwise2d_ops.h>
#include <ATen/ops/_conv_depthwise2d_ops.h>
#include <ATen/ops/slow_conv3d_ops.h>
#include <ATen/ops/slow_conv3d_ops.h>
#include <ATen/ops/special_log_ndtr_ops.h>
#include <ATen/ops/special_log_ndtr_ops.h>
#include <ATen/ops/special_erf_ops.h>
#include <ATen/ops/special_erf_ops.h>
#include <ATen/ops/special_softmax_ops.h>
#include <ATen/ops/fft_fft_ops.h>
#include <ATen/ops/fft_fft_ops.h>
#include <ATen/ops/fft_rfft_ops.h>
#include <ATen/ops/fft_rfft_ops.h>
#include <ATen/ops/fft_hfft2_ops.h>
#include <ATen/ops/fft_hfft2_ops.h>
#include <ATen/ops/fft_ifftn_ops.h>
#include <ATen/ops/fft_ifftn_ops.h>
#include <ATen/ops/fft_fftfreq_ops.h>
#include <ATen/ops/fft_fftfreq_ops.h>
#include <ATen/ops/linalg_lu_factor_ex_ops.h>
#include <ATen/ops/linalg_lu_factor_ex_ops.h>
#include <ATen/ops/linalg__powsum_ops.h>
#include <ATen/ops/_spsolve_ops.h>
#include <ATen/ops/linalg_multi_dot_ops.h>
#include <ATen/ops/linalg_multi_dot_ops.h>
#include <ATen/ops/_test_parallel_materialize_ops.h>
#include <ATen/ops/flatten_dense_tensors_ops.h>
#include <ATen/ops/_conj_copy_ops.h>
#include <ATen/ops/detach_copy_ops.h>
#include <ATen/ops/row_indices_copy_ops.h>
#include <ATen/ops/_transformer_encoder_layer_fwd_ops.h>
#include <ATen/ops/_native_multi_head_attention_ops.h>
#include <ATen/ops/_fused_sdp_choice_ops.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu_ops.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention_backward_ops.h>
#include <ATen/ops/_fill_mem_eff_dropout_mask_ops.h>
#include <ATen/ops/_triton_multi_head_attention_ops.h>
#include <ATen/ops/special_modified_bessel_i0_ops.h>
#include <ATen/ops/special_modified_bessel_i0_ops.h>
#include <ATen/ops/special_modified_bessel_k0_ops.h>
#include <ATen/ops/special_modified_bessel_k0_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_ops.h>
#include <ATen/ops/_cudnn_rnn_flatten_weight_ops.h>
#include <ATen/ops/cudnn_grid_sampler_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/grid_sampler_3d_ops.h>
#include <ATen/ops/hann_window_ops.h>
#include <ATen/ops/hann_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/hamming_window_ops.h>
#include <ATen/ops/isnan_ops.h>
#include <ATen/ops/native_layer_norm_ops.h>
#include <ATen/ops/mkldnn_max_pool2d_ops.h>
#include <ATen/ops/_mps_convolution_ops.h>
#include <ATen/ops/batch_norm_stats_ops.h>
#include <ATen/ops/batch_norm_backward_reduce_ops.h>
#include <ATen/ops/_cdist_forward_ops.h>
#include <ATen/ops/rand_like_ops.h>
#include <ATen/ops/rand_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/randint_like_ops.h>
#include <ATen/ops/_mkldnn_transpose_ops.h>
#include <ATen/ops/_nested_from_padded_and_nested_example_ops.h>
#include <ATen/ops/unique_dim_ops.h>
#include <ATen/ops/clone_ops.h>
#include <ATen/ops/resize_as_ops.h>
#include <ATen/ops/resize_as_ops.h>
#include <ATen/ops/_to_sparse_csc_ops.h>
#include <ATen/ops/_to_sparse_bsc_ops.h>
#include <ATen/ops/dequantize_ops.h>
#include <ATen/ops/dequantize_ops.h>
#include <ATen/ops/q_per_channel_zero_points_ops.h>
#include <ATen/ops/_fake_quantize_learnable_per_channel_affine_ops.h>
#include <ATen/ops/_thnn_fused_lstm_cell_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/random_ops.h>
#include <ATen/ops/cauchy_ops.h>
#include <ATen/ops/cauchy_ops.h>
#include <ATen/ops/log_normal_ops.h>
#include <ATen/ops/log_normal_ops.h>
#include <ATen/ops/unfold_backward_ops.h>
#include <ATen/ops/normal_ops.h>
#include <ATen/ops/_foreach_atan_ops.h>
#include <ATen/ops/_foreach_ceil_ops.h>
#include <ATen/ops/_foreach_erf_ops.h>
#include <ATen/ops/_foreach_log2_ops.h>
#include <ATen/ops/bucketize_ops.h>
#include <ATen/ops/glu_backward_jvp_ops.h>
#include <ATen/ops/hardswish_backward_ops.h>
#include <ATen/ops/_adaptive_avg_pool3d_backward_ops.h>
#include <ATen/ops/_conj_copy_ops.h>
#include <ATen/ops/detach_copy_ops.h>
#include <ATen/ops/row_indices_copy_ops.h>
#include <ATen/ops/_transformer_encoder_layer_fwd_ops.h>
#include <ATen/ops/_native_multi_head_attention_ops.h>
#include <ATen/ops/_triton_multi_head_attention_ops.h>
#endif

using namespace at;

namespace torch {

namespace TraceType {

namespace {
at::Tensor _cast_Double(c10::DispatchKeySet ks, const at::Tensor & self, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cast_Double");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cast_Double::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _cast_Float(c10::DispatchKeySet ks, const at::Tensor & self, bool non_blocking) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cast_Float");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "non_blocking", non_blocking);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cast_Float::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, non_blocking);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool _use_cudnn_rnn_flatten_weight(c10::DispatchKeySet ks) {
  auto result =at::_ops::_use_cudnn_rnn_flatten_weight::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer));
  return result;
}
at::Tensor _cudnn_rnn_flatten_weight(c10::DispatchKeySet ks, at::TensorList weight_arr, int64_t weight_stride0, c10::SymInt input_size, int64_t mode, c10::SymInt hidden_size, c10::SymInt proj_size, int64_t num_layers, bool batch_first, bool bidirectional) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cudnn_rnn_flatten_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_arr", weight_arr);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "proj_size", proj_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cudnn_rnn_flatten_weight::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor abs(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::abs");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::abs::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & abs_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::abs");
    } else {
      op_name = c10::Symbol::fromQualString("aten::abs_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("abs_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::abs_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & abs_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::abs");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("abs_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::abs_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor imag(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::imag");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::imag::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor resolve_conj(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::resolve_conj");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::resolve_conj::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool1d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_max_pool1d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::adaptive_max_pool1d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor addmv(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::addmv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::addmv::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat, vec, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & addmv_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::addmv");
    } else {
      op_name = c10::Symbol::fromQualString("aten::addmv_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::addmv_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat, vec, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & addmv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::addmv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat", mat);
    jit::tracer::addInputs(node, "vec", vec);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addmv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::addmv_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat, vec, beta, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor addr(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::addr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::addr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec1, vec2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & addr_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::addr");
    } else {
      op_name = c10::Symbol::fromQualString("aten::addr_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addr_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::addr_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec1, vec2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & addr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::addr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "vec1", vec1);
    jit::tracer::addInputs(node, "vec2", vec2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::addr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, vec1, vec2, beta, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor atan(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::atan::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & atan_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::atan");
    } else {
      op_name = c10::Symbol::fromQualString("aten::atan_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::atan_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & atan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::atan_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor binary_cross_entropy_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::binary_cross_entropy_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::binary_cross_entropy_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, weight, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & binary_cross_entropy_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::binary_cross_entropy_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "reduction", reduction);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("binary_cross_entropy_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::binary_cross_entropy_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, weight, reduction, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor bitwise_not(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_not");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bitwise_not::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & bitwise_not_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::bitwise_not");
    } else {
      op_name = c10::Symbol::fromQualString("aten::bitwise_not_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_not_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_not_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & bitwise_not_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bitwise_not");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bitwise_not_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bitwise_not_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor logical_not(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logical_not");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logical_not::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & logical_not_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::logical_not");
    } else {
      op_name = c10::Symbol::fromQualString("aten::logical_not_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_not_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logical_not_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & logical_not_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logical_not");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logical_not_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logical_not_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor concatenate(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::concatenate");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::concatenate::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & concatenate_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::concatenate");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("concatenate_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::concatenate_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor concatenate_names(c10::DispatchKeySet ks, at::TensorList tensors, at::Dimname dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::concatenate");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::concatenate_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & concatenate_out_names_out(c10::DispatchKeySet ks, at::TensorList tensors, at::Dimname dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::concatenate");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("concatenate_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::concatenate_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cosh(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cosh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cosh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & cosh_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::cosh");
    } else {
      op_name = c10::Symbol::fromQualString("aten::cosh_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cosh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cosh_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & cosh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cosh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cosh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cosh_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cudnn_grid_sampler(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grid) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_grid_sampler");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto output =at::_ops::cudnn_grid_sampler::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grid);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
  }
  return output;
}
void _cummin_helper(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim) {
  at::_ops::_cummin_helper::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, values, indices, dim);
}
at::Tensor div_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::div_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & div__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::div");
    } else {
      op_name = c10::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & div_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor div_Tensor_mode(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::div_Tensor_mode::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & div__Tensor_mode(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::div");
    } else {
      op_name = c10::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div__Tensor_mode::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & div_out_out_mode(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, ::std::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div_out_mode::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor div_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::div_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & div__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::div");
    } else {
      op_name = c10::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor div_Scalar_mode(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::div_Scalar_mode::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & div__Scalar_mode(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::div");
    } else {
      op_name = c10::Symbol::fromQualString("aten::div_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div__Scalar_mode::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> embedding_bag(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::embedding_bag");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "include_last_offset", include_last_offset);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::embedding_bag::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> embedding_bag_padding_idx(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const ::std::optional<at::Tensor> & per_sample_weights, bool include_last_offset, ::std::optional<int64_t> padding_idx) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::embedding_bag");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "offsets", offsets);
    jit::tracer::addInputs(node, "scale_grad_by_freq", scale_grad_by_freq);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "sparse", sparse);
    jit::tracer::addInputs(node, "per_sample_weights", per_sample_weights);
    jit::tracer::addInputs(node, "include_last_offset", include_last_offset);
    jit::tracer::addInputs(node, "padding_idx", padding_idx);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::embedding_bag_padding_idx::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
at::Tensor grid_sampler_3d(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::grid_sampler_3d");
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
  auto result =at::_ops::grid_sampler_3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, grid, interpolation_mode, padding_mode, align_corners);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hann_window(c10::DispatchKeySet ks, int64_t window_length, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hann_window");
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
  auto result =at::_ops::hann_window::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hann_window_periodic(c10::DispatchKeySet ks, int64_t window_length, bool periodic, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hann_window");
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
  auto result =at::_ops::hann_window_periodic::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hamming_window(c10::DispatchKeySet ks, int64_t window_length, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
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
  auto result =at::_ops::hamming_window::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hamming_window_periodic(c10::DispatchKeySet ks, int64_t window_length, bool periodic, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
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
  auto result =at::_ops::hamming_window_periodic::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hamming_window_periodic_alpha(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hamming_window_periodic_alpha::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, alpha, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hamming_window_periodic_alpha_beta(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, double beta, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hamming_window_periodic_alpha_beta::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, alpha, beta, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _validate_compressed_sparse_indices(c10::DispatchKeySet ks, bool is_crow, const at::Tensor & compressed_idx, const at::Tensor & plain_idx, int64_t cdim, int64_t dim, int64_t nnz) {
  at::_ops::_validate_compressed_sparse_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), is_crow, compressed_idx, plain_idx, cdim, dim, nnz);
}
int64_t _cufft_get_plan_cache_size(c10::DispatchKeySet ks, at::DeviceIndex device_index) {
  auto result =at::_ops::_cufft_get_plan_cache_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), device_index);
  return result;
}
int64_t _cufft_get_plan_cache_max_size(c10::DispatchKeySet ks, at::DeviceIndex device_index) {
  auto result =at::_ops::_cufft_get_plan_cache_max_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), device_index);
  return result;
}
at::Tensor isnan(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isnan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::isnan::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> kthvalue(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt k, int64_t dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::kthvalue::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out_values(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("kthvalue_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::kthvalue_values::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor> kthvalue_dimname(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt k, at::Dimname dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::kthvalue_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out_dimname_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt k, at::Dimname dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::kthvalue");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("kthvalue_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::kthvalue_dimname_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dim, keepdim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm(c10::DispatchKeySet ks, const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_layer_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::native_layer_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, normalized_shape, weight, bias, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor> _fused_rms_norm_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, at::IntArrayRef normalized_shape, const at::Tensor & rstd, const ::std::optional<at::Tensor> & weight, ::std::array<bool,2> output_mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fused_rms_norm_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "rstd", rstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_fused_rms_norm_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, normalized_shape, rstd, weight, output_mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> _sparse_semi_structured_apply(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & thread_masks) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_semi_structured_apply");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "thread_masks", thread_masks);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_sparse_semi_structured_apply::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, thread_masks);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor fbgemm_linear_fp16_weight_fp32_activation(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & packed_weight, const ::std::optional<at::Tensor> & bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fbgemm_linear_fp16_weight_fp32_activation");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "packed_weight", packed_weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fbgemm_linear_fp16_weight_fp32_activation::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, packed_weight, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fbgemm_linear_fp16_weight_fp32_activation_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & packed_weight, const ::std::optional<at::Tensor> & bias, at::Tensor & output) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::fbgemm_linear_fp16_weight_fp32_activation");
    } else {
      op_name = c10::Symbol::fromQualString("aten::fbgemm_linear_fp16_weight_fp32_activation");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "packed_weight", packed_weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fbgemm_linear_fp16_weight_fp32_activation_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, packed_weight, bias, output);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor xlogy_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::xlogy_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor xlogy_Scalar_Self(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::xlogy_Scalar_Self::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor xlogy_Scalar_Other(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::xlogy_Scalar_Other::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & xlogy__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::xlogy");
    } else {
      op_name = c10::Symbol::fromQualString("aten::xlogy_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("xlogy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::xlogy__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & xlogy__Scalar_Other(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::xlogy");
    } else {
      op_name = c10::Symbol::fromQualString("aten::xlogy_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("xlogy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::xlogy__Scalar_Other::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & xlogy_out_OutTensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("xlogy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::xlogy_OutTensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & xlogy_out_OutScalar_Self(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("xlogy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::xlogy_OutScalar_Self::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & xlogy_out_OutScalar_Other(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::xlogy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("xlogy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::xlogy_OutScalar_Other::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor matrix_exp_backward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::matrix_exp_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grad", grad);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::matrix_exp_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor mkldnn_max_pool2d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_max_pool2d");
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
  auto result =at::_ops::mkldnn_max_pool2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _mps_convolution(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mps_convolution");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_mps_convolution::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, padding, stride, dilation, groups);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_stats(c10::DispatchKeySet ks, const at::Tensor & input, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_stats");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::batch_norm_stats::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> batch_norm_backward_reduce(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_backward_reduce");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "input_g", input_g);
    jit::tracer::addInputs(node, "weight_g", weight_g);
    jit::tracer::addInputs(node, "bias_g", bias_g);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::batch_norm_backward_reduce::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
at::Tensor _cdist_forward(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, double p, ::std::optional<int64_t> compute_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cdist_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "compute_mode", compute_mode);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cdist_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x1, x2, p, compute_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor cosine_similarity(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, int64_t dim, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cosine_similarity");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cosine_similarity::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x1, x2, dim, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor movedim_intlist(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef source, at::IntArrayRef destination) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::movedim");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "destination", destination);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::movedim_intlist::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, destination);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor movedim_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t source, int64_t destination) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::movedim");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "destination", destination);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::movedim_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, destination);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor mH(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mH");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mH::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor rand_like(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::rand_like::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor rand_like_generator(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::rand_like_generator::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint_like(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint_like::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, high, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint_like_generator(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt high, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint_like_generator::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, high, generator, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint_like_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint_like_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, high, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint_like_Tensor_generator(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & high, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint_like_Tensor_generator::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, high, generator, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint_like_low_dtype(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint_like_low_dtype::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, low, high, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint_like_low_generator_dtype(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint_like_low_generator_dtype::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, low, high, generator, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor round(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::round");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::round::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & round_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::round");
    } else {
      op_name = c10::Symbol::fromQualString("aten::round_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::round_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & round_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::round");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::round_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor round_decimals(c10::DispatchKeySet ks, const at::Tensor & self, int64_t decimals) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::round");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "decimals", decimals);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::round_decimals::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, decimals);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & round__decimals(c10::DispatchKeySet ks, at::Tensor & self, int64_t decimals) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::round");
    } else {
      op_name = c10::Symbol::fromQualString("aten::round_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "decimals", decimals);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::round__decimals::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, decimals);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & round_out_decimals_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t decimals, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::round");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "decimals", decimals);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("round_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::round_decimals_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, decimals, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor sigmoid(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sigmoid");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sigmoid::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & sigmoid_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::sigmoid");
    } else {
      op_name = c10::Symbol::fromQualString("aten::sigmoid_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sigmoid_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sigmoid_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & sigmoid_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sigmoid");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sigmoid_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sigmoid_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::vector<at::Tensor> hsplit_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t sections) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hsplit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sections", sections);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hsplit_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, sections);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> hsplit_array(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hsplit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hsplit_array::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor stack(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::stack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::stack::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & stack_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::stack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("stack_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::stack_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor hash_tensor(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, int64_t mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hash_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "mode", mode);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hash_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hash_tensor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, int64_t mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hash_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "mode", mode);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hash_tensor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hash_tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor tile(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef dims) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tile");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::tile::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dims);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _mkldnn_transpose(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim0, int64_t dim1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mkldnn_transpose");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_mkldnn_transpose::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim0, dim1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _mkldnn_transpose_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim0, int64_t dim1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::_mkldnn_transpose");
    } else {
      op_name = c10::Symbol::fromQualString("aten::_mkldnn_transpose_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim0", dim0);
    jit::tracer::addInputs(node, "dim1", dim1);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mkldnn_transpose_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_mkldnn_transpose_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim0, dim1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor _nested_from_padded_and_nested_example(c10::DispatchKeySet ks, const at::Tensor & padded, const at::Tensor & nt_example) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_from_padded_and_nested_example");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "padded", padded);
    jit::tracer::addInputs(node, "nt_example", nt_example);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_from_padded_and_nested_example::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), padded, nt_example);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _nested_compute_contiguous_strides_offsets(c10::DispatchKeySet ks, const at::Tensor & nested_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_compute_contiguous_strides_offsets");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "nested_size", nested_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_nested_compute_contiguous_strides_offsets::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), nested_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unique_dim");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "sorted", sorted);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::unique_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, sorted, return_inverse, return_counts);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor vander(c10::DispatchKeySet ks, const at::Tensor & x, ::std::optional<int64_t> N, bool increasing) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::vander");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "N", N);
    jit::tracer::addInputs(node, "increasing", increasing);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::vander::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, N, increasing);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _philox_key_split(c10::DispatchKeySet ks, const at::Tensor & key, int64_t num_splits) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_philox_key_split");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "num_splits", num_splits);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_philox_key_split::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), key, num_splits);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> batch_norm_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, const ::std::optional<at::Tensor> & save_mean, const ::std::optional<at::Tensor> & save_var, bool update, double eps, ::std::array<bool,3> output_mask, const at::Tensor & reserve) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "save_mean", save_mean);
    jit::tracer::addInputs(node, "save_var", save_var);
    jit::tracer::addInputs(node, "update", update);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "output_mask", output_mask);
    jit::tracer::addInputs(node, "reserve", reserve);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::batch_norm_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, weight, running_mean, running_var, save_mean, save_var, update, eps, output_mask, reserve);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor clone(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clone");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::clone::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor positive(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::positive");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::positive::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _scaled_mm_v2(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, at::TensorList scale_a, at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a, at::TensorList scale_b, at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b, const ::std::optional<at::Tensor> & bias, ::std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim, bool use_fast_accum) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_mm_v2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "scale_a", scale_a);
    jit::tracer::addInputs(node, "recipe_a", recipe_a);
    jit::tracer::addInputs(node, "swizzle_a", swizzle_a);
    jit::tracer::addInputs(node, "scale_b", scale_b);
    jit::tracer::addInputs(node, "recipe_b", recipe_b);
    jit::tracer::addInputs(node, "swizzle_b", swizzle_b);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "out_dtype", out_dtype);
    jit::tracer::addInputs(node, "contraction_dim", contraction_dim);
    jit::tracer::addInputs(node, "use_fast_accum", use_fast_accum);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_scaled_mm_v2::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat2, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b, bias, out_dtype, contraction_dim, use_fast_accum);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _scaled_mm_v2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, at::TensorList scale_a, at::IntArrayRef recipe_a, at::IntArrayRef swizzle_a, at::TensorList scale_b, at::IntArrayRef recipe_b, at::IntArrayRef swizzle_b, const ::std::optional<at::Tensor> & bias, ::std::optional<at::ScalarType> out_dtype, at::IntArrayRef contraction_dim, bool use_fast_accum, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_mm_v2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mat2", mat2);
    jit::tracer::addInputs(node, "scale_a", scale_a);
    jit::tracer::addInputs(node, "recipe_a", recipe_a);
    jit::tracer::addInputs(node, "swizzle_a", swizzle_a);
    jit::tracer::addInputs(node, "scale_b", scale_b);
    jit::tracer::addInputs(node, "recipe_b", recipe_b);
    jit::tracer::addInputs(node, "swizzle_b", swizzle_b);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "out_dtype", out_dtype);
    jit::tracer::addInputs(node, "contraction_dim", contraction_dim);
    jit::tracer::addInputs(node, "use_fast_accum", use_fast_accum);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_scaled_mm_v2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_scaled_mm_v2_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mat2, scale_a, recipe_a, swizzle_a, scale_b, recipe_b, swizzle_b, bias, out_dtype, contraction_dim, use_fast_accum, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor sparse_csr_tensor_crow_col_value_size(c10::DispatchKeySet ks, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_csr_tensor");
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
  auto result =at::_ops::sparse_csr_tensor_crow_col_value_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), crow_indices, col_indices, values, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sparse_csr_tensor_crow_col_value(c10::DispatchKeySet ks, const at::Tensor & crow_indices, const at::Tensor & col_indices, const at::Tensor & values, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sparse_csr_tensor");
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
  auto result =at::_ops::sparse_csr_tensor_crow_col_value::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), crow_indices, col_indices, values, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_bsc_tensor_unsafe(c10::DispatchKeySet ks, const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_bsc_tensor_unsafe");
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
  auto result =at::_ops::_sparse_bsc_tensor_unsafe::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), ccol_indices, row_indices, values, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
int64_t dense_dim(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::dense_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
int64_t _dimV(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::_dimV::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor coalesce(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::coalesce");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::coalesce::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_indices");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _to_sparse_csc(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_csc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_to_sparse_csc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _to_sparse_bsc(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_bsc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "blocksize", blocksize);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_to_sparse_bsc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, blocksize, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor dequantize_self(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dequantize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::dequantize_self::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> dequantize_tensors(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dequantize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::dequantize_tensors::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor q_per_channel_zero_points(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::q_per_channel_zero_points");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::q_per_channel_zero_points::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fake_quantize_per_tensor_affine(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fake_quantize_per_tensor_affine::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, quant_min, quant_max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fake_quantize_per_tensor_affine_tensor_qparams(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fake_quantize_per_tensor_affine_tensor_qparams::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, quant_min, quant_max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _fake_quantize_learnable_per_channel_affine(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fake_quantize_learnable_per_channel_affine");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    jit::tracer::addInputs(node, "grad_factor", grad_factor);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_fake_quantize_learnable_per_channel_affine::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, axis, quant_min, quant_max, grad_factor);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Scalar item(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::item::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _thnn_fused_lstm_cell(c10::DispatchKeySet ks, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & cx, const ::std::optional<at::Tensor> & input_bias, const ::std::optional<at::Tensor> & hidden_bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_thnn_fused_lstm_cell");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2] =at::_ops::_thnn_fused_lstm_cell::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input_gates, hidden_gates, cx, input_bias, hidden_bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
::std::tuple<at::Tensor,at::Tensor> rnn_tanh_input(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rnn_tanh");
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
  auto [result0, result1] =at::_ops::rnn_tanh_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> rnn_tanh_data(c10::DispatchKeySet ks, const at::Tensor & data, const at::Tensor & batch_sizes, const at::Tensor & hx, at::TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rnn_tanh");
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
  auto [result0, result1] =at::_ops::rnn_tanh_data::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), data, batch_sizes, hx, params, has_biases, num_layers, dropout, train, bidirectional);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor rnn_relu_cell(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & hx, const at::Tensor & w_ih, const at::Tensor & w_hh, const ::std::optional<at::Tensor> & b_ih, const ::std::optional<at::Tensor> & b_hh) {
  auto result =at::_ops::rnn_relu_cell::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, hx, w_ih, w_hh, b_ih, b_hh);
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _pad_packed_sequence(c10::DispatchKeySet ks, const at::Tensor & data, const at::Tensor & batch_sizes, bool batch_first, const at::Scalar & padding_value, int64_t total_length) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_pad_packed_sequence");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "data", data);
    jit::tracer::addInputs(node, "batch_sizes", batch_sizes);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "padding_value", padding_value);
    jit::tracer::addInputs(node, "total_length", total_length);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_pad_packed_sequence::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), data, batch_sizes, batch_first, padding_value, total_length);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor & random__from(c10::DispatchKeySet ks, at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::random");
    } else {
      op_name = c10::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random__from::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, from, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & random__to(c10::DispatchKeySet ks, at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::random");
    } else {
      op_name = c10::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random__to::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & random_(c10::DispatchKeySet ks, at::Tensor & self, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::random");
    } else {
      op_name = c10::Symbol::fromQualString("aten::random_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & cauchy_(c10::DispatchKeySet ks, at::Tensor & self, double median, double sigma, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::cauchy");
    } else {
      op_name = c10::Symbol::fromQualString("aten::cauchy_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "median", median);
    jit::tracer::addInputs(node, "sigma", sigma);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cauchy_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cauchy_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, median, sigma, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & log_normal_(c10::DispatchKeySet ks, at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::log_normal");
    } else {
      op_name = c10::Symbol::fromQualString("aten::log_normal_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_normal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log_normal_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & cross_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, ::std::optional<int64_t> dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cross");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cross_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cross_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cross(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, ::std::optional<int64_t> dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cross");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cross::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ne_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ne");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ne_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor ne_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ne");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ne_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ne_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ne");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ne_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor ne_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ne");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::ne_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & ne__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::ne");
    } else {
      op_name = c10::Symbol::fromQualString("aten::ne_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ne__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & ne__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::ne");
    } else {
      op_name = c10::Symbol::fromQualString("aten::ne_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ne_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ne__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor _gather_sparse_backward(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & grad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_gather_sparse_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "grad", grad);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_gather_sparse_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, grad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor linalg_vander(c10::DispatchKeySet ks, const at::Tensor & x, ::std::optional<c10::SymInt> N) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_vander");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "N", N);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_vander::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, N);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor swapaxes(c10::DispatchKeySet ks, const at::Tensor & self, int64_t axis0, int64_t axis1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::swapaxes");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "axis0", axis0);
    jit::tracer::addInputs(node, "axis1", axis1);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::swapaxes::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, axis0, axis1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & swapaxes_(c10::DispatchKeySet ks, at::Tensor & self, int64_t axis0, int64_t axis1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::swapaxes");
    } else {
      op_name = c10::Symbol::fromQualString("aten::swapaxes_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "axis0", axis0);
    jit::tracer::addInputs(node, "axis1", axis1);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("swapaxes_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::swapaxes_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, axis0, axis1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & histc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::histc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("histc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::histc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, min, max, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor histc(c10::DispatchKeySet ks, const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::histc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "bins", bins);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::histc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, bins, min, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor quantile(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantile");
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
  auto result =at::_ops::quantile::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & quantile_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantile");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("quantile_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::quantile_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor quantile_scalar(c10::DispatchKeySet ks, const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantile");
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
  auto result =at::_ops::quantile_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & quantile_out_scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, double q, ::std::optional<int64_t> dim, bool keepdim, c10::string_view interpolation, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantile");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("quantile_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::quantile_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, q, dim, keepdim, interpolation, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & msort_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::msort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("msort_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::msort_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor msort(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::msort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::msort::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor argsort(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::argsort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::argsort::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor argsort_stable(c10::DispatchKeySet ks, const at::Tensor & self, bool stable, int64_t dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::argsort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stable", stable);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::argsort_stable::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, stable, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & argsort_out_stable_out(c10::DispatchKeySet ks, const at::Tensor & self, bool stable, int64_t dim, bool descending, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::argsort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "stable", stable);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("argsort_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::argsort_stable_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, stable, dim, descending, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor argsort_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, bool descending) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::argsort");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "descending", descending);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::argsort_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, descending);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor unfold_backward(c10::DispatchKeySet ks, const at::Tensor & grad_in, c10::SymIntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unfold_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_in", grad_in);
    jit::tracer::addInputs(node, "input_sizes", input_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unfold_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_in, input_sizes, dim, size, step);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & normal_(c10::DispatchKeySet ks, at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::normal");
    } else {
      op_name = c10::Symbol::fromQualString("aten::normal_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor normal_functional(c10::DispatchKeySet ks, const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal_functional");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::normal_functional::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & normal_out_Tensor_float_out(c10::DispatchKeySet ks, const at::Tensor & mean, double std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_Tensor_float_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor normal_Tensor_float(c10::DispatchKeySet ks, const at::Tensor & mean, double std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::normal_Tensor_float::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & normal_out_float_Tensor_out(c10::DispatchKeySet ks, double mean, const at::Tensor & std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_float_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor normal_float_Tensor(c10::DispatchKeySet ks, double mean, const at::Tensor & std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::normal_float_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & normal_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & mean, const at::Tensor & std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_Tensor_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor normal_Tensor_Tensor(c10::DispatchKeySet ks, const at::Tensor & mean, const at::Tensor & std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::normal_Tensor_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor normal_float_float(c10::DispatchKeySet ks, double mean, double std, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::normal_float_float::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, size, generator, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & normal_out_float_float_out(c10::DispatchKeySet ks, double mean, double std, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_float_float_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), mean, std, size, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::vector<at::Tensor> _foreach_atan(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_atan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_atan::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_atan_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_atan_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_ceil(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_ceil");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_ceil::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_ceil_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_ceil_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_erf(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_erf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_erf::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_erf_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_erf_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_log2(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_log2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_log2::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_log2_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_log2_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
at::Tensor bucketize_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bucketize_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, boundaries, out_int32, right);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & bucketize_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bucketize_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bucketize_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, boundaries, out_int32, right, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor bucketize_Scalar(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bucketize_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, boundaries, out_int32, right);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & mse_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mse_loss");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("mse_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mse_loss_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor mse_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mse_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::mse_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor l1_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::l1_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::l1_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> nll_loss2d_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, at::Tensor & output, at::Tensor & total_weight) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nll_loss2d_forward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("nll_loss2d_forward_out", output);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nll_loss2d_forward_output::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, reduction, ignore_index, output, total_weight);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  return std::forward_as_tuple(output, total_weight);
}
::std::tuple<at::Tensor,at::Tensor> nll_loss2d_forward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nll_loss2d_forward");
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
  auto [output, total_weight] =at::_ops::nll_loss2d_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, weight, reduction, ignore_index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, total_weight);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
at::Tensor & soft_margin_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::soft_margin_loss");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("soft_margin_loss_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::soft_margin_loss_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor soft_margin_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::soft_margin_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::soft_margin_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor glu_backward_jvp(c10::DispatchKeySet ks, const at::Tensor & grad_x, const at::Tensor & grad_glu, const at::Tensor & x, const at::Tensor & dgrad_glu, const at::Tensor & dx, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::glu_backward_jvp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_x", grad_x);
    jit::tracer::addInputs(node, "grad_glu", grad_glu);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "dgrad_glu", dgrad_glu);
    jit::tracer::addInputs(node, "dx", dx);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::glu_backward_jvp::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_x, grad_glu, x, dgrad_glu, dx, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hardtanh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardtanh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardtanh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardtanh_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min_val, max_val, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor hardtanh(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardtanh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hardtanh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min_val, max_val);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hardtanh_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::hardtanh");
    } else {
      op_name = c10::Symbol::fromQualString("aten::hardtanh_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min_val", min_val);
    jit::tracer::addInputs(node, "max_val", max_val);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardtanh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardtanh_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min_val, max_val);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor hardswish_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardswish_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hardswish_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & leaky_relu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::leaky_relu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("leaky_relu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::leaky_relu_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, negative_slope, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor leaky_relu(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & negative_slope) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::leaky_relu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::leaky_relu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, negative_slope);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & leaky_relu_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & negative_slope) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::leaky_relu");
    } else {
      op_name = c10::Symbol::fromQualString("aten::leaky_relu_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "negative_slope", negative_slope);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("leaky_relu_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::leaky_relu_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, negative_slope);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & log_sigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_sigmoid_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "buffer", buffer);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_sigmoid_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log_sigmoid_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, buffer, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor log_sigmoid_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_sigmoid_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "buffer", buffer);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::log_sigmoid_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, buffer);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _adaptive_avg_pool3d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_adaptive_avg_pool3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_adaptive_avg_pool3d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & adaptive_max_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_max_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::adaptive_max_pool2d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, indices, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor adaptive_max_pool2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_max_pool2d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::adaptive_max_pool2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & adaptive_max_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_max_pool3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "grad_input", grad_input);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("adaptive_max_pool3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::adaptive_max_pool3d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, indices, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor adaptive_max_pool3d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::adaptive_max_pool3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::adaptive_max_pool3d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & replication_pad2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::replication_pad2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::replication_pad2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor replication_pad2d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::replication_pad2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::replication_pad2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _pad_circular(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef pad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_pad_circular");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pad", pad);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_pad_circular::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, pad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _conv_depthwise2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_conv_depthwise2d");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_conv_depthwise2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_conv_depthwise2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, dilation, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _conv_depthwise2d(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_conv_depthwise2d");
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
  auto result =at::_ops::_conv_depthwise2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, dilation);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & slow_conv3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slow_conv3d");
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
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("slow_conv3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::slow_conv3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor slow_conv3d(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slow_conv3d");
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
  auto result =at::_ops::slow_conv3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, kernel_size, bias, stride, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_log_ndtr(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_log_ndtr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_log_ndtr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_log_ndtr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_log_ndtr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_log_ndtr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_log_ndtr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_erf(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_erf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_erf::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_erf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_erf");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_erf_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_erf_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_softmax(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_softmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_softmax::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fft_fft(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fft");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_fft::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_fft_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fft");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_fft_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_fft_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_rfft(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_rfft");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_rfft::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_rfft_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_rfft");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_rfft_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_rfft_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_hfft2(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_hfft2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_hfft2::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_hfft2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_hfft2");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_hfft2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_hfft2_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_ifftn(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ifftn");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "s", s);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_ifftn::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_ifftn_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalSymIntArrayRef s, at::OptionalIntArrayRef dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ifftn");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_ifftn_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_ifftn_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, s, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_fftfreq(c10::DispatchKeySet ks, int64_t n, double d, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fftfreq");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "d", d);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_fftfreq::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, d, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_fftfreq_out_out(c10::DispatchKeySet ks, int64_t n, double d, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_fftfreq");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "d", d);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_fftfreq_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_fftfreq_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, d, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> linalg_lu_factor_ex(c10::DispatchKeySet ks, const at::Tensor & A, bool pivot, bool check_errors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_lu_factor_ex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "pivot", pivot);
    jit::tracer::addInputs(node, "check_errors", check_errors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [LU, pivots, info] =at::_ops::linalg_lu_factor_ex::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, pivot, check_errors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, LU);
    jit::tracer::addOutput(node, pivots);
    jit::tracer::addOutput(node, info);
  }
  return std::make_tuple(std::move(LU), std::move(pivots), std::move(info));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> linalg_lu_factor_ex_out_out(c10::DispatchKeySet ks, const at::Tensor & A, bool pivot, bool check_errors, at::Tensor & LU, at::Tensor & pivots, at::Tensor & info) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_lu_factor_ex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "pivot", pivot);
    jit::tracer::addInputs(node, "check_errors", check_errors);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "LU", LU);
      jit::tracer::addInputs(node, "pivots", pivots);
      jit::tracer::addInputs(node, "info", info);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_lu_factor_ex_out", LU);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_lu_factor_ex_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, pivot, check_errors, LU, pivots, info);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, LU);
    jit::tracer::addOutput(node, pivots);
    jit::tracer::addOutput(node, info);
  }
  return std::forward_as_tuple(LU, pivots, info);
}
at::Tensor linalg__powsum(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & ord, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg__powsum");
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
  auto result =at::_ops::linalg__powsum::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, ord, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _spsolve(c10::DispatchKeySet ks, const at::Tensor & A, const at::Tensor & B, bool left) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_spsolve");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "B", B);
    jit::tracer::addInputs(node, "left", left);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_spsolve::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, B, left);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor linalg_multi_dot(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_multi_dot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_multi_dot::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_multi_dot_out_out(c10::DispatchKeySet ks, at::TensorList tensors, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_multi_dot");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_multi_dot_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_multi_dot_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _test_parallel_materialize(c10::DispatchKeySet ks, const at::Tensor & self, int64_t num_parallel, bool skip_first) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_parallel_materialize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "num_parallel", num_parallel);
    jit::tracer::addInputs(node, "skip_first", skip_first);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_test_parallel_materialize::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, num_parallel, skip_first);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor flatten_dense_tensors(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::flatten_dense_tensors");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::flatten_dense_tensors::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _conj_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_conj_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_conj_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor detach_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::detach_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::detach_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor row_indices_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::row_indices_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::row_indices_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _transformer_encoder_layer_fwd(c10::DispatchKeySet ks, const at::Tensor & src, int64_t embed_dim, int64_t num_heads, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, bool use_gelu, bool norm_first, double eps, const at::Tensor & norm_weight_1, const at::Tensor & norm_bias_1, const at::Tensor & norm_weight_2, const at::Tensor & norm_bias_2, const at::Tensor & ffn_weight_1, const at::Tensor & ffn_bias_1, const at::Tensor & ffn_weight_2, const at::Tensor & ffn_bias_2, const ::std::optional<at::Tensor> & mask, ::std::optional<int64_t> mask_type) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_transformer_encoder_layer_fwd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_heads", num_heads);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "use_gelu", use_gelu);
    jit::tracer::addInputs(node, "norm_first", norm_first);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "norm_weight_1", norm_weight_1);
    jit::tracer::addInputs(node, "norm_bias_1", norm_bias_1);
    jit::tracer::addInputs(node, "norm_weight_2", norm_weight_2);
    jit::tracer::addInputs(node, "norm_bias_2", norm_bias_2);
    jit::tracer::addInputs(node, "ffn_weight_1", ffn_weight_1);
    jit::tracer::addInputs(node, "ffn_bias_1", ffn_bias_1);
    jit::tracer::addInputs(node, "ffn_weight_2", ffn_weight_2);
    jit::tracer::addInputs(node, "ffn_bias_2", ffn_bias_2);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "mask_type", mask_type);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_transformer_encoder_layer_fwd::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias, use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask, mask_type);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _native_multi_head_attention(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask, bool need_weights, bool average_attn_weights, ::std::optional<int64_t> mask_type) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_native_multi_head_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_head", num_head);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "need_weights", need_weights);
    jit::tracer::addInputs(node, "average_attn_weights", average_attn_weights);
    jit::tracer::addInputs(node, "mask_type", mask_type);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::_native_multi_head_attention::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, need_weights, average_attn_weights, mask_type);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
int64_t _fused_sdp_choice(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const ::std::optional<at::Tensor> & attn_mask, double dropout_p, bool is_causal, ::std::optional<double> scale, bool enable_gqa) {
  auto result =at::_ops::_fused_sdp_choice::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _scaled_dot_product_flash_attention_for_cpu(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double dropout_p, bool is_causal, const ::std::optional<at::Tensor> & attn_mask, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_dot_product_flash_attention_for_cpu");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "attn_mask", attn_mask);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [output, logsumexp] =at::_ops::_scaled_dot_product_flash_attention_for_cpu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, dropout_p, is_causal, attn_mask, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, logsumexp);
  }
  return std::make_tuple(std::move(output), std::move(logsumexp));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_efficient_attention_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out_, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & attn_bias, const at::Tensor & out, const at::Tensor & logsumexp, const at::Tensor & philox_seed, const at::Tensor & philox_offset, double dropout_p, ::std::array<bool,4> grad_input_mask, bool is_causal, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_dot_product_efficient_attention_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out_", grad_out_);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "attn_bias", attn_bias);
    jit::tracer::addInputs(node, "out", out);
    jit::tracer::addInputs(node, "logsumexp", logsumexp);
    jit::tracer::addInputs(node, "philox_seed", philox_seed);
    jit::tracer::addInputs(node, "philox_offset", philox_offset);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "grad_input_mask", grad_input_mask);
    jit::tracer::addInputs(node, "is_causal", is_causal);
    jit::tracer::addInputs(node, "scale", scale);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3] =at::_ops::_scaled_dot_product_efficient_attention_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out_, query, key, value, attn_bias, out, logsumexp, philox_seed, philox_offset, dropout_p, grad_input_mask, is_causal, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
at::Tensor & _fill_mem_eff_dropout_mask_(c10::DispatchKeySet ks, at::Tensor & self, double dropout_p, int64_t seed, int64_t offset) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::_fill_mem_eff_dropout_mask");
    } else {
      op_name = c10::Symbol::fromQualString("aten::_fill_mem_eff_dropout_mask_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dropout_p", dropout_p);
    jit::tracer::addInputs(node, "seed", seed);
    jit::tracer::addInputs(node, "offset", offset);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_fill_mem_eff_dropout_mask_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_fill_mem_eff_dropout_mask_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dropout_p, seed, offset);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor _triton_multi_head_attention(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_triton_multi_head_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_head", num_head);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "mask", mask);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_triton_multi_head_attention::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_modified_bessel_i0(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_modified_bessel_i0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_modified_bessel_i0::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_modified_bessel_i0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_modified_bessel_i0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_modified_bessel_i0_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_modified_bessel_i0_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_modified_bessel_k0(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_modified_bessel_k0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_modified_bessel_k0::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_modified_bessel_k0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_modified_bessel_k0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_modified_bessel_k0_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_modified_bessel_k0_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_shifted_chebyshev_polynomial_t(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_t::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_shifted_chebyshev_polynomial_t_x_scalar(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_t_x_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_shifted_chebyshev_polynomial_t_n_scalar(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_shifted_chebyshev_polynomial_t_n_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_shifted_chebyshev_polynomial_t_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_t_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_t_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_t_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_t_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_t_x_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_shifted_chebyshev_polynomial_t_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_shifted_chebyshev_polynomial_t");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_shifted_chebyshev_polynomial_t_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_shifted_chebyshev_polynomial_t_n_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _cudnn_rnn_flatten_weight_out_out(c10::DispatchKeySet ks, at::TensorList weight_arr, int64_t weight_stride0, c10::SymInt input_size, int64_t mode, c10::SymInt hidden_size, c10::SymInt proj_size, int64_t num_layers, bool batch_first, bool bidirectional, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cudnn_rnn_flatten_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "weight_arr", weight_arr);
    jit::tracer::addInputs(node, "weight_stride0", weight_stride0);
    jit::tracer::addInputs(node, "input_size", input_size);
    jit::tracer::addInputs(node, "mode", mode);
    jit::tracer::addInputs(node, "hidden_size", hidden_size);
    jit::tracer::addInputs(node, "proj_size", proj_size);
    jit::tracer::addInputs(node, "num_layers", num_layers);
    jit::tracer::addInputs(node, "batch_first", batch_first);
    jit::tracer::addInputs(node, "bidirectional", bidirectional);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cudnn_rnn_flatten_weight_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_cudnn_rnn_flatten_weight_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), weight_arr, weight_stride0, input_size, mode, hidden_size, proj_size, num_layers, batch_first, bidirectional, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & cudnn_grid_sampler_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & grid, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_grid_sampler");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grid", grid);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cudnn_grid_sampler_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cudnn_grid_sampler_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grid, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & div_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & div_out_Scalar_mode_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, ::std::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::div");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "rounding_mode", rounding_mode);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("div_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::div_Scalar_mode_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, rounding_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & grid_sampler_3d_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::grid_sampler_3d");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("grid_sampler_3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::grid_sampler_3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, grid, interpolation_mode, padding_mode, align_corners, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hann_window_out_out(c10::DispatchKeySet ks, int64_t window_length, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hann_window");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("hann_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hann_window_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hann_window_out_periodic_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hann_window");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("hann_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hann_window_periodic_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hamming_window_out_out(c10::DispatchKeySet ks, int64_t window_length, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("hamming_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hamming_window_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hamming_window_out_periodic_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("hamming_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hamming_window_periodic_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hamming_window_out_periodic_alpha_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hamming_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hamming_window_periodic_alpha_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hamming_window_out_periodic_alpha_beta_out(c10::DispatchKeySet ks, int64_t window_length, bool periodic, double alpha, double beta, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hamming_window");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "window_length", window_length);
    jit::tracer::addInputs(node, "periodic", periodic);
    jit::tracer::addInputs(node, "alpha", alpha);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("hamming_window_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hamming_window_periodic_alpha_beta_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), window_length, periodic, alpha, beta, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & isnan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isnan");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("isnan_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::isnan_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_layer_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_layer_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "eps", eps);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_layer_norm_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::native_layer_norm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, normalized_shape, weight, bias, eps, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & mkldnn_max_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_max_pool2d");
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
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("mkldnn_max_pool2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mkldnn_max_pool2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _mps_convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mps_convolution");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_mps_convolution_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_mps_convolution_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, padding, stride, dilation, groups, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> batch_norm_stats_out_out(c10::DispatchKeySet ks, const at::Tensor & input, double eps, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_stats");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "eps", eps);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("batch_norm_stats_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::batch_norm_stats_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, eps, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> batch_norm_backward_reduce_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const ::std::optional<at::Tensor> & weight, bool input_g, bool weight_g, bool bias_g, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_backward_reduce");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "invstd", invstd);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "input_g", input_g);
    jit::tracer::addInputs(node, "weight_g", weight_g);
    jit::tracer::addInputs(node, "bias_g", bias_g);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
      jit::tracer::addInputs(node, "out3", out3);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("batch_norm_backward_reduce_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::batch_norm_backward_reduce_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g, out0, out1, out2, out3);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
    jit::tracer::addOutput(node, out3);
  }
  return std::forward_as_tuple(out0, out1, out2, out3);
}
at::Tensor & _cdist_forward_out_out(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, double p, ::std::optional<int64_t> compute_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cdist_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x1", x1);
    jit::tracer::addInputs(node, "x2", x2);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "compute_mode", compute_mode);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cdist_forward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_cdist_forward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x1, x2, p, compute_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & rand_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rand_like_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & rand_like_out_generator_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rand_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rand_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rand_like_generator_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randint_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_like_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, high, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randint_like_out_generator_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt high, ::std::optional<at::Generator> generator, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_like_generator_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, high, generator, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randint_like_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_like_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, high, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randint_like_out_Tensor_generator_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & high, ::std::optional<at::Generator> generator, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_like_Tensor_generator_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, high, generator, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randint_like_out_low_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_like_low_dtype_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, low, high, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randint_like_out_low_generator_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt low, c10::SymInt high, ::std::optional<at::Generator> generator, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_like_low_generator_dtype_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, low, high, generator, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _mkldnn_transpose_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim0, int64_t dim1, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_mkldnn_transpose");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_mkldnn_transpose_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_mkldnn_transpose_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim0, dim1, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _nested_from_padded_and_nested_example_out_out(c10::DispatchKeySet ks, const at::Tensor & padded, const at::Tensor & nt_example, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_from_padded_and_nested_example");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "padded", padded);
    jit::tracer::addInputs(node, "nt_example", nt_example);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_nested_from_padded_and_nested_example_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_nested_from_padded_and_nested_example_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), padded, nt_example, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> unique_dim_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unique_dim");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "sorted", sorted);
    jit::tracer::addInputs(node, "return_inverse", return_inverse);
    jit::tracer::addInputs(node, "return_counts", return_counts);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("unique_dim_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::unique_dim_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, sorted, return_inverse, return_counts, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & clone_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clone");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clone_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clone_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
const at::Tensor & resize_as_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & the_template, ::std::optional<at::MemoryFormat> memory_format, const at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::resize_as");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "the_template", the_template);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("resize_as_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::resize_as_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, the_template, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor resize_as(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & the_template, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::resize_as");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "the_template", the_template);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::resize_as::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, the_template, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _to_sparse_csc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dense_dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_csc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_to_sparse_csc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_to_sparse_csc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dense_dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _to_sparse_bsc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_sparse_bsc");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_to_sparse_bsc_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_to_sparse_bsc_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, blocksize, dense_dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & dequantize_out_self_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::dequantize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("dequantize_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::dequantize_self_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void dequantize_out_tensors_out(c10::DispatchKeySet ks, at::TensorList tensors, at::TensorList out) {
  at::_ops::dequantize_tensors_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, out);
}
at::Tensor & q_per_channel_zero_points_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::q_per_channel_zero_points");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("q_per_channel_zero_points_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::q_per_channel_zero_points_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _fake_quantize_learnable_per_channel_affine_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t axis, int64_t quant_min, int64_t quant_max, double grad_factor, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fake_quantize_learnable_per_channel_affine");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    jit::tracer::addInputs(node, "grad_factor", grad_factor);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_fake_quantize_learnable_per_channel_affine_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_fake_quantize_learnable_per_channel_affine_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, axis, quant_min, quant_max, grad_factor, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _thnn_fused_lstm_cell_out_out(c10::DispatchKeySet ks, const at::Tensor & input_gates, const at::Tensor & hidden_gates, const at::Tensor & cx, const ::std::optional<at::Tensor> & input_bias, const ::std::optional<at::Tensor> & hidden_bias, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_thnn_fused_lstm_cell");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input_gates", input_gates);
    jit::tracer::addInputs(node, "hidden_gates", hidden_gates);
    jit::tracer::addInputs(node, "cx", cx);
    jit::tracer::addInputs(node, "input_bias", input_bias);
    jit::tracer::addInputs(node, "hidden_bias", hidden_bias);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_fused_lstm_cell_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_thnn_fused_lstm_cell_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input_gates, hidden_gates, cx, input_bias, hidden_bias, out0, out1, out2);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
    jit::tracer::addOutput(node, out2);
  }
  return std::forward_as_tuple(out0, out1, out2);
}
at::Tensor & random_out_from_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random_from_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, from, to, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor random_from(c10::DispatchKeySet ks, const at::Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::random_from::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, from, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & random_out_to_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random_to_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, to, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor random_to(c10::DispatchKeySet ks, const at::Tensor & self, int64_t to, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::random_to::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & random_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("random_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::random_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor random(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::random");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::random::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & cauchy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double median, double sigma, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cauchy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "median", median);
    jit::tracer::addInputs(node, "sigma", sigma);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cauchy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cauchy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, median, sigma, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor cauchy(c10::DispatchKeySet ks, const at::Tensor & self, double median, double sigma, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cauchy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "median", median);
    jit::tracer::addInputs(node, "sigma", sigma);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cauchy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, median, sigma, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & log_normal_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log_normal_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor log_normal(c10::DispatchKeySet ks, const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::log_normal::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & unfold_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_in, c10::SymIntArrayRef input_sizes, int64_t dim, int64_t size, int64_t step, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unfold_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_in", grad_in);
    jit::tracer::addInputs(node, "input_sizes", input_sizes);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "step", step);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("unfold_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::unfold_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_in, input_sizes, dim, size, step, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & normal_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double mean, double std, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::normal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mean", mean);
    jit::tracer::addInputs(node, "std", std);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("normal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::normal_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mean, std, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _foreach_atan_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_atan_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_ceil_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_ceil_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_erf_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_erf_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_log2_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_log2_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
at::Tensor & bucketize_out_Scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bucketize");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "boundaries", boundaries);
    jit::tracer::addInputs(node, "out_int32", out_int32);
    jit::tracer::addInputs(node, "right", right);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bucketize_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bucketize_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, boundaries, out_int32, right, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & glu_backward_jvp_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_x, const at::Tensor & grad_glu, const at::Tensor & x, const at::Tensor & dgrad_glu, const at::Tensor & dx, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::glu_backward_jvp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_x", grad_x);
    jit::tracer::addInputs(node, "grad_glu", grad_glu);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "dgrad_glu", dgrad_glu);
    jit::tracer::addInputs(node, "dx", dx);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("glu_backward_jvp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::glu_backward_jvp_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_x, grad_glu, x, dgrad_glu, dx, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & hardswish_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hardswish_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hardswish_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hardswish_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _adaptive_avg_pool3d_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_adaptive_avg_pool3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_adaptive_avg_pool3d_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_adaptive_avg_pool3d_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _conj_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_conj_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_conj_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_conj_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & detach_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::detach_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("detach_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::detach_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & row_indices_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::row_indices_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("row_indices_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::row_indices_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _transformer_encoder_layer_fwd_out_out(c10::DispatchKeySet ks, const at::Tensor & src, int64_t embed_dim, int64_t num_heads, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, bool use_gelu, bool norm_first, double eps, const at::Tensor & norm_weight_1, const at::Tensor & norm_bias_1, const at::Tensor & norm_weight_2, const at::Tensor & norm_bias_2, const at::Tensor & ffn_weight_1, const at::Tensor & ffn_bias_1, const at::Tensor & ffn_weight_2, const at::Tensor & ffn_bias_2, const ::std::optional<at::Tensor> & mask, ::std::optional<int64_t> mask_type, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_transformer_encoder_layer_fwd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_heads", num_heads);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "use_gelu", use_gelu);
    jit::tracer::addInputs(node, "norm_first", norm_first);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "norm_weight_1", norm_weight_1);
    jit::tracer::addInputs(node, "norm_bias_1", norm_bias_1);
    jit::tracer::addInputs(node, "norm_weight_2", norm_weight_2);
    jit::tracer::addInputs(node, "norm_bias_2", norm_bias_2);
    jit::tracer::addInputs(node, "ffn_weight_1", ffn_weight_1);
    jit::tracer::addInputs(node, "ffn_bias_1", ffn_bias_1);
    jit::tracer::addInputs(node, "ffn_weight_2", ffn_weight_2);
    jit::tracer::addInputs(node, "ffn_bias_2", ffn_bias_2);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "mask_type", mask_type);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_transformer_encoder_layer_fwd_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_transformer_encoder_layer_fwd_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), src, embed_dim, num_heads, qkv_weight, qkv_bias, proj_weight, proj_bias, use_gelu, norm_first, eps, norm_weight_1, norm_bias_1, norm_weight_2, norm_bias_2, ffn_weight_1, ffn_bias_1, ffn_weight_2, ffn_bias_2, mask, mask_type, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> _native_multi_head_attention_out_out(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask, bool need_weights, bool average_attn_weights, ::std::optional<int64_t> mask_type, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_native_multi_head_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_head", num_head);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "need_weights", need_weights);
    jit::tracer::addInputs(node, "average_attn_weights", average_attn_weights);
    jit::tracer::addInputs(node, "mask_type", mask_type);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_native_multi_head_attention_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_native_multi_head_attention_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, need_weights, average_attn_weights, mask_type, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & _triton_multi_head_attention_out_out(c10::DispatchKeySet ks, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, int64_t embed_dim, int64_t num_head, const at::Tensor & qkv_weight, const at::Tensor & qkv_bias, const at::Tensor & proj_weight, const at::Tensor & proj_bias, const ::std::optional<at::Tensor> & mask, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_triton_multi_head_attention");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "embed_dim", embed_dim);
    jit::tracer::addInputs(node, "num_head", num_head);
    jit::tracer::addInputs(node, "qkv_weight", qkv_weight);
    jit::tracer::addInputs(node, "qkv_bias", qkv_bias);
    jit::tracer::addInputs(node, "proj_weight", proj_weight);
    jit::tracer::addInputs(node, "proj_bias", proj_bias);
    jit::tracer::addInputs(node, "mask", mask);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_triton_multi_head_attention_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_triton_multi_head_attention_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), query, key, value, embed_dim, num_head, qkv_weight, qkv_bias, proj_weight, proj_bias, mask, out);
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
  m.impl("_cast_Double",
         TORCH_FN(TraceType::_cast_Double)
  );
  m.impl("_cast_Float",
         TORCH_FN(TraceType::_cast_Float)
  );
  m.impl("_use_cudnn_rnn_flatten_weight",
         TORCH_FN(TraceType::_use_cudnn_rnn_flatten_weight)
  );
  m.impl("_cudnn_rnn_flatten_weight",
         TORCH_FN(TraceType::_cudnn_rnn_flatten_weight)
  );
  m.impl("abs",
         TORCH_FN(TraceType::abs)
  );
  m.impl("abs_",
         TORCH_FN(TraceType::abs_)
  );
  m.impl("abs.out",
         TORCH_FN(TraceType::abs_out_out)
  );
  m.impl("imag",
         TORCH_FN(TraceType::imag)
  );
  m.impl("resolve_conj",
         TORCH_FN(TraceType::resolve_conj)
  );
  m.impl("adaptive_max_pool1d",
         TORCH_FN(TraceType::adaptive_max_pool1d)
  );
  m.impl("addmv",
         TORCH_FN(TraceType::addmv)
  );
  m.impl("addmv_",
         TORCH_FN(TraceType::addmv_)
  );
  m.impl("addmv.out",
         TORCH_FN(TraceType::addmv_out_out)
  );
  m.impl("addr",
         TORCH_FN(TraceType::addr)
  );
  m.impl("addr_",
         TORCH_FN(TraceType::addr_)
  );
  m.impl("addr.out",
         TORCH_FN(TraceType::addr_out_out)
  );
  m.impl("atan",
         TORCH_FN(TraceType::atan)
  );
  m.impl("atan_",
         TORCH_FN(TraceType::atan_)
  );
  m.impl("atan.out",
         TORCH_FN(TraceType::atan_out_out)
  );
  m.impl("binary_cross_entropy_backward",
         TORCH_FN(TraceType::binary_cross_entropy_backward)
  );
  m.impl("binary_cross_entropy_backward.grad_input",
         TORCH_FN(TraceType::binary_cross_entropy_backward_out_grad_input)
  );
  m.impl("bitwise_not",
         TORCH_FN(TraceType::bitwise_not)
  );
  m.impl("bitwise_not_",
         TORCH_FN(TraceType::bitwise_not_)
  );
  m.impl("bitwise_not.out",
         TORCH_FN(TraceType::bitwise_not_out_out)
  );
  m.impl("logical_not",
         TORCH_FN(TraceType::logical_not)
  );
  m.impl("logical_not_",
         TORCH_FN(TraceType::logical_not_)
  );
  m.impl("logical_not.out",
         TORCH_FN(TraceType::logical_not_out_out)
  );
  m.impl("concatenate",
         TORCH_FN(TraceType::concatenate)
  );
  m.impl("concatenate.out",
         TORCH_FN(TraceType::concatenate_out_out)
  );
  m.impl("concatenate.names",
         TORCH_FN(TraceType::concatenate_names)
  );
  m.impl("concatenate.names_out",
         TORCH_FN(TraceType::concatenate_out_names_out)
  );
  m.impl("cosh",
         TORCH_FN(TraceType::cosh)
  );
  m.impl("cosh_",
         TORCH_FN(TraceType::cosh_)
  );
  m.impl("cosh.out",
         TORCH_FN(TraceType::cosh_out_out)
  );
  m.impl("cudnn_grid_sampler",
         TORCH_FN(TraceType::cudnn_grid_sampler)
  );
  m.impl("_cummin_helper",
         TORCH_FN(TraceType::_cummin_helper)
  );
  m.impl("div.Tensor",
         TORCH_FN(TraceType::div_Tensor)
  );
  m.impl("div_.Tensor",
         TORCH_FN(TraceType::div__Tensor)
  );
  m.impl("div.out",
         TORCH_FN(TraceType::div_out_out)
  );
  m.impl("div.Tensor_mode",
         TORCH_FN(TraceType::div_Tensor_mode)
  );
  m.impl("div_.Tensor_mode",
         TORCH_FN(TraceType::div__Tensor_mode)
  );
  m.impl("div.out_mode",
         TORCH_FN(TraceType::div_out_out_mode)
  );
  m.impl("div.Scalar",
         TORCH_FN(TraceType::div_Scalar)
  );
  m.impl("div_.Scalar",
         TORCH_FN(TraceType::div__Scalar)
  );
  m.impl("div.Scalar_mode",
         TORCH_FN(TraceType::div_Scalar_mode)
  );
  m.impl("div_.Scalar_mode",
         TORCH_FN(TraceType::div__Scalar_mode)
  );
  m.impl("embedding_bag",
         TORCH_FN(TraceType::embedding_bag)
  );
  m.impl("embedding_bag.padding_idx",
         TORCH_FN(TraceType::embedding_bag_padding_idx)
  );
  m.impl("grid_sampler_3d",
         TORCH_FN(TraceType::grid_sampler_3d)
  );
  m.impl("hann_window",
         TORCH_FN(TraceType::hann_window)
  );
  m.impl("hann_window.periodic",
         TORCH_FN(TraceType::hann_window_periodic)
  );
  m.impl("hamming_window",
         TORCH_FN(TraceType::hamming_window)
  );
  m.impl("hamming_window.periodic",
         TORCH_FN(TraceType::hamming_window_periodic)
  );
  m.impl("hamming_window.periodic_alpha",
         TORCH_FN(TraceType::hamming_window_periodic_alpha)
  );
  m.impl("hamming_window.periodic_alpha_beta",
         TORCH_FN(TraceType::hamming_window_periodic_alpha_beta)
  );
  m.impl("_validate_compressed_sparse_indices",
         TORCH_FN(TraceType::_validate_compressed_sparse_indices)
  );
  m.impl("_cufft_get_plan_cache_size",
         TORCH_FN(TraceType::_cufft_get_plan_cache_size)
  );
  m.impl("_cufft_get_plan_cache_max_size",
         TORCH_FN(TraceType::_cufft_get_plan_cache_max_size)
  );
  m.impl("isnan",
         TORCH_FN(TraceType::isnan)
  );
  m.impl("kthvalue",
         TORCH_FN(TraceType::kthvalue)
  );
  m.impl("kthvalue.values",
         TORCH_FN(TraceType::kthvalue_out_values)
  );
  m.impl("kthvalue.dimname",
         TORCH_FN(TraceType::kthvalue_dimname)
  );
  m.impl("kthvalue.dimname_out",
         TORCH_FN(TraceType::kthvalue_out_dimname_out)
  );
  m.impl("native_layer_norm",
         TORCH_FN(TraceType::native_layer_norm)
  );
  m.impl("_fused_rms_norm_backward",
         TORCH_FN(TraceType::_fused_rms_norm_backward)
  );
  m.impl("_sparse_semi_structured_apply",
         TORCH_FN(TraceType::_sparse_semi_structured_apply)
  );
  m.impl("fbgemm_linear_fp16_weight_fp32_activation",
         TORCH_FN(TraceType::fbgemm_linear_fp16_weight_fp32_activation)
  );
  m.impl("fbgemm_linear_fp16_weight_fp32_activation.out",
         TORCH_FN(TraceType::fbgemm_linear_fp16_weight_fp32_activation_out)
  );
  m.impl("xlogy.Tensor",
         TORCH_FN(TraceType::xlogy_Tensor)
  );
  m.impl("xlogy.Scalar_Self",
         TORCH_FN(TraceType::xlogy_Scalar_Self)
  );
  m.impl("xlogy.Scalar_Other",
         TORCH_FN(TraceType::xlogy_Scalar_Other)
  );
  m.impl("xlogy_.Tensor",
         TORCH_FN(TraceType::xlogy__Tensor)
  );
  m.impl("xlogy_.Scalar_Other",
         TORCH_FN(TraceType::xlogy__Scalar_Other)
  );
  m.impl("xlogy.OutTensor",
         TORCH_FN(TraceType::xlogy_out_OutTensor)
  );
  m.impl("xlogy.OutScalar_Self",
         TORCH_FN(TraceType::xlogy_out_OutScalar_Self)
  );
  m.impl("xlogy.OutScalar_Other",
         TORCH_FN(TraceType::xlogy_out_OutScalar_Other)
  );
  m.impl("matrix_exp_backward",
         TORCH_FN(TraceType::matrix_exp_backward)
  );
  m.impl("mkldnn_max_pool2d",
         TORCH_FN(TraceType::mkldnn_max_pool2d)
  );
  m.impl("_mps_convolution",
         TORCH_FN(TraceType::_mps_convolution)
  );
  m.impl("batch_norm_stats",
         TORCH_FN(TraceType::batch_norm_stats)
  );
  m.impl("batch_norm_backward_reduce",
         TORCH_FN(TraceType::batch_norm_backward_reduce)
  );
  m.impl("_cdist_forward",
         TORCH_FN(TraceType::_cdist_forward)
  );
  m.impl("cosine_similarity",
         TORCH_FN(TraceType::cosine_similarity)
  );
  m.impl("movedim.intlist",
         TORCH_FN(TraceType::movedim_intlist)
  );
  m.impl("movedim.int",
         TORCH_FN(TraceType::movedim_int)
  );
  m.impl("mH",
         TORCH_FN(TraceType::mH)
  );
  m.impl("rand_like",
         TORCH_FN(TraceType::rand_like)
  );
  m.impl("rand_like.generator",
         TORCH_FN(TraceType::rand_like_generator)
  );
  m.impl("randint_like",
         TORCH_FN(TraceType::randint_like)
  );
  m.impl("randint_like.generator",
         TORCH_FN(TraceType::randint_like_generator)
  );
  m.impl("randint_like.Tensor",
         TORCH_FN(TraceType::randint_like_Tensor)
  );
  m.impl("randint_like.Tensor_generator",
         TORCH_FN(TraceType::randint_like_Tensor_generator)
  );
  m.impl("randint_like.low_dtype",
         TORCH_FN(TraceType::randint_like_low_dtype)
  );
  m.impl("randint_like.low_generator_dtype",
         TORCH_FN(TraceType::randint_like_low_generator_dtype)
  );
  m.impl("round",
         TORCH_FN(TraceType::round)
  );
  m.impl("round_",
         TORCH_FN(TraceType::round_)
  );
  m.impl("round.out",
         TORCH_FN(TraceType::round_out_out)
  );
  m.impl("round.decimals",
         TORCH_FN(TraceType::round_decimals)
  );
  m.impl("round_.decimals",
         TORCH_FN(TraceType::round__decimals)
  );
  m.impl("round.decimals_out",
         TORCH_FN(TraceType::round_out_decimals_out)
  );
  m.impl("sigmoid",
         TORCH_FN(TraceType::sigmoid)
  );
  m.impl("sigmoid_",
         TORCH_FN(TraceType::sigmoid_)
  );
  m.impl("sigmoid.out",
         TORCH_FN(TraceType::sigmoid_out_out)
  );
  m.impl("hsplit.int",
         TORCH_FN(TraceType::hsplit_int)
  );
  m.impl("hsplit.array",
         TORCH_FN(TraceType::hsplit_array)
  );
  m.impl("stack",
         TORCH_FN(TraceType::stack)
  );
  m.impl("stack.out",
         TORCH_FN(TraceType::stack_out_out)
  );
  m.impl("hash_tensor",
         TORCH_FN(TraceType::hash_tensor)
  );
  m.impl("hash_tensor.out",
         TORCH_FN(TraceType::hash_tensor_out_out)
  );
  m.impl("tile",
         TORCH_FN(TraceType::tile)
  );
  m.impl("_mkldnn_transpose",
         TORCH_FN(TraceType::_mkldnn_transpose)
  );
  m.impl("_mkldnn_transpose_",
         TORCH_FN(TraceType::_mkldnn_transpose_)
  );
  m.impl("_nested_from_padded_and_nested_example",
         TORCH_FN(TraceType::_nested_from_padded_and_nested_example)
  );
  m.impl("_nested_compute_contiguous_strides_offsets",
         TORCH_FN(TraceType::_nested_compute_contiguous_strides_offsets)
  );
  m.impl("unique_dim",
         TORCH_FN(TraceType::unique_dim)
  );
  m.impl("vander",
         TORCH_FN(TraceType::vander)
  );
  m.impl("_philox_key_split",
         TORCH_FN(TraceType::_philox_key_split)
  );
  m.impl("batch_norm_backward",
         TORCH_FN(TraceType::batch_norm_backward)
  );
  m.impl("clone",
         TORCH_FN(TraceType::clone)
  );
  m.impl("positive",
         TORCH_FN(TraceType::positive)
  );
  m.impl("_scaled_mm_v2",
         TORCH_FN(TraceType::_scaled_mm_v2)
  );
  m.impl("_scaled_mm_v2.out",
         TORCH_FN(TraceType::_scaled_mm_v2_out_out)
  );
  m.impl("sparse_csr_tensor.crow_col_value_size",
         TORCH_FN(TraceType::sparse_csr_tensor_crow_col_value_size)
  );
  m.impl("sparse_csr_tensor.crow_col_value",
         TORCH_FN(TraceType::sparse_csr_tensor_crow_col_value)
  );
  m.impl("_sparse_bsc_tensor_unsafe",
         TORCH_FN(TraceType::_sparse_bsc_tensor_unsafe)
  );
  m.impl("dense_dim",
         TORCH_FN(TraceType::dense_dim)
  );
  m.impl("_dimV",
         TORCH_FN(TraceType::_dimV)
  );
  m.impl("coalesce",
         TORCH_FN(TraceType::coalesce)
  );
  m.impl("_indices",
         TORCH_FN(TraceType::_indices)
  );
  m.impl("_to_sparse_csc",
         TORCH_FN(TraceType::_to_sparse_csc)
  );
  m.impl("_to_sparse_bsc",
         TORCH_FN(TraceType::_to_sparse_bsc)
  );
  m.impl("dequantize.self",
         TORCH_FN(TraceType::dequantize_self)
  );
  m.impl("dequantize.tensors",
         TORCH_FN(TraceType::dequantize_tensors)
  );
  m.impl("q_per_channel_zero_points",
         TORCH_FN(TraceType::q_per_channel_zero_points)
  );
  m.impl("fake_quantize_per_tensor_affine",
         TORCH_FN(TraceType::fake_quantize_per_tensor_affine)
  );
  m.impl("fake_quantize_per_tensor_affine.tensor_qparams",
         TORCH_FN(TraceType::fake_quantize_per_tensor_affine_tensor_qparams)
  );
  m.impl("_fake_quantize_learnable_per_channel_affine",
         TORCH_FN(TraceType::_fake_quantize_learnable_per_channel_affine)
  );
  m.impl("item",
         TORCH_FN(TraceType::item)
  );
  m.impl("_thnn_fused_lstm_cell",
         TORCH_FN(TraceType::_thnn_fused_lstm_cell)
  );
  m.impl("rnn_tanh.input",
         TORCH_FN(TraceType::rnn_tanh_input)
  );
  m.impl("rnn_tanh.data",
         TORCH_FN(TraceType::rnn_tanh_data)
  );
  m.impl("rnn_relu_cell",
         TORCH_FN(TraceType::rnn_relu_cell)
  );
  m.impl("_pad_packed_sequence",
         TORCH_FN(TraceType::_pad_packed_sequence)
  );
  m.impl("random_.from",
         TORCH_FN(TraceType::random__from)
  );
  m.impl("random_.to",
         TORCH_FN(TraceType::random__to)
  );
  m.impl("random_",
         TORCH_FN(TraceType::random_)
  );
  m.impl("cauchy_",
         TORCH_FN(TraceType::cauchy_)
  );
  m.impl("log_normal_",
         TORCH_FN(TraceType::log_normal_)
  );
  m.impl("cross.out",
         TORCH_FN(TraceType::cross_out_out)
  );
  m.impl("cross",
         TORCH_FN(TraceType::cross)
  );
  m.impl("ne.Scalar_out",
         TORCH_FN(TraceType::ne_out_Scalar_out)
  );
  m.impl("ne.Scalar",
         TORCH_FN(TraceType::ne_Scalar)
  );
  m.impl("ne.Tensor_out",
         TORCH_FN(TraceType::ne_out_Tensor_out)
  );
  m.impl("ne.Tensor",
         TORCH_FN(TraceType::ne_Tensor)
  );
  m.impl("ne_.Scalar",
         TORCH_FN(TraceType::ne__Scalar)
  );
  m.impl("ne_.Tensor",
         TORCH_FN(TraceType::ne__Tensor)
  );
  m.impl("_gather_sparse_backward",
         TORCH_FN(TraceType::_gather_sparse_backward)
  );
  m.impl("linalg_vander",
         TORCH_FN(TraceType::linalg_vander)
  );
  m.impl("swapaxes",
         TORCH_FN(TraceType::swapaxes)
  );
  m.impl("swapaxes_",
         TORCH_FN(TraceType::swapaxes_)
  );
  m.impl("histc.out",
         TORCH_FN(TraceType::histc_out_out)
  );
  m.impl("histc",
         TORCH_FN(TraceType::histc)
  );
  m.impl("quantile",
         TORCH_FN(TraceType::quantile)
  );
  m.impl("quantile.out",
         TORCH_FN(TraceType::quantile_out_out)
  );
  m.impl("quantile.scalar",
         TORCH_FN(TraceType::quantile_scalar)
  );
  m.impl("quantile.scalar_out",
         TORCH_FN(TraceType::quantile_out_scalar_out)
  );
  m.impl("msort.out",
         TORCH_FN(TraceType::msort_out_out)
  );
  m.impl("msort",
         TORCH_FN(TraceType::msort)
  );
  m.impl("argsort",
         TORCH_FN(TraceType::argsort)
  );
  m.impl("argsort.stable",
         TORCH_FN(TraceType::argsort_stable)
  );
  m.impl("argsort.stable_out",
         TORCH_FN(TraceType::argsort_out_stable_out)
  );
  m.impl("argsort.dimname",
         TORCH_FN(TraceType::argsort_dimname)
  );
  m.impl("unfold_backward",
         TORCH_FN(TraceType::unfold_backward)
  );
  m.impl("normal_",
         TORCH_FN(TraceType::normal_)
  );
  m.impl("normal_functional",
         TORCH_FN(TraceType::normal_functional)
  );
  m.impl("normal.Tensor_float_out",
         TORCH_FN(TraceType::normal_out_Tensor_float_out)
  );
  m.impl("normal.Tensor_float",
         TORCH_FN(TraceType::normal_Tensor_float)
  );
  m.impl("normal.float_Tensor_out",
         TORCH_FN(TraceType::normal_out_float_Tensor_out)
  );
  m.impl("normal.float_Tensor",
         TORCH_FN(TraceType::normal_float_Tensor)
  );
  m.impl("normal.Tensor_Tensor_out",
         TORCH_FN(TraceType::normal_out_Tensor_Tensor_out)
  );
  m.impl("normal.Tensor_Tensor",
         TORCH_FN(TraceType::normal_Tensor_Tensor)
  );
  m.impl("normal.float_float",
         TORCH_FN(TraceType::normal_float_float)
  );
  m.impl("normal.float_float_out",
         TORCH_FN(TraceType::normal_out_float_float_out)
  );
  m.impl("_foreach_atan",
         TORCH_FN(TraceType::_foreach_atan)
  );
  m.impl("_foreach_atan_",
         TORCH_FN(TraceType::_foreach_atan_)
  );
  m.impl("_foreach_ceil",
         TORCH_FN(TraceType::_foreach_ceil)
  );
  m.impl("_foreach_ceil_",
         TORCH_FN(TraceType::_foreach_ceil_)
  );
  m.impl("_foreach_erf",
         TORCH_FN(TraceType::_foreach_erf)
  );
  m.impl("_foreach_erf_",
         TORCH_FN(TraceType::_foreach_erf_)
  );
  m.impl("_foreach_log2",
         TORCH_FN(TraceType::_foreach_log2)
  );
  m.impl("_foreach_log2_",
         TORCH_FN(TraceType::_foreach_log2_)
  );
  m.impl("bucketize.Tensor",
         TORCH_FN(TraceType::bucketize_Tensor)
  );
  m.impl("bucketize.Tensor_out",
         TORCH_FN(TraceType::bucketize_out_Tensor_out)
  );
  m.impl("bucketize.Scalar",
         TORCH_FN(TraceType::bucketize_Scalar)
  );
  m.impl("mse_loss.out",
         TORCH_FN(TraceType::mse_loss_out_out)
  );
  m.impl("mse_loss",
         TORCH_FN(TraceType::mse_loss)
  );
  m.impl("l1_loss",
         TORCH_FN(TraceType::l1_loss)
  );
  m.impl("nll_loss2d_forward.output",
         TORCH_FN(TraceType::nll_loss2d_forward_out_output)
  );
  m.impl("nll_loss2d_forward",
         TORCH_FN(TraceType::nll_loss2d_forward)
  );
  m.impl("soft_margin_loss.out",
         TORCH_FN(TraceType::soft_margin_loss_out_out)
  );
  m.impl("soft_margin_loss",
         TORCH_FN(TraceType::soft_margin_loss)
  );
  m.impl("glu_backward_jvp",
         TORCH_FN(TraceType::glu_backward_jvp)
  );
  m.impl("hardtanh.out",
         TORCH_FN(TraceType::hardtanh_out_out)
  );
  m.impl("hardtanh",
         TORCH_FN(TraceType::hardtanh)
  );
  m.impl("hardtanh_",
         TORCH_FN(TraceType::hardtanh_)
  );
  m.impl("hardswish_backward",
         TORCH_FN(TraceType::hardswish_backward)
  );
  m.impl("leaky_relu.out",
         TORCH_FN(TraceType::leaky_relu_out_out)
  );
  m.impl("leaky_relu",
         TORCH_FN(TraceType::leaky_relu)
  );
  m.impl("leaky_relu_",
         TORCH_FN(TraceType::leaky_relu_)
  );
  m.impl("log_sigmoid_backward.grad_input",
         TORCH_FN(TraceType::log_sigmoid_backward_out_grad_input)
  );
  m.impl("log_sigmoid_backward",
         TORCH_FN(TraceType::log_sigmoid_backward)
  );
  m.impl("_adaptive_avg_pool3d_backward",
         TORCH_FN(TraceType::_adaptive_avg_pool3d_backward)
  );
  m.impl("adaptive_max_pool2d_backward.grad_input",
         TORCH_FN(TraceType::adaptive_max_pool2d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool2d_backward",
         TORCH_FN(TraceType::adaptive_max_pool2d_backward)
  );
  m.impl("adaptive_max_pool3d_backward.grad_input",
         TORCH_FN(TraceType::adaptive_max_pool3d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool3d_backward",
         TORCH_FN(TraceType::adaptive_max_pool3d_backward)
  );
  m.impl("replication_pad2d.out",
         TORCH_FN(TraceType::replication_pad2d_out_out)
  );
  m.impl("replication_pad2d",
         TORCH_FN(TraceType::replication_pad2d)
  );
  m.impl("_pad_circular",
         TORCH_FN(TraceType::_pad_circular)
  );
  m.impl("_conv_depthwise2d.out",
         TORCH_FN(TraceType::_conv_depthwise2d_out_out)
  );
  m.impl("_conv_depthwise2d",
         TORCH_FN(TraceType::_conv_depthwise2d)
  );
  m.impl("slow_conv3d.out",
         TORCH_FN(TraceType::slow_conv3d_out_out)
  );
  m.impl("slow_conv3d",
         TORCH_FN(TraceType::slow_conv3d)
  );
  m.impl("special_log_ndtr",
         TORCH_FN(TraceType::special_log_ndtr)
  );
  m.impl("special_log_ndtr.out",
         TORCH_FN(TraceType::special_log_ndtr_out_out)
  );
  m.impl("special_erf",
         TORCH_FN(TraceType::special_erf)
  );
  m.impl("special_erf.out",
         TORCH_FN(TraceType::special_erf_out_out)
  );
  m.impl("special_softmax",
         TORCH_FN(TraceType::special_softmax)
  );
  m.impl("fft_fft",
         TORCH_FN(TraceType::fft_fft)
  );
  m.impl("fft_fft.out",
         TORCH_FN(TraceType::fft_fft_out_out)
  );
  m.impl("fft_rfft",
         TORCH_FN(TraceType::fft_rfft)
  );
  m.impl("fft_rfft.out",
         TORCH_FN(TraceType::fft_rfft_out_out)
  );
  m.impl("fft_hfft2",
         TORCH_FN(TraceType::fft_hfft2)
  );
  m.impl("fft_hfft2.out",
         TORCH_FN(TraceType::fft_hfft2_out_out)
  );
  m.impl("fft_ifftn",
         TORCH_FN(TraceType::fft_ifftn)
  );
  m.impl("fft_ifftn.out",
         TORCH_FN(TraceType::fft_ifftn_out_out)
  );
  m.impl("fft_fftfreq",
         TORCH_FN(TraceType::fft_fftfreq)
  );
  m.impl("fft_fftfreq.out",
         TORCH_FN(TraceType::fft_fftfreq_out_out)
  );
  m.impl("linalg_lu_factor_ex",
         TORCH_FN(TraceType::linalg_lu_factor_ex)
  );
  m.impl("linalg_lu_factor_ex.out",
         TORCH_FN(TraceType::linalg_lu_factor_ex_out_out)
  );
  m.impl("linalg__powsum",
         TORCH_FN(TraceType::linalg__powsum)
  );
  m.impl("_spsolve",
         TORCH_FN(TraceType::_spsolve)
  );
  m.impl("linalg_multi_dot",
         TORCH_FN(TraceType::linalg_multi_dot)
  );
  m.impl("linalg_multi_dot.out",
         TORCH_FN(TraceType::linalg_multi_dot_out_out)
  );
  m.impl("_test_parallel_materialize",
         TORCH_FN(TraceType::_test_parallel_materialize)
  );
  m.impl("flatten_dense_tensors",
         TORCH_FN(TraceType::flatten_dense_tensors)
  );
  m.impl("_conj_copy",
         TORCH_FN(TraceType::_conj_copy)
  );
  m.impl("detach_copy",
         TORCH_FN(TraceType::detach_copy)
  );
  m.impl("row_indices_copy",
         TORCH_FN(TraceType::row_indices_copy)
  );
  m.impl("_transformer_encoder_layer_fwd",
         TORCH_FN(TraceType::_transformer_encoder_layer_fwd)
  );
  m.impl("_native_multi_head_attention",
         TORCH_FN(TraceType::_native_multi_head_attention)
  );
  m.impl("_fused_sdp_choice",
         TORCH_FN(TraceType::_fused_sdp_choice)
  );
  m.impl("_scaled_dot_product_flash_attention_for_cpu",
         TORCH_FN(TraceType::_scaled_dot_product_flash_attention_for_cpu)
  );
  m.impl("_scaled_dot_product_efficient_attention_backward",
         TORCH_FN(TraceType::_scaled_dot_product_efficient_attention_backward)
  );
  m.impl("_fill_mem_eff_dropout_mask_",
         TORCH_FN(TraceType::_fill_mem_eff_dropout_mask_)
  );
  m.impl("_triton_multi_head_attention",
         TORCH_FN(TraceType::_triton_multi_head_attention)
  );
  m.impl("special_modified_bessel_i0",
         TORCH_FN(TraceType::special_modified_bessel_i0)
  );
  m.impl("special_modified_bessel_i0.out",
         TORCH_FN(TraceType::special_modified_bessel_i0_out_out)
  );
  m.impl("special_modified_bessel_k0",
         TORCH_FN(TraceType::special_modified_bessel_k0)
  );
  m.impl("special_modified_bessel_k0.out",
         TORCH_FN(TraceType::special_modified_bessel_k0_out_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_t",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.x_scalar",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t_x_scalar)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.n_scalar",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t_n_scalar)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t_out_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.x_scalar_out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t_out_x_scalar_out)
  );
  m.impl("special_shifted_chebyshev_polynomial_t.n_scalar_out",
         TORCH_FN(TraceType::special_shifted_chebyshev_polynomial_t_out_n_scalar_out)
  );
  m.impl("_cudnn_rnn_flatten_weight.out",
         TORCH_FN(TraceType::_cudnn_rnn_flatten_weight_out_out)
  );
  m.impl("cudnn_grid_sampler.out",
         TORCH_FN(TraceType::cudnn_grid_sampler_out_out)
  );
  m.impl("div.Scalar_out",
         TORCH_FN(TraceType::div_out_Scalar_out)
  );
  m.impl("div.Scalar_mode_out",
         TORCH_FN(TraceType::div_out_Scalar_mode_out)
  );
  m.impl("grid_sampler_3d.out",
         TORCH_FN(TraceType::grid_sampler_3d_out_out)
  );
  m.impl("hann_window.out",
         TORCH_FN(TraceType::hann_window_out_out)
  );
  m.impl("hann_window.periodic_out",
         TORCH_FN(TraceType::hann_window_out_periodic_out)
  );
  m.impl("hamming_window.out",
         TORCH_FN(TraceType::hamming_window_out_out)
  );
  m.impl("hamming_window.periodic_out",
         TORCH_FN(TraceType::hamming_window_out_periodic_out)
  );
  m.impl("hamming_window.periodic_alpha_out",
         TORCH_FN(TraceType::hamming_window_out_periodic_alpha_out)
  );
  m.impl("hamming_window.periodic_alpha_beta_out",
         TORCH_FN(TraceType::hamming_window_out_periodic_alpha_beta_out)
  );
  m.impl("isnan.out",
         TORCH_FN(TraceType::isnan_out_out)
  );
  m.impl("native_layer_norm.out",
         TORCH_FN(TraceType::native_layer_norm_out_out)
  );
  m.impl("mkldnn_max_pool2d.out",
         TORCH_FN(TraceType::mkldnn_max_pool2d_out_out)
  );
  m.impl("_mps_convolution.out",
         TORCH_FN(TraceType::_mps_convolution_out_out)
  );
  m.impl("batch_norm_stats.out",
         TORCH_FN(TraceType::batch_norm_stats_out_out)
  );
  m.impl("batch_norm_backward_reduce.out",
         TORCH_FN(TraceType::batch_norm_backward_reduce_out_out)
  );
  m.impl("_cdist_forward.out",
         TORCH_FN(TraceType::_cdist_forward_out_out)
  );
  m.impl("rand_like.out",
         TORCH_FN(TraceType::rand_like_out_out)
  );
  m.impl("rand_like.generator_out",
         TORCH_FN(TraceType::rand_like_out_generator_out)
  );
  m.impl("randint_like.out",
         TORCH_FN(TraceType::randint_like_out_out)
  );
  m.impl("randint_like.generator_out",
         TORCH_FN(TraceType::randint_like_out_generator_out)
  );
  m.impl("randint_like.Tensor_out",
         TORCH_FN(TraceType::randint_like_out_Tensor_out)
  );
  m.impl("randint_like.Tensor_generator_out",
         TORCH_FN(TraceType::randint_like_out_Tensor_generator_out)
  );
  m.impl("randint_like.low_dtype_out",
         TORCH_FN(TraceType::randint_like_out_low_dtype_out)
  );
  m.impl("randint_like.low_generator_dtype_out",
         TORCH_FN(TraceType::randint_like_out_low_generator_dtype_out)
  );
  m.impl("_mkldnn_transpose.out",
         TORCH_FN(TraceType::_mkldnn_transpose_out_out)
  );
  m.impl("_nested_from_padded_and_nested_example.out",
         TORCH_FN(TraceType::_nested_from_padded_and_nested_example_out_out)
  );
  m.impl("unique_dim.out",
         TORCH_FN(TraceType::unique_dim_out_out)
  );
  m.impl("clone.out",
         TORCH_FN(TraceType::clone_out_out)
  );
  m.impl("resize_as.out",
         TORCH_FN(TraceType::resize_as_out_out)
  );
  m.impl("resize_as",
         TORCH_FN(TraceType::resize_as)
  );
  m.impl("_to_sparse_csc.out",
         TORCH_FN(TraceType::_to_sparse_csc_out_out)
  );
  m.impl("_to_sparse_bsc.out",
         TORCH_FN(TraceType::_to_sparse_bsc_out_out)
  );
  m.impl("dequantize.self_out",
         TORCH_FN(TraceType::dequantize_out_self_out)
  );
  m.impl("dequantize.tensors_out",
         TORCH_FN(TraceType::dequantize_out_tensors_out)
  );
  m.impl("q_per_channel_zero_points.out",
         TORCH_FN(TraceType::q_per_channel_zero_points_out_out)
  );
  m.impl("_fake_quantize_learnable_per_channel_affine.out",
         TORCH_FN(TraceType::_fake_quantize_learnable_per_channel_affine_out_out)
  );
  m.impl("_thnn_fused_lstm_cell.out",
         TORCH_FN(TraceType::_thnn_fused_lstm_cell_out_out)
  );
  m.impl("random.from_out",
         TORCH_FN(TraceType::random_out_from_out)
  );
  m.impl("random.from",
         TORCH_FN(TraceType::random_from)
  );
  m.impl("random.to_out",
         TORCH_FN(TraceType::random_out_to_out)
  );
  m.impl("random.to",
         TORCH_FN(TraceType::random_to)
  );
  m.impl("random.out",
         TORCH_FN(TraceType::random_out_out)
  );
  m.impl("random",
         TORCH_FN(TraceType::random)
  );
  m.impl("cauchy.out",
         TORCH_FN(TraceType::cauchy_out_out)
  );
  m.impl("cauchy",
         TORCH_FN(TraceType::cauchy)
  );
  m.impl("log_normal.out",
         TORCH_FN(TraceType::log_normal_out_out)
  );
  m.impl("log_normal",
         TORCH_FN(TraceType::log_normal)
  );
  m.impl("unfold_backward.out",
         TORCH_FN(TraceType::unfold_backward_out_out)
  );
  m.impl("normal.out",
         TORCH_FN(TraceType::normal_out_out)
  );
  m.impl("_foreach_atan.out",
         TORCH_FN(TraceType::_foreach_atan_out_out)
  );
  m.impl("_foreach_ceil.out",
         TORCH_FN(TraceType::_foreach_ceil_out_out)
  );
  m.impl("_foreach_erf.out",
         TORCH_FN(TraceType::_foreach_erf_out_out)
  );
  m.impl("_foreach_log2.out",
         TORCH_FN(TraceType::_foreach_log2_out_out)
  );
  m.impl("bucketize.Scalar_out",
         TORCH_FN(TraceType::bucketize_out_Scalar_out)
  );
  m.impl("glu_backward_jvp.out",
         TORCH_FN(TraceType::glu_backward_jvp_out_out)
  );
  m.impl("hardswish_backward.out",
         TORCH_FN(TraceType::hardswish_backward_out_out)
  );
  m.impl("_adaptive_avg_pool3d_backward.out",
         TORCH_FN(TraceType::_adaptive_avg_pool3d_backward_out_out)
  );
  m.impl("_conj_copy.out",
         TORCH_FN(TraceType::_conj_copy_out_out)
  );
  m.impl("detach_copy.out",
         TORCH_FN(TraceType::detach_copy_out_out)
  );
  m.impl("row_indices_copy.out",
         TORCH_FN(TraceType::row_indices_copy_out_out)
  );
  m.impl("_transformer_encoder_layer_fwd.out",
         TORCH_FN(TraceType::_transformer_encoder_layer_fwd_out_out)
  );
  m.impl("_native_multi_head_attention.out",
         TORCH_FN(TraceType::_native_multi_head_attention_out_out)
  );
  m.impl("_triton_multi_head_attention.out",
         TORCH_FN(TraceType::_triton_multi_head_attention_out_out)
  );;
}

}  // namespace

} // namespace torch
