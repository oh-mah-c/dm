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
#include <ATen/ops/_new_zeros_with_same_feature_meta_ops.h>
#include <ATen/ops/_assert_scalar_ops.h>
#include <ATen/ops/_print_ops.h>
#include <ATen/ops/_functional_sym_constrain_range_for_size_ops.h>
#include <ATen/ops/_make_dep_token_ops.h>
#include <ATen/ops/_cudnn_init_dropout_state_ops.h>
#include <ATen/ops/native_dropout_ops.h>
#include <ATen/ops/angle_ops.h>
#include <ATen/ops/angle_ops.h>
#include <ATen/ops/asinh_ops.h>
#include <ATen/ops/asinh_ops.h>
#include <ATen/ops/asinh_ops.h>
#include <ATen/ops/baddbmm_ops.h>
#include <ATen/ops/baddbmm_ops.h>
#include <ATen/ops/baddbmm_ops.h>
#include <ATen/ops/baddbmm_ops.h>
#include <ATen/ops/baddbmm_ops.h>
#include <ATen/ops/batch_norm_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bilinear_ops.h>
#include <ATen/ops/bincount_ops.h>
#include <ATen/ops/unsafe_chunk_ops.h>
#include <ATen/ops/clamp_max_ops.h>
#include <ATen/ops/clamp_max_ops.h>
#include <ATen/ops/clamp_max_ops.h>
#include <ATen/ops/clamp_max_ops.h>
#include <ATen/ops/clamp_max_ops.h>
#include <ATen/ops/clamp_max_ops.h>
#include <ATen/ops/clip_ops.h>
#include <ATen/ops/clip_ops.h>
#include <ATen/ops/clip_ops.h>
#include <ATen/ops/clip_ops.h>
#include <ATen/ops/clip_ops.h>
#include <ATen/ops/clip_ops.h>
#include <ATen/ops/complex_ops.h>
#include <ATen/ops/complex_ops.h>
#include <ATen/ops/polar_ops.h>
#include <ATen/ops/polar_ops.h>
#include <ATen/ops/conv_transpose2d_ops.h>
#include <ATen/ops/count_nonzero_ops.h>
#include <ATen/ops/count_nonzero_ops.h>
#include <ATen/ops/cudnn_convolution_add_relu_ops.h>
#include <ATen/ops/cummax_ops.h>
#include <ATen/ops/cummax_ops.h>
#include <ATen/ops/cummax_ops.h>
#include <ATen/ops/cummax_ops.h>
#include <ATen/ops/_cummax_helper_ops.h>
#include <ATen/ops/_ctc_loss_backward_ops.h>
#include <ATen/ops/_ctc_loss_backward_ops.h>
#include <ATen/ops/diff_ops.h>
#include <ATen/ops/diff_ops.h>
#include <ATen/ops/gradient_ops.h>
#include <ATen/ops/gradient_ops.h>
#include <ATen/ops/gradient_ops.h>
#include <ATen/ops/gradient_ops.h>
#include <ATen/ops/gradient_ops.h>
#include <ATen/ops/gradient_ops.h>
#include <ATen/ops/gradient_ops.h>
#include <ATen/ops/einsum_ops.h>
#include <ATen/ops/embedding_renorm_ops.h>
#include <ATen/ops/empty_ops.h>
#include <ATen/ops/empty_ops.h>
#include <ATen/ops/new_ones_ops.h>
#include <ATen/ops/_empty_per_channel_affine_quantized_ops.h>
#include <ATen/ops/empty_ops.h>
#include <ATen/ops/exp2_ops.h>
#include <ATen/ops/exp2_ops.h>
#include <ATen/ops/exp2_ops.h>
#include <ATen/ops/eye_ops.h>
#include <ATen/ops/eye_ops.h>
#include <ATen/ops/eye_ops.h>
#include <ATen/ops/eye_ops.h>
#include <ATen/ops/gcd_ops.h>
#include <ATen/ops/gcd_ops.h>
#include <ATen/ops/gcd_ops.h>
#include <ATen/ops/_cufft_clear_plan_cache_ops.h>
#include <ATen/ops/isin_ops.h>
#include <ATen/ops/isin_ops.h>
#include <ATen/ops/isin_ops.h>
#include <ATen/ops/isin_ops.h>
#include <ATen/ops/isin_ops.h>
#include <ATen/ops/isin_ops.h>
#include <ATen/ops/is_conj_ops.h>
#include <ATen/ops/is_nonzero_ops.h>
#include <ATen/ops/storage_offset_ops.h>
#include <ATen/ops/layer_norm_ops.h>
#include <ATen/ops/rms_norm_ops.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_ops.h>
#include <ATen/ops/fbgemm_linear_fp16_weight_ops.h>
#include <ATen/ops/fbgemm_pack_quantized_matrix_ops.h>
#include <ATen/ops/fbgemm_pack_quantized_matrix_ops.h>
#include <ATen/ops/log_ops.h>
#include <ATen/ops/log_ops.h>
#include <ATen/ops/log_ops.h>
#include <ATen/ops/logaddexp_ops.h>
#include <ATen/ops/logaddexp_ops.h>
#include <ATen/ops/log_softmax_ops.h>
#include <ATen/ops/log_softmax_ops.h>
#include <ATen/ops/log_softmax_ops.h>
#include <ATen/ops/mkldnn_max_pool3d_ops.h>
#include <ATen/ops/quantized_max_pool3d_ops.h>
#include <ATen/ops/miopen_convolution_ops.h>
#include <ATen/ops/batch_norm_update_stats_ops.h>
#include <ATen/ops/ones_like_ops.h>
#include <ATen/ops/_pdist_forward_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randint_ops.h>
#include <ATen/ops/randn_like_ops.h>
#include <ATen/ops/randn_like_ops.h>
#include <ATen/ops/repeat_interleave_ops.h>
#include <ATen/ops/repeat_interleave_ops.h>
#include <ATen/ops/repeat_interleave_ops.h>
#include <ATen/ops/_prelu_kernel_ops.h>
#include <ATen/ops/_nested_select_backward_ops.h>
#include <ATen/ops/vsplit_ops.h>
#include <ATen/ops/vsplit_ops.h>
#include <ATen/ops/hstack_ops.h>
#include <ATen/ops/hstack_ops.h>
#include <ATen/ops/sum_ops.h>
#include <ATen/ops/sum_ops.h>
#include <ATen/ops/sum_ops.h>
#include <ATen/ops/sum_ops.h>
#include <ATen/ops/sum_ops.h>
#include <ATen/ops/rot90_ops.h>
#include <ATen/ops/_nested_tensor_strides_ops.h>
#include <ATen/ops/_nested_tensor_storage_offsets_ops.h>
#include <ATen/ops/triplet_margin_loss_ops.h>
#include <ATen/ops/var_ops.h>
#include <ATen/ops/var_ops.h>
#include <ATen/ops/var_ops.h>
#include <ATen/ops/var_ops.h>
#include <ATen/ops/var_ops.h>
#include <ATen/ops/var_ops.h>
#include <ATen/ops/var_ops.h>
#include <ATen/ops/var_ops.h>
#include <ATen/ops/var_ops.h>
#include <ATen/ops/_standard_gamma_grad_ops.h>
#include <ATen/ops/_batch_norm_with_update_ops.h>
#include <ATen/ops/_batch_norm_with_update_ops.h>
#include <ATen/ops/_sparse_csr_sum_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/nuclear_norm_ops.h>
#include <ATen/ops/nuclear_norm_ops.h>
#include <ATen/ops/nuclear_norm_ops.h>
#include <ATen/ops/nuclear_norm_ops.h>
#include <ATen/ops/_sparse_compressed_tensor_with_dims_ops.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_ops.h>
#include <ATen/ops/_validate_sparse_coo_tensor_args_ops.h>
#include <ATen/ops/_sparse_mask_projection_ops.h>
#include <ATen/ops/_to_dense_ops.h>
#include <ATen/ops/_coalesced_ops.h>
#include <ATen/ops/to_sparse_bsc_ops.h>
#include <ATen/ops/quantize_per_tensor_dynamic_ops.h>
#include <ATen/ops/quantize_per_tensor_ops.h>
#include <ATen/ops/quantize_per_tensor_ops.h>
#include <ATen/ops/quantize_per_tensor_ops.h>
#include <ATen/ops/fake_quantize_per_tensor_affine_cachemask_ops.h>
#include <ATen/ops/_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_ops.h>
#include <ATen/ops/_fake_quantize_learnable_per_tensor_affine_ops.h>
#include <ATen/ops/choose_qparams_optimized_ops.h>
#include <ATen/ops/_thnn_fused_gru_cell_backward_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/masked_scatter_backward_ops.h>
#include <ATen/ops/put_ops.h>
#include <ATen/ops/put_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/scatter_ops.h>
#include <ATen/ops/and_ops.h>
#include <ATen/ops/and_ops.h>
#include <ATen/ops/and_ops.h>
#include <ATen/ops/and_ops.h>
#include <ATen/ops/tril_ops.h>
#include <ATen/ops/uniform_ops.h>
#include <ATen/ops/tril_ops.h>
#include <ATen/ops/tril_ops.h>
#include <ATen/ops/tril_indices_ops.h>
#include <ATen/ops/less_equal_ops.h>
#include <ATen/ops/less_equal_ops.h>
#include <ATen/ops/less_equal_ops.h>
#include <ATen/ops/less_equal_ops.h>
#include <ATen/ops/less_equal_ops.h>
#include <ATen/ops/less_equal_ops.h>
#include <ATen/ops/gt_ops.h>
#include <ATen/ops/gt_ops.h>
#include <ATen/ops/gt_ops.h>
#include <ATen/ops/gt_ops.h>
#include <ATen/ops/gt_ops.h>
#include <ATen/ops/gt_ops.h>
#include <ATen/ops/lt_ops.h>
#include <ATen/ops/lt_ops.h>
#include <ATen/ops/lt_ops.h>
#include <ATen/ops/lt_ops.h>
#include <ATen/ops/lt_ops.h>
#include <ATen/ops/lt_ops.h>
#include <ATen/ops/less_ops.h>
#include <ATen/ops/less_ops.h>
#include <ATen/ops/less_ops.h>
#include <ATen/ops/less_ops.h>
#include <ATen/ops/less_ops.h>
#include <ATen/ops/less_ops.h>
#include <ATen/ops/nonzero_static_ops.h>
#include <ATen/ops/nonzero_static_ops.h>
#include <ATen/ops/addcdiv_ops.h>
#include <ATen/ops/addcdiv_ops.h>
#include <ATen/ops/addcdiv_ops.h>
#include <ATen/ops/_cholesky_solve_helper_ops.h>
#include <ATen/ops/cholesky_inverse_ops.h>
#include <ATen/ops/cholesky_inverse_ops.h>
#include <ATen/ops/_lu_with_info_ops.h>
#include <ATen/ops/atan2_ops.h>
#include <ATen/ops/atan2_ops.h>
#include <ATen/ops/atan2_ops.h>
#include <ATen/ops/fmax_ops.h>
#include <ATen/ops/fmax_ops.h>
#include <ATen/ops/_foreach_exp_ops.h>
#include <ATen/ops/_foreach_exp_ops.h>
#include <ATen/ops/_foreach_neg_ops.h>
#include <ATen/ops/_foreach_neg_ops.h>
#include <ATen/ops/_foreach_norm_ops.h>
#include <ATen/ops/_foreach_reciprocal_ops.h>
#include <ATen/ops/_foreach_reciprocal_ops.h>
#include <ATen/ops/_foreach_rsqrt_ops.h>
#include <ATen/ops/_foreach_rsqrt_ops.h>
#include <ATen/ops/_foreach_sigmoid_ops.h>
#include <ATen/ops/_foreach_sigmoid_ops.h>
#include <ATen/ops/_foreach_sin_ops.h>
#include <ATen/ops/_foreach_sin_ops.h>
#include <ATen/ops/_foreach_sqrt_ops.h>
#include <ATen/ops/_foreach_sqrt_ops.h>
#include <ATen/ops/soft_margin_loss_backward_ops.h>
#include <ATen/ops/soft_margin_loss_backward_ops.h>
#include <ATen/ops/glu_jvp_ops.h>
#include <ATen/ops/rrelu_with_noise_ops.h>
#include <ATen/ops/rrelu_with_noise_ops.h>
#include <ATen/ops/rrelu_with_noise_ops.h>
#include <ATen/ops/reflection_pad1d_backward_ops.h>
#include <ATen/ops/reflection_pad1d_backward_ops.h>
#include <ATen/ops/replication_pad1d_backward_ops.h>
#include <ATen/ops/replication_pad1d_backward_ops.h>
#include <ATen/ops/replication_pad3d_backward_ops.h>
#include <ATen/ops/replication_pad3d_backward_ops.h>
#include <ATen/ops/_upsample_nearest_exact2d_ops.h>
#include <ATen/ops/_upsample_bicubic2d_aa_backward_ops.h>
#include <ATen/ops/_upsample_bicubic2d_aa_backward_ops.h>
#include <ATen/ops/_upsample_nearest_exact2d_ops.h>
#include <ATen/ops/_upsample_nearest_exact2d_ops.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward_ops.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward_ops.h>
#include <ATen/ops/special_ndtr_ops.h>
#include <ATen/ops/special_ndtr_ops.h>
#include <ATen/ops/special_zeta_ops.h>
#include <ATen/ops/special_zeta_ops.h>
#include <ATen/ops/special_zeta_ops.h>
#include <ATen/ops/special_zeta_ops.h>
#include <ATen/ops/special_zeta_ops.h>
#include <ATen/ops/special_zeta_ops.h>
#include <ATen/ops/special_round_ops.h>
#include <ATen/ops/special_round_ops.h>
#include <ATen/ops/fft_ifft_ops.h>
#include <ATen/ops/fft_ifft_ops.h>
#include <ATen/ops/fft_ihfft_ops.h>
#include <ATen/ops/fft_ihfft_ops.h>
#include <ATen/ops/fft_ifftshift_ops.h>
#include <ATen/ops/slogdet_ops.h>
#include <ATen/ops/slogdet_ops.h>
#include <ATen/ops/linalg_eig_ops.h>
#include <ATen/ops/linalg_eig_ops.h>
#include <ATen/ops/_linalg_eigvals_ops.h>
#include <ATen/ops/linalg_eigvalsh_ops.h>
#include <ATen/ops/linalg_eigvalsh_ops.h>
#include <ATen/ops/linalg_svd_ops.h>
#include <ATen/ops/linalg_svd_ops.h>
#include <ATen/ops/_test_optional_floatlist_ops.h>
#include <ATen/ops/unflatten_dense_tensors_ops.h>
#include <ATen/ops/_sparse_broadcast_to_copy_ops.h>
#include <ATen/ops/_indices_copy_ops.h>
#include <ATen/ops/_values_copy_ops.h>
#include <ATen/ops/values_copy_ops.h>
#include <ATen/ops/view_copy_ops.h>
#include <ATen/ops/view_copy_ops.h>
#include <ATen/ops/_scaled_dot_product_fused_attention_overrideable_backward_ops.h>
#include <ATen/ops/_scaled_dot_product_cudnn_attention_backward_ops.h>
#include <ATen/ops/special_laguerre_polynomial_l_ops.h>
#include <ATen/ops/special_laguerre_polynomial_l_ops.h>
#include <ATen/ops/special_laguerre_polynomial_l_ops.h>
#include <ATen/ops/special_laguerre_polynomial_l_ops.h>
#include <ATen/ops/special_laguerre_polynomial_l_ops.h>
#include <ATen/ops/special_laguerre_polynomial_l_ops.h>
#include <ATen/ops/special_scaled_modified_bessel_k0_ops.h>
#include <ATen/ops/special_scaled_modified_bessel_k0_ops.h>
#include <ATen/ops/_fused_sgd_ops.h>
#include <ATen/ops/_fused_sgd_ops.h>
#include <ATen/ops/_propagate_xla_data_ops.h>
#include <ATen/ops/_new_zeros_with_same_feature_meta_ops.h>
#include <ATen/ops/_cudnn_init_dropout_state_ops.h>
#include <ATen/ops/native_dropout_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bernoulli_ops.h>
#include <ATen/ops/bincount_ops.h>
#include <ATen/ops/count_nonzero_ops.h>
#include <ATen/ops/count_nonzero_ops.h>
#include <ATen/ops/cudnn_convolution_add_relu_ops.h>
#include <ATen/ops/_ctc_loss_backward_ops.h>
#include <ATen/ops/embedding_renorm_ops.h>
#include <ATen/ops/embedding_renorm_ops.h>
#include <ATen/ops/empty_ops.h>
#include <ATen/ops/new_ones_ops.h>
#include <ATen/ops/_empty_per_channel_affine_quantized_ops.h>
#include <ATen/ops/mkldnn_max_pool3d_ops.h>
#include <ATen/ops/quantized_max_pool3d_ops.h>
#include <ATen/ops/miopen_convolution_ops.h>
#include <ATen/ops/batch_norm_update_stats_ops.h>
#include <ATen/ops/ones_like_ops.h>
#include <ATen/ops/_pdist_forward_ops.h>
#include <ATen/ops/randn_like_ops.h>
#include <ATen/ops/randn_like_ops.h>
#include <ATen/ops/repeat_interleave_ops.h>
#include <ATen/ops/sum_ops.h>
#include <ATen/ops/rot90_ops.h>
#include <ATen/ops/_nested_tensor_strides_ops.h>
#include <ATen/ops/_nested_tensor_storage_offsets_ops.h>
#include <ATen/ops/_standard_gamma_grad_ops.h>
#include <ATen/ops/_batch_norm_with_update_ops.h>
#include <ATen/ops/_sparse_csr_sum_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/norm_ops.h>
#include <ATen/ops/_sparse_mask_projection_ops.h>
#include <ATen/ops/_to_dense_ops.h>
#include <ATen/ops/_coalesced_ops.h>
#include <ATen/ops/_coalesced_ops.h>
#include <ATen/ops/quantize_per_tensor_dynamic_ops.h>
#include <ATen/ops/quantize_per_tensor_ops.h>
#include <ATen/ops/quantize_per_tensor_ops.h>
#include <ATen/ops/quantize_per_tensor_ops.h>
#include <ATen/ops/fake_quantize_per_tensor_affine_cachemask_ops.h>
#include <ATen/ops/_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_ops.h>
#include <ATen/ops/_fake_quantize_learnable_per_tensor_affine_ops.h>
#include <ATen/ops/_thnn_fused_gru_cell_backward_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/set_ops.h>
#include <ATen/ops/put_ops.h>
#include <ATen/ops/uniform_ops.h>
#include <ATen/ops/uniform_ops.h>
#include <ATen/ops/tril_indices_ops.h>
#include <ATen/ops/_cholesky_solve_helper_ops.h>
#include <ATen/ops/_foreach_exp_ops.h>
#include <ATen/ops/_foreach_neg_ops.h>
#include <ATen/ops/_foreach_norm_ops.h>
#include <ATen/ops/_foreach_reciprocal_ops.h>
#include <ATen/ops/_foreach_rsqrt_ops.h>
#include <ATen/ops/_foreach_sigmoid_ops.h>
#include <ATen/ops/_foreach_sin_ops.h>
#include <ATen/ops/_foreach_sqrt_ops.h>
#include <ATen/ops/glu_jvp_ops.h>
#include <ATen/ops/rrelu_with_noise_ops.h>
#include <ATen/ops/_test_optional_floatlist_ops.h>
#include <ATen/ops/_sparse_broadcast_to_copy_ops.h>
#include <ATen/ops/_indices_copy_ops.h>
#include <ATen/ops/_values_copy_ops.h>
#include <ATen/ops/values_copy_ops.h>
#include <ATen/ops/view_copy_ops.h>
#include <ATen/ops/view_copy_ops.h>
#include <ATen/ops/_fused_sgd_ops.h>
#include <ATen/ops/_fused_sgd_ops.h>
#include <ATen/ops/_fused_sgd_ops.h>
#include <ATen/ops/_fused_sgd_ops.h>
#endif

using namespace at;

namespace torch {

namespace TraceType {

namespace {
at::Tensor _new_zeros_with_same_feature_meta(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, int64_t self_num_batch_dims) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_new_zeros_with_same_feature_meta");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "self_num_batch_dims", self_num_batch_dims);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_new_zeros_with_same_feature_meta::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, self_num_batch_dims);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _assert_scalar(c10::DispatchKeySet ks, const at::Scalar & self, c10::string_view assert_msg) {
  at::_ops::_assert_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, assert_msg);
}
void _print(c10::DispatchKeySet ks, c10::string_view s) {
  at::_ops::_print::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), s);
}
at::Tensor _functional_sym_constrain_range_for_size(c10::DispatchKeySet ks, const at::Scalar & size, ::std::optional<int64_t> min, ::std::optional<int64_t> max, const at::Tensor & dep_token) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_functional_sym_constrain_range_for_size");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    jit::tracer::addInputs(node, "dep_token", dep_token);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_functional_sym_constrain_range_for_size::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, min, max, dep_token);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _make_dep_token(c10::DispatchKeySet ks, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_make_dep_token");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_make_dep_token::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _cudnn_init_dropout_state(c10::DispatchKeySet ks, double dropout, bool train, int64_t dropout_seed, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cudnn_init_dropout_state");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "dropout_seed", dropout_seed);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cudnn_init_dropout_state::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), dropout, train, dropout_seed, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> native_dropout(c10::DispatchKeySet ks, const at::Tensor & input, double p, ::std::optional<bool> train) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_dropout");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::native_dropout::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, p, train);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor angle(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::angle");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::angle::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & angle_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::angle");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("angle_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::angle_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor asinh(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::asinh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::asinh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & asinh_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::asinh");
    } else {
      op_name = c10::Symbol::fromQualString("aten::asinh_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("asinh_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::asinh_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & asinh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::asinh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("asinh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::asinh_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor baddbmm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::baddbmm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::baddbmm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, batch1, batch2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & baddbmm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::baddbmm");
    } else {
      op_name = c10::Symbol::fromQualString("aten::baddbmm_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("baddbmm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::baddbmm_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, batch1, batch2, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & baddbmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::baddbmm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("baddbmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::baddbmm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, batch1, batch2, beta, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor baddbmm_dtype(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, at::ScalarType out_dtype, const at::Scalar & beta, const at::Scalar & alpha) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::baddbmm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "out_dtype", out_dtype);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::baddbmm_dtype::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, batch1, batch2, out_dtype, beta, alpha);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & baddbmm_out_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, at::ScalarType out_dtype, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::baddbmm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "batch1", batch1);
    jit::tracer::addInputs(node, "batch2", batch2);
    jit::tracer::addInputs(node, "out_dtype", out_dtype);
    jit::tracer::addInputs(node, "beta", beta);
    jit::tracer::addInputs(node, "alpha", alpha);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("baddbmm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::baddbmm_dtype_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, batch1, batch2, out_dtype, beta, alpha, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor batch_norm(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm");
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
  auto result =at::_ops::batch_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor bernoulli(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bernoulli");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bernoulli::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & bernoulli_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bernoulli");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bernoulli_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bernoulli_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & bernoulli__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & p, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::bernoulli");
    } else {
      op_name = c10::Symbol::fromQualString("aten::bernoulli_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bernoulli_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bernoulli__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & bernoulli__float(c10::DispatchKeySet ks, at::Tensor & self, double p, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::bernoulli");
    } else {
      op_name = c10::Symbol::fromQualString("aten::bernoulli_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bernoulli_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bernoulli__float::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor bernoulli_p(c10::DispatchKeySet ks, const at::Tensor & self, double p, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bernoulli");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bernoulli_p::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor bilinear(c10::DispatchKeySet ks, const at::Tensor & input1, const at::Tensor & input2, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bilinear");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input1", input1);
    jit::tracer::addInputs(node, "input2", input2);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bilinear::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input1, input2, weight, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor bincount(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Tensor> & weights, c10::SymInt minlength) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bincount");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weights", weights);
    jit::tracer::addInputs(node, "minlength", minlength);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bincount::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weights, minlength);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> unsafe_chunk(c10::DispatchKeySet ks, const at::Tensor & self, int64_t chunks, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unsafe_chunk");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "chunks", chunks);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unsafe_chunk::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, chunks, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor clamp_max(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clamp_max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::clamp_max::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor clamp_max_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clamp_max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::clamp_max_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & clamp_max_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::clamp_max");
    } else {
      op_name = c10::Symbol::fromQualString("aten::clamp_max_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_max_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clamp_max_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & clamp_max__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::clamp_max");
    } else {
      op_name = c10::Symbol::fromQualString("aten::clamp_max_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_max_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clamp_max__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & clamp_max_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & max, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clamp_max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_max_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clamp_max_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, max, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & clamp_max_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & max, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clamp_max");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "max", max);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clamp_max_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clamp_max_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, max, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor clip(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clip");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::clip::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor clip_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clip");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::clip_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & clip_(c10::DispatchKeySet ks, at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::clip");
    } else {
      op_name = c10::Symbol::fromQualString("aten::clip_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clip_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clip_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & clip__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::clip");
    } else {
      op_name = c10::Symbol::fromQualString("aten::clip_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "min", min);
    jit::tracer::addInputs(node, "max", max);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("clip_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clip__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & clip_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & min, const ::std::optional<at::Scalar> & max, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clip");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("clip_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clip_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & clip_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Tensor> & min, const ::std::optional<at::Tensor> & max, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::clip");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("clip_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::clip_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, min, max, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor complex(c10::DispatchKeySet ks, const at::Tensor & real, const at::Tensor & imag) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::complex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "real", real);
    jit::tracer::addInputs(node, "imag", imag);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::complex::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), real, imag);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & complex_out_out(c10::DispatchKeySet ks, const at::Tensor & real, const at::Tensor & imag, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::complex");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "real", real);
    jit::tracer::addInputs(node, "imag", imag);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("complex_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::complex_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), real, imag, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor polar(c10::DispatchKeySet ks, const at::Tensor & abs, const at::Tensor & angle) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::polar");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "abs", abs);
    jit::tracer::addInputs(node, "angle", angle);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::polar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), abs, angle);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & polar_out_out(c10::DispatchKeySet ks, const at::Tensor & abs, const at::Tensor & angle, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::polar");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "abs", abs);
    jit::tracer::addInputs(node, "angle", angle);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("polar_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::polar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), abs, angle, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor conv_transpose2d_input(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef output_padding, c10::SymInt groups, c10::SymIntArrayRef dilation) {
  auto result =at::_ops::conv_transpose2d_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, stride, padding, output_padding, groups, dilation);
  return result;
}
at::Tensor count_nonzero_dim_IntList(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::count_nonzero");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::count_nonzero_dim_IntList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor count_nonzero(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::count_nonzero");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::count_nonzero::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor cudnn_convolution_add_relu(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const at::Tensor & z, const ::std::optional<at::Scalar> & alpha, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_convolution_add_relu");
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
  auto result =at::_ops::cudnn_convolution_add_relu::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, z, alpha, bias, stride, padding, dilation, groups);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> cummax(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cummax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::cummax::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> cummax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cummax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cummax_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cummax_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor,at::Tensor> cummax_dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cummax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [values, indices] =at::_ops::cummax_dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor &,at::Tensor &> cummax_out_dimname_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, at::Tensor & values, at::Tensor & indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cummax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "values", values);
      jit::tracer::addInputs(node, "indices", indices);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cummax_out", values);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cummax_dimname_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, values, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, values);
    jit::tracer::addOutput(node, indices);
  }
  return std::forward_as_tuple(values, indices);
}
void _cummax_helper(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & values, at::Tensor & indices, int64_t dim) {
  at::_ops::_cummax_helper::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, values, indices, dim);
}
at::Tensor _ctc_loss_backward(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, const at::Tensor & neg_log_likelihood, const at::Tensor & log_alpha, int64_t blank, bool zero_infinity) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_ctc_loss_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "neg_log_likelihood", neg_log_likelihood);
    jit::tracer::addInputs(node, "log_alpha", log_alpha);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_ctc_loss_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _ctc_loss_backward_Tensor(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & log_probs, const at::Tensor & targets, const at::Tensor & input_lengths, const at::Tensor & target_lengths, const at::Tensor & neg_log_likelihood, const at::Tensor & log_alpha, int64_t blank, bool zero_infinity) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_ctc_loss_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "neg_log_likelihood", neg_log_likelihood);
    jit::tracer::addInputs(node, "log_alpha", log_alpha);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_ctc_loss_backward_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor diff(c10::DispatchKeySet ks, const at::Tensor & self, int64_t n, int64_t dim, const ::std::optional<at::Tensor> & prepend, const ::std::optional<at::Tensor> & append) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::diff");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "prepend", prepend);
    jit::tracer::addInputs(node, "append", append);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::diff::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, prepend, append);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & diff_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t n, int64_t dim, const ::std::optional<at::Tensor> & prepend, const ::std::optional<at::Tensor> & append, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::diff");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "prepend", prepend);
    jit::tracer::addInputs(node, "append", append);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("diff_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::diff_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, prepend, append, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::vector<at::Tensor> gradient_scalarint(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & spacing, ::std::optional<int64_t> dim, int64_t edge_order) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gradient");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "spacing", spacing);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "edge_order", edge_order);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gradient_scalarint::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, spacing, dim, edge_order);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> gradient_scalararray(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & spacing, at::IntArrayRef dim, int64_t edge_order) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gradient");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "spacing", spacing);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "edge_order", edge_order);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gradient_scalararray::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, spacing, dim, edge_order);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> gradient_array(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t edge_order) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gradient");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "edge_order", edge_order);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gradient_array::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, edge_order);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> gradient_scalarrayint(c10::DispatchKeySet ks, const at::Tensor & self, at::ArrayRef<at::Scalar> spacing, ::std::optional<int64_t> dim, int64_t edge_order) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gradient");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "spacing", spacing);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "edge_order", edge_order);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gradient_scalarrayint::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, spacing, dim, edge_order);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> gradient_scalarrayarray(c10::DispatchKeySet ks, const at::Tensor & self, at::ArrayRef<at::Scalar> spacing, at::IntArrayRef dim, int64_t edge_order) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gradient");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "spacing", spacing);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "edge_order", edge_order);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gradient_scalarrayarray::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, spacing, dim, edge_order);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> gradient_tensorarrayint(c10::DispatchKeySet ks, const at::Tensor & self, at::TensorList spacing, ::std::optional<int64_t> dim, int64_t edge_order) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gradient");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "spacing", spacing);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "edge_order", edge_order);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gradient_tensorarrayint::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, spacing, dim, edge_order);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> gradient_tensorarray(c10::DispatchKeySet ks, const at::Tensor & self, at::TensorList spacing, at::IntArrayRef dim, int64_t edge_order) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gradient");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "spacing", spacing);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "edge_order", edge_order);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gradient_tensorarray::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, spacing, dim, edge_order);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor einsum(c10::DispatchKeySet ks, c10::string_view equation, at::TensorList tensors, at::OptionalIntArrayRef path) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::einsum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "equation", equation);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "path", path);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::einsum::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), equation, tensors, path);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & embedding_renorm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::embedding_renorm");
    } else {
      op_name = c10::Symbol::fromQualString("aten::embedding_renorm_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "max_norm", max_norm);
    jit::tracer::addInputs(node, "norm_type", norm_type);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("embedding_renorm_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::embedding_renorm_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, max_norm, norm_type);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor empty_names(c10::DispatchKeySet ks, at::IntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::empty");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::empty_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, names, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor empty_memory_format(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::empty");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::empty_memory_format::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor new_ones(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::new_ones");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::new_ones::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _empty_per_channel_affine_quantized(c10::DispatchKeySet ks, c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_empty_per_channel_affine_quantized");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "scales", scales);
    jit::tracer::addInputs(node, "zero_points", zero_points);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    jit::tracer::addInputs(node, "memory_format", memory_format);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_empty_per_channel_affine_quantized::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, scales, zero_points, axis, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & empty_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::empty");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("empty_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::empty_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor exp2(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::exp2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::exp2::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & exp2_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::exp2");
    } else {
      op_name = c10::Symbol::fromQualString("aten::exp2_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("exp2_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::exp2_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & exp2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::exp2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("exp2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::exp2_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor eye(c10::DispatchKeySet ks, c10::SymInt n, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::eye");
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
  auto result =at::_ops::eye::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor eye_m(c10::DispatchKeySet ks, c10::SymInt n, c10::SymInt m, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::eye");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "m", m);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::eye_m::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, m, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & eye_out_out(c10::DispatchKeySet ks, c10::SymInt n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::eye");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("eye_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::eye_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & eye_out_m_out(c10::DispatchKeySet ks, c10::SymInt n, c10::SymInt m, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::eye");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "m", m);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("eye_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::eye_m_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), n, m, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & gcd_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gcd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gcd_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::gcd_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor gcd(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gcd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gcd::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & gcd_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::gcd");
    } else {
      op_name = c10::Symbol::fromQualString("aten::gcd_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gcd_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::gcd_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
void _cufft_clear_plan_cache(c10::DispatchKeySet ks, at::DeviceIndex device_index) {
  at::_ops::_cufft_clear_plan_cache::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), device_index);
}
at::Tensor & isin_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "elements", elements);
    jit::tracer::addInputs(node, "test_elements", test_elements);
    jit::tracer::addInputs(node, "assume_unique", assume_unique);
    jit::tracer::addInputs(node, "invert", invert);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("isin_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::isin_Tensor_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), elements, test_elements, assume_unique, invert, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor isin_Tensor_Tensor(c10::DispatchKeySet ks, const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "elements", elements);
    jit::tracer::addInputs(node, "test_elements", test_elements);
    jit::tracer::addInputs(node, "assume_unique", assume_unique);
    jit::tracer::addInputs(node, "invert", invert);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::isin_Tensor_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), elements, test_elements, assume_unique, invert);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & isin_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "elements", elements);
    jit::tracer::addInputs(node, "test_element", test_element);
    jit::tracer::addInputs(node, "assume_unique", assume_unique);
    jit::tracer::addInputs(node, "invert", invert);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("isin_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::isin_Tensor_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), elements, test_element, assume_unique, invert, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor isin_Tensor_Scalar(c10::DispatchKeySet ks, const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "elements", elements);
    jit::tracer::addInputs(node, "test_element", test_element);
    jit::tracer::addInputs(node, "assume_unique", assume_unique);
    jit::tracer::addInputs(node, "invert", invert);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::isin_Tensor_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), elements, test_element, assume_unique, invert);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & isin_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "element", element);
    jit::tracer::addInputs(node, "test_elements", test_elements);
    jit::tracer::addInputs(node, "assume_unique", assume_unique);
    jit::tracer::addInputs(node, "invert", invert);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("isin_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::isin_Scalar_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), element, test_elements, assume_unique, invert, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor isin_Scalar_Tensor(c10::DispatchKeySet ks, const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::isin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "element", element);
    jit::tracer::addInputs(node, "test_elements", test_elements);
    jit::tracer::addInputs(node, "assume_unique", assume_unique);
    jit::tracer::addInputs(node, "invert", invert);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::isin_Scalar_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), element, test_elements, assume_unique, invert);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
bool is_conj(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::is_conj::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
bool is_nonzero(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::is_nonzero::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
int64_t storage_offset(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto result =at::_ops::storage_offset::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  return result;
}
at::Tensor layer_norm(c10::DispatchKeySet ks, const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, double eps, bool cudnn_enable) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::layer_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "cudnn_enable", cudnn_enable);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::layer_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, normalized_shape, weight, bias, eps, cudnn_enable);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor rms_norm(c10::DispatchKeySet ks, const at::Tensor & input, c10::SymIntArrayRef normalized_shape, const ::std::optional<at::Tensor> & weight, ::std::optional<double> eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rms_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "normalized_shape", normalized_shape);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "eps", eps);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::rms_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, normalized_shape, weight, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fbgemm_linear_fp16_weight(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fbgemm_linear_fp16_weight");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "packed_weight", packed_weight);
    jit::tracer::addInputs(node, "bias", bias);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fbgemm_linear_fp16_weight::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, packed_weight, bias);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fbgemm_linear_fp16_weight_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & packed_weight, const at::Tensor & bias, at::Tensor & output) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::fbgemm_linear_fp16_weight");
    } else {
      op_name = c10::Symbol::fromQualString("aten::fbgemm_linear_fp16_weight");
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
  auto result =at::_ops::fbgemm_linear_fp16_weight_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, packed_weight, bias, output);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fbgemm_pack_quantized_matrix(c10::DispatchKeySet ks, const at::Tensor & input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fbgemm_pack_quantized_matrix");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fbgemm_pack_quantized_matrix::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fbgemm_pack_quantized_matrix_KN(c10::DispatchKeySet ks, const at::Tensor & input, int64_t K, int64_t N) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fbgemm_pack_quantized_matrix");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "K", K);
    jit::tracer::addInputs(node, "N", N);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fbgemm_pack_quantized_matrix_KN::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, K, N);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor log(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::log::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & log_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::log");
    } else {
      op_name = c10::Symbol::fromQualString("aten::log_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & log_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("log_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & logaddexp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logaddexp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("logaddexp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::logaddexp_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor logaddexp(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::logaddexp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::logaddexp::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor log_softmax_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_softmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::log_softmax_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & log_softmax_out_int_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_softmax");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("log_softmax_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::log_softmax_int_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor log_softmax_Dimname(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::log_softmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::log_softmax_Dimname::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor mkldnn_max_pool3d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_max_pool3d");
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
  auto result =at::_ops::mkldnn_max_pool3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor quantized_max_pool3d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantized_max_pool3d");
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
  auto result =at::_ops::quantized_max_pool3d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor miopen_convolution(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_convolution");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
    jit::tracer::addInputs(node, "stride", stride);
    jit::tracer::addInputs(node, "dilation", dilation);
    jit::tracer::addInputs(node, "groups", groups);
    jit::tracer::addInputs(node, "benchmark", benchmark);
    jit::tracer::addInputs(node, "deterministic", deterministic);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::miopen_convolution::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_update_stats(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_update_stats");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::batch_norm_update_stats::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, running_mean, running_var, momentum);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor ones_like(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ones_like");
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
  auto result =at::_ops::ones_like::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _pdist_forward(c10::DispatchKeySet ks, const at::Tensor & self, double p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_pdist_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_pdist_forward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint(c10::DispatchKeySet ks, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), high, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint_generator(c10::DispatchKeySet ks, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint_generator::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), high, size, generator, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint_low(c10::DispatchKeySet ks, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint_low::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), low, high, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randint_low_generator(c10::DispatchKeySet ks, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "generator", generator);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::randint_low_generator::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), low, high, size, generator, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & randint_out_out(c10::DispatchKeySet ks, c10::SymInt high, c10::SymIntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "high", high);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), high, size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randint_out_generator_out(c10::DispatchKeySet ks, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "high", high);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_generator_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), high, size, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randint_out_low_out(c10::DispatchKeySet ks, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_low_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), low, high, size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randint_out_low_generator_out(c10::DispatchKeySet ks, c10::SymInt low, c10::SymInt high, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randint");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "low", low);
    jit::tracer::addInputs(node, "high", high);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("randint_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randint_low_generator_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), low, high, size, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor randn_like(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn_like");
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
  auto result =at::_ops::randn_like::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor randn_like_generator(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory, ::std::optional<at::MemoryFormat> memory_format) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn_like");
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
  auto result =at::_ops::randn_like_generator::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator, dtype, layout, device, pin_memory, memory_format);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor repeat_interleave_Tensor(c10::DispatchKeySet ks, const at::Tensor & repeats, ::std::optional<c10::SymInt> output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::repeat_interleave");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "repeats", repeats);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::repeat_interleave_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), repeats, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor repeat_interleave_self_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::repeat_interleave");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "repeats", repeats);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::repeat_interleave_self_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, repeats, dim, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor repeat_interleave_self_int(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt repeats, ::std::optional<int64_t> dim, ::std::optional<c10::SymInt> output_size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::repeat_interleave");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "repeats", repeats);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "output_size", output_size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::repeat_interleave_self_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, repeats, dim, output_size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _prelu_kernel(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_prelu_kernel");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_prelu_kernel::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _nested_select_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, c10::SymInt index) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_select_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_select_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, dim, index);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> vsplit_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t sections) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::vsplit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "sections", sections);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::vsplit_int::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, sections);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> vsplit_array(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef indices) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::vsplit");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::vsplit_array::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor hstack(c10::DispatchKeySet ks, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hstack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::hstack::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & hstack_out_out(c10::DispatchKeySet ks, at::TensorList tensors, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::hstack");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("hstack_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::hstack_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor sum(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sum::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sum_dim_IntList(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sum_dim_IntList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor sum_dim_DimnameList(c10::DispatchKeySet ks, const at::Tensor & self, at::DimnameList dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::sum_dim_DimnameList::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & sum_out_IntList_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sum");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sum_IntList_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & sum_out_DimnameList_out(c10::DispatchKeySet ks, const at::Tensor & self, at::DimnameList dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sum");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sum_DimnameList_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor rot90(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, at::IntArrayRef dims) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rot90");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dims", dims);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::rot90::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dims);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _nested_tensor_strides(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_tensor_strides");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_tensor_strides::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _nested_tensor_storage_offsets(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_tensor_storage_offsets");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_nested_tensor_storage_offsets::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor triplet_margin_loss(c10::DispatchKeySet ks, const at::Tensor & anchor, const at::Tensor & positive, const at::Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::triplet_margin_loss");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "anchor", anchor);
    jit::tracer::addInputs(node, "positive", positive);
    jit::tracer::addInputs(node, "negative", negative);
    jit::tracer::addInputs(node, "margin", margin);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "eps", eps);
    jit::tracer::addInputs(node, "swap", swap);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::triplet_margin_loss::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), anchor, positive, negative, margin, p, eps, swap, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor var(c10::DispatchKeySet ks, const at::Tensor & self, bool unbiased) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::var::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, unbiased);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor var_dim(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::var_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, unbiased, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor var_correction(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "correction", correction);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::var_correction::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, correction, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & var_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, bool unbiased, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("var_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::var_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, unbiased, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & var_out_correction_out(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim, const ::std::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "correction", correction);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("var_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::var_correction_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, correction, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor var_names_dim(c10::DispatchKeySet ks, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::var_names_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, unbiased, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & var_out_names_out(c10::DispatchKeySet ks, const at::Tensor & self, at::DimnameList dim, bool unbiased, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "unbiased", unbiased);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("var_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::var_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, unbiased, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor var_correction_names(c10::DispatchKeySet ks, const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "correction", correction);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::var_correction_names::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, correction, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & var_out_correction_names_out(c10::DispatchKeySet ks, const at::Tensor & self, at::DimnameList dim, const ::std::optional<at::Scalar> & correction, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::var");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "correction", correction);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("var_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::var_correction_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, correction, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _standard_gamma_grad(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & output) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_standard_gamma_grad");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output", output);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_standard_gamma_grad::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _batch_norm_with_update(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & running_mean, at::Tensor & running_var, double momentum, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::_batch_norm_with_update");
    } else {
      op_name = c10::Symbol::fromQualString("aten::_batch_norm_with_update");
    }
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
  auto [result0, result1, result2, result3] =at::_ops::_batch_norm_with_update::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, momentum, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _batch_norm_with_update_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, at::Tensor & running_mean, at::Tensor & running_var, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd, at::Tensor & reserve) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_batch_norm_with_update");
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
      jit::tracer::addInputs(node, "out", out);
      jit::tracer::addInputs(node, "save_mean", save_mean);
      jit::tracer::addInputs(node, "save_invstd", save_invstd);
      jit::tracer::addInputs(node, "reserve", reserve);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_batch_norm_with_update_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_batch_norm_with_update_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, momentum, eps, out, save_mean, save_invstd, reserve);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
    jit::tracer::addOutput(node, save_mean);
    jit::tracer::addOutput(node, save_invstd);
    jit::tracer::addOutput(node, reserve);
  }
  return std::forward_as_tuple(out, save_mean, save_invstd, reserve);
}
at::Tensor _sparse_csr_sum_dim_dtype(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_csr_sum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_csr_sum_dim_dtype::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor norm_ScalarOpt_dtype(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::ScalarType dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::norm_ScalarOpt_dtype::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor norm_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & p) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::norm_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor norm_ScalarOpt_dim_dtype(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
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
  auto result =at::_ops::norm_ScalarOpt_dim_dtype::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor norm_ScalarOpt_dim(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::norm_ScalarOpt_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & norm_out_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::norm_dtype_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & norm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::norm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor norm_names_ScalarOpt_dim_dtype(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
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
  auto result =at::_ops::norm_names_ScalarOpt_dim_dtype::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, keepdim, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor norm_names_ScalarOpt_dim(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::norm_names_ScalarOpt_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & norm_out_names_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::norm_names_dtype_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & norm_out_names_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::DimnameList dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::norm_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor nuclear_norm(c10::DispatchKeySet ks, const at::Tensor & self, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nuclear_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nuclear_norm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & nuclear_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nuclear_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "keepdim", keepdim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("nuclear_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nuclear_norm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor nuclear_norm_dim(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nuclear_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "keepdim", keepdim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nuclear_norm_dim::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & nuclear_norm_out_dim_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nuclear_norm");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("nuclear_norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nuclear_norm_dim_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _sparse_compressed_tensor_with_dims(c10::DispatchKeySet ks, int64_t nnz, int64_t dense_dim, at::IntArrayRef size, at::IntArrayRef blocksize, at::ScalarType index_dtype, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_compressed_tensor_with_dims");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "nnz", nnz);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "blocksize", blocksize);
    jit::tracer::addInputs(node, "index_dtype", index_dtype);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "layout", layout);
    jit::tracer::addInputs(node, "device", device);
    jit::tracer::addInputs(node, "pin_memory", pin_memory);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_compressed_tensor_with_dims::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), nnz, dense_dim, size, blocksize, index_dtype, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_csc_tensor_unsafe(c10::DispatchKeySet ks, const at::Tensor & ccol_indices, const at::Tensor & row_indices, const at::Tensor & values, at::IntArrayRef size, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_csc_tensor_unsafe");
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
  auto result =at::_ops::_sparse_csc_tensor_unsafe::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), ccol_indices, row_indices, values, size, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _validate_sparse_coo_tensor_args(c10::DispatchKeySet ks, const at::Tensor & indices, const at::Tensor & values, at::IntArrayRef size, ::std::optional<bool> is_coalesced, ::std::optional<bool> check_pinning) {
  at::_ops::_validate_sparse_coo_tensor_args::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), indices, values, size, is_coalesced, check_pinning);
}
at::Tensor _sparse_mask_projection(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, bool accumulate_matches) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_mask_projection");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "accumulate_matches", accumulate_matches);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_mask_projection::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, accumulate_matches);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _to_dense(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<bool> masked_grad) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_dense");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "masked_grad", masked_grad);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_to_dense::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, masked_grad);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _coalesced_(c10::DispatchKeySet ks, at::Tensor & self, bool coalesced) {
  at::_ops::_coalesced_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, coalesced);
  return self;
}
at::Tensor to_sparse_bsc(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef blocksize, ::std::optional<int64_t> dense_dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::to_sparse_bsc");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "blocksize", blocksize);
    jit::tracer::addInputs(node, "dense_dim", dense_dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::to_sparse_bsc::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, blocksize, dense_dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor quantize_per_tensor_dynamic(c10::DispatchKeySet ks, const at::Tensor & self, at::ScalarType dtype, bool reduce_range) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantize_per_tensor_dynamic");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "reduce_range", reduce_range);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::quantize_per_tensor_dynamic::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, reduce_range);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor quantize_per_tensor(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point, at::ScalarType dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantize_per_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::quantize_per_tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor quantize_per_tensor_tensor_qparams(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, at::ScalarType dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantize_per_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::quantize_per_tensor_tensor_qparams::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> quantize_per_tensor_tensors(c10::DispatchKeySet ks, at::TensorList tensors, const at::Tensor & scales, const at::Tensor & zero_points, at::ScalarType dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantize_per_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "tensors", tensors);
    jit::tracer::addInputs(node, "scales", scales);
    jit::tracer::addInputs(node, "zero_points", zero_points);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::quantize_per_tensor_tensors::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, scales, zero_points, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> fake_quantize_per_tensor_affine_cachemask(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine_cachemask");
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
  auto [output, mask] =at::_ops::fake_quantize_per_tensor_affine_cachemask::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, quant_min, quant_max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, mask);
  }
  return std::make_tuple(std::move(output), std::move(mask));
}
::std::tuple<at::Tensor,at::Tensor> _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, const at::Tensor & fake_quant_enabled, int64_t quant_min, int64_t quant_max) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "fake_quant_enabled", fake_quant_enabled);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [output, mask] =at::_ops::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, fake_quant_enabled, quant_min, quant_max);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, output);
    jit::tracer::addOutput(node, mask);
  }
  return std::make_tuple(std::move(output), std::move(mask));
}
at::Tensor _fake_quantize_learnable_per_tensor_affine(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fake_quantize_learnable_per_tensor_affine");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    jit::tracer::addInputs(node, "grad_factor", grad_factor);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_fake_quantize_learnable_per_tensor_affine::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, quant_min, quant_max, grad_factor);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> choose_qparams_optimized(c10::DispatchKeySet ks, const at::Tensor & input, int64_t numel, int64_t n_bins, double ratio, int64_t bit_width) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::choose_qparams_optimized");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "numel", numel);
    jit::tracer::addInputs(node, "n_bins", n_bins);
    jit::tracer::addInputs(node, "ratio", ratio);
    jit::tracer::addInputs(node, "bit_width", bit_width);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1] =at::_ops::choose_qparams_optimized::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, numel, n_bins, ratio, bit_width);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _thnn_fused_gru_cell_backward(c10::DispatchKeySet ks, const at::Tensor & grad_hy, const at::Tensor & workspace, bool has_bias) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_thnn_fused_gru_cell_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "workspace", workspace);
    jit::tracer::addInputs(node, "has_bias", has_bias);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, result1, result2, result3, result4] =at::_ops::_thnn_fused_gru_cell_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_hy, workspace, has_bias);
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
at::Tensor & set__source_Storage(c10::DispatchKeySet ks, at::Tensor & self, at::Storage source) {
  at::_ops::set__source_Storage::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source);
  return self;
}
at::Tensor & set__source_Storage_storage_offset(c10::DispatchKeySet ks, at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  at::_ops::set__source_Storage_storage_offset::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, storage_offset, size, stride);
  return self;
}
at::Tensor & set__source_Tensor_storage_offset(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::set");
    } else {
      op_name = c10::Symbol::fromQualString("aten::set_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "storage_offset", storage_offset);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "stride", stride);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("set_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::set__source_Tensor_storage_offset::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, storage_offset, size, stride);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & set__source_Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & source) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::set");
    } else {
      op_name = c10::Symbol::fromQualString("aten::set_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("set_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::set__source_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & set_(c10::DispatchKeySet ks, at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::set");
    } else {
      op_name = c10::Symbol::fromQualString("aten::set_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("set_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::set_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor masked_scatter_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & mask, c10::SymIntArrayRef sizes) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::masked_scatter_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "sizes", sizes);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::masked_scatter_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, mask, sizes);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & put_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::put");
    } else {
      op_name = c10::Symbol::fromQualString("aten::put_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("put_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::put_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, index, source, accumulate);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor put(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::put");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "accumulate", accumulate);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::put::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, index, source, accumulate);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor scatter_src(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::scatter_src::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & scatter__src(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::scatter");
    } else {
      op_name = c10::Symbol::fromQualString("aten::scatter_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter__src::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & scatter_out_src_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter_src_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor scatter_value(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::scatter_value::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & scatter__value(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::scatter");
    } else {
      op_name = c10::Symbol::fromQualString("aten::scatter_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter__value::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & scatter_out_value_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter_value_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor scatter_reduce(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "reduce", reduce);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::scatter_reduce::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src, reduce);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & scatter__reduce(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::scatter");
    } else {
      op_name = c10::Symbol::fromQualString("aten::scatter_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "reduce", reduce);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter__reduce::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src, reduce);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & scatter_out_reduce_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    jit::tracer::addInputs(node, "reduce", reduce);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter_reduce_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src, reduce, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor scatter_value_reduce(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "reduce", reduce);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::scatter_value_reduce::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value, reduce);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & scatter__value_reduce(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::scatter");
    } else {
      op_name = c10::Symbol::fromQualString("aten::scatter_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "reduce", reduce);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter__value_reduce::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value, reduce);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & scatter_out_value_reduce_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "reduce", reduce);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("scatter_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::scatter_value_reduce_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value, reduce, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor scatter_dimname_src(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Tensor & src) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "src", src);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::scatter_dimname_src::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, src);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor scatter_dimname_value(c10::DispatchKeySet ks, const at::Tensor & self, at::Dimname dim, const at::Tensor & index, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::scatter");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::scatter_dimname_value::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, index, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor __and___Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__and__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::__and___Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor __and___Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__and__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::__and___Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & __iand___Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__iand__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__iand___Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & __iand___Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::__iand__");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::__iand___Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & tril_(c10::DispatchKeySet ks, at::Tensor & self, c10::SymInt diagonal) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::tril");
    } else {
      op_name = c10::Symbol::fromQualString("aten::tril_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tril_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::tril_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, diagonal);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & uniform_(c10::DispatchKeySet ks, at::Tensor & self, double from, double to, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::uniform");
    } else {
      op_name = c10::Symbol::fromQualString("aten::uniform_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("uniform_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::uniform_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, from, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & tril_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt diagonal, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tril");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("tril_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::tril_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, diagonal, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor tril(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt diagonal) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tril");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "diagonal", diagonal);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::tril::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, diagonal);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor tril_indices(c10::DispatchKeySet ks, int64_t row, int64_t col, int64_t offset, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tril_indices");
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
  auto result =at::_ops::tril_indices::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), row, col, offset, dtype, layout, device, pin_memory);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & less_equal_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::less_equal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("less_equal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::less_equal_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor less_equal_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::less_equal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::less_equal_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & less_equal_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::less_equal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("less_equal_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::less_equal_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor less_equal_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::less_equal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::less_equal_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & less_equal__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::less_equal");
    } else {
      op_name = c10::Symbol::fromQualString("aten::less_equal_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("less_equal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::less_equal__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & less_equal__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::less_equal");
    } else {
      op_name = c10::Symbol::fromQualString("aten::less_equal_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("less_equal_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::less_equal__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & gt_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::gt_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor gt_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gt_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & gt_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::gt_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor gt_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::gt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::gt_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & gt__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::gt");
    } else {
      op_name = c10::Symbol::fromQualString("aten::gt_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::gt__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & gt__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::gt");
    } else {
      op_name = c10::Symbol::fromQualString("aten::gt_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("gt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::gt__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & lt_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::lt_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor lt_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::lt_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & lt_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::lt_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor lt_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::lt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::lt_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & lt__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::lt");
    } else {
      op_name = c10::Symbol::fromQualString("aten::lt_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::lt__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & lt__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::lt");
    } else {
      op_name = c10::Symbol::fromQualString("aten::lt_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("lt_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::lt__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & less_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::less");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("less_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::less_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor less_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::less");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::less_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & less_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::less");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("less_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::less_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor less_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::less");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::less_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & less__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::less");
    } else {
      op_name = c10::Symbol::fromQualString("aten::less_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("less_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::less__Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & less__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::less");
    } else {
      op_name = c10::Symbol::fromQualString("aten::less_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("less_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::less__Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & nonzero_static_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt size, int64_t fill_value, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nonzero_static");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("nonzero_static_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::nonzero_static_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, fill_value, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor nonzero_static(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymInt size, int64_t fill_value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::nonzero_static");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "fill_value", fill_value);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::nonzero_static::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, fill_value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & addcdiv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::addcdiv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcdiv_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::addcdiv_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, value, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor addcdiv(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::addcdiv");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::addcdiv::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & addcdiv_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::addcdiv");
    } else {
      op_name = c10::Symbol::fromQualString("aten::addcdiv_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "tensor1", tensor1);
    jit::tracer::addInputs(node, "tensor2", tensor2);
    jit::tracer::addInputs(node, "value", value);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("addcdiv_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::addcdiv_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, tensor1, tensor2, value);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor _cholesky_solve_helper(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & A, bool upper) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cholesky_solve_helper");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_cholesky_solve_helper::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, A, upper);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor cholesky_inverse(c10::DispatchKeySet ks, const at::Tensor & self, bool upper) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cholesky_inverse");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::cholesky_inverse::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, upper);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & cholesky_inverse_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool upper, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cholesky_inverse");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "upper", upper);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cholesky_inverse_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cholesky_inverse_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, upper, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _lu_with_info(c10::DispatchKeySet ks, const at::Tensor & self, bool pivot, bool check_errors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_lu_with_info");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "pivot", pivot);
    jit::tracer::addInputs(node, "check_errors", check_errors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [LU, pivots, info] =at::_ops::_lu_with_info::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, pivot, check_errors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, LU);
    jit::tracer::addOutput(node, pivots);
    jit::tracer::addOutput(node, info);
  }
  return std::make_tuple(std::move(LU), std::move(pivots), std::move(info));
}
at::Tensor & atan2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atan2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan2_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::atan2_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & atan2_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::atan2");
    } else {
      op_name = c10::Symbol::fromQualString("aten::atan2_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("atan2_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::atan2_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor atan2(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::atan2");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::atan2::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor fmax(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fmax::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fmax");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fmax_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fmax_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::vector<at::Tensor> _foreach_exp(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_exp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_exp::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_exp_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_exp_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_neg(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_neg");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_neg::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_neg_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_neg_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_norm_Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & ord, ::std::optional<at::ScalarType> dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "ord", ord);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_norm_Scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, ord, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> _foreach_reciprocal(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_reciprocal");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_reciprocal::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_reciprocal_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_reciprocal_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_rsqrt(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_rsqrt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_rsqrt::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_rsqrt_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_rsqrt_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_sigmoid(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_sigmoid");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_sigmoid::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_sigmoid_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_sigmoid_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_sin(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_sin");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_sin::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_sin_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_sin_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
::std::vector<at::Tensor> _foreach_sqrt(c10::DispatchKeySet ks, at::TensorList self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_foreach_sqrt");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_foreach_sqrt::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
void _foreach_sqrt_(c10::DispatchKeySet ks, at::TensorList self) {
  at::_ops::_foreach_sqrt_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
}
at::Tensor & soft_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::soft_margin_loss_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("soft_margin_loss_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::soft_margin_loss_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, reduction, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor soft_margin_loss_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::soft_margin_loss_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "target", target);
    jit::tracer::addInputs(node, "reduction", reduction);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::soft_margin_loss_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, target, reduction);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor glu_jvp(c10::DispatchKeySet ks, const at::Tensor & glu, const at::Tensor & x, const at::Tensor & dx, int64_t dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::glu_jvp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "glu", glu);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "dx", dx);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::glu_jvp::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), glu, x, dx, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & rrelu_with_noise_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rrelu_with_noise");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_with_noise_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rrelu_with_noise_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, noise, lower, upper, training, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor rrelu_with_noise(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::rrelu_with_noise");
    } else {
      op_name = c10::Symbol::fromQualString("aten::rrelu_with_noise");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::rrelu_with_noise::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, noise, lower, upper, training, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & rrelu_with_noise_(c10::DispatchKeySet ks, at::Tensor & self, at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;

    if (tracer_state->force_outplace) {
      op_name = c10::Symbol::fromQualString("aten::rrelu_with_noise");
    } else {
      op_name = c10::Symbol::fromQualString("aten::rrelu_with_noise_");
    }
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rrelu_with_noise_", self);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rrelu_with_noise_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, noise, lower, upper, training, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self);
  }
  return self;
}
at::Tensor & reflection_pad1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reflection_pad1d_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("reflection_pad1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::reflection_pad1d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, padding, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor reflection_pad1d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::reflection_pad1d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::reflection_pad1d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & replication_pad1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::replication_pad1d_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad1d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::replication_pad1d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, padding, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor replication_pad1d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::replication_pad1d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::replication_pad1d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & replication_pad3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::replication_pad3d_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("replication_pad3d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::replication_pad3d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, padding, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor replication_pad3d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::SymIntArrayRef padding) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::replication_pad3d_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_output", grad_output);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "padding", padding);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::replication_pad3d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, self, padding);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _upsample_nearest_exact2d_vec(c10::DispatchKeySet ks, const at::Tensor & input, at::OptionalSymIntArrayRef output_size, ::std::optional<at::ArrayRef<double>> scale_factors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scale_factors", scale_factors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_upsample_nearest_exact2d_vec::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, output_size, scale_factors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _upsample_bicubic2d_aa_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_bicubic2d_aa_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_upsample_bicubic2d_aa_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_upsample_bicubic2d_aa_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor _upsample_bicubic2d_aa_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, bool align_corners, ::std::optional<double> scales_h, ::std::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_bicubic2d_aa_backward");
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
  auto result =at::_ops::_upsample_bicubic2d_aa_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _upsample_nearest_exact2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_upsample_nearest_exact2d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_upsample_nearest_exact2d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, scales_h, scales_w, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor _upsample_nearest_exact2d(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef output_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact2d");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output_size", output_size);
    jit::tracer::addInputs(node, "scales_h", scales_h);
    jit::tracer::addInputs(node, "scales_w", scales_w);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_upsample_nearest_exact2d::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output_size, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & _upsample_nearest_exact2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w, at::Tensor & grad_input) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact2d_backward");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_upsample_nearest_exact2d_backward_out", grad_input);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_upsample_nearest_exact2d_backward_grad_input::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, scales_h, scales_w, grad_input);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_input);
  }
  return grad_input;
}
at::Tensor _upsample_nearest_exact2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size, ::std::optional<double> scales_h, ::std::optional<double> scales_w) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_upsample_nearest_exact2d_backward");
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
  auto result =at::_ops::_upsample_nearest_exact2d_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_output, output_size, input_size, scales_h, scales_w);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_ndtr(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_ndtr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_ndtr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_ndtr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_ndtr");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_ndtr_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_ndtr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_zeta(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_zeta");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_zeta::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_zeta_self_scalar(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_zeta");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_zeta_self_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_zeta_other_scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_zeta");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_zeta_other_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_zeta_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_zeta");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_zeta_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_zeta_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_zeta_out_self_scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_zeta");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_zeta_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_zeta_self_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_zeta_out_other_scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_zeta");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_zeta_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_zeta_other_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_round(c10::DispatchKeySet ks, const at::Tensor & self, int64_t decimals) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_round");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "decimals", decimals);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_round::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, decimals);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_round_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t decimals, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_round");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "decimals", decimals);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_round_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_round_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, decimals, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_ifft(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ifft");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_ifft::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_ifft_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ifft");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_ifft_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_ifft_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_ihfft(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ihfft");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "n", n);
    jit::tracer::addInputs(node, "dim", dim);
    jit::tracer::addInputs(node, "norm", norm);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_ihfft::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & fft_ihfft_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<c10::SymInt> n, int64_t dim, ::std::optional<c10::string_view> norm, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ihfft");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("fft_ihfft_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fft_ihfft_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, n, dim, norm, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor fft_ifftshift(c10::DispatchKeySet ks, const at::Tensor & self, at::OptionalIntArrayRef dim) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fft_ifftshift");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::fft_ifftshift::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> slogdet(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slogdet");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [sign, logabsdet] =at::_ops::slogdet::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, sign);
    jit::tracer::addOutput(node, logabsdet);
  }
  return std::make_tuple(std::move(sign), std::move(logabsdet));
}
::std::tuple<at::Tensor &,at::Tensor &> slogdet_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & sign, at::Tensor & logabsdet) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::slogdet");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "sign", sign);
      jit::tracer::addInputs(node, "logabsdet", logabsdet);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("slogdet_out", sign);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::slogdet_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, sign, logabsdet);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, sign);
    jit::tracer::addOutput(node, logabsdet);
  }
  return std::forward_as_tuple(sign, logabsdet);
}
::std::tuple<at::Tensor,at::Tensor> linalg_eig(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_eig");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [eigenvalues, eigenvectors] =at::_ops::linalg_eig::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, eigenvalues);
    jit::tracer::addOutput(node, eigenvectors);
  }
  return std::make_tuple(std::move(eigenvalues), std::move(eigenvectors));
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_eig_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & eigenvalues, at::Tensor & eigenvectors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_eig");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "eigenvalues", eigenvalues);
      jit::tracer::addInputs(node, "eigenvectors", eigenvectors);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_eig_out", eigenvalues);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_eig_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, eigenvalues, eigenvectors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, eigenvalues);
    jit::tracer::addOutput(node, eigenvectors);
  }
  return std::forward_as_tuple(eigenvalues, eigenvectors);
}
at::Tensor _linalg_eigvals(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_linalg_eigvals");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_linalg_eigvals::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor linalg_eigvalsh(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view UPLO) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_eigvalsh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "UPLO", UPLO);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::linalg_eigvalsh::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, UPLO);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & linalg_eigvalsh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view UPLO, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_eigvalsh");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "UPLO", UPLO);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_eigvalsh_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_eigvalsh_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, UPLO, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> linalg_svd(c10::DispatchKeySet ks, const at::Tensor & A, bool full_matrices, ::std::optional<c10::string_view> driver) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_svd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "full_matrices", full_matrices);
    jit::tracer::addInputs(node, "driver", driver);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [U, S, Vh] =at::_ops::linalg_svd::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, full_matrices, driver);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, U);
    jit::tracer::addOutput(node, S);
    jit::tracer::addOutput(node, Vh);
  }
  return std::make_tuple(std::move(U), std::move(S), std::move(Vh));
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> linalg_svd_out_U(c10::DispatchKeySet ks, const at::Tensor & A, bool full_matrices, ::std::optional<c10::string_view> driver, at::Tensor & U, at::Tensor & S, at::Tensor & Vh) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::linalg_svd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "full_matrices", full_matrices);
    jit::tracer::addInputs(node, "driver", driver);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "U", U);
      jit::tracer::addInputs(node, "S", S);
      jit::tracer::addInputs(node, "Vh", Vh);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("linalg_svd_out", U);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::linalg_svd_U::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), A, full_matrices, driver, U, S, Vh);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, U);
    jit::tracer::addOutput(node, S);
    jit::tracer::addOutput(node, Vh);
  }
  return std::forward_as_tuple(U, S, Vh);
}
at::Tensor _test_optional_floatlist(c10::DispatchKeySet ks, const at::Tensor & values, ::std::optional<at::ArrayRef<double>> addends) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_optional_floatlist");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "addends", addends);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_test_optional_floatlist::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), values, addends);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::vector<at::Tensor> unflatten_dense_tensors(c10::DispatchKeySet ks, const at::Tensor & flat, at::TensorList tensors) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::unflatten_dense_tensors");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "flat", flat);
    jit::tracer::addInputs(node, "tensors", tensors);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::unflatten_dense_tensors::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), flat, tensors);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _sparse_broadcast_to_copy(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_broadcast_to_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_sparse_broadcast_to_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _indices_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_indices_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_indices_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor _values_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_values_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::_values_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor values_copy(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::values_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::values_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor view_copy(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::view_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::view_copy::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor view_copy_dtype(c10::DispatchKeySet ks, const at::Tensor & self, at::ScalarType dtype) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::view_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::view_copy_dtype::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_fused_attention_overrideable_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & attn_bias, ::std::array<bool,4> grad_input_mask, const at::Tensor & out, const at::Tensor & logsumexp, const at::Tensor & cum_seq_q, const at::Tensor & cum_seq_k, c10::SymInt max_q, c10::SymInt max_k, double dropout_p, bool is_causal, const at::Tensor & philox_seed, const at::Tensor & philox_offset, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_dot_product_fused_attention_overrideable_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_out", grad_out);
    jit::tracer::addInputs(node, "query", query);
    jit::tracer::addInputs(node, "key", key);
    jit::tracer::addInputs(node, "value", value);
    jit::tracer::addInputs(node, "attn_bias", attn_bias);
    jit::tracer::addInputs(node, "grad_input_mask", grad_input_mask);
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
  auto [grad_query, grad_key, grad_value, grad_attn_bias] =at::_ops::_scaled_dot_product_fused_attention_overrideable_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, query, key, value, attn_bias, grad_input_mask, out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, philox_seed, philox_offset, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, grad_query);
    jit::tracer::addOutput(node, grad_key);
    jit::tracer::addOutput(node, grad_value);
    jit::tracer::addOutput(node, grad_attn_bias);
  }
  return std::make_tuple(std::move(grad_query), std::move(grad_key), std::move(grad_value), std::move(grad_attn_bias));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _scaled_dot_product_cudnn_attention_backward(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & out, const at::Tensor & logsumexp, const at::Tensor & philox_seed, const at::Tensor & philox_offset, const at::Tensor & attn_bias, const at::Tensor & cum_seq_q, const at::Tensor & cum_seq_k, c10::SymInt max_q, c10::SymInt max_k, double dropout_p, bool is_causal, ::std::optional<double> scale) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_scaled_dot_product_cudnn_attention_backward");
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
  auto [result0, result1, result2] =at::_ops::_scaled_dot_product_cudnn_attention_backward::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_out, query, key, value, out, logsumexp, philox_seed, philox_offset, attn_bias, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, is_causal, scale);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor special_laguerre_polynomial_l(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_laguerre_polynomial_l");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_laguerre_polynomial_l::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_laguerre_polynomial_l_x_scalar(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_laguerre_polynomial_l");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_laguerre_polynomial_l_x_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor special_laguerre_polynomial_l_n_scalar(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_laguerre_polynomial_l");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_laguerre_polynomial_l_n_scalar::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_laguerre_polynomial_l_out_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_laguerre_polynomial_l");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_laguerre_polynomial_l_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_laguerre_polynomial_l_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_laguerre_polynomial_l_out_x_scalar_out(c10::DispatchKeySet ks, const at::Scalar & x, const at::Tensor & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_laguerre_polynomial_l");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_laguerre_polynomial_l_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_laguerre_polynomial_l_x_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & special_laguerre_polynomial_l_out_n_scalar_out(c10::DispatchKeySet ks, const at::Tensor & x, const at::Scalar & n, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_laguerre_polynomial_l");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "n", n);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_laguerre_polynomial_l_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_laguerre_polynomial_l_n_scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, n, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor special_scaled_modified_bessel_k0(c10::DispatchKeySet ks, const at::Tensor & x) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_scaled_modified_bessel_k0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::special_scaled_modified_bessel_k0::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & special_scaled_modified_bessel_k0_out_out(c10::DispatchKeySet ks, const at::Tensor & x, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::special_scaled_modified_bessel_k0");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "x", x);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("special_scaled_modified_bessel_k0_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::special_scaled_modified_bessel_k0_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), x, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _fused_sgd_(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  at::_ops::_fused_sgd_::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step, grad_scale, found_inf);
}
void _fused_sgd__tensor_lr(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, const at::Tensor & lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  at::_ops::_fused_sgd__tensor_lr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step, grad_scale, found_inf);
}
void _propagate_xla_data(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & output) {
  at::_ops::_propagate_xla_data::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, output);
}
at::Tensor & _new_zeros_with_same_feature_meta_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, int64_t self_num_batch_dims, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_new_zeros_with_same_feature_meta");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "other", other);
    jit::tracer::addInputs(node, "self_num_batch_dims", self_num_batch_dims);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_new_zeros_with_same_feature_meta_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_new_zeros_with_same_feature_meta_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, other, self_num_batch_dims, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _cudnn_init_dropout_state_out_out(c10::DispatchKeySet ks, double dropout, bool train, int64_t dropout_seed, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cudnn_init_dropout_state");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "dropout", dropout);
    jit::tracer::addInputs(node, "train", train);
    jit::tracer::addInputs(node, "dropout_seed", dropout_seed);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cudnn_init_dropout_state_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_cudnn_init_dropout_state_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), dropout, train, dropout_seed, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> native_dropout_out_out(c10::DispatchKeySet ks, const at::Tensor & input, double p, ::std::optional<bool> train, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::native_dropout");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "train", train);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("native_dropout_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::native_dropout_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, p, train, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & bernoulli_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & p, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bernoulli");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bernoulli_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bernoulli_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor bernoulli_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & p, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bernoulli");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::bernoulli_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & bernoulli_out_float_out(c10::DispatchKeySet ks, const at::Tensor & self, double p, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bernoulli");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "generator", generator);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bernoulli_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bernoulli_float_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & bincount_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Tensor> & weights, c10::SymInt minlength, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::bincount");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weights", weights);
    jit::tracer::addInputs(node, "minlength", minlength);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("bincount_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::bincount_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weights, minlength, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & count_nonzero_out_dim_IntList_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::count_nonzero");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("count_nonzero_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::count_nonzero_dim_IntList_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & count_nonzero_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<int64_t> dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::count_nonzero");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("count_nonzero_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::count_nonzero_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & cudnn_convolution_add_relu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const at::Tensor & z, const ::std::optional<at::Scalar> & alpha, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, c10::SymInt groups, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::cudnn_convolution_add_relu");
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

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("cudnn_convolution_add_relu_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::cudnn_convolution_add_relu_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, z, alpha, bias, stride, padding, dilation, groups, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _ctc_loss_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, const at::Tensor & neg_log_likelihood, const at::Tensor & log_alpha, int64_t blank, bool zero_infinity, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_ctc_loss_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad", grad);
    jit::tracer::addInputs(node, "log_probs", log_probs);
    jit::tracer::addInputs(node, "targets", targets);
    jit::tracer::addInputs(node, "input_lengths", input_lengths);
    jit::tracer::addInputs(node, "target_lengths", target_lengths);
    jit::tracer::addInputs(node, "neg_log_likelihood", neg_log_likelihood);
    jit::tracer::addInputs(node, "log_alpha", log_alpha);
    jit::tracer::addInputs(node, "blank", blank);
    jit::tracer::addInputs(node, "zero_infinity", zero_infinity);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_ctc_loss_backward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_ctc_loss_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & embedding_renorm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::embedding_renorm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "max_norm", max_norm);
    jit::tracer::addInputs(node, "norm_type", norm_type);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("embedding_renorm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::embedding_renorm_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, max_norm, norm_type, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor embedding_renorm(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::embedding_renorm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "indices", indices);
    jit::tracer::addInputs(node, "max_norm", max_norm);
    jit::tracer::addInputs(node, "norm_type", norm_type);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::embedding_renorm::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, indices, max_norm, norm_type);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & empty_out_names_out(c10::DispatchKeySet ks, at::IntArrayRef size, ::std::optional<at::DimnameList> names, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::empty");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "names", names);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {
      jit::tracer::addInputs(node, "out", c10::optTypeMetaToScalarType(out.options().dtype_opt()));
      jit::tracer::addInputs(node, "out", out.options().layout());
      jit::tracer::addInputs(node, "out", out.options().device());
      jit::tracer::addInputs(node, "out", out.options().pinned_memory());
    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("empty_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::empty_names_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, names, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & new_ones_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::new_ones");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("new_ones_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::new_ones_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _empty_per_channel_affine_quantized_out_out(c10::DispatchKeySet ks, c10::SymIntArrayRef size, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_empty_per_channel_affine_quantized");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "size", size);
    jit::tracer::addInputs(node, "scales", scales);
    jit::tracer::addInputs(node, "zero_points", zero_points);
    jit::tracer::addInputs(node, "axis", axis);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_empty_per_channel_affine_quantized_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_empty_per_channel_affine_quantized_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), size, scales, zero_points, axis, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & mkldnn_max_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::mkldnn_max_pool3d");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("mkldnn_max_pool3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::mkldnn_max_pool3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & quantized_max_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantized_max_pool3d");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("quantized_max_pool3d_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::quantized_max_pool3d_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, kernel_size, stride, padding, dilation, ceil_mode, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & miopen_convolution_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const ::std::optional<at::Tensor> & bias, c10::SymIntArrayRef padding, c10::SymIntArrayRef stride, c10::SymIntArrayRef dilation, c10::SymInt groups, bool benchmark, bool deterministic, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::miopen_convolution");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "weight", weight);
    jit::tracer::addInputs(node, "bias", bias);
    jit::tracer::addInputs(node, "padding", padding);
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
    jit::tracer::ensureUniqueIfOutOfPlaced("miopen_convolution_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::miopen_convolution_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> batch_norm_update_stats_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, double momentum, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::batch_norm_update_stats");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "input", input);
    jit::tracer::addInputs(node, "running_mean", running_mean);
    jit::tracer::addInputs(node, "running_var", running_var);
    jit::tracer::addInputs(node, "momentum", momentum);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("batch_norm_update_stats_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::batch_norm_update_stats_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, running_mean, running_var, momentum, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & ones_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::ones_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("ones_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::ones_like_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _pdist_forward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double p, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_pdist_forward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_pdist_forward_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_pdist_forward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randn_like_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn_like");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "memory_format", memory_format);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randn_like_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & randn_like_out_generator_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::Generator> generator, ::std::optional<at::MemoryFormat> memory_format, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::randn_like");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("randn_like_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::randn_like_generator_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, generator, memory_format, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & repeat_interleave_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & repeats, ::std::optional<c10::SymInt> output_size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::repeat_interleave");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "repeats", repeats);
    jit::tracer::addInputs(node, "output_size", output_size);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("repeat_interleave_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::repeat_interleave_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), repeats, output_size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & sum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::sum");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("sum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::sum_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & rot90_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, at::IntArrayRef dims, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rot90");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "k", k);
    jit::tracer::addInputs(node, "dims", dims);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("rot90_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::rot90_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, k, dims, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _nested_tensor_strides_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_tensor_strides");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_nested_tensor_strides_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_nested_tensor_strides_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _nested_tensor_storage_offsets_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_nested_tensor_storage_offsets");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_nested_tensor_storage_offsets_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_nested_tensor_storage_offsets_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _standard_gamma_grad_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & output, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_standard_gamma_grad");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "output", output);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_standard_gamma_grad_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_standard_gamma_grad_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, output, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _batch_norm_with_update_functional(c10::DispatchKeySet ks, const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const at::Tensor & running_mean, const at::Tensor & running_var, double momentum, double eps) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_batch_norm_with_update_functional");
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
  auto [result0, result1, result2, result3, running_mean_out, running_var_out] =at::_ops::_batch_norm_with_update_functional::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), input, weight, bias, running_mean, running_var, momentum, eps);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, result1);
    jit::tracer::addOutput(node, result2);
    jit::tracer::addOutput(node, result3);
    jit::tracer::addOutput(node, running_mean_out);
    jit::tracer::addOutput(node, running_var_out);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(running_mean_out), std::move(running_var_out));
}
at::Tensor & _sparse_csr_sum_out_dim_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, ::std::optional<at::ScalarType> dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_csr_sum");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_csr_sum_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_sparse_csr_sum_dim_dtype_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dim, keepdim, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & norm_out_ScalarOpt_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, const ::std::optional<at::Scalar> & p, at::ScalarType dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::norm_ScalarOpt_dtype_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & norm_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & p, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::norm");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "p", p);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("norm_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::norm_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, p, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _sparse_mask_projection_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, bool accumulate_matches, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_mask_projection");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "mask", mask);
    jit::tracer::addInputs(node, "accumulate_matches", accumulate_matches);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_mask_projection_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_sparse_mask_projection_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, mask, accumulate_matches, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _to_dense_out_out(c10::DispatchKeySet ks, const at::Tensor & self, ::std::optional<at::ScalarType> dtype, ::std::optional<bool> masked_grad, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_to_dense");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "masked_grad", masked_grad);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_to_dense_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_to_dense_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, masked_grad, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _coalesced_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool coalesced, at::Tensor & out) {
  at::_ops::_coalesced_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, coalesced, out);
  return out;
}
at::Tensor _coalesced(c10::DispatchKeySet ks, const at::Tensor & self, bool coalesced) {
  auto result =at::_ops::_coalesced::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, coalesced);
  return result;
}
at::Tensor & quantize_per_tensor_dynamic_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::ScalarType dtype, bool reduce_range, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantize_per_tensor_dynamic");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);
    jit::tracer::addInputs(node, "reduce_range", reduce_range);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("quantize_per_tensor_dynamic_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::quantize_per_tensor_dynamic_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, reduce_range, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & quantize_per_tensor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point, at::ScalarType dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantize_per_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("quantize_per_tensor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::quantize_per_tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & quantize_per_tensor_out_tensor_qparams_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, at::ScalarType dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::quantize_per_tensor");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("quantize_per_tensor_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::quantize_per_tensor_tensor_qparams_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void quantize_per_tensor_out_tensors_out(c10::DispatchKeySet ks, at::TensorList tensors, const at::Tensor & scales, const at::Tensor & zero_points, at::ScalarType dtype, at::TensorList out) {
  at::_ops::quantize_per_tensor_tensors_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), tensors, scales, zero_points, dtype, out);
}
::std::tuple<at::Tensor &,at::Tensor &> fake_quantize_per_tensor_affine_cachemask_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::fake_quantize_per_tensor_affine_cachemask");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("fake_quantize_per_tensor_affine_cachemask_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::fake_quantize_per_tensor_affine_cachemask_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, quant_min, quant_max, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
::std::tuple<at::Tensor &,at::Tensor &> _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, const at::Tensor & fake_quant_enabled, int64_t quant_min, int64_t quant_max, at::Tensor & out0, at::Tensor & out1) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "fake_quant_enabled", fake_quant_enabled);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, fake_quant_enabled, quant_min, quant_max, out0, out1);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out0);
    jit::tracer::addOutput(node, out1);
  }
  return std::forward_as_tuple(out0, out1);
}
at::Tensor & _fake_quantize_learnable_per_tensor_affine_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scale, const at::Tensor & zero_point, int64_t quant_min, int64_t quant_max, double grad_factor, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fake_quantize_learnable_per_tensor_affine");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "scale", scale);
    jit::tracer::addInputs(node, "zero_point", zero_point);
    jit::tracer::addInputs(node, "quant_min", quant_min);
    jit::tracer::addInputs(node, "quant_max", quant_max);
    jit::tracer::addInputs(node, "grad_factor", grad_factor);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_fake_quantize_learnable_per_tensor_affine_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_fake_quantize_learnable_per_tensor_affine_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, scale, zero_point, quant_min, quant_max, grad_factor, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> _thnn_fused_gru_cell_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_hy, const at::Tensor & workspace, bool has_bias, at::Tensor & out0, at::Tensor & out1, at::Tensor & out2, at::Tensor & out3, at::Tensor & out4) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_thnn_fused_gru_cell_backward");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "grad_hy", grad_hy);
    jit::tracer::addInputs(node, "workspace", workspace);
    jit::tracer::addInputs(node, "has_bias", has_bias);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out0", out0);
      jit::tracer::addInputs(node, "out1", out1);
      jit::tracer::addInputs(node, "out2", out2);
      jit::tracer::addInputs(node, "out3", out3);
      jit::tracer::addInputs(node, "out4", out4);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_thnn_fused_gru_cell_backward_out", out0);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_thnn_fused_gru_cell_backward_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), grad_hy, workspace, has_bias, out0, out1, out2, out3, out4);
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
at::Tensor & set_out_source_Storage_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Storage source, at::Tensor & out) {
  at::_ops::set_source_Storage_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, out);
  return out;
}
at::Tensor set_source_Storage(c10::DispatchKeySet ks, const at::Tensor & self, at::Storage source) {
  auto result =at::_ops::set_source_Storage::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source);
  return result;
}
at::Tensor & set_out_source_Storage_storage_offset_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, at::Tensor & out) {
  at::_ops::set_source_Storage_storage_offset_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, storage_offset, size, stride, out);
  return out;
}
at::Tensor set_source_Storage_storage_offset(c10::DispatchKeySet ks, const at::Tensor & self, at::Storage source, c10::SymInt storage_offset, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  auto result =at::_ops::set_source_Storage_storage_offset::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, storage_offset, size, stride);
  return result;
}
at::Tensor & set_out_source_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & source, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::set");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("set_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::set_source_Tensor_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor set_source_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & source) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::set");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "source", source);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::set_source_Tensor::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, source);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & set_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::set");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("set_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::set_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor set(c10::DispatchKeySet ks, const at::Tensor & self) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::set");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::set::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & put_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::put");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "index", index);
    jit::tracer::addInputs(node, "source", source);
    jit::tracer::addInputs(node, "accumulate", accumulate);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("put_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::put_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, index, source, accumulate, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & uniform_out_out(c10::DispatchKeySet ks, const at::Tensor & self, double from, double to, ::std::optional<at::Generator> generator, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::uniform");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("uniform_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::uniform_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, from, to, generator, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor uniform(c10::DispatchKeySet ks, const at::Tensor & self, double from, double to, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::uniform");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "from", from);
    jit::tracer::addInputs(node, "to", to);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto result =at::_ops::uniform::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, from, to, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result);
  }
  return result;
}
at::Tensor & tril_indices_out_out(c10::DispatchKeySet ks, int64_t row, int64_t col, int64_t offset, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::tril_indices");
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
    jit::tracer::ensureUniqueIfOutOfPlaced("tril_indices_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::tril_indices_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), row, col, offset, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _cholesky_solve_helper_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & A, bool upper, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_cholesky_solve_helper");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "A", A);
    jit::tracer::addInputs(node, "upper", upper);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_cholesky_solve_helper_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_cholesky_solve_helper_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, A, upper, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _foreach_exp_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_exp_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_neg_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_neg_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_norm_out_Scalar_out(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & ord, ::std::optional<at::ScalarType> dtype, at::TensorList out) {
  at::_ops::_foreach_norm_Scalar_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, ord, dtype, out);
}
void _foreach_reciprocal_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_reciprocal_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_rsqrt_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_rsqrt_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_sigmoid_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_sigmoid_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_sin_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_sin_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
void _foreach_sqrt_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList out) {
  at::_ops::_foreach_sqrt_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
}
at::Tensor & glu_jvp_out_out(c10::DispatchKeySet ks, const at::Tensor & glu, const at::Tensor & x, const at::Tensor & dx, int64_t dim, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::glu_jvp");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "glu", glu);
    jit::tracer::addInputs(node, "x", x);
    jit::tracer::addInputs(node, "dx", dx);
    jit::tracer::addInputs(node, "dim", dim);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("glu_jvp_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::glu_jvp_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), glu, x, dx, dim, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
::std::tuple<at::Tensor,at::Tensor> rrelu_with_noise_functional(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, ::std::optional<at::Generator> generator) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::rrelu_with_noise_functional");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "noise", noise);
    jit::tracer::addInputs(node, "lower", lower);
    jit::tracer::addInputs(node, "upper", upper);
    jit::tracer::addInputs(node, "training", training);
    jit::tracer::addInputs(node, "generator", generator);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [result0, noise_out] =at::_ops::rrelu_with_noise_functional::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, noise, lower, upper, training, generator);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, result0);
    jit::tracer::addOutput(node, noise_out);
  }
  return std::make_tuple(std::move(result0), std::move(noise_out));
}
at::Tensor & _test_optional_floatlist_out_out(c10::DispatchKeySet ks, const at::Tensor & values, ::std::optional<at::ArrayRef<double>> addends, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_test_optional_floatlist");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "values", values);
    jit::tracer::addInputs(node, "addends", addends);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_test_optional_floatlist_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_test_optional_floatlist_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), values, addends, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _sparse_broadcast_to_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_sparse_broadcast_to_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_sparse_broadcast_to_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_sparse_broadcast_to_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _indices_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_indices_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_indices_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_indices_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & _values_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_values_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("_values_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::_values_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & values_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::values_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("values_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::values_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & view_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::SymIntArrayRef size, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::view_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "size", size);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("view_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::view_copy_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, size, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
at::Tensor & view_copy_out_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, at::ScalarType dtype, at::Tensor & out) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::view_copy");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "dtype", dtype);

    if (tracer_state->force_outplace) {

    } else {
      jit::tracer::addInputs(node, "out", out);
    }
    tracer_state->insertNode(node);
    jit::tracer::ensureUniqueIfOutOfPlaced("view_copy_out", out);
    jit::tracer::setTracingState(nullptr);
  }
  at::_ops::view_copy_dtype_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, dtype, out);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, out);
  }
  return out;
}
void _fused_sgd_out_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf, at::TensorList out) {
  at::_ops::_fused_sgd_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step, grad_scale, found_inf, out);
}
::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>> _fused_sgd(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fused_sgd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grads", grads);
    jit::tracer::addInputs(node, "momentum_buffer_list", momentum_buffer_list);
    jit::tracer::addInputs(node, "weight_decay", weight_decay);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "lr", lr);
    jit::tracer::addInputs(node, "dampening", dampening);
    jit::tracer::addInputs(node, "nesterov", nesterov);
    jit::tracer::addInputs(node, "maximize", maximize);
    jit::tracer::addInputs(node, "is_first_step", is_first_step);
    jit::tracer::addInputs(node, "grad_scale", grad_scale);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [self_out, grads_out, momentum_buffer_list_out] =at::_ops::_fused_sgd::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step, grad_scale, found_inf);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self_out);
    jit::tracer::addOutput(node, grads_out);
    jit::tracer::addOutput(node, momentum_buffer_list_out);
  }
  return std::make_tuple(std::move(self_out), std::move(grads_out), std::move(momentum_buffer_list_out));
}
void _fused_sgd_out_tensor_lr_out(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, const at::Tensor & lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf, at::TensorList out) {
  at::_ops::_fused_sgd_tensor_lr_out::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step, grad_scale, found_inf, out);
}
::std::tuple<::std::vector<at::Tensor>,::std::vector<at::Tensor>,::std::vector<at::Tensor>> _fused_sgd_tensor_lr(c10::DispatchKeySet ks, at::TensorList self, at::TensorList grads, at::TensorList momentum_buffer_list, double weight_decay, double momentum, const at::Tensor & lr, double dampening, bool nesterov, bool maximize, bool is_first_step, const ::std::optional<at::Tensor> & grad_scale, const ::std::optional<at::Tensor> & found_inf) {
  torch::jit::Node* node = nullptr;
  std::shared_ptr<jit::tracer::TracingState> tracer_state;
  if (jit::tracer::isTracing()) {
    tracer_state = jit::tracer::getTracingState();
    at::Symbol op_name;
    op_name = c10::Symbol::fromQualString("aten::_fused_sgd");
    node = tracer_state->createNode(op_name, /*num_outputs=*/0);
    jit::tracer::recordSourceLocation(node);
    jit::tracer::addInputs(node, "self", self);
    jit::tracer::addInputs(node, "grads", grads);
    jit::tracer::addInputs(node, "momentum_buffer_list", momentum_buffer_list);
    jit::tracer::addInputs(node, "weight_decay", weight_decay);
    jit::tracer::addInputs(node, "momentum", momentum);
    jit::tracer::addInputs(node, "lr", lr);
    jit::tracer::addInputs(node, "dampening", dampening);
    jit::tracer::addInputs(node, "nesterov", nesterov);
    jit::tracer::addInputs(node, "maximize", maximize);
    jit::tracer::addInputs(node, "is_first_step", is_first_step);
    jit::tracer::addInputs(node, "grad_scale", grad_scale);
    jit::tracer::addInputs(node, "found_inf", found_inf);
    tracer_state->insertNode(node);

    jit::tracer::setTracingState(nullptr);
  }
  auto [self_out, grads_out, momentum_buffer_list_out] =at::_ops::_fused_sgd_tensor_lr::redispatch(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer), self, grads, momentum_buffer_list, weight_decay, momentum, lr, dampening, nesterov, maximize, is_first_step, grad_scale, found_inf);
  if (tracer_state) {
    jit::tracer::setTracingState(std::move(tracer_state));
    jit::tracer::addOutput(node, self_out);
    jit::tracer::addOutput(node, grads_out);
    jit::tracer::addOutput(node, momentum_buffer_list_out);
  }
  return std::make_tuple(std::move(self_out), std::move(grads_out), std::move(momentum_buffer_list_out));
}
}  // namespace
}  // namespace TraceType

namespace {

TORCH_LIBRARY_IMPL(aten, Tracer, m) {
  m.impl("_new_zeros_with_same_feature_meta",
         TORCH_FN(TraceType::_new_zeros_with_same_feature_meta)
  );
  m.impl("_assert_scalar",
         TORCH_FN(TraceType::_assert_scalar)
  );
  m.impl("_print",
         TORCH_FN(TraceType::_print)
  );
  m.impl("_functional_sym_constrain_range_for_size",
         TORCH_FN(TraceType::_functional_sym_constrain_range_for_size)
  );
  m.impl("_make_dep_token",
         TORCH_FN(TraceType::_make_dep_token)
  );
  m.impl("_cudnn_init_dropout_state",
         TORCH_FN(TraceType::_cudnn_init_dropout_state)
  );
  m.impl("native_dropout",
         TORCH_FN(TraceType::native_dropout)
  );
  m.impl("angle",
         TORCH_FN(TraceType::angle)
  );
  m.impl("angle.out",
         TORCH_FN(TraceType::angle_out_out)
  );
  m.impl("asinh",
         TORCH_FN(TraceType::asinh)
  );
  m.impl("asinh_",
         TORCH_FN(TraceType::asinh_)
  );
  m.impl("asinh.out",
         TORCH_FN(TraceType::asinh_out_out)
  );
  m.impl("baddbmm",
         TORCH_FN(TraceType::baddbmm)
  );
  m.impl("baddbmm_",
         TORCH_FN(TraceType::baddbmm_)
  );
  m.impl("baddbmm.out",
         TORCH_FN(TraceType::baddbmm_out_out)
  );
  m.impl("baddbmm.dtype",
         TORCH_FN(TraceType::baddbmm_dtype)
  );
  m.impl("baddbmm.dtype_out",
         TORCH_FN(TraceType::baddbmm_out_dtype_out)
  );
  m.impl("batch_norm",
         TORCH_FN(TraceType::batch_norm)
  );
  m.impl("bernoulli",
         TORCH_FN(TraceType::bernoulli)
  );
  m.impl("bernoulli.out",
         TORCH_FN(TraceType::bernoulli_out_out)
  );
  m.impl("bernoulli_.Tensor",
         TORCH_FN(TraceType::bernoulli__Tensor)
  );
  m.impl("bernoulli_.float",
         TORCH_FN(TraceType::bernoulli__float)
  );
  m.impl("bernoulli.p",
         TORCH_FN(TraceType::bernoulli_p)
  );
  m.impl("bilinear",
         TORCH_FN(TraceType::bilinear)
  );
  m.impl("bincount",
         TORCH_FN(TraceType::bincount)
  );
  m.impl("unsafe_chunk",
         TORCH_FN(TraceType::unsafe_chunk)
  );
  m.impl("clamp_max",
         TORCH_FN(TraceType::clamp_max)
  );
  m.impl("clamp_max.Tensor",
         TORCH_FN(TraceType::clamp_max_Tensor)
  );
  m.impl("clamp_max_",
         TORCH_FN(TraceType::clamp_max_)
  );
  m.impl("clamp_max_.Tensor",
         TORCH_FN(TraceType::clamp_max__Tensor)
  );
  m.impl("clamp_max.out",
         TORCH_FN(TraceType::clamp_max_out_out)
  );
  m.impl("clamp_max.Tensor_out",
         TORCH_FN(TraceType::clamp_max_out_Tensor_out)
  );
  m.impl("clip",
         TORCH_FN(TraceType::clip)
  );
  m.impl("clip.Tensor",
         TORCH_FN(TraceType::clip_Tensor)
  );
  m.impl("clip_",
         TORCH_FN(TraceType::clip_)
  );
  m.impl("clip_.Tensor",
         TORCH_FN(TraceType::clip__Tensor)
  );
  m.impl("clip.out",
         TORCH_FN(TraceType::clip_out_out)
  );
  m.impl("clip.Tensor_out",
         TORCH_FN(TraceType::clip_out_Tensor_out)
  );
  m.impl("complex",
         TORCH_FN(TraceType::complex)
  );
  m.impl("complex.out",
         TORCH_FN(TraceType::complex_out_out)
  );
  m.impl("polar",
         TORCH_FN(TraceType::polar)
  );
  m.impl("polar.out",
         TORCH_FN(TraceType::polar_out_out)
  );
  m.impl("conv_transpose2d.input",
         TORCH_FN(TraceType::conv_transpose2d_input)
  );
  m.impl("count_nonzero.dim_IntList",
         TORCH_FN(TraceType::count_nonzero_dim_IntList)
  );
  m.impl("count_nonzero",
         TORCH_FN(TraceType::count_nonzero)
  );
  m.impl("cudnn_convolution_add_relu",
         TORCH_FN(TraceType::cudnn_convolution_add_relu)
  );
  m.impl("cummax",
         TORCH_FN(TraceType::cummax)
  );
  m.impl("cummax.out",
         TORCH_FN(TraceType::cummax_out_out)
  );
  m.impl("cummax.dimname",
         TORCH_FN(TraceType::cummax_dimname)
  );
  m.impl("cummax.dimname_out",
         TORCH_FN(TraceType::cummax_out_dimname_out)
  );
  m.impl("_cummax_helper",
         TORCH_FN(TraceType::_cummax_helper)
  );
  m.impl("_ctc_loss_backward",
         TORCH_FN(TraceType::_ctc_loss_backward)
  );
  m.impl("_ctc_loss_backward.Tensor",
         TORCH_FN(TraceType::_ctc_loss_backward_Tensor)
  );
  m.impl("diff",
         TORCH_FN(TraceType::diff)
  );
  m.impl("diff.out",
         TORCH_FN(TraceType::diff_out_out)
  );
  m.impl("gradient.scalarint",
         TORCH_FN(TraceType::gradient_scalarint)
  );
  m.impl("gradient.scalararray",
         TORCH_FN(TraceType::gradient_scalararray)
  );
  m.impl("gradient.array",
         TORCH_FN(TraceType::gradient_array)
  );
  m.impl("gradient.scalarrayint",
         TORCH_FN(TraceType::gradient_scalarrayint)
  );
  m.impl("gradient.scalarrayarray",
         TORCH_FN(TraceType::gradient_scalarrayarray)
  );
  m.impl("gradient.tensorarrayint",
         TORCH_FN(TraceType::gradient_tensorarrayint)
  );
  m.impl("gradient.tensorarray",
         TORCH_FN(TraceType::gradient_tensorarray)
  );
  m.impl("einsum",
         TORCH_FN(TraceType::einsum)
  );
  m.impl("embedding_renorm_",
         TORCH_FN(TraceType::embedding_renorm_)
  );
  m.impl("empty.names",
         TORCH_FN(TraceType::empty_names)
  );
  m.impl("empty.memory_format",
         TORCH_FN(TraceType::empty_memory_format)
  );
  m.impl("new_ones",
         TORCH_FN(TraceType::new_ones)
  );
  m.impl("_empty_per_channel_affine_quantized",
         TORCH_FN(TraceType::_empty_per_channel_affine_quantized)
  );
  m.impl("empty.out",
         TORCH_FN(TraceType::empty_out_out)
  );
  m.impl("exp2",
         TORCH_FN(TraceType::exp2)
  );
  m.impl("exp2_",
         TORCH_FN(TraceType::exp2_)
  );
  m.impl("exp2.out",
         TORCH_FN(TraceType::exp2_out_out)
  );
  m.impl("eye",
         TORCH_FN(TraceType::eye)
  );
  m.impl("eye.m",
         TORCH_FN(TraceType::eye_m)
  );
  m.impl("eye.out",
         TORCH_FN(TraceType::eye_out_out)
  );
  m.impl("eye.m_out",
         TORCH_FN(TraceType::eye_out_m_out)
  );
  m.impl("gcd.out",
         TORCH_FN(TraceType::gcd_out_out)
  );
  m.impl("gcd",
         TORCH_FN(TraceType::gcd)
  );
  m.impl("gcd_",
         TORCH_FN(TraceType::gcd_)
  );
  m.impl("_cufft_clear_plan_cache",
         TORCH_FN(TraceType::_cufft_clear_plan_cache)
  );
  m.impl("isin.Tensor_Tensor_out",
         TORCH_FN(TraceType::isin_out_Tensor_Tensor_out)
  );
  m.impl("isin.Tensor_Tensor",
         TORCH_FN(TraceType::isin_Tensor_Tensor)
  );
  m.impl("isin.Tensor_Scalar_out",
         TORCH_FN(TraceType::isin_out_Tensor_Scalar_out)
  );
  m.impl("isin.Tensor_Scalar",
         TORCH_FN(TraceType::isin_Tensor_Scalar)
  );
  m.impl("isin.Scalar_Tensor_out",
         TORCH_FN(TraceType::isin_out_Scalar_Tensor_out)
  );
  m.impl("isin.Scalar_Tensor",
         TORCH_FN(TraceType::isin_Scalar_Tensor)
  );
  m.impl("is_conj",
         TORCH_FN(TraceType::is_conj)
  );
  m.impl("is_nonzero",
         TORCH_FN(TraceType::is_nonzero)
  );
  m.impl("storage_offset",
         TORCH_FN(TraceType::storage_offset)
  );
  m.impl("layer_norm",
         TORCH_FN(TraceType::layer_norm)
  );
  m.impl("rms_norm",
         TORCH_FN(TraceType::rms_norm)
  );
  m.impl("fbgemm_linear_fp16_weight",
         TORCH_FN(TraceType::fbgemm_linear_fp16_weight)
  );
  m.impl("fbgemm_linear_fp16_weight.out",
         TORCH_FN(TraceType::fbgemm_linear_fp16_weight_out)
  );
  m.impl("fbgemm_pack_quantized_matrix",
         TORCH_FN(TraceType::fbgemm_pack_quantized_matrix)
  );
  m.impl("fbgemm_pack_quantized_matrix.KN",
         TORCH_FN(TraceType::fbgemm_pack_quantized_matrix_KN)
  );
  m.impl("log",
         TORCH_FN(TraceType::log)
  );
  m.impl("log_",
         TORCH_FN(TraceType::log_)
  );
  m.impl("log.out",
         TORCH_FN(TraceType::log_out_out)
  );
  m.impl("logaddexp.out",
         TORCH_FN(TraceType::logaddexp_out_out)
  );
  m.impl("logaddexp",
         TORCH_FN(TraceType::logaddexp)
  );
  m.impl("log_softmax.int",
         TORCH_FN(TraceType::log_softmax_int)
  );
  m.impl("log_softmax.int_out",
         TORCH_FN(TraceType::log_softmax_out_int_out)
  );
  m.impl("log_softmax.Dimname",
         TORCH_FN(TraceType::log_softmax_Dimname)
  );
  m.impl("mkldnn_max_pool3d",
         TORCH_FN(TraceType::mkldnn_max_pool3d)
  );
  m.impl("quantized_max_pool3d",
         TORCH_FN(TraceType::quantized_max_pool3d)
  );
  m.impl("miopen_convolution",
         TORCH_FN(TraceType::miopen_convolution)
  );
  m.impl("batch_norm_update_stats",
         TORCH_FN(TraceType::batch_norm_update_stats)
  );
  m.impl("ones_like",
         TORCH_FN(TraceType::ones_like)
  );
  m.impl("_pdist_forward",
         TORCH_FN(TraceType::_pdist_forward)
  );
  m.impl("randint",
         TORCH_FN(TraceType::randint)
  );
  m.impl("randint.generator",
         TORCH_FN(TraceType::randint_generator)
  );
  m.impl("randint.low",
         TORCH_FN(TraceType::randint_low)
  );
  m.impl("randint.low_generator",
         TORCH_FN(TraceType::randint_low_generator)
  );
  m.impl("randint.out",
         TORCH_FN(TraceType::randint_out_out)
  );
  m.impl("randint.generator_out",
         TORCH_FN(TraceType::randint_out_generator_out)
  );
  m.impl("randint.low_out",
         TORCH_FN(TraceType::randint_out_low_out)
  );
  m.impl("randint.low_generator_out",
         TORCH_FN(TraceType::randint_out_low_generator_out)
  );
  m.impl("randn_like",
         TORCH_FN(TraceType::randn_like)
  );
  m.impl("randn_like.generator",
         TORCH_FN(TraceType::randn_like_generator)
  );
  m.impl("repeat_interleave.Tensor",
         TORCH_FN(TraceType::repeat_interleave_Tensor)
  );
  m.impl("repeat_interleave.self_Tensor",
         TORCH_FN(TraceType::repeat_interleave_self_Tensor)
  );
  m.impl("repeat_interleave.self_int",
         TORCH_FN(TraceType::repeat_interleave_self_int)
  );
  m.impl("_prelu_kernel",
         TORCH_FN(TraceType::_prelu_kernel)
  );
  m.impl("_nested_select_backward",
         TORCH_FN(TraceType::_nested_select_backward)
  );
  m.impl("vsplit.int",
         TORCH_FN(TraceType::vsplit_int)
  );
  m.impl("vsplit.array",
         TORCH_FN(TraceType::vsplit_array)
  );
  m.impl("hstack",
         TORCH_FN(TraceType::hstack)
  );
  m.impl("hstack.out",
         TORCH_FN(TraceType::hstack_out_out)
  );
  m.impl("sum",
         TORCH_FN(TraceType::sum)
  );
  m.impl("sum.dim_IntList",
         TORCH_FN(TraceType::sum_dim_IntList)
  );
  m.impl("sum.dim_DimnameList",
         TORCH_FN(TraceType::sum_dim_DimnameList)
  );
  m.impl("sum.IntList_out",
         TORCH_FN(TraceType::sum_out_IntList_out)
  );
  m.impl("sum.DimnameList_out",
         TORCH_FN(TraceType::sum_out_DimnameList_out)
  );
  m.impl("rot90",
         TORCH_FN(TraceType::rot90)
  );
  m.impl("_nested_tensor_strides",
         TORCH_FN(TraceType::_nested_tensor_strides)
  );
  m.impl("_nested_tensor_storage_offsets",
         TORCH_FN(TraceType::_nested_tensor_storage_offsets)
  );
  m.impl("triplet_margin_loss",
         TORCH_FN(TraceType::triplet_margin_loss)
  );
  m.impl("var",
         TORCH_FN(TraceType::var)
  );
  m.impl("var.dim",
         TORCH_FN(TraceType::var_dim)
  );
  m.impl("var.correction",
         TORCH_FN(TraceType::var_correction)
  );
  m.impl("var.out",
         TORCH_FN(TraceType::var_out_out)
  );
  m.impl("var.correction_out",
         TORCH_FN(TraceType::var_out_correction_out)
  );
  m.impl("var.names_dim",
         TORCH_FN(TraceType::var_names_dim)
  );
  m.impl("var.names_out",
         TORCH_FN(TraceType::var_out_names_out)
  );
  m.impl("var.correction_names",
         TORCH_FN(TraceType::var_correction_names)
  );
  m.impl("var.correction_names_out",
         TORCH_FN(TraceType::var_out_correction_names_out)
  );
  m.impl("_standard_gamma_grad",
         TORCH_FN(TraceType::_standard_gamma_grad)
  );
  m.impl("_batch_norm_with_update",
         TORCH_FN(TraceType::_batch_norm_with_update)
  );
  m.impl("_batch_norm_with_update.out",
         TORCH_FN(TraceType::_batch_norm_with_update_out_out)
  );
  m.impl("_sparse_csr_sum.dim_dtype",
         TORCH_FN(TraceType::_sparse_csr_sum_dim_dtype)
  );
  m.impl("norm.ScalarOpt_dtype",
         TORCH_FN(TraceType::norm_ScalarOpt_dtype)
  );
  m.impl("norm.Scalar",
         TORCH_FN(TraceType::norm_Scalar)
  );
  m.impl("norm.ScalarOpt_dim_dtype",
         TORCH_FN(TraceType::norm_ScalarOpt_dim_dtype)
  );
  m.impl("norm.ScalarOpt_dim",
         TORCH_FN(TraceType::norm_ScalarOpt_dim)
  );
  m.impl("norm.dtype_out",
         TORCH_FN(TraceType::norm_out_dtype_out)
  );
  m.impl("norm.out",
         TORCH_FN(TraceType::norm_out_out)
  );
  m.impl("norm.names_ScalarOpt_dim_dtype",
         TORCH_FN(TraceType::norm_names_ScalarOpt_dim_dtype)
  );
  m.impl("norm.names_ScalarOpt_dim",
         TORCH_FN(TraceType::norm_names_ScalarOpt_dim)
  );
  m.impl("norm.names_dtype_out",
         TORCH_FN(TraceType::norm_out_names_dtype_out)
  );
  m.impl("norm.names_out",
         TORCH_FN(TraceType::norm_out_names_out)
  );
  m.impl("nuclear_norm",
         TORCH_FN(TraceType::nuclear_norm)
  );
  m.impl("nuclear_norm.out",
         TORCH_FN(TraceType::nuclear_norm_out_out)
  );
  m.impl("nuclear_norm.dim",
         TORCH_FN(TraceType::nuclear_norm_dim)
  );
  m.impl("nuclear_norm.dim_out",
         TORCH_FN(TraceType::nuclear_norm_out_dim_out)
  );
  m.impl("_sparse_compressed_tensor_with_dims",
         TORCH_FN(TraceType::_sparse_compressed_tensor_with_dims)
  );
  m.impl("_sparse_csc_tensor_unsafe",
         TORCH_FN(TraceType::_sparse_csc_tensor_unsafe)
  );
  m.impl("_validate_sparse_coo_tensor_args",
         TORCH_FN(TraceType::_validate_sparse_coo_tensor_args)
  );
  m.impl("_sparse_mask_projection",
         TORCH_FN(TraceType::_sparse_mask_projection)
  );
  m.impl("_to_dense",
         TORCH_FN(TraceType::_to_dense)
  );
  m.impl("_coalesced_",
         TORCH_FN(TraceType::_coalesced_)
  );
  m.impl("to_sparse_bsc",
         TORCH_FN(TraceType::to_sparse_bsc)
  );
  m.impl("quantize_per_tensor_dynamic",
         TORCH_FN(TraceType::quantize_per_tensor_dynamic)
  );
  m.impl("quantize_per_tensor",
         TORCH_FN(TraceType::quantize_per_tensor)
  );
  m.impl("quantize_per_tensor.tensor_qparams",
         TORCH_FN(TraceType::quantize_per_tensor_tensor_qparams)
  );
  m.impl("quantize_per_tensor.tensors",
         TORCH_FN(TraceType::quantize_per_tensor_tensors)
  );
  m.impl("fake_quantize_per_tensor_affine_cachemask",
         TORCH_FN(TraceType::fake_quantize_per_tensor_affine_cachemask)
  );
  m.impl("_fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
         TORCH_FN(TraceType::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams)
  );
  m.impl("_fake_quantize_learnable_per_tensor_affine",
         TORCH_FN(TraceType::_fake_quantize_learnable_per_tensor_affine)
  );
  m.impl("choose_qparams_optimized",
         TORCH_FN(TraceType::choose_qparams_optimized)
  );
  m.impl("_thnn_fused_gru_cell_backward",
         TORCH_FN(TraceType::_thnn_fused_gru_cell_backward)
  );
  m.impl("set_.source_Storage",
         TORCH_FN(TraceType::set__source_Storage)
  );
  m.impl("set_.source_Storage_storage_offset",
         TORCH_FN(TraceType::set__source_Storage_storage_offset)
  );
  m.impl("set_.source_Tensor_storage_offset",
         TORCH_FN(TraceType::set__source_Tensor_storage_offset)
  );
  m.impl("set_.source_Tensor",
         TORCH_FN(TraceType::set__source_Tensor)
  );
  m.impl("set_",
         TORCH_FN(TraceType::set_)
  );
  m.impl("masked_scatter_backward",
         TORCH_FN(TraceType::masked_scatter_backward)
  );
  m.impl("put_",
         TORCH_FN(TraceType::put_)
  );
  m.impl("put",
         TORCH_FN(TraceType::put)
  );
  m.impl("scatter.src",
         TORCH_FN(TraceType::scatter_src)
  );
  m.impl("scatter_.src",
         TORCH_FN(TraceType::scatter__src)
  );
  m.impl("scatter.src_out",
         TORCH_FN(TraceType::scatter_out_src_out)
  );
  m.impl("scatter.value",
         TORCH_FN(TraceType::scatter_value)
  );
  m.impl("scatter_.value",
         TORCH_FN(TraceType::scatter__value)
  );
  m.impl("scatter.value_out",
         TORCH_FN(TraceType::scatter_out_value_out)
  );
  m.impl("scatter.reduce",
         TORCH_FN(TraceType::scatter_reduce)
  );
  m.impl("scatter_.reduce",
         TORCH_FN(TraceType::scatter__reduce)
  );
  m.impl("scatter.reduce_out",
         TORCH_FN(TraceType::scatter_out_reduce_out)
  );
  m.impl("scatter.value_reduce",
         TORCH_FN(TraceType::scatter_value_reduce)
  );
  m.impl("scatter_.value_reduce",
         TORCH_FN(TraceType::scatter__value_reduce)
  );
  m.impl("scatter.value_reduce_out",
         TORCH_FN(TraceType::scatter_out_value_reduce_out)
  );
  m.impl("scatter.dimname_src",
         TORCH_FN(TraceType::scatter_dimname_src)
  );
  m.impl("scatter.dimname_value",
         TORCH_FN(TraceType::scatter_dimname_value)
  );
  m.impl("__and__.Scalar",
         TORCH_FN(TraceType::__and___Scalar)
  );
  m.impl("__and__.Tensor",
         TORCH_FN(TraceType::__and___Tensor)
  );
  m.impl("__iand__.Scalar",
         TORCH_FN(TraceType::__iand___Scalar)
  );
  m.impl("__iand__.Tensor",
         TORCH_FN(TraceType::__iand___Tensor)
  );
  m.impl("tril_",
         TORCH_FN(TraceType::tril_)
  );
  m.impl("uniform_",
         TORCH_FN(TraceType::uniform_)
  );
  m.impl("tril.out",
         TORCH_FN(TraceType::tril_out_out)
  );
  m.impl("tril",
         TORCH_FN(TraceType::tril)
  );
  m.impl("tril_indices",
         TORCH_FN(TraceType::tril_indices)
  );
  m.impl("less_equal.Scalar_out",
         TORCH_FN(TraceType::less_equal_out_Scalar_out)
  );
  m.impl("less_equal.Scalar",
         TORCH_FN(TraceType::less_equal_Scalar)
  );
  m.impl("less_equal.Tensor_out",
         TORCH_FN(TraceType::less_equal_out_Tensor_out)
  );
  m.impl("less_equal.Tensor",
         TORCH_FN(TraceType::less_equal_Tensor)
  );
  m.impl("less_equal_.Scalar",
         TORCH_FN(TraceType::less_equal__Scalar)
  );
  m.impl("less_equal_.Tensor",
         TORCH_FN(TraceType::less_equal__Tensor)
  );
  m.impl("gt.Scalar_out",
         TORCH_FN(TraceType::gt_out_Scalar_out)
  );
  m.impl("gt.Scalar",
         TORCH_FN(TraceType::gt_Scalar)
  );
  m.impl("gt.Tensor_out",
         TORCH_FN(TraceType::gt_out_Tensor_out)
  );
  m.impl("gt.Tensor",
         TORCH_FN(TraceType::gt_Tensor)
  );
  m.impl("gt_.Scalar",
         TORCH_FN(TraceType::gt__Scalar)
  );
  m.impl("gt_.Tensor",
         TORCH_FN(TraceType::gt__Tensor)
  );
  m.impl("lt.Scalar_out",
         TORCH_FN(TraceType::lt_out_Scalar_out)
  );
  m.impl("lt.Scalar",
         TORCH_FN(TraceType::lt_Scalar)
  );
  m.impl("lt.Tensor_out",
         TORCH_FN(TraceType::lt_out_Tensor_out)
  );
  m.impl("lt.Tensor",
         TORCH_FN(TraceType::lt_Tensor)
  );
  m.impl("lt_.Scalar",
         TORCH_FN(TraceType::lt__Scalar)
  );
  m.impl("lt_.Tensor",
         TORCH_FN(TraceType::lt__Tensor)
  );
  m.impl("less.Scalar_out",
         TORCH_FN(TraceType::less_out_Scalar_out)
  );
  m.impl("less.Scalar",
         TORCH_FN(TraceType::less_Scalar)
  );
  m.impl("less.Tensor_out",
         TORCH_FN(TraceType::less_out_Tensor_out)
  );
  m.impl("less.Tensor",
         TORCH_FN(TraceType::less_Tensor)
  );
  m.impl("less_.Scalar",
         TORCH_FN(TraceType::less__Scalar)
  );
  m.impl("less_.Tensor",
         TORCH_FN(TraceType::less__Tensor)
  );
  m.impl("nonzero_static.out",
         TORCH_FN(TraceType::nonzero_static_out_out)
  );
  m.impl("nonzero_static",
         TORCH_FN(TraceType::nonzero_static)
  );
  m.impl("addcdiv.out",
         TORCH_FN(TraceType::addcdiv_out_out)
  );
  m.impl("addcdiv",
         TORCH_FN(TraceType::addcdiv)
  );
  m.impl("addcdiv_",
         TORCH_FN(TraceType::addcdiv_)
  );
  m.impl("_cholesky_solve_helper",
         TORCH_FN(TraceType::_cholesky_solve_helper)
  );
  m.impl("cholesky_inverse",
         TORCH_FN(TraceType::cholesky_inverse)
  );
  m.impl("cholesky_inverse.out",
         TORCH_FN(TraceType::cholesky_inverse_out_out)
  );
  m.impl("_lu_with_info",
         TORCH_FN(TraceType::_lu_with_info)
  );
  m.impl("atan2.out",
         TORCH_FN(TraceType::atan2_out_out)
  );
  m.impl("atan2_",
         TORCH_FN(TraceType::atan2_)
  );
  m.impl("atan2",
         TORCH_FN(TraceType::atan2)
  );
  m.impl("fmax",
         TORCH_FN(TraceType::fmax)
  );
  m.impl("fmax.out",
         TORCH_FN(TraceType::fmax_out_out)
  );
  m.impl("_foreach_exp",
         TORCH_FN(TraceType::_foreach_exp)
  );
  m.impl("_foreach_exp_",
         TORCH_FN(TraceType::_foreach_exp_)
  );
  m.impl("_foreach_neg",
         TORCH_FN(TraceType::_foreach_neg)
  );
  m.impl("_foreach_neg_",
         TORCH_FN(TraceType::_foreach_neg_)
  );
  m.impl("_foreach_norm.Scalar",
         TORCH_FN(TraceType::_foreach_norm_Scalar)
  );
  m.impl("_foreach_reciprocal",
         TORCH_FN(TraceType::_foreach_reciprocal)
  );
  m.impl("_foreach_reciprocal_",
         TORCH_FN(TraceType::_foreach_reciprocal_)
  );
  m.impl("_foreach_rsqrt",
         TORCH_FN(TraceType::_foreach_rsqrt)
  );
  m.impl("_foreach_rsqrt_",
         TORCH_FN(TraceType::_foreach_rsqrt_)
  );
  m.impl("_foreach_sigmoid",
         TORCH_FN(TraceType::_foreach_sigmoid)
  );
  m.impl("_foreach_sigmoid_",
         TORCH_FN(TraceType::_foreach_sigmoid_)
  );
  m.impl("_foreach_sin",
         TORCH_FN(TraceType::_foreach_sin)
  );
  m.impl("_foreach_sin_",
         TORCH_FN(TraceType::_foreach_sin_)
  );
  m.impl("_foreach_sqrt",
         TORCH_FN(TraceType::_foreach_sqrt)
  );
  m.impl("_foreach_sqrt_",
         TORCH_FN(TraceType::_foreach_sqrt_)
  );
  m.impl("soft_margin_loss_backward.grad_input",
         TORCH_FN(TraceType::soft_margin_loss_backward_out_grad_input)
  );
  m.impl("soft_margin_loss_backward",
         TORCH_FN(TraceType::soft_margin_loss_backward)
  );
  m.impl("glu_jvp",
         TORCH_FN(TraceType::glu_jvp)
  );
  m.impl("rrelu_with_noise.out",
         TORCH_FN(TraceType::rrelu_with_noise_out_out)
  );
  m.impl("rrelu_with_noise",
         TORCH_FN(TraceType::rrelu_with_noise)
  );
  m.impl("rrelu_with_noise_",
         TORCH_FN(TraceType::rrelu_with_noise_)
  );
  m.impl("reflection_pad1d_backward.grad_input",
         TORCH_FN(TraceType::reflection_pad1d_backward_out_grad_input)
  );
  m.impl("reflection_pad1d_backward",
         TORCH_FN(TraceType::reflection_pad1d_backward)
  );
  m.impl("replication_pad1d_backward.grad_input",
         TORCH_FN(TraceType::replication_pad1d_backward_out_grad_input)
  );
  m.impl("replication_pad1d_backward",
         TORCH_FN(TraceType::replication_pad1d_backward)
  );
  m.impl("replication_pad3d_backward.grad_input",
         TORCH_FN(TraceType::replication_pad3d_backward_out_grad_input)
  );
  m.impl("replication_pad3d_backward",
         TORCH_FN(TraceType::replication_pad3d_backward)
  );
  m.impl("_upsample_nearest_exact2d.vec",
         TORCH_FN(TraceType::_upsample_nearest_exact2d_vec)
  );
  m.impl("_upsample_bicubic2d_aa_backward.grad_input",
         TORCH_FN(TraceType::_upsample_bicubic2d_aa_backward_out_grad_input)
  );
  m.impl("_upsample_bicubic2d_aa_backward",
         TORCH_FN(TraceType::_upsample_bicubic2d_aa_backward)
  );
  m.impl("_upsample_nearest_exact2d.out",
         TORCH_FN(TraceType::_upsample_nearest_exact2d_out_out)
  );
  m.impl("_upsample_nearest_exact2d",
         TORCH_FN(TraceType::_upsample_nearest_exact2d)
  );
  m.impl("_upsample_nearest_exact2d_backward.grad_input",
         TORCH_FN(TraceType::_upsample_nearest_exact2d_backward_out_grad_input)
  );
  m.impl("_upsample_nearest_exact2d_backward",
         TORCH_FN(TraceType::_upsample_nearest_exact2d_backward)
  );
  m.impl("special_ndtr",
         TORCH_FN(TraceType::special_ndtr)
  );
  m.impl("special_ndtr.out",
         TORCH_FN(TraceType::special_ndtr_out_out)
  );
  m.impl("special_zeta",
         TORCH_FN(TraceType::special_zeta)
  );
  m.impl("special_zeta.self_scalar",
         TORCH_FN(TraceType::special_zeta_self_scalar)
  );
  m.impl("special_zeta.other_scalar",
         TORCH_FN(TraceType::special_zeta_other_scalar)
  );
  m.impl("special_zeta.out",
         TORCH_FN(TraceType::special_zeta_out_out)
  );
  m.impl("special_zeta.self_scalar_out",
         TORCH_FN(TraceType::special_zeta_out_self_scalar_out)
  );
  m.impl("special_zeta.other_scalar_out",
         TORCH_FN(TraceType::special_zeta_out_other_scalar_out)
  );
  m.impl("special_round",
         TORCH_FN(TraceType::special_round)
  );
  m.impl("special_round.out",
         TORCH_FN(TraceType::special_round_out_out)
  );
  m.impl("fft_ifft",
         TORCH_FN(TraceType::fft_ifft)
  );
  m.impl("fft_ifft.out",
         TORCH_FN(TraceType::fft_ifft_out_out)
  );
  m.impl("fft_ihfft",
         TORCH_FN(TraceType::fft_ihfft)
  );
  m.impl("fft_ihfft.out",
         TORCH_FN(TraceType::fft_ihfft_out_out)
  );
  m.impl("fft_ifftshift",
         TORCH_FN(TraceType::fft_ifftshift)
  );
  m.impl("slogdet",
         TORCH_FN(TraceType::slogdet)
  );
  m.impl("slogdet.out",
         TORCH_FN(TraceType::slogdet_out_out)
  );
  m.impl("linalg_eig",
         TORCH_FN(TraceType::linalg_eig)
  );
  m.impl("linalg_eig.out",
         TORCH_FN(TraceType::linalg_eig_out_out)
  );
  m.impl("_linalg_eigvals",
         TORCH_FN(TraceType::_linalg_eigvals)
  );
  m.impl("linalg_eigvalsh",
         TORCH_FN(TraceType::linalg_eigvalsh)
  );
  m.impl("linalg_eigvalsh.out",
         TORCH_FN(TraceType::linalg_eigvalsh_out_out)
  );
  m.impl("linalg_svd",
         TORCH_FN(TraceType::linalg_svd)
  );
  m.impl("linalg_svd.U",
         TORCH_FN(TraceType::linalg_svd_out_U)
  );
  m.impl("_test_optional_floatlist",
         TORCH_FN(TraceType::_test_optional_floatlist)
  );
  m.impl("unflatten_dense_tensors",
         TORCH_FN(TraceType::unflatten_dense_tensors)
  );
  m.impl("_sparse_broadcast_to_copy",
         TORCH_FN(TraceType::_sparse_broadcast_to_copy)
  );
  m.impl("_indices_copy",
         TORCH_FN(TraceType::_indices_copy)
  );
  m.impl("_values_copy",
         TORCH_FN(TraceType::_values_copy)
  );
  m.impl("values_copy",
         TORCH_FN(TraceType::values_copy)
  );
  m.impl("view_copy",
         TORCH_FN(TraceType::view_copy)
  );
  m.impl("view_copy.dtype",
         TORCH_FN(TraceType::view_copy_dtype)
  );
  m.impl("_scaled_dot_product_fused_attention_overrideable_backward",
         TORCH_FN(TraceType::_scaled_dot_product_fused_attention_overrideable_backward)
  );
  m.impl("_scaled_dot_product_cudnn_attention_backward",
         TORCH_FN(TraceType::_scaled_dot_product_cudnn_attention_backward)
  );
  m.impl("special_laguerre_polynomial_l",
         TORCH_FN(TraceType::special_laguerre_polynomial_l)
  );
  m.impl("special_laguerre_polynomial_l.x_scalar",
         TORCH_FN(TraceType::special_laguerre_polynomial_l_x_scalar)
  );
  m.impl("special_laguerre_polynomial_l.n_scalar",
         TORCH_FN(TraceType::special_laguerre_polynomial_l_n_scalar)
  );
  m.impl("special_laguerre_polynomial_l.out",
         TORCH_FN(TraceType::special_laguerre_polynomial_l_out_out)
  );
  m.impl("special_laguerre_polynomial_l.x_scalar_out",
         TORCH_FN(TraceType::special_laguerre_polynomial_l_out_x_scalar_out)
  );
  m.impl("special_laguerre_polynomial_l.n_scalar_out",
         TORCH_FN(TraceType::special_laguerre_polynomial_l_out_n_scalar_out)
  );
  m.impl("special_scaled_modified_bessel_k0",
         TORCH_FN(TraceType::special_scaled_modified_bessel_k0)
  );
  m.impl("special_scaled_modified_bessel_k0.out",
         TORCH_FN(TraceType::special_scaled_modified_bessel_k0_out_out)
  );
  m.impl("_fused_sgd_",
         TORCH_FN(TraceType::_fused_sgd_)
  );
  m.impl("_fused_sgd_.tensor_lr",
         TORCH_FN(TraceType::_fused_sgd__tensor_lr)
  );
  m.impl("_propagate_xla_data",
         TORCH_FN(TraceType::_propagate_xla_data)
  );
  m.impl("_new_zeros_with_same_feature_meta.out",
         TORCH_FN(TraceType::_new_zeros_with_same_feature_meta_out_out)
  );
  m.impl("_cudnn_init_dropout_state.out",
         TORCH_FN(TraceType::_cudnn_init_dropout_state_out_out)
  );
  m.impl("native_dropout.out",
         TORCH_FN(TraceType::native_dropout_out_out)
  );
  m.impl("bernoulli.Tensor_out",
         TORCH_FN(TraceType::bernoulli_out_Tensor_out)
  );
  m.impl("bernoulli.Tensor",
         TORCH_FN(TraceType::bernoulli_Tensor)
  );
  m.impl("bernoulli.float_out",
         TORCH_FN(TraceType::bernoulli_out_float_out)
  );
  m.impl("bincount.out",
         TORCH_FN(TraceType::bincount_out_out)
  );
  m.impl("count_nonzero.dim_IntList_out",
         TORCH_FN(TraceType::count_nonzero_out_dim_IntList_out)
  );
  m.impl("count_nonzero.out",
         TORCH_FN(TraceType::count_nonzero_out_out)
  );
  m.impl("cudnn_convolution_add_relu.out",
         TORCH_FN(TraceType::cudnn_convolution_add_relu_out_out)
  );
  m.impl("_ctc_loss_backward.out",
         TORCH_FN(TraceType::_ctc_loss_backward_out_out)
  );
  m.impl("embedding_renorm.out",
         TORCH_FN(TraceType::embedding_renorm_out_out)
  );
  m.impl("embedding_renorm",
         TORCH_FN(TraceType::embedding_renorm)
  );
  m.impl("empty.names_out",
         TORCH_FN(TraceType::empty_out_names_out)
  );
  m.impl("new_ones.out",
         TORCH_FN(TraceType::new_ones_out_out)
  );
  m.impl("_empty_per_channel_affine_quantized.out",
         TORCH_FN(TraceType::_empty_per_channel_affine_quantized_out_out)
  );
  m.impl("mkldnn_max_pool3d.out",
         TORCH_FN(TraceType::mkldnn_max_pool3d_out_out)
  );
  m.impl("quantized_max_pool3d.out",
         TORCH_FN(TraceType::quantized_max_pool3d_out_out)
  );
  m.impl("miopen_convolution.out",
         TORCH_FN(TraceType::miopen_convolution_out_out)
  );
  m.impl("batch_norm_update_stats.out",
         TORCH_FN(TraceType::batch_norm_update_stats_out_out)
  );
  m.impl("ones_like.out",
         TORCH_FN(TraceType::ones_like_out_out)
  );
  m.impl("_pdist_forward.out",
         TORCH_FN(TraceType::_pdist_forward_out_out)
  );
  m.impl("randn_like.out",
         TORCH_FN(TraceType::randn_like_out_out)
  );
  m.impl("randn_like.generator_out",
         TORCH_FN(TraceType::randn_like_out_generator_out)
  );
  m.impl("repeat_interleave.Tensor_out",
         TORCH_FN(TraceType::repeat_interleave_out_Tensor_out)
  );
  m.impl("sum.out",
         TORCH_FN(TraceType::sum_out_out)
  );
  m.impl("rot90.out",
         TORCH_FN(TraceType::rot90_out_out)
  );
  m.impl("_nested_tensor_strides.out",
         TORCH_FN(TraceType::_nested_tensor_strides_out_out)
  );
  m.impl("_nested_tensor_storage_offsets.out",
         TORCH_FN(TraceType::_nested_tensor_storage_offsets_out_out)
  );
  m.impl("_standard_gamma_grad.out",
         TORCH_FN(TraceType::_standard_gamma_grad_out_out)
  );
  m.impl("_batch_norm_with_update_functional",
         TORCH_FN(TraceType::_batch_norm_with_update_functional)
  );
  m.impl("_sparse_csr_sum.dim_dtype_out",
         TORCH_FN(TraceType::_sparse_csr_sum_out_dim_dtype_out)
  );
  m.impl("norm.ScalarOpt_dtype_out",
         TORCH_FN(TraceType::norm_out_ScalarOpt_dtype_out)
  );
  m.impl("norm.Scalar_out",
         TORCH_FN(TraceType::norm_out_Scalar_out)
  );
  m.impl("_sparse_mask_projection.out",
         TORCH_FN(TraceType::_sparse_mask_projection_out_out)
  );
  m.impl("_to_dense.out",
         TORCH_FN(TraceType::_to_dense_out_out)
  );
  m.impl("_coalesced.out",
         TORCH_FN(TraceType::_coalesced_out_out)
  );
  m.impl("_coalesced",
         TORCH_FN(TraceType::_coalesced)
  );
  m.impl("quantize_per_tensor_dynamic.out",
         TORCH_FN(TraceType::quantize_per_tensor_dynamic_out_out)
  );
  m.impl("quantize_per_tensor.out",
         TORCH_FN(TraceType::quantize_per_tensor_out_out)
  );
  m.impl("quantize_per_tensor.tensor_qparams_out",
         TORCH_FN(TraceType::quantize_per_tensor_out_tensor_qparams_out)
  );
  m.impl("quantize_per_tensor.tensors_out",
         TORCH_FN(TraceType::quantize_per_tensor_out_tensors_out)
  );
  m.impl("fake_quantize_per_tensor_affine_cachemask.out",
         TORCH_FN(TraceType::fake_quantize_per_tensor_affine_cachemask_out_out)
  );
  m.impl("_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.out",
         TORCH_FN(TraceType::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams_out_out)
  );
  m.impl("_fake_quantize_learnable_per_tensor_affine.out",
         TORCH_FN(TraceType::_fake_quantize_learnable_per_tensor_affine_out_out)
  );
  m.impl("_thnn_fused_gru_cell_backward.out",
         TORCH_FN(TraceType::_thnn_fused_gru_cell_backward_out_out)
  );
  m.impl("set.source_Storage_out",
         TORCH_FN(TraceType::set_out_source_Storage_out)
  );
  m.impl("set.source_Storage",
         TORCH_FN(TraceType::set_source_Storage)
  );
  m.impl("set.source_Storage_storage_offset_out",
         TORCH_FN(TraceType::set_out_source_Storage_storage_offset_out)
  );
  m.impl("set.source_Storage_storage_offset",
         TORCH_FN(TraceType::set_source_Storage_storage_offset)
  );
  m.impl("set.source_Tensor_out",
         TORCH_FN(TraceType::set_out_source_Tensor_out)
  );
  m.impl("set.source_Tensor",
         TORCH_FN(TraceType::set_source_Tensor)
  );
  m.impl("set.out",
         TORCH_FN(TraceType::set_out_out)
  );
  m.impl("set",
         TORCH_FN(TraceType::set)
  );
  m.impl("put.out",
         TORCH_FN(TraceType::put_out_out)
  );
  m.impl("uniform.out",
         TORCH_FN(TraceType::uniform_out_out)
  );
  m.impl("uniform",
         TORCH_FN(TraceType::uniform)
  );
  m.impl("tril_indices.out",
         TORCH_FN(TraceType::tril_indices_out_out)
  );
  m.impl("_cholesky_solve_helper.out",
         TORCH_FN(TraceType::_cholesky_solve_helper_out_out)
  );
  m.impl("_foreach_exp.out",
         TORCH_FN(TraceType::_foreach_exp_out_out)
  );
  m.impl("_foreach_neg.out",
         TORCH_FN(TraceType::_foreach_neg_out_out)
  );
  m.impl("_foreach_norm.Scalar_out",
         TORCH_FN(TraceType::_foreach_norm_out_Scalar_out)
  );
  m.impl("_foreach_reciprocal.out",
         TORCH_FN(TraceType::_foreach_reciprocal_out_out)
  );
  m.impl("_foreach_rsqrt.out",
         TORCH_FN(TraceType::_foreach_rsqrt_out_out)
  );
  m.impl("_foreach_sigmoid.out",
         TORCH_FN(TraceType::_foreach_sigmoid_out_out)
  );
  m.impl("_foreach_sin.out",
         TORCH_FN(TraceType::_foreach_sin_out_out)
  );
  m.impl("_foreach_sqrt.out",
         TORCH_FN(TraceType::_foreach_sqrt_out_out)
  );
  m.impl("glu_jvp.out",
         TORCH_FN(TraceType::glu_jvp_out_out)
  );
  m.impl("rrelu_with_noise_functional",
         TORCH_FN(TraceType::rrelu_with_noise_functional)
  );
  m.impl("_test_optional_floatlist.out",
         TORCH_FN(TraceType::_test_optional_floatlist_out_out)
  );
  m.impl("_sparse_broadcast_to_copy.out",
         TORCH_FN(TraceType::_sparse_broadcast_to_copy_out_out)
  );
  m.impl("_indices_copy.out",
         TORCH_FN(TraceType::_indices_copy_out_out)
  );
  m.impl("_values_copy.out",
         TORCH_FN(TraceType::_values_copy_out_out)
  );
  m.impl("values_copy.out",
         TORCH_FN(TraceType::values_copy_out_out)
  );
  m.impl("view_copy.out",
         TORCH_FN(TraceType::view_copy_out_out)
  );
  m.impl("view_copy.dtype_out",
         TORCH_FN(TraceType::view_copy_out_dtype_out)
  );
  m.impl("_fused_sgd.out",
         TORCH_FN(TraceType::_fused_sgd_out_out)
  );
  m.impl("_fused_sgd",
         TORCH_FN(TraceType::_fused_sgd)
  );
  m.impl("_fused_sgd.tensor_lr_out",
         TORCH_FN(TraceType::_fused_sgd_out_tensor_lr_out)
  );
  m.impl("_fused_sgd.tensor_lr",
         TORCH_FN(TraceType::_fused_sgd_tensor_lr)
  );;
}

}  // namespace

} // namespace torch
