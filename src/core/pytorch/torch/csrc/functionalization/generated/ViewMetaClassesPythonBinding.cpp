#include <ATen/ViewMetaClasses.h>
#include <torch/csrc/functionalization/Module.h>

namespace torch::functionalization {

void initGenerated(PyObject* module) {
  auto functionalization = py::handle(module).cast<py::module>();
    create_binding_with_pickle<at::functionalization::_fw_primal_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::_make_dual_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::view_as_real_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::view_as_complex_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::_conj_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::_neg_view_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::as_strided_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::as_strided__ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::_sparse_broadcast_to_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::diagonal_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::expand_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::permute_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::_reshape_alias_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::select_int_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::detach_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::detach__ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::slice_Tensor_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::slice_inverse_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::split_Tensor_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::split_with_sizes_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::squeeze_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::squeeze__ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::squeeze_dim_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::squeeze__dim_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::squeeze_dims_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::squeeze__dims_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::t_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::t__ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::transpose_int_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::transpose__ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::_nested_view_from_buffer_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::_nested_view_from_jagged_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::_nested_get_values_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::unsqueeze_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::unsqueeze__ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::_indices_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::_values_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::indices_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::values_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::crow_indices_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::col_indices_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::ccol_indices_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::row_indices_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::unbind_int_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::lift_fresh_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::view_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::view_dtype_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::unfold_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::alias_ViewMeta>(functionalization);
    create_binding_with_pickle<at::functionalization::_test_autograd_multiple_dispatch_view_ViewMeta>(functionalization);
}

} // namespace torch::functionalization
