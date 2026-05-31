#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/DispatchStub.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>

namespace at {

// NB: this is explicitly copied here (via codegen) rather than
// included via NativeFunctions.h to avoid recompiling this file when
// NativeFunctions.h changes
namespace meta {
struct TORCH_API structured_add_Tensor : public TensorIteratorBase {


    void meta(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
};
}

namespace native {
struct TORCH_API structured_ufunc_add_CPU : public at::meta::structured_add_Tensor {
void impl(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out);
};

using add_fn = void(*)(TensorIteratorBase&, const at::Scalar &);
DECLARE_DISPATCH(add_fn, add_stub)
DEFINE_DISPATCH(add_stub);

TORCH_IMPL_FUNC(ufunc_add_CPU)(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out) {
  add_stub(device_type(), *this, alpha);
}
}} // namespace at::native
