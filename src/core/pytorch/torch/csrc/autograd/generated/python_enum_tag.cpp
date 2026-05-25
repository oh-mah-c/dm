#include <torch/csrc/autograd/python_enum_tag.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <ATen/core/enum_tag.h>

namespace py = pybind11;
namespace torch {
    namespace autograd {
    void initEnumTag(PyObject* module) {
        auto m = py::handle(module).cast<py::module>();
        py::enum_<at::Tag>(m, "Tag")

        .value("core", at::Tag::core)
        .value("cudagraph_unsafe", at::Tag::cudagraph_unsafe)
        .value("data_dependent_output", at::Tag::data_dependent_output)
        .value("dynamic_output_shape", at::Tag::dynamic_output_shape)
        .value("flexible_layout", at::Tag::flexible_layout)
        .value("generated", at::Tag::generated)
        .value("inplace", at::Tag::inplace)
        .value("inplace_view", at::Tag::inplace_view)
        .value("maybe_aliasing_or_mutating", at::Tag::maybe_aliasing_or_mutating)
        .value("needs_contiguous_strides", at::Tag::needs_contiguous_strides)
        .value("needs_exact_strides", at::Tag::needs_exact_strides)
        .value("needs_fixed_stride_order", at::Tag::needs_fixed_stride_order)
        .value("nondeterministic_bitwise", at::Tag::nondeterministic_bitwise)
        .value("nondeterministic_seeded", at::Tag::nondeterministic_seeded)
        .value("out", at::Tag::out)
        .value("out_variant", at::Tag::out_variant)
        .value("pointwise", at::Tag::pointwise)
        .value("pt2_compliant_tag", at::Tag::pt2_compliant_tag)
        .value("reduction", at::Tag::reduction)
        .value("view_copy", at::Tag::view_copy);
        m.doc() = "An Enum that contains tags that can be assigned to an operator registered in C++.";
    }
}}
