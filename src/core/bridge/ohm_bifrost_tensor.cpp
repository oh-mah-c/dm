#include "ohm_bifrost_tensor.h"
#include <cstring>

namespace ohm {
namespace bridge {

at::Tensor BifrostTensor::from_mujoco_qpos(mjModel* m, mjData* d) {
    // mjData->qpos is an array of size m->nq of type mjtNum (usually double)
    // We use from_blob to map it directly without allocating new memory!
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    return torch::from_blob(d->qpos, {m->nq}, options);
}

at::Tensor BifrostTensor::from_mujoco_qvel(mjModel* m, mjData* d) {
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    return torch::from_blob(d->qvel, {m->nv}, options);
}

at::Tensor BifrostTensor::from_mujoco_ctrl(mjModel* m, mjData* d) {
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    return torch::from_blob(d->ctrl, {m->nu}, options);
}

void BifrostTensor::to_mujoco_ctrl(const at::Tensor& t, mjModel* m, mjData* d) {
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(t.dtype() == torch::kFloat64, "Tensor must be Float64");
    TORCH_CHECK(t.numel() == m->nu, "Tensor size must match MuJoCo model's nu");
    
    // Copy safely to MuJoCo's control buffer
    std::memcpy(d->ctrl, t.data_ptr<double>(), m->nu * sizeof(double));
}

} // namespace bridge
} // namespace ohm
