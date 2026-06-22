#pragma once
#include <torch/torch.h>
#include <mujoco/mujoco.h>

namespace ohm {
namespace bridge {

/**
 * @brief Bifrost Bridge: Zero-copy integration between MuJoCo physics data
 * and dm_torch (PyTorch Native) Tensors.
 */
class BifrostTensor {
public:
    // Create a 1D tensor viewing MuJoCo's generalized coordinates (qpos)
    static at::Tensor from_mujoco_qpos(mjModel* m, mjData* d);
    
    // Create a 1D tensor viewing MuJoCo's generalized velocities (qvel)
    static at::Tensor from_mujoco_qvel(mjModel* m, mjData* d);
    
    // Create a 1D tensor viewing MuJoCo's control inputs (ctrl)
    static at::Tensor from_mujoco_ctrl(mjModel* m, mjData* d);
    
    // Safe utility to copy data from a tensor to MuJoCo control array 
    // (If modifying the tensor returned by from_mujoco_ctrl in-place, this is not needed!)
    static void to_mujoco_ctrl(const at::Tensor& t, mjModel* m, mjData* d);
};

} // namespace bridge
} // namespace ohm
