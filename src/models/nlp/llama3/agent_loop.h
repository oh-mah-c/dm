#pragma once

#include "weight_loader.h"
#include "experience_replay.h"
#include "../../../core/bridge/ohm_bifrost_tensor.h"
#include "models/nlp/llama3/llama3.h"
#include "embodied_head.h"
#include <mujoco/mujoco.h>
#include <memory>
#include <unordered_map>
#include <string>

namespace ohm {
namespace nlp {

enum class AgentState {
    IDLE,
    OBSERVING,
    THINKING,
    ACTING,
    LEARNING
};

class AgentLoop {
public:
    AgentLoop(const std::string& model_path);
    ~AgentLoop();

    // Attach MuJoCo context to the agent via Bifrost Bridge
    void attach_physics(mjModel* m, mjData* d);

    // Run one iteration of the cognitive loop
    void step();

    // Get current state
    AgentState get_state() const { return state_; }
    
    // Get Replay Buffer for RL Trainer Thread
    ExperienceReplay* get_replay_buffer() const { return replay_buffer_.get(); }
    
    // Get Neural Modules for the RL Trainer
    dm::models::nlp::Llama3Model& get_llm() { return core_llm_; }
    StateEncoder& get_state_encoder() { return state_encoder_; }
    ActionHead& get_action_head() { return action_head_; }

private:
    AgentState state_ = AgentState::IDLE;
    
    // Core AI (Memory & Processing)
    std::unique_ptr<GGUFLoader> weight_loader_;
    std::unordered_map<std::string, at::Tensor> model_weights_;
    
    // Telemetry Shared Memory
    std::unique_ptr<ExperienceReplay> replay_buffer_;
    at::Tensor prev_state_;
    at::Tensor prev_action_;
    
    // Mathematical Neural Integration (Phase 6)
    dm::models::nlp::Llama3Model core_llm_{nullptr};
    StateEncoder state_encoder_{nullptr};
    ActionHead action_head_{nullptr};
    
    // Physics Bridge (Senses & Actuators)
    mjModel* m_ = nullptr;
    mjData* d_  = nullptr;
};

} // namespace nlp
} // namespace ohm
