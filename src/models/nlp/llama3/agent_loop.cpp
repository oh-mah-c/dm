#include "agent_loop.h"
#include <iostream>

namespace ohm {
namespace nlp {

AgentLoop::AgentLoop(const std::string& model_path) {
    std::cout << "[AgentLoop] Initializing Core Cognitive Engine...\n";
    weight_loader_ = std::make_unique<GGUFLoader>(model_path);
    model_weights_ = weight_loader_->load();
    replay_buffer_ = std::make_unique<ExperienceReplay>(10000); // 10K transition capacity
    
    // Phase 6: Initialize Neural Math Layers
    // Using a Tiny Transformer core for the RL loop to save memory, 
    // StateEncoder and ActionHead will be dynamically sized when physics attaches!
    core_llm_ = dm::models::nlp::make_llama3_tiny();
    
    std::cout << "[AgentLoop] Weights injected. Total tensors: " << model_weights_.size() << "\n";
}

AgentLoop::~AgentLoop() = default;

void AgentLoop::attach_physics(mjModel* m, mjData* d) {
    m_ = m;
    d_ = d;
    std::cout << "[AgentLoop] Bifrost Bridge connected. Agent has senses & actuators.\n";
    
    // Initialize Encoders based on MuJoCo dimensions
    state_encoder_ = StateEncoder(m_->nq, core_llm_->cfg.dim);
    action_head_   = ActionHead(core_llm_->cfg.dim, m_->nu);
    
    // Move to eval mode for inference
    core_llm_->eval();
    state_encoder_->eval();
    action_head_->eval();
    
    state_ = AgentState::OBSERVING;
}

void AgentLoop::step() {
    if (!m_ || !d_) {
        std::cerr << "[AgentLoop] Cannot step: Physics not attached.\n";
        return;
    }

    switch (state_) {
        case AgentState::OBSERVING: {
            // Read generalized coordinates (qpos) representing agent's joint states
            at::Tensor senses = ohm::bridge::BifrostTensor::from_mujoco_qpos(m_, d_);
            prev_state_ = senses.clone();
            
            // Transition to THINKING
            state_ = AgentState::THINKING;
            break;
        }
        case AgentState::THINKING: {
            torch::NoGradGuard no_grad; // Disable gradient tracking in forward loop
            
            // 1. Convert senses [1, nq] -> Llama3 embeddings [1, 1, dim]
            auto state_input = prev_state_.unsqueeze(0).to(torch::kFloat32); // [1, nq]
            auto embeds = state_encoder_->forward(state_input).unsqueeze(1); // [1, 1, dim]
            
            // 2. LLaMA3 Transformer block (Frozen feature extractor)
            auto hidden = core_llm_->forward_embeds(embeds); // [1, 1, dim]
            
            // 3. Action Head translates hidden state to motor commands [1, nu]
            auto action = action_head_->forward(hidden.squeeze(1)); // [1, nu]
            
            prev_action_ = action.squeeze(0).clone(); // Save for telemetry
            
            // Transition to ACTING
            state_ = AgentState::ACTING;
            break;
        }
        case AgentState::ACTING: {
            // Retrieve actuators tensor to send commands
            at::Tensor controls = ohm::bridge::BifrostTensor::from_mujoco_ctrl(m_, d_);
            
            // Write impulses from THINKING into `controls` tensor (zero-copy sync to MuJoCo)
            controls.copy_(prev_action_);
            
            // Transition to LEARNING
            state_ = AgentState::LEARNING;
            break;
        }
        case AgentState::LEARNING: {
            // Evaluate reward function from physics state (Phase 6)
            // Goal: Stand up and walk forward
            float reward = 0.0f; 
            
            // 1. Z-axis Height Reward (Standing up)
            // Assuming the first 3 coordinates of qpos are usually x, y, z of the free root joint (e.g. torso)
            if (m_->nq >= 3) {
                float z_height = d_->qpos[2];
                reward += (z_height * 10.0f); // Positive reward for higher Z
            }
            
            // 2. X-axis Velocity Reward (Walking forward)
            if (m_->nv >= 3) {
                float x_vel = d_->qvel[0];
                reward += (x_vel * 5.0f); // Positive reward for moving forward
            }
            
            // 3. Energy Penalty (Avoid chaotic thrashing)
            float energy = prev_action_.pow(2).sum().item<float>();
            reward -= (energy * 0.1f);
            
            // Push real-time telemetry to Shared Memory Replay Buffer
            if (prev_state_.defined() && prev_action_.defined()) {
                at::Tensor current_state = ohm::bridge::BifrostTensor::from_mujoco_qpos(m_, d_);
                bool done = false; // Add custom logic if robot falls below certain height
                if (m_->nq >= 3 && d_->qpos[2] < 0.2) done = true;
                
                Transition t = {prev_state_, prev_action_, reward, current_state, done};
                replay_buffer_->push(t);
            }
            
            // Loop back to OBSERVE
            state_ = AgentState::OBSERVING;
            break;
        }
        default:
            break;
    }
}

} // namespace nlp
} // namespace ohm
