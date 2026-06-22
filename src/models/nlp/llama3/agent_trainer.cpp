#include "agent_trainer.h"
#include <iostream>
#include <chrono>

namespace ohm {
namespace nlp {

AgentTrainer::AgentTrainer(ExperienceReplay* replay_buffer,
                           dm::models::nlp::Llama3Model llm, 
                           StateEncoder state_encoder, 
                           ActionHead action_head) 
    : replay_buffer_(replay_buffer), llm_(llm), state_encoder_(state_encoder), action_head_(action_head) {}

AgentTrainer::~AgentTrainer() {
    stop();
}

void AgentTrainer::start() {
    if (!running_) {
        running_ = true;
        trainer_thread_ = std::thread(&AgentTrainer::training_loop, this);
        std::cout << "[AgentTrainer] RL Background Thread started. Listening for transitions...\n";
    }
}

void AgentTrainer::stop() {
    if (running_) {
        running_ = false;
        if (trainer_thread_.joinable()) {
            trainer_thread_.join();
        }
        std::cout << "\n[AgentTrainer] RL Background Thread stopped.\n";
    }
}

void AgentTrainer::training_loop() {
    size_t batch_size = 32;
    int epochs = 0;
    
    // Set trainable modules to train mode
    state_encoder_->train();
    action_head_->train();
    
    // Gather parameters from Trainable Heads
    std::vector<torch::optim::OptimizerParamGroup> groups;
    std::vector<torch::Tensor> trainable_params;
    
    for (const auto& p : state_encoder_->parameters()) trainable_params.push_back(p);
    for (const auto& p : action_head_->parameters()) trainable_params.push_back(p);
    
    // Create AdamW optimizer for the Action and Encoder heads
    auto opts = std::make_unique<torch::optim::AdamWOptions>(1e-3);
    groups.emplace_back(trainable_params, std::move(opts));
    torch::optim::AdamW optimizer(groups);

    while (running_) {
        // We need the modules to be fully initialized
        if (!state_encoder_->layer1) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        if (replay_buffer_->size() >= batch_size) {
            auto batch = replay_buffer_->sample(batch_size);
            
            optimizer.zero_grad();
            
            float total_reward = 0.0f;
            torch::Tensor total_loss = torch::zeros({1}, torch::kFloat32);
            
            for(const auto& t : batch) {
                total_reward += t.reward;
                
                // 1. Forward Pass again to get computational graph
                auto state_input = t.state.unsqueeze(0).to(torch::kFloat32); // [1, nq]
                auto embeds = state_encoder_->forward(state_input).unsqueeze(1); // [1, 1, dim]
                
                // Freeze LLM during backward
                torch::Tensor hidden;
                {
                    torch::NoGradGuard no_grad;
                    hidden = llm_->forward_embeds(embeds);
                }
                
                auto action_pred = action_head_->forward(hidden.squeeze(1)); // [1, nu]
                
                // 2. Simple REINFORCE Loss: 
                // Normally we'd use log_prob * Advantage. Since action output is deterministic tanh here,
                // we treat the 'target' as the action taken, but we can't do exact REINFORCE without action sampling noise.
                // For simplicity of this Prototype, we just do MSE(action, pred) * Reward
                auto loss = torch::mse_loss(action_pred, t.action.unsqueeze(0)) * -t.reward;
                total_loss += loss;
            }
            
            // 3. Backward Pass
            total_loss = total_loss / batch_size;
            total_loss.backward();
            
            // 4. Optimizer Step
            optimizer.step();
            
            float avg_reward = total_reward / batch_size;
            epochs++;
            
            std::cout << "\r[AgentTrainer] Epoch " << epochs 
                      << " | Loss: " << total_loss.item<float>()
                      << " | Avg Reward: " << avg_reward << "    " << std::flush;
                      
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

} // namespace nlp
} // namespace ohm
