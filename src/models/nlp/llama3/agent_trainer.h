#pragma once
#include "models/nlp/llama3/experience_replay.h"
#include "models/nlp/llama3/llama3.h"
#include "embodied_head.h"
#include <thread>
#include <atomic>

namespace ohm {
namespace nlp {

class AgentTrainer {
public:
    AgentTrainer(ExperienceReplay* replay_buffer, 
                 dm::models::nlp::Llama3Model llm, 
                 StateEncoder state_encoder, 
                 ActionHead action_head);
    ~AgentTrainer();

    void start();
    void stop();

private:
    void training_loop();

    ExperienceReplay* replay_buffer_;
    std::thread trainer_thread_;
    std::atomic<bool> running_{false};
    
    dm::models::nlp::Llama3Model llm_;
    StateEncoder state_encoder_;
    ActionHead action_head_;
};

} // namespace nlp
} // namespace ohm
