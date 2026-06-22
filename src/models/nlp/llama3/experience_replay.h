#pragma once
#include <torch/torch.h>
#include <vector>
#include <mutex>

namespace ohm {
namespace nlp {

// Represents a single step in the simulation for Ohm-Miner RL
struct Transition {
    at::Tensor state;
    at::Tensor action;
    float reward;
    at::Tensor next_state;
    bool done;
};

class ExperienceReplay {
public:
    ExperienceReplay(size_t capacity);
    ~ExperienceReplay() = default;

    // Thread-safe push for real-time telemetry from the Agent Loop
    void push(const Transition& transition);

    // Sample a batch of transitions for training
    std::vector<Transition> sample(size_t batch_size);

    size_t size() const;

private:
    size_t capacity_;
    size_t head_ = 0;
    std::vector<Transition> buffer_;
    mutable std::mutex mutex_;
};

} // namespace nlp
} // namespace ohm
