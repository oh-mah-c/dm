#include "experience_replay.h"
#include <algorithm>
#include <random>

namespace ohm {
namespace nlp {

ExperienceReplay::ExperienceReplay(size_t capacity) : capacity_(capacity) {
    buffer_.reserve(capacity);
}

void ExperienceReplay::push(const Transition& transition) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Clone tensors to detach them from the zero-copy Bifrost buffer.
    // This ensures historical states aren't overwritten when MuJoCo advances.
    Transition cloned_transition = {
        transition.state.clone(),
        transition.action.clone(),
        transition.reward,
        transition.next_state.clone(),
        transition.done
    };

    if (buffer_.size() < capacity_) {
        buffer_.push_back(cloned_transition);
    } else {
        buffer_[head_] = cloned_transition;
        head_ = (head_ + 1) % capacity_;
    }
}

std::vector<Transition> ExperienceReplay::sample(size_t batch_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<Transition> batch;
    size_t current_size = buffer_.size();
    if (current_size == 0) return batch;
    
    size_t actual_batch_size = std::min(batch_size, current_size);
    batch.reserve(actual_batch_size);
    
    // Random sampling without replacement
    std::vector<size_t> indices(current_size);
    std::iota(indices.begin(), indices.end(), 0);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    for (size_t i = 0; i < actual_batch_size; ++i) {
        batch.push_back(buffer_[indices[i]]);
    }
    
    return batch;
}

size_t ExperienceReplay::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size();
}

} // namespace nlp
} // namespace ohm
