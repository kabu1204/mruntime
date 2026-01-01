#include "sync.h"

#include <stdexcept>

namespace mruntime {

BlockingCounter::BlockingCounter() : count_(0) {}

void BlockingCounter::Reset(int initial_count) {
    int old_count_value = count_.load(std::memory_order_relaxed);
    if (old_count_value != 0) {
        throw std::runtime_error("BlockingCounter: Reset called with non-zero initial count");
    }
    (void)old_count_value;
    count_.store(initial_count, std::memory_order_release);
}

bool BlockingCounter::DecrementCount() {
    int old_count_value = count_.fetch_sub(1, std::memory_order_acq_rel);
    if (old_count_value <= 0) {
        throw std::runtime_error("BlockingCounter: DecrementCount called with non-positive count");
    }
    int count_value = old_count_value - 1;
    bool hit_zero = (count_value == 0);
    if (hit_zero) {
        count_cond_.notify_all();
    }
    return hit_zero;
}

void BlockingCounter::Wait(Duration spin_duration) {
    const auto& condition = [this]() {
        return count_.load(std::memory_order_acquire) == 0;
    };
    ::mruntime::Wait(condition, spin_duration, count_cond_, count_mutex_);
}

}  // namespace mruntime


