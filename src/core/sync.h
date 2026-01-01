#pragma once

#include "timer.h"

#include <atomic>
#include <condition_variable>
#include <mutex>

namespace mruntime {

// Waits until `condition()` becomes true.
//
// First, attempts busy-waiting for `spin_duration` (if non-zero), then falls
// back to passive waiting on `condvar`.
template <typename Predicate>
void Wait(Predicate condition,
          const Duration& spin_duration,
          std::condition_variable& condvar,
          std::mutex& mutex) {
    // First, trivial case where the `condition` is already true.
    if (condition()) {
        return;
    }

    // Then, if spin_duration is nonzero, try busy-waiting.
    if (spin_duration != Duration::zero()) {
        const TimePoint wait_end = Now() + spin_duration;
        while (Now() < wait_end) {
            if (condition()) {
                return;
            }
        }
    }

    // Finally, do real passive waiting.
    std::unique_lock<std::mutex> lock(mutex);
    condvar.wait(lock, condition);
}

// A BlockingCounter lets one thread wait for N events to occur.
// This is how the master thread waits for all the worker threads
// to have finished working.
class BlockingCounter {
  public:
    BlockingCounter();

    BlockingCounter(const BlockingCounter&) = delete;
    BlockingCounter& operator=(const BlockingCounter&) = delete;
    BlockingCounter(BlockingCounter&&) = delete;
    BlockingCounter& operator=(BlockingCounter&&) = delete;

    // Sets/resets the counter; initial_count is the number of
    // decrementing events that the Wait() call will be waiting for.
    void Reset(int initial_count);

    // Decrements the counter; if the counter hits zero, signals
    // the threads that were waiting for that, and returns true.
    // Otherwise (if the decremented count is still nonzero),
    // returns false.
    bool DecrementCount();

    // Waits for the N other threads (N having been set by Reset())
    // to hit the BlockingCounter.
    //
    // Will first spin-wait for `spin_duration` before reverting to passive
    // wait.
    void Wait(Duration spin_duration);

  private:
    std::atomic<int> count_;
    std::condition_variable count_cond_;
    std::mutex count_mutex_;
};

}  // namespace mruntime


