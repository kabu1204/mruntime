#include "thread_pool.h"
#include "mruntime/logging.h"

#include <atomic>
#include <cstddef>
#include <thread>

namespace mruntime {

class Thread {
  public:
    Thread(BlockingCounter* count_busy_threads, const Duration* spin_duration)
        : state_(State::Startup),
          count_busy_threads_(count_busy_threads),
          spin_duration_(spin_duration) {
        thread_ = std::thread(&Thread::ThreadFuncImpl, this);
    }

    ~Thread() {
        RequestExitAsSoonAsPossible();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    Thread(const Thread&) = delete;
    Thread& operator=(const Thread&) = delete;
    Thread(Thread&&) = delete;
    Thread& operator=(Thread&&) = delete;

    void RequestExitAsSoonAsPossible() {
        ChangeStateFromOutsideThread(State::ExitAsSoonAsPossible);
    }

    // Called by the outside thread to give work to the worker thread.
    void StartWork(Task* task) {
        ChangeStateFromOutsideThread(State::HasWork, task);
    }

  private:
    enum class State {
        Startup,  // The initial state before the thread loop runs.
        Ready,    // Is not working, has not yet received new work to do.
        HasWork,  // Has work to do.
        ExitAsSoonAsPossible  // Should exit at earliest convenience.
    };

    // Implements the state_ change to State::Ready, which is where we consume
    // task_. Only called on the worker thread.
    void RevertToReadyState() {
        // See task_ member comment for the ordering of accesses.
        if (task_) {
            task_->Run();
            task_ = nullptr;
        }
        // Relaxed order: task_ ordering is provided by state_ (release/acquire)
        // and by count_busy_threads_ for the between-batch barrier.
        state_.store(State::Ready, std::memory_order_relaxed);
        count_busy_threads_->DecrementCount();
    }

    // Changes State, from outside thread.
    //
    // new_task is only used with State::HasWork.
    void ChangeStateFromOutsideThread(State new_state, Task* new_task = nullptr) {
        // See task_ member comment for the ordering of accesses.
        if (new_state == State::HasWork) {
            task_ = new_task;
        }
        state_.store(new_state, std::memory_order_release);
        state_cond_.notify_one();
    }

    // Waits for state_ to be different from State::Ready, and returns that
    // new value.
    State GetNewStateOtherThanReady() {
        State new_state = State::Ready;
        const auto& new_state_not_ready = [this, &new_state]() {
            new_state = state_.load(std::memory_order_acquire);
            return new_state != State::Ready;
        };
        ::mruntime::Wait(new_state_not_ready, *spin_duration_, state_cond_, state_cond_mutex_);
        return new_state;
    }

    // Thread entry point.
    void ThreadFuncImpl() {
        MRUNTIME_LOG_INFO("ThreadFuncImpl: Starting thread");
        RevertToReadyState();

        // Thread loop
        while (GetNewStateOtherThanReady() == State::HasWork) {
            RevertToReadyState();
        }
    }

    // The task to be worked on.
    //
    // The ordering of reads and writes to task_ is as follows.
    //
    // 1. The outside thread gives new work by calling
    //      ChangeStateFromOutsideThread(State::HasWork, new_task);
    //    That does:
    //    - a. Write task_ = new_task (non-atomic).
    //    - b. Store state_ = State::HasWork (memory_order_release).
    // 2. The worker thread picks up the new state by calling
    //      GetNewStateOtherThanReady()
    //    That does:
    //    - c. Load state (memory_order_acquire).
    //    The worker thread then reads the new task in RevertToReadyState().
    //    That does:
    //    - d. Read task_ (non-atomic).
    // 3. The worker thread, still in RevertToReadyState(), consumes the task_ and
    //    does:
    //    - e. Write task_ = nullptr (non-atomic).
    //    And then calls count_busy_threads_->DecrementCount()
    //    which does
    //    - f. Store count_busy_threads_ (memory_order_release via acq_rel RMW).
    // 4. The outside thread, in ThreadPool::ExecuteImpl, finally waits for worker
    //    threads by calling count_busy_threads_->Wait(), which does:
    //    - g. Load count_busy_threads_ (memory_order_acquire).
    //
    // Thus the non-atomic write-then-read accesses to task_ (a. -> d.) are
    // ordered by the release-acquire relationship of accesses to state_ (b. ->
    // c.), and the non-atomic write accesses to task_ (e. -> a.) are ordered by
    // the release-acquire relationship of accesses to count_busy_threads_ (f. ->
    // g.).
    Task* task_ = nullptr;

    std::condition_variable state_cond_;
    std::mutex state_cond_mutex_;

    std::atomic<State> state_;

    BlockingCounter* const count_busy_threads_;
    const Duration* const spin_duration_;

    std::thread thread_;
};

ThreadPool::ThreadPool() = default;

ThreadPool::~ThreadPool() {
    // Send all exit requests upfront so threads can work on them in parallel.
    for (auto& w : threads_) {
        w->RequestExitAsSoonAsPossible();
    }
    threads_.clear();
}

void ThreadPool::CreateThreads(int threads_count) {
    if (threads_count < 0) {
        return;
    }
    const int current_threads = static_cast<int>(threads_.size());
    if (current_threads >= threads_count) {
        return;
    }

    const int new_threads = threads_count - current_threads;
    count_busy_threads_.Reset(new_threads);
    threads_.reserve(static_cast<size_t>(threads_count));
    while (static_cast<int>(threads_.size()) < threads_count) {
        threads_.push_back(std::make_unique<Thread>(&count_busy_threads_, &spin_duration_));
    }
    count_busy_threads_.Wait(spin_duration_);
}

void ThreadPool::ExecuteImpl(int task_count, int stride, Task* tasks) {
    if (task_count <= 0) {
        return;
    }
    if (tasks == nullptr) {
        return;
    }

    // Single-thread case.
    if (task_count == 1) {
        tasks[0].Run();
        return;
    }

    const int worker_count = task_count - 1;
    CreateThreads(worker_count);

    count_busy_threads_.Reset(worker_count);
    for (int i = 1; i < task_count; ++i) {
        auto* task_address = reinterpret_cast<std::byte*>(tasks) +
                             static_cast<std::ptrdiff_t>(i) * stride;
        threads_[static_cast<size_t>(i - 1)]->StartWork(reinterpret_cast<Task*>(task_address));
    }

    // Run the 0-th task on the main thread.
    tasks[0].Run();

    // Wait for worker threads to finish.
    count_busy_threads_.Wait(spin_duration_);
}

} // namespace mruntime

