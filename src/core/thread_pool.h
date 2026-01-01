#pragma once

#include "sync.h"

#include <memory>
#include <type_traits>
#include <vector>

namespace mruntime {

// A workload for a thread.
struct Task {
    virtual ~Task() = default;
    virtual void Run() = 0;
};

class Thread;

// A simple pool of threads, that only allows the very
// specific parallelization pattern that we use here:
// One thread, which we call the 'main thread', calls Execute, distributing
// a Task each to N threads, being N-1 'worker threads' and the main thread
// itself. After the main thread has completed its own Task, it waits for
// the worker threads to have all completed. That is the only synchronization
// performed by this ThreadPool.
//
// In particular, there is a naive 1:1 mapping of Tasks to threads.
// This ThreadPool considers it outside of its own scope to try to work
// with fewer threads than there are Tasks. The idea is that such N:M mappings
// of tasks to threads can be implemented as a higher-level feature on top of
// the present low-level 1:1 threadpool. For example, a user might have a
// Task subclass referencing a shared atomic counter indexing into a vector of
// finer-granularity subtasks. Different threads would then concurrently
// increment this atomic counter, getting each their own subtasks to work on.
class ThreadPool {
  public:
    ThreadPool();
    ~ThreadPool();

    // Executes task_count tasks on task_count threads.
    // Grows the threadpool as needed to have at least (task_count-1) threads.
    // The 0-th task is run on the thread on which Execute is called: that
    // is by definition what we call the "main thread". Synchronization of all
    // threads is performed before this function returns.
    //
    // As explained in the class comment, there is a 1:1 mapping of tasks to
    // threads. If you need something smarter than that, for instance if you
    // want to run an unbounded number of tasks on a bounded number of threads,
    // then you need something higher-level than this ThreadPool, that can
    // be layered on top of it by appropriately subclassing Tasks.
    //
    // TaskType must be a subclass of mruntime::Task. That is implicitly guarded by
    // the static_cast in this inline implementation.
    template <typename TaskType>
    void Execute(int task_count, TaskType* tasks) {
        static_assert(std::is_base_of<Task, TaskType>::value,
                      "TaskType must derive from mruntime::Task");
        ExecuteImpl(task_count, sizeof(TaskType), static_cast<Task*>(tasks));
    }

    void set_spin_milliseconds(float milliseconds) {
        spin_duration_ = DurationFromMilliseconds(milliseconds);
    }

    float spin_milliseconds() const {
        return ToFloatMilliseconds(spin_duration_);
    }

  private:
    // Ensures that the pool has at least the given count of threads.
    // If any new thread has to be created, this function waits for it to
    // be ready.
    void CreateThreads(int threads_count);

    // Non-templatized implementation of the public Execute method.
    // See the inline implementation of Execute for how this is used.
    void ExecuteImpl(int task_count, int stride, Task* tasks);

    // copy construction disallowed
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    // The BlockingCounter used to wait for the threads.
    BlockingCounter count_busy_threads_;

    // Spin duration for worker-thread waits.
    Duration spin_duration_ = DurationFromMilliseconds(2);

    // The worker threads in this pool. They are owned by the pool.
    std::vector<std::unique_ptr<Thread>> threads_;
};

} // namespace mruntime


