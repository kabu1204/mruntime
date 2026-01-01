#include <algorithm>
#include <atomic>
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

#include "thread_pool.h"

using namespace mruntime;

namespace {

struct IdAndIncTask : Task {
    std::atomic<int>* counter = nullptr;
    int delta = 0;
    std::thread::id* thread_id_out = nullptr;

    void Run() override {
        if (thread_id_out) {
            *thread_id_out = std::this_thread::get_id();
        }
        if (counter) {
            counter->fetch_add(delta, std::memory_order_relaxed);
        }
    }
};

} // namespace

void test_execute_single_task_runs_on_main_thread() {
    ThreadPool pool;
    pool.set_spin_milliseconds(0.0f);

    std::atomic<int> sum{0};
    std::thread::id tid{};

    IdAndIncTask task;
    task.counter = &sum;
    task.delta = 7;
    task.thread_id_out = &tid;

    pool.Execute(1, &task);

    assert(sum.load(std::memory_order_relaxed) == 7);
    assert(tid == std::this_thread::get_id());
    std::cout << "test_execute_single_task_runs_on_main_thread PASSED\n";
}

void test_execute_two_tasks_uses_worker_thread_and_waits() {
    ThreadPool pool;
    pool.set_spin_milliseconds(0.0f);

    std::atomic<int> sum{0};
    std::thread::id tids[2] = {};

    IdAndIncTask tasks[2];
    tasks[0].counter = &sum;
    tasks[0].delta = 1;
    tasks[0].thread_id_out = &tids[0];

    tasks[1].counter = &sum;
    tasks[1].delta = 2;
    tasks[1].thread_id_out = &tids[1];

    pool.Execute(2, tasks);

    assert(sum.load(std::memory_order_relaxed) == 3);
    assert(tids[0] != std::thread::id{});
    assert(tids[1] != std::thread::id{});
    assert(tids[0] != tids[1]);
    std::cout << "test_execute_two_tasks_uses_worker_thread_and_waits PASSED\n";
}

void test_execute_many_tasks_and_reuse_pool() {
    ThreadPool pool;
    pool.set_spin_milliseconds(0.0f);

    // First run: small.
    {
        std::atomic<int> sum{0};
        std::thread::id tids[3] = {};
        IdAndIncTask tasks[3];
        for (int i = 0; i < 3; ++i) {
            tasks[i].counter = &sum;
            tasks[i].delta = 1;
            tasks[i].thread_id_out = &tids[i];
        }
        pool.Execute(3, tasks);
        assert(sum.load(std::memory_order_relaxed) == 3);
        std::vector<std::thread::id> uniq(std::begin(tids), std::end(tids));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        assert(uniq.size() >= 2); // main + at least one worker
    }

    // Second run: larger (forces growth).
    {
        std::atomic<int> sum{0};
        std::thread::id tids[5] = {};
        IdAndIncTask tasks[5];
        for (int i = 0; i < 5; ++i) {
            tasks[i].counter = &sum;
            tasks[i].delta = i;
            tasks[i].thread_id_out = &tids[i];
        }
        pool.Execute(5, tasks);
        assert(sum.load(std::memory_order_relaxed) == (0 + 1 + 2 + 3 + 4));
        std::vector<std::thread::id> uniq(std::begin(tids), std::end(tids));
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        assert(uniq.size() >= 2);
    }

    std::cout << "test_execute_many_tasks_and_reuse_pool PASSED\n";
}

int main() {
    test_execute_single_task_runs_on_main_thread();
    test_execute_two_tasks_uses_worker_thread_and_waits();
    test_execute_many_tasks_and_reuse_pool();

    std::cout << "\nAll thread pool tests passed!\n";
    return 0;
}


