#pragma once

#include <pthreadpool.h>
#include <exception>
#include <type_traits>

#include <cstddef>
#include <utility>

namespace mruntime {

// Move-only RAII wrapper for the C pthreadpool library.
//
// Notes:
// - If creation fails, the wrapper is empty (get() == nullptr).
// - For an empty wrapper, threads_count() returns 1 to match the common
//   pthreadpool convention of executing work serially on the calling thread
//   when threadpool == NULL.
class PThreadPool {
  public:
    PThreadPool() = default;

    static PThreadPool Create(size_t threads_count) noexcept {
        return PThreadPool(pthreadpool_create(threads_count));
    }

    ~PThreadPool() {
        reset();
    }

    PThreadPool(const PThreadPool&) = delete;
    PThreadPool& operator=(const PThreadPool&) = delete;

    PThreadPool(PThreadPool&& other) noexcept : pool_(std::exchange(other.pool_, nullptr)) {}

    PThreadPool& operator=(PThreadPool&& other) noexcept {
        if (this != &other) {
            reset(std::exchange(other.pool_, nullptr));
        }
        return *this;
    }

    void reset() noexcept {
        reset(nullptr);
    }

    void reset(pthreadpool_t new_pool) noexcept {
        if (new_pool == pool_) return;
        if (pool_ != nullptr) {
            pthreadpool_destroy(pool_);
        }
        pool_ = new_pool;
    }

    pthreadpool_t get() const noexcept {
        return pool_;
    }

    explicit operator bool() const noexcept {
        return pool_ != nullptr;
    }

    size_t threads_count() const noexcept {
        return pool_ != nullptr ? pthreadpool_get_threads_count(pool_) : size_t{1};
    }

    size_t set_threads_count(size_t num_threads) noexcept {
        return pool_ != nullptr ? pthreadpool_set_threads_count(pool_, num_threads) : size_t{1};
    }

    void parallelize_1d(
        pthreadpool_task_1d_t func,
        void* context,
        size_t range,
        uint32_t flags = 0) const noexcept
    {
        pthreadpool_parallelize_1d(pool_, func, context, range, flags);
    }

    template <class F>
    void parallelize_1d(size_t range, F& f, uint32_t flags = 0) const {
        static_assert(std::is_invocable_v<F&, size_t>, "f must be callable as f(size_t)");
        pthreadpool_parallelize_1d(pool_, &invoke_1d<F>, &f, range, flags);
    }

  private:
    explicit PThreadPool(pthreadpool_t pool) noexcept : pool_(pool) {}

    template <class F>
    static void invoke_1d(void* ctx, size_t i) noexcept {
        auto& f = *static_cast<F*>(ctx);
        if constexpr (noexcept(f(i))) {
        f(i);
        } else {
        // Don’t let exceptions cross the C callback boundary.
        try { f(i); } catch (...) { std::terminate(); }
        }
    }

    pthreadpool_t pool_ = nullptr;
};

} // namespace mruntime
