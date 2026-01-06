#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

#include "mruntime/pthreadpool_raii.h"

using namespace mruntime;

void test_create_move_reset() {
    PThreadPool a = PThreadPool::Create(0);
    assert(a.threads_count() >= 1);

    const pthreadpool_t raw = a.get();
    // Creation can fail (NULL) on allocation failure; the wrapper should stay safe.
    if (raw != nullptr) {
        assert(static_cast<bool>(a));
    } else {
        assert(!static_cast<bool>(a));
    }

    PThreadPool b = std::move(a);
    assert(a.get() == nullptr);
    assert(b.get() == raw);

    PThreadPool c;
    assert(c.get() == nullptr);
    assert(c.threads_count() == 1);
    c = std::move(b);
    assert(b.get() == nullptr);
    assert(c.get() == raw);

    c.reset();
    assert(c.get() == nullptr);
    assert(c.threads_count() == 1);

    std::cout << "test_create_move_reset PASSED\n";
}

void test_parallelize_1d_elementwise_add() {
    PThreadPool pool = PThreadPool::Create(4);

    constexpr size_t n = 1024;
    std::vector<float> a(n);
    std::vector<float> b(n);
    std::vector<float> out(n, 0.0f);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    auto add = [&](size_t i) { out[i] = a[i] + b[i]; };
    pool.parallelize_1d(n, add);

    for (size_t i = 0; i < n; ++i) {
        assert(out[i] == a[i] + b[i]);
    }

    std::cout << "test_parallelize_1d_elementwise_add PASSED\n";
}

int main() {
    test_create_move_reset();
    test_parallelize_1d_elementwise_add();
    std::cout << "\nAll pthreadpool wrapper tests passed!\n";
    return 0;
}
