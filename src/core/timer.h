#pragma once

#include <chrono>
#include <cstdint>
#include <ratio>

#ifdef __linux__
#include <time.h>
#endif

namespace mruntime {

using InternalDefaultClock = std::chrono::steady_clock;

using TimePoint = InternalDefaultClock::time_point;
using Duration = InternalDefaultClock::duration;

template <typename RepresentationType>
Duration DurationFromSeconds(RepresentationType representation) {
    return std::chrono::duration_cast<Duration>(
        std::chrono::duration<RepresentationType>(representation));
}

template <typename RepresentationType>
Duration DurationFromMilliseconds(RepresentationType representation) {
    return std::chrono::duration_cast<Duration>(
        std::chrono::duration<RepresentationType, std::milli>(representation));
}

template <typename RepresentationType>
Duration DurationFromNanoseconds(RepresentationType representation) {
    return std::chrono::duration_cast<Duration>(
        std::chrono::duration<RepresentationType, std::nano>(representation));
}

inline float ToFloatSeconds(const Duration& duration) {
    return std::chrono::duration_cast<std::chrono::duration<float>>(duration)
        .count();
}

inline float ToFloatMilliseconds(const Duration& duration) {
    return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(duration)
        .count();
}

inline std::int64_t ToInt64Nanoseconds(const Duration& duration) {
    return std::chrono::duration_cast<
        std::chrono::duration<std::int64_t, std::nano>>(duration).count();
}

inline TimePoint Now() { return InternalDefaultClock::now(); }

inline TimePoint CoarseNow() {
#ifdef __linux__
    timespec t;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &t);
    return TimePoint(
        DurationFromNanoseconds(1000000000LL * t.tv_sec + t.tv_nsec));
#else
    return Now();
#endif
}

}  // namespace mruntime


