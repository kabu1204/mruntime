#pragma once

#include <chrono>
#include <cstdint>
#include <string_view>
#include <vector>
#include <mutex>
#include <thread>
#include <fstream>

namespace mruntime {

// Trace event types
enum class TraceEventType : uint8_t {
    Begin,      // B - duration begin
    End,        // E - duration end
    Complete,   // X - complete event (has duration)
    Instant,    // i - instant event
    Counter,    // C - counter event
};

// Single trace event (Chrome Tracing format compatible)
struct TraceEvent {
    const char* name;           // Event name (must be string literal or persistent)
    const char* category;       // Category (e.g., "gemm", "attention", "mlp")
    int64_t timestamp_us;       // Microseconds since trace start
    int64_t duration_us;        // Duration for Complete events
    uint32_t thread_id;
    TraceEventType type;
    int64_t counter_value;      // For Counter events
};

// Global trace collector
class TraceCollector {
public:
    static TraceCollector& instance() {
        static TraceCollector inst;
        return inst;
    }

    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
        start_time_ = std::chrono::steady_clock::now();
    }

    void add_event(TraceEvent event) {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(event);
    }

    int64_t now_us() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            now - start_time_).count();
    }

    static uint32_t current_thread_id() {
        return static_cast<uint32_t>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFFFFF);
    }

    // Export to Chrome Tracing JSON format (chrome://tracing)
    bool export_chrome_json(const char* filename) const {
        std::ofstream out(filename);
        if (!out) return false;

        out << "{\n\"traceEvents\": [\n";
        bool first = true;
        for (const auto& e : events_) {
            if (!first) out << ",\n";
            first = false;

            out << "{\"name\":\"" << e.name << "\""
                << ",\"cat\":\"" << (e.category ? e.category : "default") << "\""
                << ",\"ph\":\"" << event_type_char(e.type) << "\""
                << ",\"ts\":" << e.timestamp_us
                << ",\"pid\":1"
                << ",\"tid\":" << e.thread_id;

            if (e.type == TraceEventType::Complete) {
                out << ",\"dur\":" << e.duration_us;
            }
            if (e.type == TraceEventType::Counter) {
                out << ",\"args\":{\"value\":" << e.counter_value << "}";
            }
            out << "}";
        }
        out << "\n]\n}\n";
        return true;
    }

    // Print summary statistics
    void print_summary() const;

    const std::vector<TraceEvent>& events() const { return events_; }

private:
    TraceCollector() : start_time_(std::chrono::steady_clock::now()) {}

    static char event_type_char(TraceEventType type) {
        switch (type) {
            case TraceEventType::Begin: return 'B';
            case TraceEventType::End: return 'E';
            case TraceEventType::Complete: return 'X';
            case TraceEventType::Instant: return 'i';
            case TraceEventType::Counter: return 'C';
        }
        return '?';
    }

    bool enabled_ = false;
    std::chrono::steady_clock::time_point start_time_;
    std::vector<TraceEvent> events_;
    mutable std::mutex mutex_;
};

// RAII scoped trace - records Complete event on destruction
class ScopedTrace {
public:
    ScopedTrace(const char* name, const char* category = nullptr)
        : name_(name), category_(category), enabled_(TraceCollector::instance().is_enabled()) {
        if (enabled_) {
            start_us_ = TraceCollector::instance().now_us();
            thread_id_ = TraceCollector::instance().current_thread_id();
        }
    }

    ~ScopedTrace() {
        if (enabled_) {
            int64_t end_us = TraceCollector::instance().now_us();
            TraceCollector::instance().add_event({
                .name = name_,
                .category = category_,
                .timestamp_us = start_us_,
                .duration_us = end_us - start_us_,
                .thread_id = thread_id_,
                .type = TraceEventType::Complete,
                .counter_value = 0
            });
        }
    }

    // Non-copyable, non-movable
    ScopedTrace(const ScopedTrace&) = delete;
    ScopedTrace& operator=(const ScopedTrace&) = delete;

private:
    const char* name_;
    const char* category_;
    int64_t start_us_ = 0;
    uint32_t thread_id_ = 0;
    bool enabled_;
};

// Manual begin/end tracing
inline void trace_begin(const char* name, const char* category = nullptr) {
    if (!TraceCollector::instance().is_enabled()) return;
    TraceCollector::instance().add_event({
        .name = name,
        .category = category,
        .timestamp_us = TraceCollector::instance().now_us(),
        .duration_us = 0,
        .thread_id = TraceCollector::instance().current_thread_id(),
        .type = TraceEventType::Begin,
        .counter_value = 0
    });
}

inline void trace_end(const char* name, const char* category = nullptr) {
    if (!TraceCollector::instance().is_enabled()) return;
    TraceCollector::instance().add_event({
        .name = name,
        .category = category,
        .timestamp_us = TraceCollector::instance().now_us(),
        .duration_us = 0,
        .thread_id = TraceCollector::instance().current_thread_id(),
        .type = TraceEventType::End,
        .counter_value = 0
    });
}

inline void trace_instant(const char* name, const char* category = nullptr) {
    if (!TraceCollector::instance().is_enabled()) return;
    TraceCollector::instance().add_event({
        .name = name,
        .category = category,
        .timestamp_us = TraceCollector::instance().now_us(),
        .duration_us = 0,
        .thread_id = TraceCollector::instance().current_thread_id(),
        .type = TraceEventType::Instant,
        .counter_value = 0
    });
}

inline void trace_counter(const char* name, int64_t value, const char* category = nullptr) {
    if (!TraceCollector::instance().is_enabled()) return;
    TraceCollector::instance().add_event({
        .name = name,
        .category = category,
        .timestamp_us = TraceCollector::instance().now_us(),
        .duration_us = 0,
        .thread_id = TraceCollector::instance().current_thread_id(),
        .type = TraceEventType::Counter,
        .counter_value = value
    });
}

}  // namespace mruntime

// Convenience macros - can be disabled at compile time
#ifndef MRUNTIME_TRACE_DISABLED

#define TRACE_SCOPE(name) \
    ::mruntime::ScopedTrace _trace_scope_##__LINE__(name)

#define TRACE_SCOPE_CAT(name, category) \
    ::mruntime::ScopedTrace _trace_scope_##__LINE__(name, category)

#define TRACE_BEGIN(name) ::mruntime::trace_begin(name)
#define TRACE_END(name) ::mruntime::trace_end(name)
#define TRACE_BEGIN_CAT(name, cat) ::mruntime::trace_begin(name, cat)
#define TRACE_END_CAT(name, cat) ::mruntime::trace_end(name, cat)
#define TRACE_INSTANT(name) ::mruntime::trace_instant(name)
#define TRACE_COUNTER(name, value) ::mruntime::trace_counter(name, value)

#else

#define TRACE_SCOPE(name) ((void)0)
#define TRACE_SCOPE_CAT(name, category) ((void)0)
#define TRACE_BEGIN(name) ((void)0)
#define TRACE_END(name) ((void)0)
#define TRACE_BEGIN_CAT(name, cat) ((void)0)
#define TRACE_END_CAT(name, cat) ((void)0)
#define TRACE_INSTANT(name) ((void)0)
#define TRACE_COUNTER(name, value) ((void)0)

#endif
