#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <initializer_list>
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

struct TraceArg {
    const char* key;    // must be string literal or persistent
    int64_t value;
};

inline constexpr size_t kMaxTraceArgs = 8;

inline TraceArg trace_arg(const char* key, int64_t value) {
    return {.key = key, .value = value};
}

// Single trace event (Chrome Tracing format compatible)
struct TraceEvent {
    const char* name;           // Event name (must be string literal or persistent)
    const char* category;       // Category (e.g., "gemm", "attention", "mlp")
    int64_t timestamp_us;       // Microseconds since trace start
    int64_t duration_us;        // Duration for Complete events
    uint32_t thread_id;
    uint64_t id;                // Unique scope id for Complete events from ScopedTrace
    uint64_t parent_id;         // Parent scope id (0 when none/unknown)
    TraceEventType type;
    int64_t counter_value;      // For Counter events
    uint8_t args_count = 0;
    TraceArg args[kMaxTraceArgs]{};
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
        next_id_.store(1, std::memory_order_relaxed);
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
            if (e.type == TraceEventType::Counter || e.id != 0 || e.parent_id != 0 || e.args_count != 0) {
                out << ",\"args\":{";
                bool first_arg = true;
                if (e.type == TraceEventType::Counter) {
                    out << "\"value\":" << e.counter_value;
                    first_arg = false;
                }
                if (e.id != 0 || e.parent_id != 0) {
                    if (!first_arg) out << ",";
                    out << "\"id\":" << e.id
                        << ",\"parent_id\":" << e.parent_id;
                    first_arg = false;
                }
                for (uint8_t i = 0; i < e.args_count && i < kMaxTraceArgs; ++i) {
                    const TraceArg& a = e.args[i];
                    if (a.key == nullptr) continue;
                    const std::string_view key(a.key);
                    if (key == "value" || key == "id" || key == "parent_id") continue;
                    if (!first_arg) out << ",";
                    out << "\"" << a.key << "\":" << a.value;
                    first_arg = false;
                }
                out << "}";
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

    friend class ScopedTrace;
    uint64_t alloc_id() { return next_id_.fetch_add(1, std::memory_order_relaxed); }

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
    std::atomic<uint64_t> next_id_{1};
};

namespace detail {

inline std::vector<uint64_t>& trace_stack() {
    static thread_local std::vector<uint64_t> stack;
    return stack;
}

}  // namespace detail

// RAII scoped trace - records Complete event on destruction
class ScopedTrace {
public:
    ScopedTrace(
        const char* name,
        const char* category = nullptr,
        std::initializer_list<TraceArg> args = {}
    ) : name_(name), category_(category), enabled_(TraceCollector::instance().is_enabled()) {
        if (enabled_) {
            for (const auto& a : args) {
                if (args_count_ < kMaxTraceArgs) {
                    args_[args_count_++] = a;
                }
            }

            auto& stack = detail::trace_stack();
            id_ = TraceCollector::instance().alloc_id();
            parent_id_ = stack.empty() ? 0 : stack.back();
            stack.push_back(id_);

            start_us_ = TraceCollector::instance().now_us();
            thread_id_ = TraceCollector::instance().current_thread_id();
        }
    }

    ~ScopedTrace() {
        if (enabled_) {
            int64_t end_us = TraceCollector::instance().now_us();
            auto& stack = detail::trace_stack();
            if (!stack.empty() && stack.back() == id_) {
                stack.pop_back();
            } else {
                auto it = std::find(stack.rbegin(), stack.rend(), id_);
                if (it != stack.rend()) {
                    stack.erase(it.base() - 1);
                }
            }

            TraceEvent event{
                .name = name_,
                .category = category_,
                .timestamp_us = start_us_,
                .duration_us = end_us - start_us_,
                .thread_id = thread_id_,
                .id = id_,
                .parent_id = parent_id_,
                .type = TraceEventType::Complete,
                .counter_value = 0,
                .args_count = args_count_,
            };
            for (uint8_t i = 0; i < args_count_ && i < kMaxTraceArgs; ++i) {
                event.args[i] = args_[i];
            }
            TraceCollector::instance().add_event(event);
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
    uint64_t id_ = 0;
    uint64_t parent_id_ = 0;
    uint8_t args_count_ = 0;
    TraceArg args_[kMaxTraceArgs]{};
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
        .id = 0,
        .parent_id = 0,
        .type = TraceEventType::Begin,
        .counter_value = 0,
        .args_count = 0
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
        .id = 0,
        .parent_id = 0,
        .type = TraceEventType::End,
        .counter_value = 0,
        .args_count = 0
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
        .id = 0,
        .parent_id = 0,
        .type = TraceEventType::Instant,
        .counter_value = 0,
        .args_count = 0
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
        .id = 0,
        .parent_id = 0,
        .type = TraceEventType::Counter,
        .counter_value = value,
        .args_count = 0
    });
}

}  // namespace mruntime

// Convenience macros - can be disabled at compile time
#ifndef MRUNTIME_TRACE_DISABLED

#define TRACE_SCOPE(name) \
    ::mruntime::ScopedTrace _trace_scope_##__LINE__(name)

#define TRACE_SCOPE_CAT(name, category) \
    ::mruntime::ScopedTrace _trace_scope_##__LINE__(name, category)

#define TRACE_SCOPE_ARGS_CAT(name, category, ...) \
    ::mruntime::ScopedTrace _trace_scope_##__LINE__(name, category, {__VA_ARGS__})

#define TRACE_BEGIN(name) ::mruntime::trace_begin(name)
#define TRACE_END(name) ::mruntime::trace_end(name)
#define TRACE_BEGIN_CAT(name, cat) ::mruntime::trace_begin(name, cat)
#define TRACE_END_CAT(name, cat) ::mruntime::trace_end(name, cat)
#define TRACE_INSTANT(name) ::mruntime::trace_instant(name)
#define TRACE_COUNTER(name, value) ::mruntime::trace_counter(name, value)

#else

#define TRACE_SCOPE(name) ((void)0)
#define TRACE_SCOPE_CAT(name, category) ((void)0)
#define TRACE_SCOPE_ARGS_CAT(name, category, ...) ((void)0)
#define TRACE_BEGIN(name) ((void)0)
#define TRACE_END(name) ((void)0)
#define TRACE_BEGIN_CAT(name, cat) ((void)0)
#define TRACE_END_CAT(name, cat) ((void)0)
#define TRACE_INSTANT(name) ((void)0)
#define TRACE_COUNTER(name, value) ((void)0)

#endif
