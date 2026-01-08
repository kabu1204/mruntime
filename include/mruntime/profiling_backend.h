#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "mruntime/backend.h"

namespace mruntime {

enum class BackendOp : uint8_t {
    Gemm,
    FlashAttention,
    Rmsnorm,
    Rope,
    Silu,
    ElementwiseMul,
    Add,
    Softmax,
    EmbeddingLookup,
    Count,
};

inline const char* backend_op_name(BackendOp op) {
    switch (op) {
        case BackendOp::Gemm: return "gemm";
        case BackendOp::FlashAttention: return "flash_attention";
        case BackendOp::Rmsnorm: return "rmsnorm";
        case BackendOp::Rope: return "rope";
        case BackendOp::Silu: return "silu";
        case BackendOp::ElementwiseMul: return "elementwise_mul";
        case BackendOp::Add: return "add";
        case BackendOp::Softmax: return "softmax";
        case BackendOp::EmbeddingLookup: return "embedding_lookup";
        case BackendOp::Count: return "<count>";
    }
    return "<unknown>";
}

struct BackendOpSnapshot {
    uint64_t calls = 0;
    uint64_t total_ns = 0;
    uint64_t min_ns = 0;
    uint64_t max_ns = 0;
};

namespace detail {

inline void atomic_update_max(std::atomic<uint64_t>& target, uint64_t value) {
    uint64_t prev = target.load(std::memory_order_relaxed);
    while (value > prev && !target.compare_exchange_weak(prev, value, std::memory_order_relaxed)) {
    }
}

inline void atomic_update_min(std::atomic<uint64_t>& target, uint64_t value) {
    uint64_t prev = target.load(std::memory_order_relaxed);
    while (value < prev && !target.compare_exchange_weak(prev, value, std::memory_order_relaxed)) {
    }
}

inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::FP16: return "fp16";
        case DType::BF16: return "bf16";
        case DType::FP32: return "fp32";
    }
    return "<unknown>";
}

inline std::string shape_to_string(const Tensor& t) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < t.ndim(); ++i) {
        if (i) ss << "x";
        ss << t.dim(i);
    }
    ss << "]";
    return ss.str();
}

inline std::string json_escape(std::string_view in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    std::ostringstream ss;
                    ss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(static_cast<unsigned char>(c));
                    out += ss.str();
                } else {
                    out += c;
                }
        }
    }
    return out;
}

class JsonObject {
public:
    void add_string(std::string_view key, std::string_view value) {
        add_key(key);
        out_ += '"';
        out_ += json_escape(value);
        out_ += '"';
    }

    void add_bool(std::string_view key, bool value) {
        add_key(key);
        out_ += value ? "true" : "false";
    }

    void add_uint(std::string_view key, uint64_t value) {
        add_key(key);
        out_ += std::to_string(value);
    }

    void add_int(std::string_view key, int64_t value) {
        add_key(key);
        out_ += std::to_string(value);
    }

    void add_double(std::string_view key, double value) {
        add_key(key);
        std::ostringstream ss;
        ss.setf(std::ios::fixed);
        ss << std::setprecision(6) << value;
        out_ += ss.str();
    }

    std::string finish() {
        out_ += "}";
        return std::move(out_);
    }

private:
    void add_key(std::string_view key) {
        if (!first_) {
            out_ += ',';
        }
        first_ = false;
        out_ += '"';
        out_ += json_escape(key);
        out_ += "\":";
    }

    bool first_ = true;
    std::string out_ = "{";
};

}  // namespace detail

class BackendProfiler {
public:
    BackendProfiler() { reset(); }

    void reset() {
        for (auto& s : stats_) {
            s.calls.store(0, std::memory_order_relaxed);
            s.total_ns.store(0, std::memory_order_relaxed);
            s.min_ns.store(std::numeric_limits<uint64_t>::max(), std::memory_order_relaxed);
            s.max_ns.store(0, std::memory_order_relaxed);
        }
    }

    void record(BackendOp op, uint64_t ns) {
        const size_t idx = static_cast<size_t>(op);
        if (idx >= stats_.size()) return;
        auto& s = stats_[idx];
        s.calls.fetch_add(1, std::memory_order_relaxed);
        s.total_ns.fetch_add(ns, std::memory_order_relaxed);
        detail::atomic_update_max(s.max_ns, ns);
        detail::atomic_update_min(s.min_ns, ns);
    }

    BackendOpSnapshot snapshot(BackendOp op) const {
        const size_t idx = static_cast<size_t>(op);
        if (idx >= stats_.size()) return {};
        const auto& s = stats_[idx];
        BackendOpSnapshot snap;
        snap.calls = s.calls.load(std::memory_order_relaxed);
        snap.total_ns = s.total_ns.load(std::memory_order_relaxed);
        const uint64_t min = s.min_ns.load(std::memory_order_relaxed);
        snap.min_ns = (min == std::numeric_limits<uint64_t>::max()) ? 0 : min;
        snap.max_ns = s.max_ns.load(std::memory_order_relaxed);
        return snap;
    }

    std::vector<std::pair<BackendOp, BackendOpSnapshot>> snapshot_all() const {
        std::vector<std::pair<BackendOp, BackendOpSnapshot>> out;
        out.reserve(static_cast<size_t>(BackendOp::Count));
        for (size_t i = 0; i < static_cast<size_t>(BackendOp::Count); ++i) {
            auto op = static_cast<BackendOp>(i);
            out.emplace_back(op, snapshot(op));
        }
        return out;
    }

    std::string format_report() const {
        auto entries = snapshot_all();

        uint64_t total_ns = 0;
        for (const auto& e : entries) {
            total_ns += e.second.total_ns;
        }

        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            return a.second.total_ns > b.second.total_ns;
        });

        std::ostringstream ss;
        ss << "Backend op profile (" << std::fixed << std::setprecision(3)
           << (static_cast<double>(total_ns) / 1e6) << " ms total)\n";
        ss << "op\tcalls\ttotal_ms\tavg_us\tmin_us\tmax_us\tpct\n";

        for (const auto& [op, snap] : entries) {
            if (op == BackendOp::Count) continue;
            if (snap.calls == 0) continue;

            const double op_ms = static_cast<double>(snap.total_ns) / 1e6;
            const double avg_us = (snap.calls == 0)
                ? 0.0
                : (static_cast<double>(snap.total_ns) / 1e3) / static_cast<double>(snap.calls);
            const double min_us = static_cast<double>(snap.min_ns) / 1e3;
            const double max_us = static_cast<double>(snap.max_ns) / 1e3;
            const double pct = (total_ns == 0)
                ? 0.0
                : (100.0 * static_cast<double>(snap.total_ns) / static_cast<double>(total_ns));

            ss << backend_op_name(op) << '\t'
               << snap.calls << '\t'
               << std::fixed << std::setprecision(3) << op_ms << '\t'
               << std::fixed << std::setprecision(3) << avg_us << '\t'
               << std::fixed << std::setprecision(3) << min_us << '\t'
               << std::fixed << std::setprecision(3) << max_us << '\t'
               << std::fixed << std::setprecision(1) << pct << "\n";
        }

        return ss.str();
    }

private:
    struct BackendOpStats {
        std::atomic<uint64_t> calls{0};
        std::atomic<uint64_t> total_ns{0};
        std::atomic<uint64_t> min_ns{std::numeric_limits<uint64_t>::max()};
        std::atomic<uint64_t> max_ns{0};
    };

    std::array<BackendOpStats, static_cast<size_t>(BackendOp::Count)> stats_{};
};

class TraceWriter {
public:
    using Clock = std::chrono::steady_clock;

    struct Options {
        bool enabled = false;
        std::string path;
    };

    TraceWriter() : TraceWriter(Options{}) {}

    explicit TraceWriter(Options options)
        : options_(std::move(options)), origin_(Clock::now()) {}

    ~TraceWriter() {
        if (!options_.enabled || options_.path.empty()) return;
        try {
            flush_to_file(nullptr);
        } catch (...) {
        }
    }

    bool enabled() const { return options_.enabled; }
    const std::string& path() const { return options_.path; }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
        origin_ = Clock::now();
    }

    void record_event(
        std::string_view name,
        Clock::time_point start,
        Clock::time_point end,
        std::string_view args_json
    ) {
        if (!options_.enabled) return;

        const uint64_t ts_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(start - origin_).count()
        );
        const uint64_t dur_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
        );

        const uint64_t tid = static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));

        std::string event;
        event.reserve(256);
        event += "{\"name\":\"";
        event += detail::json_escape(name);
        event += "\",\"cat\":\"backend\",\"ph\":\"X\",\"ts\":";
        event += std::to_string(ts_us);
        event += ",\"dur\":";
        event += std::to_string(dur_us);
        event += ",\"pid\":0,\"tid\":";
        event += std::to_string(tid);
        event += ",\"args\":";
        if (args_json.empty()) {
            event += "{}";
        } else {
            event += args_json;
        }
        event += "}";

        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(std::move(event));
    }

    size_t event_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_.size();
    }

    std::string to_json() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ostringstream ss;
        ss << "{\"displayTimeUnit\":\"us\",\"traceEvents\":[\n";
        for (size_t i = 0; i < events_.size(); ++i) {
            ss << events_[i];
            if (i + 1 < events_.size()) ss << ',';
            ss << '\n';
        }
        ss << "]}";
        return ss.str();
    }

    bool flush_to_file(std::string* error) const {
        if (!options_.enabled || options_.path.empty()) return true;

        std::ofstream out(options_.path, std::ios::out | std::ios::trunc);
        if (!out.good()) {
            if (error) {
                *error = "Failed to open trace file: " + options_.path;
            }
            return false;
        }
        out << to_json();
        if (!out.good()) {
            if (error) {
                *error = "Failed to write trace file: " + options_.path;
            }
            return false;
        }
        return true;
    }

private:
    Options options_;
    mutable std::mutex mutex_;
    mutable Clock::time_point origin_;
    std::vector<std::string> events_;
};

class ProfilingBackend : public Backend {
public:
    struct Options {
        bool enabled = true;
        bool trace_enabled = false;
        std::string trace_path;
        bool trace_args = true;
    };

    explicit ProfilingBackend(Backend& inner)
        : ProfilingBackend(inner, Options{}) {}

    explicit ProfilingBackend(Backend& inner, Options options)
        : inner_(inner),
          options_(std::move(options)),
          trace_(TraceWriter::Options{options_.trace_enabled, options_.trace_path}) {}

    BackendProfiler& profiler() { return profiler_; }
    const BackendProfiler& profiler() const { return profiler_; }

    TraceWriter& trace_writer() { return trace_; }
    const TraceWriter& trace_writer() const { return trace_; }

    void reset() {
        profiler_.reset();
        trace_.reset();
    }

    bool flush_trace_to_file(std::string* error = nullptr) const {
        return trace_.flush_to_file(error);
    }

    void gemm(
        const Tensor& A,
        const Tensor& B,
        Tensor& C,
        float alpha = 1.0f,
        float beta = 0.0f,
        bool trans_a = false,
        bool trans_b = false
    ) override {
        profile_call(
            BackendOp::Gemm,
            "backend.gemm",
            [&]() {
                detail::JsonObject args;
                args.add_string("A_shape", detail::shape_to_string(A));
                args.add_string("B_shape", detail::shape_to_string(B));
                args.add_string("C_shape", detail::shape_to_string(C));
                args.add_string("A_dtype", detail::dtype_name(A.dtype()));
                args.add_string("B_dtype", detail::dtype_name(B.dtype()));
                args.add_string("C_dtype", detail::dtype_name(C.dtype()));
                args.add_bool("trans_a", trans_a);
                args.add_bool("trans_b", trans_b);
                args.add_double("alpha", alpha);
                args.add_double("beta", beta);
                return args.finish();
            },
            [&]() {
                inner_.gemm(A, B, C, alpha, beta, trans_a, trans_b);
            }
        );
    }

    void flash_attention(
        const Tensor& Q,
        const Tensor& K,
        const Tensor& V,
        Tensor& output,
        float scale,
        bool causal = true
    ) override {
        profile_call(
            BackendOp::FlashAttention,
            "backend.flash_attention",
            [&]() {
                detail::JsonObject args;
                args.add_string("Q_shape", detail::shape_to_string(Q));
                args.add_string("K_shape", detail::shape_to_string(K));
                args.add_string("V_shape", detail::shape_to_string(V));
                args.add_string("O_shape", detail::shape_to_string(output));
                args.add_string("Q_dtype", detail::dtype_name(Q.dtype()));
                args.add_string("K_dtype", detail::dtype_name(K.dtype()));
                args.add_string("V_dtype", detail::dtype_name(V.dtype()));
                args.add_string("O_dtype", detail::dtype_name(output.dtype()));
                args.add_double("scale", scale);
                args.add_bool("causal", causal);
                return args.finish();
            },
            [&]() {
                inner_.flash_attention(Q, K, V, output, scale, causal);
            }
        );
    }

    void rmsnorm(
        const Tensor& input,
        const Tensor& weight,
        Tensor& output,
        float eps = 1e-6f
    ) override {
        profile_call(
            BackendOp::Rmsnorm,
            "backend.rmsnorm",
            [&]() {
                detail::JsonObject args;
                args.add_string("input_shape", detail::shape_to_string(input));
                args.add_string("weight_shape", detail::shape_to_string(weight));
                args.add_string("output_shape", detail::shape_to_string(output));
                args.add_string("input_dtype", detail::dtype_name(input.dtype()));
                args.add_string("weight_dtype", detail::dtype_name(weight.dtype()));
                args.add_string("output_dtype", detail::dtype_name(output.dtype()));
                args.add_double("eps", eps);
                return args.finish();
            },
            [&]() {
                inner_.rmsnorm(input, weight, output, eps);
            }
        );
    }

    void rope(
        Tensor& Q,
        Tensor& K,
        size_t position_offset,
        float theta = 10000.0f
    ) override {
        profile_call(
            BackendOp::Rope,
            "backend.rope",
            [&]() {
                detail::JsonObject args;
                args.add_string("Q_shape", detail::shape_to_string(Q));
                args.add_string("K_shape", detail::shape_to_string(K));
                args.add_string("Q_dtype", detail::dtype_name(Q.dtype()));
                args.add_string("K_dtype", detail::dtype_name(K.dtype()));
                args.add_uint("position_offset", position_offset);
                args.add_double("theta", theta);
                return args.finish();
            },
            [&]() {
                inner_.rope(Q, K, position_offset, theta);
            }
        );
    }

    void silu(
        const Tensor& input,
        Tensor& output
    ) override {
        profile_call(
            BackendOp::Silu,
            "backend.silu",
            [&]() {
                detail::JsonObject args;
                args.add_string("input_shape", detail::shape_to_string(input));
                args.add_string("output_shape", detail::shape_to_string(output));
                args.add_string("input_dtype", detail::dtype_name(input.dtype()));
                args.add_string("output_dtype", detail::dtype_name(output.dtype()));
                return args.finish();
            },
            [&]() {
                inner_.silu(input, output);
            }
        );
    }

    void elementwise_mul(
        const Tensor& a,
        const Tensor& b,
        Tensor& output
    ) override {
        profile_call(
            BackendOp::ElementwiseMul,
            "backend.elementwise_mul",
            [&]() {
                detail::JsonObject args;
                args.add_string("a_shape", detail::shape_to_string(a));
                args.add_string("b_shape", detail::shape_to_string(b));
                args.add_string("output_shape", detail::shape_to_string(output));
                args.add_string("a_dtype", detail::dtype_name(a.dtype()));
                args.add_string("b_dtype", detail::dtype_name(b.dtype()));
                args.add_string("output_dtype", detail::dtype_name(output.dtype()));
                return args.finish();
            },
            [&]() {
                inner_.elementwise_mul(a, b, output);
            }
        );
    }

    void add(
        const Tensor& a,
        const Tensor& b,
        Tensor& output
    ) override {
        profile_call(
            BackendOp::Add,
            "backend.add",
            [&]() {
                detail::JsonObject args;
                args.add_string("a_shape", detail::shape_to_string(a));
                args.add_string("b_shape", detail::shape_to_string(b));
                args.add_string("output_shape", detail::shape_to_string(output));
                args.add_string("a_dtype", detail::dtype_name(a.dtype()));
                args.add_string("b_dtype", detail::dtype_name(b.dtype()));
                args.add_string("output_dtype", detail::dtype_name(output.dtype()));
                return args.finish();
            },
            [&]() {
                inner_.add(a, b, output);
            }
        );
    }

    void softmax(
        const Tensor& input,
        Tensor& output,
        int dim = -1
    ) override {
        profile_call(
            BackendOp::Softmax,
            "backend.softmax",
            [&]() {
                detail::JsonObject args;
                args.add_string("input_shape", detail::shape_to_string(input));
                args.add_string("output_shape", detail::shape_to_string(output));
                args.add_string("input_dtype", detail::dtype_name(input.dtype()));
                args.add_string("output_dtype", detail::dtype_name(output.dtype()));
                args.add_int("dim", dim);
                return args.finish();
            },
            [&]() {
                inner_.softmax(input, output, dim);
            }
        );
    }

    void embedding_lookup(
        const Tensor& weight,
        const std::vector<int>& token_ids,
        Tensor& output
    ) override {
        profile_call(
            BackendOp::EmbeddingLookup,
            "backend.embedding_lookup",
            [&]() {
                detail::JsonObject args;
                args.add_string("weight_shape", detail::shape_to_string(weight));
                args.add_string("output_shape", detail::shape_to_string(output));
                args.add_string("weight_dtype", detail::dtype_name(weight.dtype()));
                args.add_string("output_dtype", detail::dtype_name(output.dtype()));
                args.add_uint("token_count", static_cast<uint64_t>(token_ids.size()));
                if (!token_ids.empty()) {
                    args.add_int("first_token", token_ids.front());
                    args.add_int("last_token", token_ids.back());
                }
                return args.finish();
            },
            [&]() {
                inner_.embedding_lookup(weight, token_ids, output);
            }
        );
    }

private:
    using Clock = TraceWriter::Clock;

    template <typename ArgsFn, typename Fn>
    void profile_call(BackendOp op, std::string_view trace_name, ArgsFn&& args_fn, Fn&& fn) {
        if (!options_.enabled) {
            fn();
            return;
        }

        const auto start = Clock::now();
        fn();
        const auto end = Clock::now();

        const uint64_t ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
        );
        profiler_.record(op, ns);

        if (trace_.enabled()) {
            if (options_.trace_args) {
                trace_.record_event(trace_name, start, end, args_fn());
            } else {
                trace_.record_event(trace_name, start, end, "{}");
            }
        }
    }

    Backend& inner_;
    Options options_;
    BackendProfiler profiler_;
    TraceWriter trace_;
};

}  // namespace mruntime
