#pragma once

#include <memory>
#include <mutex>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace mruntime {

namespace detail {

// Returns the filename portion of a path (handles both '/' and '\\').
// Safe for string literals like __FILE__ (returns a pointer into the original string).
constexpr const char* basename(const char* path) {
    const char* file = path;
    for (const char* p = path; *p != '\0'; ++p) {
        if (*p == '/' || *p == '\\') {
            file = p + 1;
        }
    }
    return file;
}

} // namespace detail

inline std::shared_ptr<spdlog::logger> get_logger() {
    static std::once_flag init_flag;
    static std::shared_ptr<spdlog::logger> logger;
    std::call_once(init_flag, []() {
        logger = spdlog::get("mruntime");
        if (!logger) {
            // Create+register the logger in spdlog's global registry so it can be
            // retrieved later via spdlog::get("mruntime").
            logger = spdlog::stdout_color_mt("mruntime");
        }
        // Always set as default + apply mruntime's preferred formatting/settings.
        logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%s:%#] %v");
        logger->set_level(spdlog::level::info);
        logger->flush_on(spdlog::level::warn);
        spdlog::set_default_logger(logger);
    });
    return logger;
}

inline void set_log_level(spdlog::level::level_enum level) {
    auto logger = get_logger();
    logger->set_level(level);
    logger->flush_on(level);
}

} // namespace mruntime

// Convenience logging helpers that always initialize/obtain the mruntime logger.
#define MRUNTIME_LOG_CALL_(level, ...)                                                           \
    (::mruntime::get_logger()->log(                                                              \
        spdlog::source_loc{::mruntime::detail::basename(__FILE__), __LINE__, SPDLOG_FUNCTION},   \
        (level),                                                                                 \
        __VA_ARGS__))

#define MRUNTIME_LOG_TRACE(...) MRUNTIME_LOG_CALL_(spdlog::level::trace, __VA_ARGS__)
#define MRUNTIME_LOG_DEBUG(...) MRUNTIME_LOG_CALL_(spdlog::level::debug, __VA_ARGS__)
#define MRUNTIME_LOG_INFO(...) MRUNTIME_LOG_CALL_(spdlog::level::info, __VA_ARGS__)
#define MRUNTIME_LOG_WARN(...) MRUNTIME_LOG_CALL_(spdlog::level::warn, __VA_ARGS__)
#define MRUNTIME_LOG_ERROR(...) MRUNTIME_LOG_CALL_(spdlog::level::err, __VA_ARGS__)
#define MRUNTIME_LOG_CRITICAL(...) MRUNTIME_LOG_CALL_(spdlog::level::critical, __VA_ARGS__)
