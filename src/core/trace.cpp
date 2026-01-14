#include "mruntime/trace.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>

namespace mruntime {

void TraceCollector::print_summary() const {
    if (events_.empty()) {
        printf("No trace events recorded.\n");
        return;
    }

    // Aggregate Complete events by name
    struct Stats {
        int64_t total_us = 0;
        int64_t min_us = INT64_MAX;
        int64_t max_us = 0;
        size_t count = 0;
    };
    std::map<std::string, Stats> by_name;

    for (const auto& e : events_) {
        if (e.type == TraceEventType::Complete) {
            auto& s = by_name[e.name];
            s.total_us += e.duration_us;
            s.min_us = std::min(s.min_us, e.duration_us);
            s.max_us = std::max(s.max_us, e.duration_us);
            s.count++;
        }
    }

    if (by_name.empty()) {
        printf("No Complete events to summarize.\n");
        return;
    }

    // Sort by total time descending
    std::vector<std::pair<std::string, Stats>> sorted(by_name.begin(), by_name.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.total_us > b.second.total_us; });

    // Calculate total time
    int64_t total_time_us = 0;
    for (const auto& [name, stats] : sorted) {
        total_time_us += stats.total_us;
    }

    printf("\n=== Trace Summary ===\n");
    printf("%-30s %10s %10s %10s %10s %8s\n",
           "Name", "Total(ms)", "Avg(ms)", "Min(ms)", "Max(ms)", "Count");
    printf("%-30s %10s %10s %10s %10s %8s\n",
           "------------------------------", "----------", "----------", "----------", "----------", "--------");

    for (const auto& [name, stats] : sorted) {
        double total_ms = stats.total_us / 1000.0;
        double avg_ms = total_ms / stats.count;
        double min_ms = stats.min_us / 1000.0;
        double max_ms = stats.max_us / 1000.0;
        double pct = 100.0 * stats.total_us / total_time_us;

        printf("%-30s %9.3f %10.3f %10.3f %10.3f %8zu  (%.1f%%)\n",
               name.c_str(), total_ms, avg_ms, min_ms, max_ms, stats.count, pct);
    }

    printf("%-30s %9.3f\n", "TOTAL", total_time_us / 1000.0);
    printf("\nEvents recorded: %zu\n", events_.size());
}

}  // namespace mruntime
