#include "mruntime/trace.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <string>

namespace mruntime {

void TraceCollector::print_summary() const {
    if (events_.empty()) {
        printf("No trace events recorded.\n");
        return;
    }

    struct CompleteEventView {
        const char* name = nullptr;
        uint32_t thread_id = 0;
        int64_t start_us = 0;
        int64_t end_us = 0;
        int64_t duration_us = 0;
        uint64_t id = 0;
        uint64_t parent_id = 0;
    };

    struct InclusiveStats {
        int64_t total_us = 0;
        int64_t min_us = INT64_MAX;
        int64_t max_us = 0;
        size_t count = 0;
    };

    std::vector<CompleteEventView> complete_events;
    complete_events.reserve(events_.size());

    std::map<std::string, InclusiveStats> inclusive_by_name;
    int64_t min_start_us = INT64_MAX;
    int64_t max_end_us = 0;

    for (const auto& e : events_) {
        if (e.type != TraceEventType::Complete) {
            continue;
        }
        if (e.duration_us < 0) {
            continue;
        }
        int64_t start_us = e.timestamp_us;
        int64_t end_us = e.timestamp_us + e.duration_us;
        complete_events.push_back({e.name, e.thread_id, start_us, end_us, e.duration_us, e.id, e.parent_id});

        auto& s = inclusive_by_name[e.name];
        s.total_us += e.duration_us;
        s.min_us = std::min(s.min_us, e.duration_us);
        s.max_us = std::max(s.max_us, e.duration_us);
        s.count++;

        min_start_us = std::min(min_start_us, start_us);
        max_end_us = std::max(max_end_us, end_us);
    }

    if (inclusive_by_name.empty()) {
        printf("No Complete events to summarize.\n");
        return;
    }

    // Compute per-event parent relationships and exclusive/self time using the explicit parent_id
    // recorded by ScopedTrace's per-thread live stack.
    constexpr size_t kNoParent = static_cast<size_t>(-1);
    std::vector<size_t> parent_idx(complete_events.size(), kNoParent);
    std::vector<std::vector<size_t>> children_idx(complete_events.size());
    std::vector<size_t> roots;
    roots.reserve(complete_events.size());

    std::map<uint64_t, size_t> id_to_index;
    for (size_t i = 0; i < complete_events.size(); ++i) {
        if (complete_events[i].id != 0) {
            id_to_index[complete_events[i].id] = i;
        }
    }

    for (size_t i = 0; i < complete_events.size(); ++i) {
        size_t p = kNoParent;
        if (complete_events[i].parent_id != 0) {
            auto it = id_to_index.find(complete_events[i].parent_id);
            if (it != id_to_index.end()) {
                p = it->second;
            }
        }

        parent_idx[i] = p;
        if (p != kNoParent) {
            children_idx[p].push_back(i);
        } else {
            roots.push_back(i);
        }
    }

    std::vector<int64_t> self_us(complete_events.size(), 0);
    for (size_t i = 0; i < complete_events.size(); ++i) {
        int64_t child_sum_us = 0;
        for (size_t c : children_idx[i]) {
            child_sum_us += complete_events[c].duration_us;
        }
        int64_t self = complete_events[i].duration_us - child_sum_us;
        if (self < 0) {
            // Should not happen for properly nested scopes; clamp defensively.
            self = 0;
        }
        self_us[i] = self;
    }

    std::map<std::string, int64_t> self_by_name;
    int64_t total_self_us = 0;
    for (size_t i = 0; i < complete_events.size(); ++i) {
        total_self_us += self_us[i];
        self_by_name[complete_events[i].name] += self_us[i];
    }

    struct Row {
        std::string name;
        InclusiveStats inclusive;
        int64_t self_total_us = 0;
    };
    std::vector<Row> rows;
    rows.reserve(inclusive_by_name.size());
    for (const auto& [name, stats] : inclusive_by_name) {
        int64_t self_total = 0;
        auto it = self_by_name.find(name);
        if (it != self_by_name.end()) {
            self_total = it->second;
        }
        rows.push_back({name, stats, self_total});
    }

    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        return a.inclusive.total_us > b.inclusive.total_us;
    });

    int64_t inclusive_sum_us = 0;
    for (const auto& r : rows) {
        inclusive_sum_us += r.inclusive.total_us;
    }

    // Build an aggregated call tree using the inferred parent relationships.
    struct TreeNode {
        std::string name;
        int64_t inclusive_us = 0;
        int64_t self_us = 0;
        size_t count = 0;
        std::map<std::string, std::unique_ptr<TreeNode>> children;
    };

    TreeNode root;
    root.name = "<root>";
    auto get_or_create_child = [](TreeNode* parent, const char* child_name) -> TreeNode* {
        const std::string key = (child_name != nullptr) ? std::string(child_name) : std::string("<null>");
        auto& slot = parent->children[key];
        if (!slot) {
            slot = std::make_unique<TreeNode>();
            slot->name = key;
        }
        return slot.get();
    };

    // Build an aggregated call tree by walking per-instance relationships.
    auto build_tree = [&](auto&& self, size_t event_idx, TreeNode* parent_node) -> void {
        TreeNode* node = get_or_create_child(parent_node, complete_events[event_idx].name);
        node->inclusive_us += complete_events[event_idx].duration_us;
        node->self_us += self_us[event_idx];
        node->count += 1;

        for (size_t child : children_idx[event_idx]) {
            self(self, child, node);
        }
    };

    // Optional deterministic ordering: walk roots by start time (outer-first).
    std::sort(roots.begin(), roots.end(), [&](size_t a, size_t b) {
        const auto& ea = complete_events[a];
        const auto& eb = complete_events[b];
        if (ea.start_us != eb.start_us) return ea.start_us < eb.start_us;
        if (ea.end_us != eb.end_us) return ea.end_us > eb.end_us;
        return a < b;
    });

    for (size_t r : roots) {
        build_tree(build_tree, r, &root);
    }

    printf("\n=== Trace Summary ===\n");
    printf("%-30s %10s %7s %10s %7s %10s %10s %10s %8s\n",
           "Name", "Total(ms)", "Incl%", "Self(ms)", "Self%", "Avg(ms)", "Min(ms)", "Max(ms)", "Count");
    printf("%-30s %10s %7s %10s %7s %10s %10s %10s %8s\n",
           "------------------------------", "----------", "------", "----------", "------", "----------", "----------", "----------", "--------");

    for (const auto& r : rows) {
        const auto& stats = r.inclusive;
        double total_ms = stats.total_us / 1000.0;
        double self_ms = r.self_total_us / 1000.0;
        double avg_ms = (stats.count > 0) ? (total_ms / stats.count) : 0.0;
        double min_ms = (stats.min_us != INT64_MAX) ? (stats.min_us / 1000.0) : 0.0;
        double max_ms = stats.max_us / 1000.0;
        double incl_pct = (total_self_us > 0) ? (100.0 * stats.total_us / total_self_us) : 0.0;
        double self_pct = (total_self_us > 0) ? (100.0 * r.self_total_us / total_self_us) : 0.0;

        printf("%-30s %10.3f %6.1f%% %10.3f %6.1f%% %10.3f %10.3f %10.3f %8zu\n",
               r.name.c_str(),
               total_ms, incl_pct,
               self_ms, self_pct,
               avg_ms, min_ms, max_ms, stats.count);
    }

    double traced_total_ms = total_self_us / 1000.0;
    double wall_span_ms =
        (min_start_us != INT64_MAX && max_end_us >= min_start_us) ? ((max_end_us - min_start_us) / 1000.0) : 0.0;
    double inclusive_sum_ms = inclusive_sum_us / 1000.0;

    printf("\n=== Trace Call Tree (Inclusive) ===\n");
    printf("%-50s %10s %7s %10s %7s %8s\n", "Name", "Total(ms)", "Incl%", "Self(ms)", "Self%", "Count");
    printf("%-50s %10s %7s %10s %7s %8s\n",
           "--------------------------------------------------", "----------", "------", "----------", "------", "--------");

    auto print_tree = [&](auto&& self, const TreeNode& node, int depth) -> void {
        std::vector<const TreeNode*> children;
        children.reserve(node.children.size());
        for (const auto& [_, child] : node.children) {
            children.push_back(child.get());
        }
        std::sort(children.begin(), children.end(), [](const TreeNode* a, const TreeNode* b) {
            return a->inclusive_us > b->inclusive_us;
        });

        for (const TreeNode* child : children) {
            std::string name = std::string(static_cast<size_t>(depth) * 2, ' ') + child->name;
            double incl_ms = child->inclusive_us / 1000.0;
            double self_ms = child->self_us / 1000.0;
            double incl_pct = (total_self_us > 0) ? (100.0 * child->inclusive_us / total_self_us) : 0.0;
            double self_pct = (total_self_us > 0) ? (100.0 * child->self_us / total_self_us) : 0.0;

            printf("%-50s %10.3f %6.1f%% %10.3f %6.1f%% %8zu\n",
                   name.c_str(), incl_ms, incl_pct, self_ms, self_pct, child->count);

            self(self, *child, depth + 1);
        }
    };

    print_tree(print_tree, root, /*depth=*/0);

    printf("\nTRACED_TOTAL(ms):   %.3f\n", traced_total_ms);
    printf("WALL_SPAN(ms):      %.3f\n", wall_span_ms);
    printf("INCLUSIVE_SUM(ms):  %.3f\n", inclusive_sum_ms);
    printf("\nEvents recorded: %zu\n", events_.size());
}

}  // namespace mruntime
