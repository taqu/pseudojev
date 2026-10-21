#include "r3_analysis.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "r1_metrics.h"

namespace pjev
{
namespace r3
{

using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// Parsing / alignment / statistics
// ---------------------------------------------------------------------------

static double num(const json& j, const char* k, double def = 0.0)
{
    return (j.contains(k) && j[k].is_number()) ? j[k].get<double>() : def;
}
static std::string str(const json& j, const char* k)
{
    return (j.contains(k) && j[k].is_string()) ? j[k].get<std::string>() : std::string();
}

bool parse_seed_record(const json& j, SeedRecord& r)
{
    if (!j.is_object() || !j.contains("query_id") || !j.contains("r3") || !j.contains("exemplar")) return false;
    r = SeedRecord{};
    r.query_id     = str(j, "query_id");
    r.kind         = str(j, "kind");
    r.seed         = j.value("seed", uint64_t{0});
    r.ground_truth = static_cast<int>(num(j, "ground_truth_index", -1));
    r.ground_truth_label = str(j, "ground_truth_label");
    const json& ex = j["exemplar"];
    r.exemplar_status = str(ex, "status");
    r.example_id      = str(ex, "example_id");
    r.demo_label      = str(ex, "answer_label");
    r.same_as_nearest = ex.value("same_as_nearest", false);
    if (ex.contains("estimated_cosine") && ex["estimated_cosine"].is_number()) {
        r.has_similarity = true;
        r.similarity = ex["estimated_cosine"].get<double>();
        r.hamming    = num(ex, "hamming_distance");
    }
    r.select_ms     = num(ex, "select_ms");
    r.fetch_ms      = num(ex, "fetch_ms");
    r.decompress_ms = num(ex, "decompress_ms");
    const json& c = j["r3"];
    r.status = str(c, "status");
    if (r.status == "ok") {
        r.prediction       = static_cast<int>(num(c, "prediction_index", -1));
        r.prediction_label = str(c, "prediction_label");
        r.correct          = c.value("correct", false);
    }
    r.prompt_tokens = static_cast<int>(num(c, "prompt_tokens"));
    r.decision_ms   = num(c, "latency_ms");
    r.prompt_ms     = num(c, "prompt_ms");
    r.eval_ms       = num(c, "eval_ms");
    return !r.query_id.empty();
}

Alignment align(const std::vector<std::string>& expected, const std::map<std::string, RefRecord>& ref,
                const std::map<uint64_t, std::vector<SeedRecord>>& seeds)
{
    Alignment a;
    const std::unordered_set<std::string> want(expected.begin(), expected.end());
    std::map<uint64_t, std::unordered_map<std::string, const SeedRecord*>> idx;
    for (const auto& [seed, recs] : seeds) {
        auto& m = idx[seed];
        a.failed_per_seed[seed] = 0;
        for (const auto& r : recs) {
            if (!want.count(r.query_id)) { a.unexpected_in_seed++; continue; }
            if (!m.emplace(r.query_id, &r).second) a.duplicate_in_seed++;
        }
    }
    for (const auto& id : expected) {
        auto rit = ref.find(id);
        bool complete = rit != ref.end();
        for (const auto& [seed, m] : idx) {
            auto it = m.find(id);
            if (it == m.end()) { a.missing_in_seed++; complete = false; continue; }
            if (rit != ref.end() && (it->second->kind != rit->second.kind || it->second->ground_truth != rit->second.ground_truth))
                a.ground_truth_mismatch++;
        }
        if (!complete) continue;
        a.query_ids.push_back(id);
        bool all_ok = true;
        if (!rit->second.s1_ok || !rit->second.r1_ok) { a.reference_failed++; all_ok = false; }
        for (const auto& [seed, m] : idx) {
            if (!m.at(id)->ok()) { a.failed_per_seed[seed]++; all_ok = false; }
        }
        if (all_ok) a.analysed.push_back(id);
    }
    return a;
}

SeedSummary summarize(const std::vector<double>& v)
{
    SeedSummary s;
    s.n = static_cast<int>(v.size());
    if (v.empty()) return s;
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (double x : v) sum += x;
    s.mean = sum / v.size();
    if (v.size() > 1) {
        double ss = 0.0;
        for (double x : v) ss += (x - s.mean) * (x - s.mean);
        s.sd = std::sqrt(ss / (v.size() - 1));
    }
    return s;
}

ClusterCI cluster_bootstrap(const std::vector<std::string>& cluster, const std::vector<double>& a,
                            const std::vector<double>& b, int reps, uint64_t seed)
{
    Resampler rs(cluster, reps, seed);
    return paired_ci(rs, a, b);
}

} // namespace r3
} // namespace pjev
