#include "value_profile.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace pjev
{

// ---------------------------------------------------------------------------
// PerformanceStats
// ---------------------------------------------------------------------------

PerformanceStats compute_perf_stats(int64_t load_ms,
                                     int64_t rss_bytes,
                                     const std::vector<int64_t>& warm_ms,
                                     const std::string& os,
                                     const std::string& cpu_info,
                                     int n_threads)
{
    PerformanceStats ps;
    ps.model_load_ms       = load_ms;
    ps.rss_post_load_bytes = rss_bytes;
    ps.os                  = os;
    ps.cpu_info            = cpu_info;
    ps.n_threads           = n_threads;
    ps.n_warm_samples      = (int)warm_ms.size();

    if (!warm_ms.empty()) {
        std::vector<int64_t> sorted = warm_ms;
        std::sort(sorted.begin(), sorted.end());
        auto pctile = [&](double p) -> double {
            double idx = p * (sorted.size() - 1);
            size_t lo = (size_t)idx;
            size_t hi = lo + 1 < sorted.size() ? lo + 1 : lo;
            double frac = idx - (double)lo;
            return (double)sorted[lo] * (1.0 - frac) + (double)sorted[hi] * frac;
        };
        ps.warm_p50_ms = pctile(0.50);
        ps.warm_p90_ms = pctile(0.90);
        ps.warm_p95_ms = pctile(0.95);
    }
    return ps;
}

nlohmann::json PerformanceStats::to_json() const
{
    return {
        {"model_load_ms",       model_load_ms},
        {"warm_p50_ms",         warm_p50_ms},
        {"warm_p90_ms",         warm_p90_ms},
        {"warm_p95_ms",         warm_p95_ms},
        {"n_warm_samples",      n_warm_samples},
        {"mean_prompt_tokens",  mean_prompt_tokens},
        {"rss_post_load_bytes", rss_post_load_bytes},
        {"os",                  os},
        {"cpu_info",            cpu_info},
        {"n_threads",           n_threads},
        {"llama_backend_tag",   llama_backend_tag}
    };
}

// ---------------------------------------------------------------------------
// QualityStats
// ---------------------------------------------------------------------------

QualityStats compute_quality_stats(const RunResult& result)
{
    QualityStats qs;

    auto fill = [&](PrimitiveQuality& pq, const TypeMetrics& tm,
                    const std::string& type, int n_levels) {
        pq.n        = tm.labeled;
        pq.accuracy = tm.accuracy();
        pq.mae      = tm.mae();
        pq.n_levels = n_levels;
        if (type == "score" && n_levels >= 2)
            pq.qwk = tm.qwk(n_levels);

        // Difficulty breakdown
        std::map<std::string, int> n_d, corr_d;
        for (const auto& ir : result.items) {
            if (ir.type != type || !ir.ok || ir.difficulty.empty()) continue;
            n_d[ir.difficulty]++;
            if (ir.correct) corr_d[ir.difficulty]++;
        }
        for (const auto& kv : n_d) {
            pq.n_by_difficulty[kv.first]       = kv.second;
            int c = corr_d.count(kv.first) ? corr_d.at(kv.first) : 0;
            pq.accuracy_by_difficulty[kv.first] = (double)c / kv.second;
        }
    };

    fill(qs.noul,   result.metrics.noul,   "noul",   0);
    fill(qs.choice, result.metrics.choice, "choice", 0);
    fill(qs.score,  result.metrics.score,  "score",  result.metrics.score_n_levels);

    return qs;
}

nlohmann::json PrimitiveQuality::to_json() const
{
    nlohmann::json j = {
        {"n",        n},
        {"accuracy", accuracy},
        {"mae",      mae},
        {"qwk",      qwk},
        {"n_levels", n_levels}
    };
    if (!accuracy_by_difficulty.empty()) {
        nlohmann::json abd = nlohmann::json::object();
        nlohmann::json nbd = nlohmann::json::object();
        for (const auto& kv : accuracy_by_difficulty) abd[kv.first] = kv.second;
        for (const auto& kv : n_by_difficulty)         nbd[kv.first] = kv.second;
        j["accuracy_by_difficulty"] = abd;
        j["n_by_difficulty"]        = nbd;
    }
    return j;
}

nlohmann::json QualityStats::to_json() const
{
    return {
        {"noul",   noul.to_json()},
        {"choice", choice.to_json()},
        {"score",  score.to_json()}
    };
}

// ---------------------------------------------------------------------------
// CommonValidSubset
// ---------------------------------------------------------------------------

CommonValidSubset build_common_valid_subset(
    const std::map<std::string, RunResult>& results,
    const std::map<std::string, std::vector<std::string>>& incompatible_ids)
{
    CommonValidSubset cvs;
    for (const auto& kv : results) cvs.model_ids.push_back(kv.first);

    // Build union of all item IDs in any result
    std::set<std::string> all_ids;
    for (const auto& kv : results)
        for (const auto& ir : kv.second.items)
            all_ids.insert(ir.id);
    cvs.total_items = (int)all_ids.size();

    // Build per-model incompatible set
    std::map<std::string, std::set<std::string>> incompat;
    for (const auto& kv : incompatible_ids)
        incompat[kv.first] = std::set<std::string>(kv.second.begin(), kv.second.end());

    // Valid = not incompatible in any model
    std::set<std::string> common_valid = all_ids;
    for (const auto& kv : incompat)
        for (const auto& id : kv.second)
            common_valid.erase(id);
    cvs.valid_items = (int)common_valid.size();

    // Per-model stats
    for (const auto& kv : results) {
        const std::string& mid = kv.first;
        const RunResult& rr = kv.second;

        int n_full_corr = 0, n_full_tot = 0;
        int n_sub_corr  = 0, n_sub_tot  = 0;
        int n_incompat  = 0;

        const auto* incompat_set = incompat.count(mid) ? &incompat.at(mid) : nullptr;

        for (const auto& ir : rr.items) {
            if (!ir.ok) continue;
            bool is_incompat = incompat_set && incompat_set->count(ir.id);
            if (is_incompat) { n_incompat++; continue; }
            n_full_tot++;
            if (ir.correct) n_full_corr++;
            if (common_valid.count(ir.id)) {
                n_sub_tot++;
                if (ir.correct) n_sub_corr++;
            }
        }

        cvs.full_accuracy[mid]   = (n_full_tot > 0) ? (double)n_full_corr / n_full_tot : -1.0;
        cvs.subset_accuracy[mid] = (n_sub_tot  > 0) ? (double)n_sub_corr  / n_sub_tot  : -1.0;
        cvs.n_compat_issues[mid] = incompat_set ? (int)incompat_set->size() : 0;
    }
    return cvs;
}

nlohmann::json CommonValidSubset::to_json() const
{
    nlohmann::json fa = nlohmann::json::object();
    nlohmann::json sa = nlohmann::json::object();
    nlohmann::json nc = nlohmann::json::object();
    for (const auto& kv : full_accuracy)   fa[kv.first] = kv.second;
    for (const auto& kv : subset_accuracy) sa[kv.first] = kv.second;
    for (const auto& kv : n_compat_issues) nc[kv.first] = kv.second;
    return {
        {"model_ids",       model_ids},
        {"total_items",     total_items},
        {"valid_items",     valid_items},
        {"full_accuracy",   fa},
        {"subset_accuracy", sa},
        {"n_compat_issues", nc}
    };
}

// ---------------------------------------------------------------------------
// ValueProfile
// ---------------------------------------------------------------------------

nlohmann::json ValueProfile::to_json() const
{
    return {
        {"version",               version},
        {"timestamp",             timestamp},
        {"pjev_commit",           pjev_commit},
        {"llamacpp_revision",     llamacpp_revision},
        {"model",                 model.to_json()},
        {"frozen_config", {
            {"layout",            frozen_config.layout_str()},
            {"scheme",            frozen_config.scheme_str()},
            {"option_order",      frozen_config.order_str()},
            {"prior_correction",  frozen_config.prior_correction}
        }},
        {"quality_en",            quality_en.to_json()},
        {"quality_ja",            quality_ja.to_json()},
        {"calibration_en",        calibration_en.to_json()},
        {"calibration_ja",        calibration_ja.to_json()},
        {"multilingual",          multilingual.to_json()},
        {"performance",           performance.to_json()},
        {"candidate_compat",      candidate_compat.to_json()},
        {"compatibility_issues",  compatibility_issues}
    };
}

// ---------------------------------------------------------------------------
// Phase5Decision
// ---------------------------------------------------------------------------

nlohmann::json Phase5Decision::to_json() const
{
    return {
        {"phase",                  phase},
        {"timestamp",              timestamp},
        {"baseline_model",         baseline_model},
        {"evaluated_models",       evaluated_models},
        {"selected_default_model", selected_default_model},
        {"evidence_artifacts",     evidence_artifacts},
        {"notes",                  notes}
    };
}

bool Phase5Decision::from_json(const nlohmann::json& j, Phase5Decision& out, std::string& err)
{
    try {
        if (!j.contains("phase") || j["phase"].get<int>() != 5) {
            err = "missing or wrong \"phase\" field (expected 5)";
            return false;
        }
        out.phase = 5;
        if (j.contains("timestamp"))              out.timestamp              = j["timestamp"].get<std::string>();
        if (j.contains("baseline_model"))         out.baseline_model         = j["baseline_model"].get<std::string>();
        if (j.contains("selected_default_model")) out.selected_default_model = j["selected_default_model"].get<std::string>();
        if (j.contains("evaluated_models"))
            for (const auto& x : j["evaluated_models"]) out.evaluated_models.push_back(x.get<std::string>());
        if (j.contains("evidence_artifacts"))
            for (const auto& x : j["evidence_artifacts"]) out.evidence_artifacts.push_back(x.get<std::string>());
        if (j.contains("notes"))
            for (const auto& x : j["notes"]) out.notes.push_back(x.get<std::string>());
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

} // namespace pjev
