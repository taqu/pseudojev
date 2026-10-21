#pragma once

#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "r1_metrics.h"

#include "nlohmann/json.hpp"

namespace pjev
{
namespace r3
{

// One R3 inference (one query, one seed) as written by `pjev-r3 --mode run`.
struct SeedRecord {
    std::string query_id;
    std::string kind;
    uint64_t    seed           = 0;
    int         ground_truth   = -1;
    std::string ground_truth_label;
    std::string exemplar_status;
    std::string example_id;
    std::string demo_label;
    bool        same_as_nearest = false;  // R3-L: the draw is R1's nearest exemplar
    bool        has_similarity = false;
    double      hamming        = 0.0;
    double      similarity     = 0.0;   // estimated cosine
    std::string status;                 // r3 condition status
    int         prediction     = -1;
    std::string prediction_label;
    bool        correct        = false;
    int         prompt_tokens  = 0;
    double      select_ms = 0.0, fetch_ms = 0.0, decompress_ms = 0.0, decision_ms = 0.0, prompt_ms = 0.0,
                eval_ms = 0.0;

    bool ok() const { return status == "ok" && exemplar_status == "ok"; }
    double total_ms() const { return select_ms + fetch_ms + decompress_ms + decision_ms; }
};

bool parse_seed_record(const nlohmann::ordered_json& j, SeedRecord& r);

// Reference S1 / R1 outcome of one held-out query (from the R1 predictions.jsonl).
struct RefRecord {
    std::string query_id;
    std::string kind;
    std::string group_id;
    int         ground_truth = -1;
    bool        s1_ok = false, r1_ok = false;
    bool        s1_correct = false, r1_correct = false;
    int         s1_pred = -1, r1_pred = -1;
    std::string s1_label, r1_label, demo_label;
    double      similarity = 0.0;
    int         s1_tokens = 0, r1_tokens = 0;
    double      s1_ms = 0.0, retrieval_ms = 0.0, r1_ms = 0.0;
};

// Alignment of every seed with the reference on the same query set.
struct Alignment {
    std::vector<std::string> query_ids;          // evaluated by every strategy (all statuses)
    std::vector<std::string> analysed;           // S1, R1 and every seed ok
    int missing_in_seed     = 0;                 // expected query absent from a seed file
    int unexpected_in_seed  = 0;                 // seed query outside the expected set
    int duplicate_in_seed   = 0;
    int ground_truth_mismatch = 0;               // seed and reference disagree on kind / ground truth
    std::map<uint64_t, int> failed_per_seed;     // R3 not ok
    int reference_failed    = 0;                 // S1 or R1 not ok
    bool exact() const
    {
        return missing_in_seed == 0 && unexpected_in_seed == 0 && duplicate_in_seed == 0 && ground_truth_mismatch == 0;
    }
};

// `expected`: held-out query IDs (all of them, or the run's subset); `seeds[s]` the records of seed s.
Alignment align(const std::vector<std::string>& expected, const std::map<std::string, RefRecord>& ref,
                const std::map<uint64_t, std::vector<SeedRecord>>& seeds);

struct SeedSummary {
    double mean = 0.0, sd = 0.0, min = 0.0, max = 0.0;
    int    n = 0;
};
// Sample standard deviation (n - 1); sd = 0 for a single value.
SeedSummary summarize(const std::vector<double>& v);

struct ClusterCI {
    double diff = 0.0, lo = 0.0, hi = 0.0;
    int    clusters = 0, reps = 0;
};

// Paired cluster bootstrap of mean(a - b) over queries; queries sharing a cluster key are resampled
// together. Values are per-query (a seed-averaged correctness may be fractional), so every seed of a
// sampled query stays in the replicate.
ClusterCI cluster_bootstrap(const std::vector<std::string>& cluster, const std::vector<double>& a,
                            const std::vector<double>& b, int reps, uint64_t seed);

// Cluster resampling shared by every bootstrap statistic: replicate k draws clusters with replacement
// from one seeded generator, so all statistics of one report use the same replicates.
class Resampler
{
public:
    Resampler(const std::vector<std::string>& cluster, int reps, uint64_t seed) : reps_(reps)
    {
        std::unordered_map<std::string, size_t> index;
        of_.reserve(cluster.size());
        for (const auto& c : cluster) {
            auto [it, ins] = index.emplace(c, index.size());
            of_.push_back(it->second);
        }
        n_clusters_ = index.size();
        std::mt19937_64 rng(seed);
        counts_.assign(static_cast<size_t>(reps), std::vector<int>(n_clusters_, 0));
        if (n_clusters_ == 0) return;
        std::uniform_int_distribution<size_t> pick(0, n_clusters_ - 1);
        for (int k = 0; k < reps; ++k)
            for (size_t c = 0; c < n_clusters_; ++c) counts_[static_cast<size_t>(k)][pick(rng)]++;
    }
    size_t clusters() const { return n_clusters_; }
    int    reps() const { return reps_; }
    // Per-cluster sums of a per-query series.
    std::vector<double> sums(const std::vector<double>& x) const
    {
        std::vector<double> s(n_clusters_, 0.0);
        for (size_t i = 0; i < x.size(); ++i) s[of_[i]] += x[i];
        return s;
    }
    double weighted(int k, const std::vector<double>& cluster_sums) const
    {
        double t = 0.0;
        const auto& w = counts_[static_cast<size_t>(k)];
        for (size_t c = 0; c < n_clusters_; ++c) t += w[c] * cluster_sums[c];
        return t;
    }

private:
    int reps_;
    size_t n_clusters_ = 0;
    std::vector<size_t> of_;
    std::vector<std::vector<int>> counts_;
};

inline ClusterCI paired_ci(const Resampler& rs, const std::vector<double>& a, const std::vector<double>& b)
{
    ClusterCI ci;
    ci.reps = rs.reps();
    ci.clusters = static_cast<int>(rs.clusters());
    if (a.empty()) return ci;
    std::vector<double> d(a.size()), ones(a.size(), 1.0);
    double total = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { d[i] = a[i] - b[i]; total += d[i]; }
    ci.diff = total / a.size();
    const auto ds = rs.sums(d), ns = rs.sums(ones);
    std::vector<double> reps;
    reps.reserve(static_cast<size_t>(rs.reps()));
    for (int k = 0; k < rs.reps(); ++k) {
        const double n = rs.weighted(k, ns);
        reps.push_back(n > 0 ? rs.weighted(k, ds) / n : 0.0);
    }
    ci.lo = r1::percentile(reps, 2.5);
    ci.hi = r1::percentile(reps, 97.5);
    return ci;
}


// 2x2 agreement of two correctness vectors.
struct Agreement {
    int both_correct = 0, a_only = 0, b_only = 0, both_incorrect = 0;
    void add(bool a, bool b)
    {
        if (a && b) both_correct++;
        else if (a) a_only++;
        else if (b) b_only++;
        else both_incorrect++;
    }
    int n() const { return both_correct + a_only + b_only + both_incorrect; }
};

// `pjev-r3 --mode aggregate`.
int aggregate_main(int argc, char** argv);

} // namespace r3
} // namespace pjev
