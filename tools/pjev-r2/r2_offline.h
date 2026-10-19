#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "r1_metrics.h"
#include "r2_gate.h"

#include "nlohmann/json.hpp"

namespace pjev
{
namespace r2
{

enum class Partition { Unassigned = -1, Tuning = 0, Evaluation = 1 };
const char* to_string(Partition p);

// One query of an R1 (or gated online) predictions.jsonl, reduced to what R2 needs.
struct Record {
    std::string query_id;
    std::string kind;
    int         number_of_levels = 0;
    int         ground_truth     = -1;
    std::vector<std::string> option_labels;

    bool        s1_ok      = false;
    int         s1_pred    = -1;
    bool        s1_correct = false;
    double      s1_margin  = 0.0;   // kept for R3; never used by the R2 gate
    double      s1_ms      = 0.0;
    int         s1_tokens  = 0;

    std::string retrieval_status;
    bool        retrieval_ok = false;
    double      hamming      = 0.0;
    double      similarity   = 0.0; // estimated cosine, recomputed from the Hamming distance
    std::string example_id;
    std::string demo_label;
    bool        same_group   = false;
    double      retrieval_ms = 0.0; // query embedding + vector search + row fetch + LZ4 decompression

    std::string r1_status;
    bool        r1_ok      = false;
    int         r1_pred    = -1;
    bool        r1_correct = false;
    std::vector<float> r1_logits;
    double      r1_ms      = 0.0;   // 1-shot decision: prompt construction + evaluation + scoring
    int         r1_tokens  = 0;

    // Present only in gated online runs.
    bool        has_gate = false;
    double      gate_threshold = 0.0;
    std::string gate_decision;

    // Filled by apply_partition().
    std::string group_id;
    std::string unit;               // partition unit: "family:<scenario_family>" or "group:<group_id>"
    std::string question;           // short query summary for examples.md
    Partition   partition = Partition::Unassigned;

    // A record usable for the three-way S1 / R1 / R2 comparison.
    bool paired() const { return s1_ok && retrieval_ok && r1_ok && valid_similarity(similarity); }
};

struct LoadStats {
    int lines             = 0;
    int parse_errors      = 0;
    int similarity_mismatch = 0;  // stored estimated_cosine disagrees with cos(pi*h/dim)
    int invalid_similarity  = 0;
    std::map<std::string, int> retrieval_status;
    std::map<std::string, int> s1_status;
    std::map<std::string, int> r1_status;
};

// Parse predictions.jsonl written by pjev-r1. `dim` is the embedding dimension used for
// cos(pi * hamming / dim). Returns false with `err` on I/O failure or duplicate query IDs.
bool load_predictions(const std::string& path, int dim, std::vector<Record>& out, LoadStats& stats,
                      std::string& err);
bool parse_record(const nlohmann::ordered_json& j, int dim, Record& r, LoadStats& stats);

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

struct Policy {
    enum class Type { AllS1, AllR1, Global, PerKind };
    Type        type = Type::AllS1;
    double      threshold = 0.0;                              // Global
    std::map<std::string, std::optional<double>> per_kind;    // PerKind; nullopt = never activate
    std::string name;

    static Policy all_s1();
    static Policy all_r1();
    static Policy global(double t);
};

// Whether `p` would run the 1-shot pass for `r`. Uses only kind, retrieval status and similarity.
bool activates(const Policy& p, const Record& r);

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

struct Stats {
    int n = 0;
    int activated = 0;
    int s1_correct = 0, r1_correct = 0, r2_correct = 0;
    r1::TransitionCounts r1;   // S1 -> R1
    r1::TransitionCounts r2;   // S1 -> R2
    r1::TransitionCounts r2_vs_r1;  // "s1" slot = R1, "r1" slot = R2
    int rescue_retained = 0, rescue_lost = 0, harm_retained = 0, harm_prevented = 0;
    double sim_activated_sum = 0.0, sim_rejected_sum = 0.0;
    // Per-query measured costs (ms), summed.
    double s1_ms = 0.0, retrieval_ms = 0.0, r1_pass_ms = 0.0, r1_pass_ms_activated = 0.0;

    int    rejected() const { return n - activated; }
    double rate(int k) const { return n ? static_cast<double>(k) / n : 0.0; }
    double s1_accuracy() const { return rate(s1_correct); }
    double r1_accuracy() const { return rate(r1_correct); }
    double r2_accuracy() const { return rate(r2_correct); }
    double activation_rate() const { return rate(activated); }
    double mean_sim_activated() const { return activated ? sim_activated_sum / activated : 0.0; }
    double mean_sim_rejected() const { return rejected() ? sim_rejected_sum / rejected() : 0.0; }
    // Estimated per-query cost models from R1 measurements.
    double est_s1_ms() const { return n ? s1_ms / n : 0.0; }
    double est_r1_deployed_ms() const { return n ? (retrieval_ms + r1_pass_ms) / n : 0.0; }
    double est_r2_ms() const { return n ? (s1_ms + retrieval_ms + r1_pass_ms_activated) / n : 0.0; }
};

using RecordSet = std::vector<const Record*>;

// All records must be paired().
Stats evaluate(const RecordSet& set, const Policy& p);

int r2_prediction(const Policy& p, const Record& r);

// ---------------------------------------------------------------------------
// Sweeps and selection
// ---------------------------------------------------------------------------

struct SweepRow {
    std::string sweep;      // endpoint | grid | exact | quantile
    std::string label;
    double      threshold = 0.0;
    bool        has_threshold = true;
    Stats       stats;
};

std::vector<SweepRow> sweep_endpoints(const RecordSet& set);
std::vector<SweepRow> sweep_grid(const RecordSet& set, double lo, double hi, double step);
// One row per distinct observed similarity (each yields a distinct activation set), highest first.
std::vector<SweepRow> sweep_exact(const RecordSet& set);
std::vector<SweepRow> sweep_quantile(const RecordSet& set, int n_quantiles);

struct Selection {
    bool   all_s1     = false;  // no threshold beat never-activate
    double threshold  = 0.0;
    int    candidates = 0;
    Stats  tuning;
};

// Maximise R2 correct count on `tuning`; ties go to the lower activation count (never-activate
// counts as zero activations). Candidate thresholds are the observed tuning similarities.
Selection select_threshold(const RecordSet& tuning);

// ---------------------------------------------------------------------------
// Partition
// ---------------------------------------------------------------------------

struct PartitionInfo {
    std::string dataset_path;
    std::string dataset_sha256;
    std::vector<std::string> tuning_families, evaluation_families;
    int matched = 0, unmatched = 0;
    int group_overlap = 0;          // groups present in both partitions (must be 0)
    int train_split_records = 0;    // query records whose split is "train" (must be 0)
};

// Deterministic group-disjoint tuning / evaluation split of the queries in `records`, using the
// source dataset's metadata. Unit = scenario family when present (whole families alternate by
// sorted name), otherwise group_id (xxHash64 parity).
bool apply_partition(const std::string& dataset_path, std::vector<Record>& records, PartitionInfo& info,
                     std::string& err);

// ---------------------------------------------------------------------------
// Uncertainty
// ---------------------------------------------------------------------------

struct BootstrapCI {
    double diff = 0.0, lo = 0.0, hi = 0.0;
    int    reps = 0;
    int    clusters = 0;
};

// Paired cluster bootstrap of mean(a) - mean(b); `cluster` keys are resampled together.
BootstrapCI paired_bootstrap(const std::vector<std::string>& cluster, const std::vector<int>& a,
                             const std::vector<int>& b, int reps, uint64_t seed);

// ---------------------------------------------------------------------------
// Compatibility
// ---------------------------------------------------------------------------

// Keys of pjev-r1 config.json that must match before predictions may be combined or compared.
const std::vector<std::string>& identity_keys();
// Returns the differing keys (empty = compatible).
std::vector<std::string> config_differences(const nlohmann::ordered_json& a, const nlohmann::ordered_json& b);

} // namespace r2
} // namespace pjev
