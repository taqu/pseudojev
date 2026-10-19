#include "r2_offline.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <set>
#include <unordered_set>

#include "id_hash.h"

namespace pjev
{
namespace r2
{

using json = nlohmann::ordered_json;

const char* to_string(Partition p)
{
    switch (p) {
    case Partition::Tuning:     return "tuning";
    case Partition::Evaluation: return "evaluation";
    default:                    return "unassigned";
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

static double num(const json& j, const char* key, double def = 0.0)
{
    auto it = j.find(key);
    return (it != j.end() && it->is_number()) ? it->get<double>() : def;
}

static std::string str(const json& j, const char* key)
{
    auto it = j.find(key);
    return (it != j.end() && it->is_string()) ? it->get<std::string>() : std::string();
}

bool parse_record(const json& j, int dim, Record& r, LoadStats& stats)
{
    if (!j.contains("query_id") || !j.contains("s1") || !j.contains("r1") || !j.contains("retrieval")) {
        return false;
    }
    r.query_id         = j["query_id"].get<std::string>();
    r.kind             = str(j, "kind");
    r.number_of_levels = static_cast<int>(num(j, "number_of_levels"));
    r.ground_truth     = static_cast<int>(num(j, "ground_truth_index", -1));
    if (j.contains("option_labels") && j["option_labels"].is_array()) {
        for (const auto& l : j["option_labels"]) r.option_labels.push_back(l.get<std::string>());
    }

    const json& s1 = j["s1"];
    const std::string s1_status = str(s1, "status");
    stats.s1_status[s1_status]++;
    r.s1_ok      = (s1_status == "ok");
    r.s1_pred    = static_cast<int>(num(s1, "prediction_index", -1));
    r.s1_correct = r.s1_ok && s1.value("correct", false);
    r.s1_margin  = num(s1, "margin");
    r.s1_ms      = num(s1, "latency_ms");
    r.s1_tokens  = static_cast<int>(num(s1, "prompt_tokens"));

    const json& rt = j["retrieval"];
    r.retrieval_status = str(rt, "status");
    stats.retrieval_status[r.retrieval_status]++;
    r.retrieval_ok = (r.retrieval_status == "ok");
    r.retrieval_ms = num(rt, "embed_ms") + num(rt, "search_ms") + num(rt, "fetch_ms") + num(rt, "decompress_ms");
    if (r.retrieval_ok) {
        r.hamming    = num(rt, "hamming_distance", std::nan(""));
        r.similarity = r1::estimated_cosine(r.hamming, dim);
        if (!valid_similarity(r.similarity)) stats.invalid_similarity++;
        // Stored values come from the same function; any disagreement means a different dim or metric.
        if (rt.contains("estimated_cosine") && std::fabs(num(rt, "estimated_cosine") - r.similarity) > 1e-12) {
            stats.similarity_mismatch++;
        }
        r.example_id = str(rt, "example_id");
        r.demo_label = str(rt, "answer_label");
        r.same_group = rt.value("same_group", false);
    }

    const json& r1j = j["r1"];
    r.r1_status = str(r1j, "status");
    stats.r1_status[r.r1_status]++;
    r.r1_ok      = (r.r1_status == "ok");
    r.r1_pred    = static_cast<int>(num(r1j, "prediction_index", -1));
    r.r1_correct = r.r1_ok && r1j.value("correct", false);
    r.r1_ms      = num(r1j, "latency_ms");
    r.r1_tokens  = static_cast<int>(num(r1j, "prompt_tokens"));
    if (r1j.contains("logits") && r1j["logits"].is_array()) {
        for (const auto& v : r1j["logits"]) r.r1_logits.push_back(v.get<float>());
    }

    if (j.contains("gate") && j["gate"].is_object()) {
        r.has_gate       = true;
        r.gate_threshold = num(j["gate"], "threshold");
        r.gate_decision  = str(j["gate"], "decision");
    }
    return true;
}

bool load_predictions(const std::string& path, int dim, std::vector<Record>& out, LoadStats& stats,
                      std::string& err)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        err = "cannot open " + path;
        return false;
    }
    std::unordered_set<std::string> seen;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        stats.lines++;
        Record r;
        try {
            if (!parse_record(json::parse(line), dim, r, stats)) { stats.parse_errors++; continue; }
        } catch (const std::exception&) {
            stats.parse_errors++;
            continue;
        }
        if (!seen.insert(r.query_id).second) {
            err = "duplicate query_id in " + path + ": " + r.query_id;
            return false;
        }
        out.push_back(std::move(r));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Policies
// ---------------------------------------------------------------------------

Policy Policy::all_s1()
{
    Policy p;
    p.type = Type::AllS1;
    p.name = "all_s1";
    return p;
}

Policy Policy::all_r1()
{
    Policy p;
    p.type = Type::AllR1;
    p.name = "all_eligible_r1";
    return p;
}

Policy Policy::global(double t)
{
    Policy p;
    p.type      = Type::Global;
    p.threshold = t;
    p.name      = "global";
    return p;
}

bool activates(const Policy& p, const Record& r)
{
    switch (p.type) {
    case Policy::Type::AllS1: return false;
    case Policy::Type::AllR1: return r.retrieval_ok;
    case Policy::Type::Global:
        return gate_decide(r.retrieval_ok, r.similarity, p.threshold) == GateDecision::Activated;
    case Policy::Type::PerKind: {
        auto it = p.per_kind.find(r.kind);
        if (it == p.per_kind.end() || !it->second) return false;
        return gate_decide(r.retrieval_ok, r.similarity, *it->second) == GateDecision::Activated;
    }
    }
    return false;
}

int r2_prediction(const Policy& p, const Record& r)
{
    return (activates(p, r) && r.r1_ok) ? r.r1_pred : r.s1_pred;
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

Stats evaluate(const RecordSet& set, const Policy& p)
{
    Stats s;
    for (const Record* r : set) {
        const bool act = activates(p, *r);
        const bool r2_correct = act ? r->r1_correct : r->s1_correct;
        s.n++;
        s.s1_correct += r->s1_correct;
        s.r1_correct += r->r1_correct;
        s.r2_correct += r2_correct;
        const r1::Transition t1 = r1::classify(r->s1_correct, r->r1_correct);
        s.r1.add(t1);
        s.r2.add(r1::classify(r->s1_correct, r2_correct));
        s.r2_vs_r1.add(r1::classify(r->r1_correct, r2_correct));
        if (t1 == r1::Transition::Rescue) (act ? s.rescue_retained : s.rescue_lost)++;
        if (t1 == r1::Transition::Harm)   (act ? s.harm_retained : s.harm_prevented)++;
        s.s1_ms        += r->s1_ms;
        s.retrieval_ms += r->retrieval_ms;
        s.r1_pass_ms   += r->r1_ms;
        if (act) {
            s.activated++;
            s.sim_activated_sum    += r->similarity;
            s.r1_pass_ms_activated += r->r1_ms;
        } else {
            s.sim_rejected_sum += r->similarity;
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Sweeps
// ---------------------------------------------------------------------------

std::vector<SweepRow> sweep_endpoints(const RecordSet& set)
{
    std::vector<SweepRow> rows;
    SweepRow a{"endpoint", "all_s1", 0.0, false, evaluate(set, Policy::all_s1())};
    SweepRow b{"endpoint", "all_eligible_r1", 0.0, false, evaluate(set, Policy::all_r1())};
    rows.push_back(a);
    rows.push_back(b);
    return rows;
}

std::vector<SweepRow> sweep_grid(const RecordSet& set, double lo, double hi, double step)
{
    std::vector<SweepRow> rows;
    if (!(step > 0.0)) return rows;
    const int n = static_cast<int>(std::floor((hi - lo) / step + 1e-9));
    for (int i = 0; i <= n; ++i) {
        // Computed from the index, not accumulated, so grid values are exact decimal steps.
        const double t = lo + step * i;
        char label[32];
        snprintf(label, sizeof(label), "%.4f", t);
        rows.push_back({"grid", label, t, true, evaluate(set, Policy::global(t))});
    }
    return rows;
}

static std::vector<double> distinct_similarities_desc(const RecordSet& set)
{
    std::set<double> s;
    for (const Record* r : set) if (r->retrieval_ok && valid_similarity(r->similarity)) s.insert(r->similarity);
    return std::vector<double>(s.rbegin(), s.rend());
}

std::vector<SweepRow> sweep_exact(const RecordSet& set)
{
    std::vector<SweepRow> rows;
    for (double t : distinct_similarities_desc(set)) {
        char label[32];
        snprintf(label, sizeof(label), "%.17g", t);
        rows.push_back({"exact", label, t, true, evaluate(set, Policy::global(t))});
    }
    return rows;
}

std::vector<SweepRow> sweep_quantile(const RecordSet& set, int n_quantiles)
{
    std::vector<SweepRow> rows;
    std::vector<double> v;
    for (const Record* r : set) if (r->retrieval_ok && valid_similarity(r->similarity)) v.push_back(r->similarity);
    if (v.empty() || n_quantiles < 1) return rows;
    std::sort(v.begin(), v.end());
    // Threshold = the q-th quantile of observed similarity (nearest observed value at or below it),
    // so roughly (1 - q) of queries activate.
    for (int i = 0; i <= n_quantiles; ++i) {
        const double q   = static_cast<double>(i) / n_quantiles;
        const size_t idx = std::min(v.size() - 1, static_cast<size_t>(q * (v.size() - 1)));
        char label[32];
        snprintf(label, sizeof(label), "q%.2f", q);
        rows.push_back({"quantile", label, v[idx], true, evaluate(set, Policy::global(v[idx]))});
    }
    return rows;
}

Selection select_threshold(const RecordSet& tuning)
{
    Selection best;
    best.all_s1 = true;
    best.tuning = evaluate(tuning, Policy::all_s1());
    best.candidates = 1;
    for (double t : distinct_similarities_desc(tuning)) {
        Stats s = evaluate(tuning, Policy::global(t));
        best.candidates++;
        // Strictly better accuracy wins; equal accuracy keeps the earlier (lower-activation) choice,
        // because candidates are visited in increasing activation order.
        if (s.r2_correct > best.tuning.r2_correct) {
            best.all_s1    = false;
            best.threshold = t;
            best.tuning    = s;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Partition
// ---------------------------------------------------------------------------

bool apply_partition(const std::string& dataset_path, std::vector<Record>& records, PartitionInfo& info,
                     std::string& err)
{
    info.dataset_path = dataset_path;
    std::unordered_map<std::string, Record*> by_id;
    for (auto& r : records) by_id[r.query_id] = &r;

    std::ifstream f(dataset_path, std::ios::binary);
    if (!f.is_open()) {
        err = "cannot open " + dataset_path;
        return false;
    }
    std::string line;
    std::set<std::string> families;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        json j;
        try { j = json::parse(line); } catch (...) { continue; }
        const std::string id = str(j, "id");
        auto it = by_id.find(id);
        if (it == by_id.end()) continue;
        Record& r = *it->second;
        info.matched++;
        if (str(j, "split") == "train") info.train_split_records++;
        r.group_id = str(j, "group_id");
        std::string fam;
        if (j.contains("metadata") && j["metadata"].is_object()) fam = str(j["metadata"], "scenario_family");
        r.unit = fam.empty() ? "group:" + r.group_id : "family:" + fam;
        if (!fam.empty()) families.insert(fam);
        std::string q = str(j, "question");
        if (q.size() > 200) q = q.substr(0, 200) + "...";
        r.question = q;
    }
    info.unmatched = static_cast<int>(records.size()) - info.matched;

    // Whole families alternate in sorted order so both sides receive families of every kind.
    std::unordered_map<std::string, Partition> family_side;
    int idx = 0;
    for (const auto& fam : families) {
        Partition p = (idx++ % 2 == 0) ? Partition::Tuning : Partition::Evaluation;
        family_side[fam] = p;
        (p == Partition::Tuning ? info.tuning_families : info.evaluation_families).push_back(fam);
    }
    std::unordered_map<std::string, int> group_sides;  // bit 1 = tuning, bit 2 = evaluation
    for (auto& r : records) {
        if (r.unit.empty()) continue;
        if (r.unit.rfind("family:", 0) == 0) {
            r.partition = family_side[r.unit.substr(7)];
        } else {
            const uint64_t h = static_cast<uint64_t>(filldb::hash_exemplar_id(r.group_id));
            r.partition = (h & 1u) ? Partition::Evaluation : Partition::Tuning;
        }
        group_sides[r.group_id] |= (r.partition == Partition::Tuning) ? 1 : 2;
    }
    for (const auto& [g, sides] : group_sides) if (sides == 3) info.group_overlap++;
    return true;
}

// ---------------------------------------------------------------------------
// Uncertainty
// ---------------------------------------------------------------------------

BootstrapCI paired_bootstrap(const std::vector<std::string>& cluster, const std::vector<int>& a,
                             const std::vector<int>& b, int reps, uint64_t seed)
{
    BootstrapCI ci;
    ci.reps = reps;
    std::unordered_map<std::string, size_t> index;
    std::vector<double> d_sum;
    std::vector<int>    n_sum;
    double total_d = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        auto [it, inserted] = index.emplace(cluster[i], d_sum.size());
        if (inserted) { d_sum.push_back(0.0); n_sum.push_back(0); }
        d_sum[it->second] += a[i] - b[i];
        n_sum[it->second] += 1;
        total_d += a[i] - b[i];
    }
    ci.clusters = static_cast<int>(d_sum.size());
    if (a.empty()) return ci;
    ci.diff = total_d / a.size();

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, d_sum.size() - 1);
    std::vector<double> diffs;
    diffs.reserve(static_cast<size_t>(reps));
    for (int k = 0; k < reps; ++k) {
        double d = 0.0;
        long   n = 0;
        for (size_t c = 0; c < d_sum.size(); ++c) {
            const size_t j = pick(rng);
            d += d_sum[j];
            n += n_sum[j];
        }
        diffs.push_back(n ? d / n : 0.0);
    }
    ci.lo = r1::percentile(diffs, 2.5);
    ci.hi = r1::percentile(diffs, 97.5);
    return ci;
}

// ---------------------------------------------------------------------------
// Compatibility
// ---------------------------------------------------------------------------

const std::vector<std::string>& identity_keys()
{
    static const std::vector<std::string> keys = {
        "db_path", "train_path", "inference_model_identity", "embedding_fingerprint", "embedding_dim",
        "db_info", "query_prefix", "prompt_format_version", "prompt_format", "scheme", "layout",
        "ensemble", "calibration", "n_ctx", "top_k", "leakage_policy", "max_fetch_k"};
    return keys;
}

std::vector<std::string> config_differences(const json& a, const json& b)
{
    std::vector<std::string> diff;
    for (const auto& k : identity_keys()) {
        const bool ha = a.contains(k), hb = b.contains(k);
        if (!ha || !hb || a[k] != b[k]) diff.push_back(k);
    }
    return diff;
}

} // namespace r2
} // namespace pjev
