// pjev-r2: R2 similarity-gated retrieved 1-shot.
//   offline: replay paired S1/R1 predictions from pjev-r1 under similarity thresholds, select a
//            threshold on a tuning partition only, and report the frozen threshold on held-out data.
//   online:  the pjev-r1 harness with the gate inserted before the 1-shot pass (rejected queries
//            never run the second inference).
//   parity:  check that an online gated run reproduces the offline replay of a reference R1 run.

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "model/model_identity.h"

#include "id_hash.h"
#include "r1_harness.h"
#include "r1_metrics.h"
#include "r2_gate.h"
#include "r2_offline.h"

#include "sqlite/sqlite3.h"

#include "nlohmann/json.hpp"

using namespace pjev;
using namespace pjev::r2;
using json = nlohmann::ordered_json;

static const char* kKinds[] = {"all", "choice", "noul", "score"};
static const char* kSelectionPolicy =
    "Candidates: never activate (all S1) plus every distinct Top-1 similarity observed on the tuning "
    "partition (gate: similarity >= threshold). Objective: maximise paired R2 accuracy on tuning. "
    "Ties: lower activation count wins (never-activate counts as zero). Fixed before any held-out "
    "result was computed; held-out data never enters selection.";

static void print_usage(const char* prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s --mode offline (--input PRED.jsonl [--split-dataset DATA.jsonl]\n"
        "                     | --tuning-input PRED.jsonl --evaluation-input PRED.jsonl)\n"
        "        [--select-threshold] [--per-kind-thresholds] [--output DIR]\n"
        "        [--grid-lo X] [--grid-hi X] [--grid-step X] [--quantiles N]\n"
        "        [--bootstrap N] [--seed N] [--db PATH] [--train PATH] [--allow-unverified-config]\n"
        "  %s --mode online (--threshold T | --threshold-from selected_threshold.json) [pjev-r1 options]\n"
        "  %s --mode parity --online PRED.jsonl --reference PRED.jsonl [--output DIR]\n"
        "\n"
        "Offline:\n"
        "  --input PATH             pjev-r1 predictions.jsonl (config.json is read from the same directory)\n"
        "  --split-dataset PATH     Source dataset of --input; builds the deterministic group-disjoint\n"
        "                           tuning/evaluation partition (scenario family, else group_id)\n"
        "  --tuning-input / --evaluation-input  Separate, compatible pjev-r1 runs instead of a split\n"
        "  --select-threshold       Select the global threshold on tuning and freeze it for evaluation\n"
        "  --per-kind-thresholds    Also select per-kind thresholds (secondary diagnostic)\n"
        "  --grid-lo/--grid-hi/--grid-step   Fixed grid (default 0.00 .. 1.00 step 0.05)\n"
        "  --quantiles N            Quantile sweep resolution (default 10)\n"
        "  --bootstrap N            Paired cluster-bootstrap replicates (default 2000)\n"
        "  --seed N                 Bootstrap seed (default 20261016)\n"
        "  --db PATH                Exemplar DB (read-only) for exemplar summaries in examples.md\n"
        "  --train PATH             Training split; verifies no query is a training record\n"
        "  --allow-unverified-config  Accept predictions without config.json (marked unverified)\n"
        "  --output DIR             Output directory (default: results/r2)\n"
        "\n"
        "Online: every option of pjev-r1 is accepted; --output defaults to results/r2-online.\n"
        "  No default threshold exists: one must be given explicitly.\n",
        prog, prog, prog);
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static std::string fmt(double v, int prec = 2)
{
    char b[64];
    snprintf(b, sizeof(b), "%.*f", prec, v);
    return b;
}
static std::string pct(double v, int prec = 2) { return fmt(v * 100.0, prec) + "%"; }
static std::string pp(double v, int prec = 2)
{
    char b[64];
    snprintf(b, sizeof(b), "%+.*f pp", prec, v * 100.0);
    return b;
}
static std::string full(double v)
{
    char b[64];
    snprintf(b, sizeof(b), "%.17g", v);
    return b;
}

static bool read_json_file(const std::string& path, json& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    try { out = json::parse(f); } catch (...) { return false; }
    return true;
}

static std::string sibling(const std::string& path, const char* name)
{
    return (std::filesystem::path(path).parent_path() / name).string();
}

static bool open_out(const std::string& dir, const char* name, std::ofstream& f)
{
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    f.open((std::filesystem::path(dir) / name).string(), std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "error: cannot write %s/%s\n", dir.c_str(), name);
        return false;
    }
    return true;
}

static RecordSet filter_kind(const RecordSet& set, const std::string& kind)
{
    if (kind == "all") return set;
    RecordSet out;
    for (const Record* r : set) if (r->kind == kind) out.push_back(r);
    return out;
}

static int64_t parse_hex_id(const std::string& hex)
{
    return static_cast<int64_t>(std::strtoull(hex.c_str(), nullptr, 16));
}

// Wald / McNemar from the existing R1 paired test; `t` maps "s1" -> baseline, "r1" -> candidate.
static std::string mcnemar_text(const r1::TransitionCounts& t)
{
    const r1::PairedTest p = r1::paired_test(t);
    char b[160];
    snprintf(b, sizeof(b), "Wald 95%% [%+.2f, %+.2f] pp, McNemar p=%.3g (exact %.3g)",
             p.ci_low * 100, p.ci_high * 100, p.mcnemar_p, p.exact_p);
    return b;
}

// ---------------------------------------------------------------------------
// CSV rows
// ---------------------------------------------------------------------------

static const char* kSweepHeader =
    "split,kind,sweep,label,threshold,n_queries,n_activated,activation_rate,s1_accuracy,r1_accuracy,"
    "r2_accuracy,delta_vs_s1,delta_vs_r1,rescue,harm,net_gain,rescue_lost,harm_prevented,"
    "mean_activated_similarity,est_r2_latency_ms,selected\n";

static void sweep_csv_row(std::ostream& o, const std::string& split, const std::string& kind,
                          const SweepRow& row, bool selected)
{
    const Stats& s = row.stats;
    o << split << "," << kind << "," << row.sweep << "," << row.label << ","
      << (row.has_threshold ? full(row.threshold) : "") << "," << s.n << "," << s.activated << ","
      << full(s.activation_rate()) << "," << full(s.s1_accuracy()) << "," << full(s.r1_accuracy()) << ","
      << full(s.r2_accuracy()) << "," << full(s.r2_accuracy() - s.s1_accuracy()) << ","
      << full(s.r2_accuracy() - s.r1_accuracy()) << "," << s.r2.rescue << "," << s.r2.harm << ","
      << s.r2.net() << "," << s.rescue_lost << "," << s.harm_prevented << ","
      << full(s.mean_sim_activated()) << "," << full(s.est_r2_ms()) << "," << (selected ? 1 : 0) << "\n";
}

struct SweepOptions {
    double grid_lo = 0.0, grid_hi = 1.0, grid_step = 0.05;
    int    quantiles = 10;
};

static std::vector<SweepRow> all_sweeps(const RecordSet& set, const SweepOptions& o)
{
    std::vector<SweepRow> rows = sweep_endpoints(set);
    for (auto& r : sweep_grid(set, o.grid_lo, o.grid_hi, o.grid_step)) rows.push_back(r);
    for (auto& r : sweep_exact(set))                                   rows.push_back(r);
    for (auto& r : sweep_quantile(set, o.quantiles))                   rows.push_back(r);
    return rows;
}

// ---------------------------------------------------------------------------
// Offline
// ---------------------------------------------------------------------------

struct InputRun {
    std::string         path;
    json                config;
    bool                config_found = false;
    std::vector<Record> records;
    LoadStats           stats;
    int                 dim = 768;
};

static bool load_input(const std::string& path, bool allow_unverified, InputRun& run)
{
    run.path = path;
    run.config_found = read_json_file(sibling(path, "config.json"), run.config);
    if (!run.config_found && !allow_unverified) {
        fprintf(stderr, "error: %s has no config.json beside it, so model / DB / prompt identity cannot be "
                        "verified (use --allow-unverified-config to analyse it anyway)\n", path.c_str());
        return false;
    }
    if (run.config_found && run.config.contains("embedding_dim")) run.dim = run.config["embedding_dim"].get<int>();
    std::string err;
    if (!load_predictions(path, run.dim, run.records, run.stats, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return false;
    }
    if (run.stats.similarity_mismatch > 0) {
        fprintf(stderr, "error: %d records in %s store an estimated_cosine that does not match "
                        "cos(pi*h/%d); refusing to gate on an unknown metric\n",
                run.stats.similarity_mismatch, path.c_str(), run.dim);
        return false;
    }
    printf("Loaded %zu records from %s (parse errors %d, invalid similarity %d, config %s)\n",
           run.records.size(), path.c_str(), run.stats.parse_errors, run.stats.invalid_similarity,
           run.config_found ? "found" : "MISSING - identity unverified");
    return true;
}

static std::string exclusion_reason(const Record& r)
{
    if (!r.s1_ok)                          return "s1_unavailable";
    if (!r.retrieval_ok)                   return "retrieval_" + r.retrieval_status;
    if (!valid_similarity(r.similarity))   return "invalid_similarity";
    if (!r.r1_ok)                          return "r1_" + r.r1_status;
    return "";
}

struct ExemplarLookup {
    sqlite3*      db = nullptr;
    sqlite3_stmt* st = nullptr;
    explicit ExemplarLookup(const std::string& path)
    {
        if (path.empty()) return;
        if (sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
            sqlite3_close(db);
            db = nullptr;
            return;
        }
        sqlite3_prepare_v2(db, "SELECT question, family FROM examples WHERE id=?", -1, &st, nullptr);
    }
    ~ExemplarLookup()
    {
        if (st) sqlite3_finalize(st);
        if (db) sqlite3_close(db);
    }
    std::string summary(const std::string& hex)
    {
        if (!st || hex.empty()) return "(exemplar text not loaded; pass --db)";
        sqlite3_reset(st);
        sqlite3_bind_int64(st, 1, parse_hex_id(hex));
        if (sqlite3_step(st) != SQLITE_ROW) return "(exemplar not found)";
        auto col = [&](int c) {
            const char* v = reinterpret_cast<const char*>(sqlite3_column_text(st, c));
            return std::string(v ? v : "");
        };
        std::string q = col(0);
        if (q.size() > 200) q = q.substr(0, 200) + "...";
        return q + " [family: " + col(1) + "]";
    }
};

static std::string label_of(const Record& r, int idx)
{
    if (idx < 0) return "n/a";
    if (static_cast<size_t>(idx) < r.option_labels.size()) return r.option_labels[static_cast<size_t>(idx)];
    return std::to_string(idx);
}

static void example_block(std::ostream& o, const Record& r, const Policy& p, ExemplarLookup& ex)
{
    const bool act = activates(p, r);
    o << "#### " << r.query_id << "\n\n";
    o << "| Field | Value |\n|---|---|\n";
    o << "| Kind | " << r.kind << (r.kind == "score" ? " (" + std::to_string(r.number_of_levels) + " levels)" : "")
      << " |\n";
    o << "| Similarity (est. cosine) | " << fmt(r.similarity, 4) << " (Hamming " << fmt(r.hamming, 0) << ") |\n";
    o << "| Gate | " << (act ? "activated" : "rejected") << " at " << fmt(p.threshold, 4) << " |\n";
    o << "| Ground truth | " << label_of(r, r.ground_truth) << " |\n";
    o << "| S1 / R1 / R2 | " << label_of(r, r.s1_pred) << " / " << label_of(r, r.r1_pred) << " / "
      << label_of(r, r2_prediction(p, r)) << " |\n";
    o << "| Demo answer label | " << r.demo_label << (r.same_group ? " (same group)" : "") << " |\n\n";
    o << "- Query: " << (r.question.empty() ? "(not joined)" : r.question) << "\n";
    o << "- Exemplar " << r.example_id << ": " << ex.summary(r.example_id) << "\n\n";
}

static void write_examples(const std::string& dir, const RecordSet& eval, const Policy& p,
                           const std::string& db_path)
{
    std::ofstream o;
    if (!open_out(dir, "examples.md", o)) return;
    ExemplarLookup ex(db_path);
    o << "# R2 examples (held-out partition, frozen threshold " << full(p.threshold) << ")\n\n";
    o << "Representative cases, not cherry-picked successes: each category takes the first matches in "
         "file order, except the similarity extremes which take the highest / lowest similarity.\n\n";

    struct Cat { const char* title; std::function<bool(const Record&)> pred; };
    const std::vector<Cat> cats = {
        {"Activated rescue (S1 wrong, R1 right, gate on)",
         [&](const Record& r) { return activates(p, r) && !r.s1_correct && r.r1_correct; }},
        {"Activated harm (S1 right, R1 wrong, gate on)",
         [&](const Record& r) { return activates(p, r) && r.s1_correct && !r.r1_correct; }},
        {"Rejected rescue opportunity (R1 would have rescued, gate off)",
         [&](const Record& r) { return !activates(p, r) && !r.s1_correct && r.r1_correct; }},
        {"Prevented harm (R1 would have harmed, gate off)",
         [&](const Record& r) { return !activates(p, r) && r.s1_correct && !r.r1_correct; }},
    };
    for (const auto& c : cats) {
        o << "## " << c.title << "\n\n";
        int k = 0;
        for (const Record* r : eval) {
            if (!c.pred(*r)) continue;
            example_block(o, *r, p, ex);
            if (++k >= 3) break;
        }
        if (k == 0) o << "_No cases._\n\n";
    }
    auto extremes = [&](const char* title, bool want_harm, bool highest) {
        std::vector<const Record*> v;
        for (const Record* r : eval) {
            const bool harm = r->s1_correct && !r->r1_correct, rescue = !r->s1_correct && r->r1_correct;
            if (want_harm ? harm : rescue) v.push_back(r);
        }
        std::stable_sort(v.begin(), v.end(), [&](const Record* a, const Record* b) {
            return highest ? a->similarity > b->similarity : a->similarity < b->similarity;
        });
        o << "## " << title << "\n\n";
        for (size_t i = 0; i < v.size() && i < 3; ++i) example_block(o, *v[i], p, ex);
        if (v.empty()) o << "_No cases._\n\n";
    };
    extremes("High-similarity harm (R1 harms with the highest similarity)", true, true);
    extremes("Low-similarity rescue (R1 rescues with the lowest similarity)", false, false);
}

static json stats_json(const Stats& s)
{
    return {{"n", s.n}, {"activated", s.activated}, {"activation_rate", s.activation_rate()},
            {"s1_accuracy", s.s1_accuracy()}, {"r1_accuracy", s.r1_accuracy()},
            {"r2_accuracy", s.r2_accuracy()}, {"rescue", s.r2.rescue}, {"harm", s.r2.harm},
            {"rescue_lost", s.rescue_lost}, {"harm_prevented", s.harm_prevented}};
}

struct OfflineOptions {
    std::string input, split_dataset, tuning_input, evaluation_input, output = "results/r2";
    std::string db_path, train_path;
    bool select = false, per_kind = false, allow_unverified = false;
    SweepOptions sweep;
    int bootstrap = 2000;
    uint64_t seed = 20261016;
};

static int run_offline(const OfflineOptions& opt)
{
    const bool split_mode = !opt.tuning_input.empty() || !opt.evaluation_input.empty();
    if (split_mode && (opt.tuning_input.empty() || opt.evaluation_input.empty())) {
        fprintf(stderr, "error: --tuning-input and --evaluation-input must be given together\n");
        return 1;
    }
    if (!split_mode && opt.input.empty()) {
        fprintf(stderr, "error: --input (or --tuning-input/--evaluation-input) is required\n");
        return 1;
    }
    if (!(opt.sweep.grid_step > 0.0) || !valid_threshold(opt.sweep.grid_lo) || !valid_threshold(opt.sweep.grid_hi)) {
        fprintf(stderr, "error: grid must lie in [-1, 1] with a positive step\n");
        return 1;
    }

    // --- Load ---
    std::vector<InputRun> runs(split_mode ? 2 : 1);
    if (!load_input(split_mode ? opt.tuning_input : opt.input, opt.allow_unverified, runs[0])) return 1;
    if (split_mode && !load_input(opt.evaluation_input, opt.allow_unverified, runs[1])) return 1;

    json identity;
    identity["verified"] = runs[0].config_found && (!split_mode || runs[1].config_found);
    if (split_mode && runs[0].config_found && runs[1].config_found) {
        auto diff = config_differences(runs[0].config, runs[1].config);
        if (!diff.empty()) {
            fprintf(stderr, "error: tuning and evaluation runs are not comparable; differing keys:");
            for (const auto& k : diff) fprintf(stderr, " %s", k.c_str());
            fprintf(stderr, "\n");
            return 1;
        }
        identity["tuning_vs_evaluation"] = "identical on all identity keys";
    }
    if (runs[0].config_found && runs[0].config.value("preliminary", false)) {
        identity["warning"] = "source R1 run is preliminary (--limit/--offset/--seed/--kind subset)";
    }

    // --- Partition ---
    PartitionInfo pinfo;
    std::vector<Record*> all;
    if (split_mode) {
        std::unordered_set<std::string> tuning_ids;
        for (auto& r : runs[0].records) { r.partition = Partition::Tuning; tuning_ids.insert(r.query_id); }
        int overlap = 0;
        for (auto& r : runs[1].records) {
            r.partition = Partition::Evaluation;
            overlap += tuning_ids.count(r.query_id) > 0;
        }
        if (overlap > 0) {
            fprintf(stderr, "error: %d query IDs appear in both tuning and evaluation inputs\n", overlap);
            return 1;
        }
    } else if (!opt.split_dataset.empty()) {
        std::string err;
        if (!apply_partition(opt.split_dataset, runs[0].records, pinfo, err)) {
            fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        pinfo.dataset_sha256 = sha256_file(opt.split_dataset);
        if (pinfo.unmatched > 0) {
            fprintf(stderr, "error: %d prediction records are not in %s; wrong --split-dataset?\n",
                    pinfo.unmatched, opt.split_dataset.c_str());
            return 1;
        }
        if (pinfo.group_overlap > 0) {
            fprintf(stderr, "error: %d groups straddle the tuning/evaluation partition\n", pinfo.group_overlap);
            return 1;
        }
        if (pinfo.train_split_records > 0) {
            fprintf(stderr, "error: %d query records are labelled split=train\n", pinfo.train_split_records);
            return 1;
        }
        if (runs[0].config_found && runs[0].config.contains("input_sha256") &&
            runs[0].config["input_sha256"].get<std::string>() != pinfo.dataset_sha256) {
            fprintf(stderr, "error: --split-dataset differs from the dataset the R1 run evaluated (sha256)\n");
            return 1;
        }
    } else if (opt.select) {
        fprintf(stderr, "error: --select-threshold needs a tuning partition: --split-dataset or "
                        "--tuning-input/--evaluation-input\n");
        return 1;
    }
    for (auto& run : runs) for (auto& r : run.records) all.push_back(&r);

    // Leakage: no query may be a training record (exemplar bank).
    int train_overlap = -1;
    if (!opt.train_path.empty()) {
        std::unordered_set<std::string> ids;
        std::ifstream tf(opt.train_path, std::ios::binary);
        std::string line;
        while (std::getline(tf, line)) {
            try {
                auto j = json::parse(line);
                if (j.value("split", "") == "train") ids.insert(j.value("id", ""));
            } catch (...) {}
        }
        train_overlap = 0;
        for (const Record* r : all) train_overlap += ids.count(r->query_id) > 0;
        if (train_overlap > 0) {
            fprintf(stderr, "error: %d queries are training records\n", train_overlap);
            return 1;
        }
    }

    // --- Paired sets ---
    const bool has_partition = split_mode || !opt.split_dataset.empty();
    RecordSet tuning, evaluation, paired_all;
    std::map<std::string, int> excluded;
    for (const Record* r : all) {
        const std::string why = exclusion_reason(*r);
        if (!why.empty()) { excluded[why]++; continue; }
        paired_all.push_back(r);
        if (r->partition == Partition::Tuning)     tuning.push_back(r);
        if (r->partition == Partition::Evaluation) evaluation.push_back(r);
    }
    printf("Paired records: %zu (tuning %zu, evaluation %zu); excluded %zu\n", paired_all.size(),
           tuning.size(), evaluation.size(), all.size() - paired_all.size());
    if (paired_all.empty()) {
        fprintf(stderr, "error: no paired S1/R1 records\n");
        return 1;
    }

    // Endpoint self-check: replay must reproduce the observed S1 and R1 accuracies.
    {
        const Stats s1 = evaluate(paired_all, Policy::all_s1());
        const Stats r1 = evaluate(paired_all, Policy::all_r1());
        if (s1.r2_correct != s1.s1_correct || r1.r2_correct != r1.r1_correct || r1.activated != r1.n) {
            fprintf(stderr, "internal error: endpoint replay disagrees with observed S1/R1\n");
            return 3;
        }
    }

    // --- Selection (tuning only) ---
    Selection sel;
    std::map<std::string, Selection> sel_kind;
    Policy frozen = Policy::all_s1();
    Policy frozen_kind;
    frozen_kind.type = Policy::Type::PerKind;
    frozen_kind.name = "per_kind";
    if (opt.select) {
        sel = select_threshold(tuning);
        frozen = sel.all_s1 ? Policy::all_s1() : Policy::global(sel.threshold);
        if (opt.per_kind) {
            for (const char* k : {"choice", "noul", "score"}) {
                sel_kind[k] = select_threshold(filter_kind(tuning, k));
                frozen_kind.per_kind[k] = sel_kind[k].all_s1 ? std::nullopt
                                                             : std::optional<double>(sel_kind[k].threshold);
            }
        }
        printf("Selected on tuning: %s (tuning R2 %.2f%%, activation %.1f%%)\n",
               sel.all_s1 ? "never activate (all S1)" : ("threshold " + full(sel.threshold)).c_str(),
               sel.tuning.r2_accuracy() * 100, sel.tuning.activation_rate() * 100);
    }

    // --- Outputs ---
    const std::string& out = opt.output;
    const std::string sweep_split = has_partition ? "tuning" : "all";
    const RecordSet& sweep_set    = has_partition ? tuning : paired_all;

    {
        std::ofstream f;
        if (!open_out(out, "threshold_sweep.csv", f)) return 1;
        f << kSweepHeader;
        for (const char* k : kKinds) {
            for (const auto& row : all_sweeps(filter_kind(sweep_set, k), opt.sweep)) {
                const bool selected = opt.select && std::string(k) == "all" &&
                                      ((sel.all_s1 && row.label == "all_s1") ||
                                       (!sel.all_s1 && row.sweep == "exact" && row.threshold == sel.threshold));
                sweep_csv_row(f, sweep_split, k, row, selected);
            }
        }
    }
    if (has_partition) {
        // Exploratory only: written after selection and never read back by it.
        std::ofstream f;
        if (!open_out(out, "heldout_sweep_exploratory.csv", f)) return 1;
        f << kSweepHeader;
        for (const char* k : kKinds) {
            for (const auto& row : all_sweeps(filter_kind(evaluation, k), opt.sweep)) {
                sweep_csv_row(f, "evaluation_exploratory", k, row, false);
            }
        }
    }

    if (opt.select) {
        json s;
        s["policy"]          = sel.all_s1 ? "all_s1" : "global_threshold";
        s["threshold"]       = sel.all_s1 ? json(nullptr) : json(sel.threshold);
        s["threshold_hamming_equivalent"] = sel.all_s1 ? json(nullptr)
            : json(std::acos(sel.threshold) / 3.14159265358979323846 * runs[0].dim);
        s["metric"]          = "estimated_cosine = cos(pi * hamming / " + std::to_string(runs[0].dim) + ")";
        s["rule"]            = "similarity >= threshold";
        s["selection_policy"] = kSelectionPolicy;
        s["candidates"]      = sel.candidates;
        s["tuning"]          = stats_json(sel.tuning);
        if (opt.per_kind) {
            json pk;
            for (const auto& [k, ks] : sel_kind) {
                pk[k] = {{"policy", ks.all_s1 ? "all_s1" : "threshold"},
                         {"threshold", ks.all_s1 ? json(nullptr) : json(ks.threshold)},
                         {"tuning", stats_json(ks.tuning)}};
            }
            s["per_kind_diagnostic"] = pk;
        }
        std::ofstream f;
        if (!open_out(out, "selected_threshold.json", f)) return 1;
        f << s.dump(2) << "\n";
    }

    // Per-query predictions under the frozen policy.
    {
        std::ofstream f;
        if (!open_out(out, "predictions.jsonl", f)) return 1;
        for (const Record* r : all) {
            json j;
            j["query_id"]  = r->query_id;
            j["kind"]      = r->kind;
            if (r->kind == "score") j["number_of_levels"] = r->number_of_levels;
            j["partition"] = to_string(r->partition);
            j["unit"]      = r->unit;
            j["similarity"] = r->retrieval_ok ? json(r->similarity) : json(nullptr);
            j["hamming_distance"] = r->retrieval_ok ? json(r->hamming) : json(nullptr);
            j["threshold"] = (opt.select && !sel.all_s1) ? json(sel.threshold) : json(nullptr);
            j["policy"]    = frozen.name;
            j["gate_decision"] = (frozen.type == Policy::Type::Global)
                ? to_string(gate_decide(r->retrieval_ok, r->similarity, frozen.threshold))
                : (r->retrieval_ok ? "rejected" : "no_exemplar");
            const bool act = activates(frozen, *r);
            j["gate_activated"] = act;
            j["ground_truth"]   = r->ground_truth;
            j["s1_prediction"]  = r->s1_ok ? json(r->s1_pred) : json(nullptr);
            j["r1_prediction"]  = r->r1_ok ? json(r->r1_pred) : json(nullptr);
            // R2 needs R1 only when activated; a missing R1 then makes the R2 result unavailable.
            const bool r2_ok = r->s1_ok && (!act || r->r1_ok);
            j["r2_prediction"]  = r2_ok ? json(r2_prediction(frozen, *r)) : json(nullptr);
            j["s1_correct"]     = r->s1_ok ? json(r->s1_correct) : json(nullptr);
            j["r1_correct"]     = r->r1_ok ? json(r->r1_correct) : json(nullptr);
            j["r2_correct"]     = r2_ok ? json(act ? r->r1_correct : r->s1_correct) : json(nullptr);
            const std::string why = exclusion_reason(*r);
            j["paired"]         = why.empty();
            if (!why.empty()) j["excluded_reason"] = why;
            if (why.empty()) {
                j["r1_transition"] = r1::to_string(r1::classify(r->s1_correct, r->r1_correct));
                j["r2_transition"] = r1::to_string(r1::classify(r->s1_correct, act ? r->r1_correct : r->s1_correct));
            }
            if (opt.per_kind && opt.select) j["per_kind_r2_prediction"] = r2_prediction(frozen_kind, *r);
            j["s1_margin"] = r->s1_ok ? json(r->s1_margin) : json(nullptr);  // for R3; unused by the gate
            f << j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
        }
    }

    // Report set: held-out evaluation under the frozen threshold, or everything for a plain sweep.
    const RecordSet& report_set = has_partition ? evaluation : paired_all;
    const std::string report_name = has_partition ? "held-out evaluation partition" : "all paired records";

    // Transitions and latency CSVs.
    {
        std::ofstream t, l;
        if (!open_out(out, "transitions.csv", t) || !open_out(out, "latency.csv", l)) return 1;
        t << "split,kind,condition,n,both_correct,rescue,harm,both_incorrect,net_gain,rescue_retained,"
             "rescue_lost,harm_retained,harm_prevented,activation_rate\n";
        l << "split,kind,policy,n,activation_rate,est_s1_ms,est_retrieval_ms,est_r1_pass_ms,"
             "est_r1_deployed_ms,est_r2_ms,r2_over_s1,r2_over_r1_deployed,note\n";
        std::vector<std::pair<std::string, RecordSet>> splits;
        if (has_partition) { splits.push_back({"tuning", tuning}); splits.push_back({"evaluation", evaluation}); }
        else               { splits.push_back({"all", paired_all}); }
        std::vector<Policy> policies = {Policy::all_s1(), Policy::all_r1()};
        if (opt.select) policies.push_back(frozen);
        if (opt.select && opt.per_kind) policies.push_back(frozen_kind);
        for (const auto& [split, set] : splits) {
            for (const char* k : kKinds) {
                const RecordSet sub = filter_kind(set, k);
                for (const Policy& p : policies) {
                    const Stats s = evaluate(sub, p);
                    const std::string cond = (p.type == Policy::Type::AllR1) ? "R1" : "R2_" + p.name;
                    if (p.type != Policy::Type::AllS1) {
                        const auto& tc = (p.type == Policy::Type::AllR1) ? s.r1 : s.r2;
                        t << split << "," << k << "," << cond << "," << s.n << "," << tc.both_correct << ","
                          << tc.rescue << "," << tc.harm << "," << tc.both_incorrect << "," << tc.net() << ","
                          << s.rescue_retained << "," << s.rescue_lost << "," << s.harm_retained << ","
                          << s.harm_prevented << "," << full(s.activation_rate()) << "\n";
                    }
                    const double s1 = s.est_s1_ms(), r1d = s.est_r1_deployed_ms(), r2 = s.est_r2_ms();
                    l << split << "," << k << "," << p.name << "," << s.n << "," << full(s.activation_rate()) << ","
                      << fmt(s1, 3) << "," << fmt(s.n ? s.retrieval_ms / s.n : 0, 3) << ","
                      << fmt(s.n ? s.r1_pass_ms / s.n : 0, 3) << "," << fmt(r1d, 3) << "," << fmt(r2, 3) << ","
                      << fmt(s1 > 0 ? r2 / s1 : 0, 4) << "," << fmt(r1d > 0 ? r2 / r1d : 0, 4)
                      << ",estimated from R1 per-query measurements (warm models)\n";
                }
            }
        }
    }

    if (opt.select) write_examples(out, report_set, frozen, opt.db_path);

    // --- config.json ---
    {
        json c;
        c["experiment"]      = "R2 similarity-gated retrieved 1-shot (offline replay)";
        c["mode"]            = "offline";
        c["inputs"]          = split_mode ? json{{"tuning", opt.tuning_input}, {"evaluation", opt.evaluation_input}}
                                          : json{{"predictions", opt.input}};
        c["input_sha256"]    = split_mode ? json{{"tuning", sha256_file(opt.tuning_input)},
                                                 {"evaluation", sha256_file(opt.evaluation_input)}}
                                          : json{{"predictions", sha256_file(opt.input)}};
        c["source_r1_config"] = runs[0].config_found ? runs[0].config : json(nullptr);
        c["identity"]        = identity;
        c["similarity_metric"] = "estimated cosine cos(pi * hamming / dim) of the Top-1 exemplar; sqlite-vec "
                                 "stores bit[768] vectors so its KNN distance is Hamming, and 1 - distance "
                                 "is not a cosine similarity here";
        c["embedding_dim"]   = runs[0].dim;
        c["gate_rule"]       = "similarity >= threshold (equality activates)";
        c["selection_policy"] = kSelectionPolicy;
        c["selected"]        = opt.select ? (sel.all_s1 ? json("all_s1") : json(sel.threshold)) : json(nullptr);
        if (has_partition && !split_mode) {
            c["partition"] = {
                {"kind", "internal deterministic group-disjoint split of one validation run"},
                {"dataset", pinfo.dataset_path}, {"dataset_sha256", pinfo.dataset_sha256},
                {"unit", "scenario_family when present (whole families alternate by sorted name), else "
                         "group_id (xxHash64 low bit: 0 = tuning)"},
                {"tuning_families", pinfo.tuning_families}, {"evaluation_families", pinfo.evaluation_families},
                {"group_overlap", pinfo.group_overlap}, {"train_split_records", pinfo.train_split_records}};
        }
        c["train_overlap_check"] = train_overlap < 0 ? json("not run (no --train)") : json(train_overlap);
        c["counts"] = {{"records", all.size()}, {"paired", paired_all.size()}, {"tuning", tuning.size()},
                       {"evaluation", evaluation.size()}};
        json ex;
        for (const auto& [k, v] : excluded) ex[k] = v;
        c["excluded"] = ex;
        c["grid"] = {{"lo", opt.sweep.grid_lo}, {"hi", opt.sweep.grid_hi}, {"step", opt.sweep.grid_step}};
        c["quantiles"] = opt.sweep.quantiles;
        c["bootstrap"] = {{"replicates", opt.bootstrap}, {"seed", opt.seed},
                          {"cluster", has_partition ? "group_id" : "query_id"}};
        std::ofstream f;
        if (!open_out(out, "config.json", f)) return 1;
        f << c.dump(2, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
    }

    // --- summary.md ---
    std::ofstream o;
    if (!open_out(out, "summary.md", o)) return 1;
    o << "# R2 — similarity-gated retrieved 1-shot (offline replay)\n\n";
    if (!identity["verified"].get<bool>()) {
        o << "> **Identity unverified:** the source R1 run has no config.json; model, DB and prompt "
             "configuration could not be checked. Treat these numbers as development output only.\n\n";
    }
    if (identity.contains("warning")) o << "> **Warning:** " << identity["warning"].get<std::string>() << "\n\n";
    if (!has_partition) {
        o << "> **Exploratory sweep over all records.** No tuning partition was given, so nothing here is "
             "a held-out result and no threshold was selected.\n\n";
    }

    o << "## Configuration\n\n| Setting | Value |\n|---|---|\n";
    if (runs[0].config_found) {
        const json& rc = runs[0].config;
        for (const char* k : {"input_path", "inference_model", "embedding_model", "db_path", "train_path",
                              "prompt_format_version", "n_ctx", "run_timestamp"}) {
            if (rc.contains(k)) o << "| R1 " << k << " | " << (rc[k].is_string() ? rc[k].get<std::string>() : rc[k].dump()) << " |\n";
        }
        if (rc.contains("inference_model_identity") && rc["inference_model_identity"].contains("sha256")) {
            o << "| R1 inference model sha256 | " << rc["inference_model_identity"]["sha256"].get<std::string>() << " |\n";
        }
    }
    o << "| Predictions | " << (split_mode ? opt.tuning_input + " + " + opt.evaluation_input : opt.input) << " |\n";
    o << "| Similarity | estimated cosine cos(pi·h/" << runs[0].dim << ") from the Top-1 Hamming distance h |\n";
    o << "| Gate | similarity >= threshold (equality activates) |\n\n";

    o << "## Data separation\n\n";
    if (has_partition && !split_mode) {
        o << "No official calibration split exists in this repository (validation records are all labelled "
             "`validation`; `calibration/original.jsonl` is unrelated), so the R1 validation run is split "
             "into an **internal**, deterministic, group-disjoint tuning / evaluation partition:\n\n";
        o << "- Unit: scenario family when present (whole families, alternating by sorted name), otherwise "
             "group_id (xxHash64 parity). Counterfactual variants share a group and never straddle.\n";
        o << "- Tuning families: ";
        for (const auto& f : pinfo.tuning_families) o << "`" << f << "` ";
        o << "\n- Evaluation families: ";
        for (const auto& f : pinfo.evaluation_families) o << "`" << f << "` ";
        o << "\n- Groups in both partitions: " << pinfo.group_overlap
          << "; query records labelled train: " << pinfo.train_split_records << "\n";
    } else if (split_mode) {
        o << "Separate tuning and evaluation runs; identity keys match and query IDs are disjoint.\n";
    }
    o << "- Queries that are training records: "
      << (train_overlap < 0 ? std::string("not checked (no --train)") : std::to_string(train_overlap)) << "\n";
    o << "- Paired records: " << paired_all.size() << " (tuning " << tuning.size() << ", evaluation "
      << evaluation.size() << "). Excluded:";
    if (excluded.empty()) o << " none";
    for (const auto& [k, v] : excluded) o << " " << k << "=" << v;
    o << "\n\n";

    if (opt.select) {
        o << "## Threshold selection (tuning only)\n\n" << kSelectionPolicy << "\n\n";
        o << "| Tuning n | Candidates | Selected | Tuning S1 | Tuning R1 | Tuning R2 | Tuning activation |\n"
             "|---:|---:|---|---:|---:|---:|---:|\n";
        o << "| " << sel.tuning.n << " | " << sel.candidates << " | "
          << (sel.all_s1 ? std::string("never activate (all S1)") : full(sel.threshold) + " (Hamming <= " +
              fmt(std::acos(sel.threshold) / 3.14159265358979323846 * runs[0].dim, 1) + ")")
          << " | " << pct(sel.tuning.s1_accuracy()) << " | " << pct(sel.tuning.r1_accuracy()) << " | "
          << pct(sel.tuning.r2_accuracy()) << " | " << pct(sel.tuning.activation_rate(), 1) << " |\n\n";
        o << "Tuning accuracy is optimistically biased (the threshold was fitted to it). Only the held-out "
             "rows below support an effectiveness claim.\n\n";
    }

    auto three_way = [&](const RecordSet& set, const Policy& p, const std::string& title) {
        o << "### " << title << "\n\n";
        o << "| Kind | n | S1 | R1 | R2 | R2−S1 | R2−R1 | Activated | Rescue kept | Rescue lost | "
             "Harm kept | Harm prevented |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        for (const char* k : kKinds) {
            const Stats s = evaluate(filter_kind(set, k), p);
            if (s.n == 0) continue;
            o << "| " << k << " | " << s.n << " | " << pct(s.s1_accuracy()) << " | " << pct(s.r1_accuracy())
              << " | " << pct(s.r2_accuracy()) << " | " << pp(s.r2_accuracy() - s.s1_accuracy()) << " | "
              << pp(s.r2_accuracy() - s.r1_accuracy()) << " | " << s.activated << " ("
              << pct(s.activation_rate(), 1) << ") | " << s.rescue_retained << " | " << s.rescue_lost << " | "
              << s.harm_retained << " | " << s.harm_prevented << " |\n";
        }
        o << "\n";
    };

    if (opt.select) {
        o << "## Held-out result at the frozen threshold\n\n";
        three_way(evaluation, frozen, "Global threshold (primary)");
        const Stats s = evaluate(evaluation, frozen);
        std::vector<std::string> cl;
        std::vector<int> s1c, r1c, r2c;
        for (const Record* r : evaluation) {
            cl.push_back(r->group_id.empty() ? r->query_id : r->group_id);
            s1c.push_back(r->s1_correct);
            r1c.push_back(r->r1_correct);
            r2c.push_back(activates(frozen, *r) ? r->r1_correct : r->s1_correct);
        }
        const BootstrapCI vs_s1 = paired_bootstrap(cl, r2c, s1c, opt.bootstrap, opt.seed);
        const BootstrapCI vs_r1 = paired_bootstrap(cl, r2c, r1c, opt.bootstrap, opt.seed + 1);
        const BootstrapCI r1_s1 = paired_bootstrap(cl, r1c, s1c, opt.bootstrap, opt.seed + 2);
        o << "Paired uncertainty (held-out, " << s.n << " queries, " << vs_s1.clusters << " group clusters, "
          << opt.bootstrap << " cluster-bootstrap replicates):\n\n";
        o << "| Comparison | Difference | Cluster bootstrap 95% | Wald / McNemar |\n|---|---:|---|---|\n";
        o << "| R2 − S1 | " << pp(vs_s1.diff) << " | [" << pp(vs_s1.lo) << ", " << pp(vs_s1.hi) << "] | "
          << mcnemar_text(s.r2) << " |\n";
        o << "| R2 − R1 | " << pp(vs_r1.diff) << " | [" << pp(vs_r1.lo) << ", " << pp(vs_r1.hi) << "] | "
          << mcnemar_text(s.r2_vs_r1) << " |\n";
        o << "| R1 − S1 | " << pp(r1_s1.diff) << " | [" << pp(r1_s1.lo) << ", " << pp(r1_s1.hi) << "] | "
          << mcnemar_text(s.r1) << " |\n\n";
        o << "Gate on held-out: activated " << s.activated << " (" << pct(s.activation_rate(), 1)
          << "), rejected " << s.rejected() << "; mean similarity activated " << fmt(s.mean_sim_activated(), 4)
          << ", rejected " << fmt(s.mean_sim_rejected(), 4) << ". Queries without an eligible exemplar are "
          << "excluded from the paired set (counted above) and would keep S1.\n\n";

        o << "### Strategy comparison (held-out)\n\n| Strategy | n | Accuracy | Activation |\n|---|---:|---:|---:|\n";
        std::vector<Policy> strategies = {Policy::all_s1(), Policy::all_r1(), frozen};
        if (opt.per_kind) strategies.push_back(frozen_kind);
        for (const Policy& p : strategies) {
            const Stats ps = evaluate(evaluation, p);
            o << "| " << p.name << " | " << ps.n << " | " << pct(ps.r2_accuracy()) << " | "
              << pct(ps.activation_rate(), 1) << " |\n";
        }
        o << "\n";
        if (opt.per_kind) {
            o << "Per-kind thresholds (secondary diagnostic, selected on tuning per kind): ";
            for (const auto& [k, ks] : sel_kind) {
                o << k << "=" << (ks.all_s1 ? "never" : fmt(ks.threshold, 4)) << " (tuning n=" << ks.tuning.n
                  << ") ";
            }
            o << "\n\n";
            three_way(evaluation, frozen_kind, "Per-kind thresholds (diagnostic)");
        }

        // Score level stratification.
        std::map<int, RecordSet> by_levels;
        for (const Record* r : evaluation) if (r->kind == "score") by_levels[r->number_of_levels].push_back(r);
        if (!by_levels.empty()) {
            o << "Score by level count (held-out, frozen global threshold):\n\n| Levels | n | S1 | R1 | R2 | Activation |\n"
                 "|---:|---:|---:|---:|---:|---:|\n";
            for (const auto& [lv, set] : by_levels) {
                const Stats ls = evaluate(set, frozen);
                o << "| " << lv << " | " << ls.n << " | " << pct(ls.s1_accuracy()) << " | " << pct(ls.r1_accuracy())
                  << " | " << pct(ls.r2_accuracy()) << " | " << pct(ls.activation_rate(), 1) << " |\n";
            }
            o << "\n";
        }
    }

    // Threshold range analysis.
    auto range_analysis = [&](const RecordSet& set, const std::string& title) {
        const auto rows = sweep_exact(set);
        int beat_both = 0;
        double lo = 2, hi = -2;
        const Stats s1 = evaluate(set, Policy::all_s1()), r1 = evaluate(set, Policy::all_r1());
        const int best_base = std::max(s1.s1_correct, r1.r1_correct);
        for (const auto& row : rows) {
            if (row.stats.r2_correct > best_base) {
                beat_both++;
                lo = std::min(lo, row.threshold);
                hi = std::max(hi, row.threshold);
            }
        }
        o << "- " << title << ": " << beat_both << " of " << rows.size()
          << " distinct thresholds beat both S1 and R1";
        if (beat_both) o << " (thresholds " << fmt(lo, 4) << " – " << fmt(hi, 4) << ")";
        o << ".\n";
    };
    o << "## Threshold ranges\n\n";
    if (has_partition) {
        range_analysis(tuning, "Tuning");
        range_analysis(evaluation, "Held-out (exploratory only; not used for selection)");
    } else {
        range_analysis(paired_all, "All records");
    }
    o << "\nFull curves (accuracy, activation, rescue/harm, estimated latency vs threshold) are in "
         "`threshold_sweep.csv`" << (has_partition ? " (tuning) and `heldout_sweep_exploratory.csv`" : "")
      << "; plot columns threshold against r2_accuracy / activation_rate / rescue,harm / est_r2_latency_ms, "
         "and activation_rate against r2_accuracy for the trade-off.\n\n";

    // Does similarity rank beneficial retrieval? R1 net gain by similarity quintile.
    {
        o << "## Similarity as a signal (" << report_name << ")\n\n";
        std::vector<const Record*> v(report_set.begin(), report_set.end());
        std::stable_sort(v.begin(), v.end(), [](const Record* a, const Record* b) { return a->similarity > b->similarity; });
        o << "R1 rescue/harm by Top-1 similarity quintile (highest first). A useful gate needs net gain to "
             "fall with similarity.\n\n| Quintile | Similarity range | n | Rescue | Harm | Net | R1−S1 |\n"
             "|---|---|---:|---:|---:|---:|---:|\n";
        for (int qi = 0; qi < 5 && !v.empty(); ++qi) {
            const size_t a = v.size() * qi / 5, b = v.size() * (qi + 1) / 5;
            RecordSet part(v.begin() + a, v.begin() + b);
            if (part.empty()) continue;
            const Stats s = evaluate(part, Policy::all_r1());
            o << "| Q" << qi + 1 << " | " << fmt(part.back()->similarity, 4) << " – " << fmt(part.front()->similarity, 4)
              << " | " << s.n << " | " << s.r1.rescue << " | " << s.r1.harm << " | " << s.r1.net() << " | "
              << pp(s.r1_accuracy() - s.s1_accuracy()) << " |\n";
        }
        o << "\n";
    }

    // Cost.
    {
        const Policy& p = opt.select ? frozen : Policy::all_r1();
        const Stats s = evaluate(report_set, p);
        o << "## Cost (estimated from R1 measurements, " << report_name << ")\n\n";
        o << "| Model | Mean ms / query |\n|---|---:|\n";
        o << "| S1 only | " << fmt(s.est_s1_ms(), 1) << " |\n";
        o << "| Deployed R1 (retrieval + 1-shot, no S1 pass) | " << fmt(s.est_r1_deployed_ms(), 1) << " |\n";
        o << "| R2 (S1 + retrieval + activation × 1-shot) | " << fmt(s.est_r2_ms(), 1) << " |\n";
        o << "| Retrieval overhead (embed + search + fetch + LZ4) | " << fmt(s.n ? s.retrieval_ms / s.n : 0, 1) << " |\n\n";
        o << "R2/S1 = " << fmt(s.est_s1_ms() > 0 ? s.est_r2_ms() / s.est_s1_ms() : 0, 2) << "×, R2/deployed R1 = "
          << fmt(s.est_r1_deployed_ms() > 0 ? s.est_r2_ms() / s.est_r1_deployed_ms() : 0, 2)
          << "×. These are predictions from per-query timings of the R1 harness (warm models, sequential, "
             "S1 pass counted once); the online gated run measures them directly.\n";
    }
    printf("Wrote %s\n", out.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// Parity
// ---------------------------------------------------------------------------

static int run_parity(const std::string& online_path, const std::string& reference_path, const std::string& out)
{
    InputRun on, ref;
    if (!load_input(online_path, false, on) || !load_input(reference_path, false, ref)) return 1;
    auto diff = config_differences(on.config, ref.config);
    if (!on.config.contains("gate")) {
        fprintf(stderr, "error: %s is not a gated online run (config.json has no gate)\n", online_path.c_str());
        return 1;
    }
    const double thr = on.config["gate"]["threshold"].get<double>();
    std::unordered_map<std::string, const Record*> by_id;
    for (const auto& r : ref.records) by_id[r.query_id] = &r;

    int matched = 0, decision_mismatch = 0, s1_mismatch = 0, r2_mismatch = 0, sim_mismatch = 0;
    int activated = 0, rejected = 0, ran_second_pass_when_rejected = 0, missing_second_pass = 0;
    int r1_logit_diff = 0;
    double max_logit_abs = 0.0;
    std::vector<std::string> examples;
    const Policy p = Policy::global(thr);
    for (const auto& r : on.records) {
        auto it = by_id.find(r.query_id);
        if (it == by_id.end()) continue;
        const Record& f = *it->second;
        matched++;
        const bool online_act = r.gate_decision == "activated";
        const bool offline_act = activates(p, f);
        online_act ? activated++ : rejected++;
        if (!online_act && r.r1_status != "gate_rejected" && r.retrieval_ok) ran_second_pass_when_rejected++;
        if (online_act && !r.r1_ok) missing_second_pass++;
        if (r.hamming != f.hamming) sim_mismatch++;
        if (online_act != offline_act) { decision_mismatch++; if (examples.size() < 10) examples.push_back(r.query_id + " gate"); }
        if (r.s1_pred != f.s1_pred) { s1_mismatch++; if (examples.size() < 10) examples.push_back(r.query_id + " s1"); }
        const int online_r2  = online_act ? r.r1_pred : r.s1_pred;
        const int offline_r2 = r2_prediction(p, f);
        if (online_r2 != offline_r2) { r2_mismatch++; if (examples.size() < 10) examples.push_back(r.query_id + " r2"); }
        if (online_act && r.r1_logits.size() == f.r1_logits.size()) {
            double m = 0.0;
            for (size_t i = 0; i < r.r1_logits.size(); ++i) m = std::max(m, (double)std::fabs(r.r1_logits[i] - f.r1_logits[i]));
            max_logit_abs = std::max(max_logit_abs, m);
            r1_logit_diff += m > 0.0;
        }
    }
    const int second_passes = on.config["gate"].value("second_pass_calls", -1);
    const bool ok = diff.empty() && matched == static_cast<int>(on.records.size()) && decision_mismatch == 0 &&
                    s1_mismatch == 0 && r2_mismatch == 0 && sim_mismatch == 0 &&
                    ran_second_pass_when_rejected == 0 && missing_second_pass == 0 && second_passes == activated;

    // Measured online latency.
    double s1 = 0, retr = 0, r1p = 0;
    for (const auto& r : on.records) { s1 += r.s1_ms; retr += r.retrieval_ms; r1p += r.r1_ok ? r.r1_ms : 0; }
    const double n = on.records.empty() ? 1 : static_cast<double>(on.records.size());
    double ref_s1 = 0, ref_retr = 0, ref_r1 = 0;
    int ref_n = 0;
    for (const auto& r : on.records) {
        auto it = by_id.find(r.query_id);
        if (it == by_id.end()) continue;
        ref_s1 += it->second->s1_ms; ref_retr += it->second->retrieval_ms; ref_r1 += it->second->r1_ms; ref_n++;
    }

    json j;
    j["online"] = online_path;
    j["reference"] = reference_path;
    j["threshold"] = thr;
    j["identity_differences"] = diff;
    j["records"] = on.records.size();
    j["matched"] = matched;
    j["activated"] = activated;
    j["rejected"] = rejected;
    j["second_pass_calls"] = second_passes;
    j["rejected_with_second_pass"] = ran_second_pass_when_rejected;
    j["activated_without_second_pass"] = missing_second_pass;
    j["hamming_mismatch"] = sim_mismatch;
    j["gate_decision_mismatch"] = decision_mismatch;
    j["s1_prediction_mismatch"] = s1_mismatch;
    j["r2_prediction_mismatch"] = r2_mismatch;
    j["r1_logit_vectors_differing"] = r1_logit_diff;
    j["r1_max_abs_logit_diff"] = max_logit_abs;
    j["mismatch_examples"] = examples;
    j["measured_ms_per_query"] = {
        {"s1", s1 / n}, {"retrieval", retr / n}, {"one_shot_pass_amortised", r1p / n},
        {"r2_total", (s1 + retr + r1p) / n},
        {"r2_over_s1", s1 > 0 ? (s1 + retr + r1p) / s1 : 0.0},
        {"reference_deployed_r1", ref_n ? (ref_retr + ref_r1) / ref_n : 0.0},
        {"r2_over_reference_deployed_r1", (ref_retr + ref_r1) > 0 ? (s1 + retr + r1p) / (ref_retr + ref_r1) * ref_n / n : 0.0}};
    j["passed"] = ok;
    std::ofstream f;
    if (!open_out(out, "parity.json", f)) return 1;
    f << j.dump(2) << "\n";
    printf("Parity %s: matched %d, gate mismatches %d, S1 mismatches %d, R2 mismatches %d, second passes %d "
           "(activated %d), max |R1 logit diff| %.3g\n",
           ok ? "PASSED" : "FAILED", matched, decision_mismatch, s1_mismatch, r2_mismatch, second_passes,
           activated, max_logit_abs);
    return ok ? 0 : 2;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    std::string mode;
    for (int i = 1; i + 1 < argc; ++i) if (strcmp(argv[i], "--mode") == 0) mode = argv[i + 1];
    if (argc < 2 || mode.empty()) {
        for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { print_usage(argv[0]); return 0; }
        print_usage(argv[0]);
        return 1;
    }

    if (mode == "online") {
        // Strip R2-only flags; everything else goes to the shared R1 harness unchanged.
        std::vector<char*> args = {argv[0]};
        bool has_thr = false;
        double thr = 0.0;
        for (int i = 1; i < argc; ++i) {
            if (!strcmp(argv[i], "--mode")) { ++i; continue; }
            if (!strcmp(argv[i], "--threshold") && i + 1 < argc) {
                char* end = nullptr;
                thr = std::strtod(argv[++i], &end);
                if (!end || *end) { fprintf(stderr, "error: --threshold is not a number\n"); return 1; }
                has_thr = true;
                continue;
            }
            if (!strcmp(argv[i], "--threshold-from") && i + 1 < argc) {
                json s;
                if (!read_json_file(argv[++i], s)) { fprintf(stderr, "error: cannot read %s\n", argv[i]); return 1; }
                if (s.value("policy", "") != "global_threshold") {
                    fprintf(stderr, "error: selected policy is '%s'; an all-S1 policy needs no gated run\n",
                            s.value("policy", "").c_str());
                    return 1;
                }
                thr = s["threshold"].get<double>();
                has_thr = true;
                continue;
            }
            args.push_back(argv[i]);
        }
        if (!has_thr) {
            fprintf(stderr, "error: online R2 needs an explicit --threshold or --threshold-from; there is no default\n");
            return 1;
        }
        if (!valid_threshold(thr)) {
            fprintf(stderr, "error: threshold %g is outside the estimated-cosine range [-1, 1]\n", thr);
            return 1;
        }
        r1::HarnessHooks hooks;
        hooks.gate_enabled   = true;
        hooks.gate_threshold = thr;
        hooks.experiment     = "R2 similarity-gated retrieved 1-shot (online)";
        hooks.default_output = "results/r2-online";
        printf("R2 online: gate threshold %.17g (similarity >= threshold activates)\n", thr);
        return r1::run_harness(static_cast<int>(args.size()), args.data(), hooks);
    }

    OfflineOptions opt;
    std::string online_path, reference_path;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s requires an argument\n", flag); exit(1); }
            return argv[++i];
        };
        if (arg("--help") || arg("-h"))          { print_usage(argv[0]); return 0; }
        else if (arg("--mode"))                   next("--mode");
        else if (arg("--input"))                  opt.input = next("--input");
        else if (arg("--split-dataset"))          opt.split_dataset = next("--split-dataset");
        else if (arg("--tuning-input"))           opt.tuning_input = next("--tuning-input");
        else if (arg("--evaluation-input"))       opt.evaluation_input = next("--evaluation-input");
        else if (arg("--output"))                 opt.output = next("--output");
        else if (arg("--sweep"))                  {}  // sweeps are always written
        else if (arg("--select-threshold"))       opt.select = true;
        else if (arg("--per-kind-thresholds"))    opt.per_kind = true;
        else if (arg("--grid-lo"))                opt.sweep.grid_lo = std::atof(next("--grid-lo"));
        else if (arg("--grid-hi"))                opt.sweep.grid_hi = std::atof(next("--grid-hi"));
        else if (arg("--grid-step"))              opt.sweep.grid_step = std::atof(next("--grid-step"));
        else if (arg("--quantiles"))              opt.sweep.quantiles = std::atoi(next("--quantiles"));
        else if (arg("--bootstrap"))              opt.bootstrap = std::atoi(next("--bootstrap"));
        else if (arg("--seed"))                   opt.seed = std::strtoull(next("--seed"), nullptr, 10);
        else if (arg("--db"))                     opt.db_path = next("--db");
        else if (arg("--train"))                  opt.train_path = next("--train");
        else if (arg("--allow-unverified-config")) opt.allow_unverified = true;
        else if (arg("--online"))                 online_path = next("--online");
        else if (arg("--reference"))              reference_path = next("--reference");
        else {
            fprintf(stderr, "error: unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    if (mode == "offline") return run_offline(opt);
    if (mode == "parity") {
        if (online_path.empty() || reference_path.empty()) {
            fprintf(stderr, "error: parity needs --online and --reference\n");
            return 1;
        }
        return run_parity(online_path, reference_path, opt.output == "results/r2" ? "results/r2-parity" : opt.output);
    }
    fprintf(stderr, "error: --mode must be offline, online or parity\n");
    return 1;
}
