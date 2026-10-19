// Unit tests for R2 similarity-gated retrieved 1-shot (no model needed).
//
//  Gate:        above / below / equal threshold, missing exemplar, invalid similarity, negative
//               cosine, threshold range, zero and full activation.
//  Replay:      all-S1 and all-eligible-R1 endpoints, R2 prediction selection, missing R1
//               predictions, rescue / harm retained / lost / prevented, activation rate, sweeps.
//  Selection:   maximises tuning accuracy, ties -> lower activation, all-S1 wins ties, held-out
//               labels cannot change the selected threshold.
//  Separation:  group-disjoint partition, family alternation, straddling-group detection,
//               train-split detection.
//  Misc:        prediction parsing / similarity recomputation, bootstrap, config identity.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "r1_metrics.h"
#include "r2_gate.h"
#include "r2_offline.h"

#include "nlohmann/json.hpp"

using namespace pjev;
using namespace pjev::r2;
using json = nlohmann::ordered_json;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const char* desc)
{
    if (cond) { printf("  PASS: %s\n", desc); g_pass++; }
    else      { printf("  FAIL: %s\n", desc); g_fail++; }
}

static constexpr int kDim = 768;

// Paired record with similarity from a Hamming distance, as pjev-r1 would produce it.
static Record rec(const std::string& id, const std::string& kind, double hamming, bool s1c, bool r1c)
{
    Record r;
    r.query_id     = id;
    r.kind         = kind;
    r.ground_truth = 0;
    r.s1_ok = r.r1_ok = r.retrieval_ok = true;
    r.retrieval_status = "ok";
    r.r1_status = "ok";
    r.hamming    = hamming;
    r.similarity = r1::estimated_cosine(hamming, kDim);
    r.s1_correct = s1c;
    r.r1_correct = r1c;
    r.s1_pred = s1c ? 0 : 1;
    r.r1_pred = r1c ? 0 : 2;
    r.s1_ms = 100;
    r.retrieval_ms = 10;
    r.r1_ms = 300;
    return r;
}

static RecordSet ptrs(const std::vector<Record>& v)
{
    RecordSet s;
    for (const auto& r : v) s.push_back(&r);
    return s;
}

// ---------------------------------------------------------------------------

static void test_gate()
{
    printf("\n[Gate]\n");
    check(gate_decide(true, 0.95, 0.90) == GateDecision::Activated, "similarity above threshold activates");
    check(gate_decide(true, 0.85, 0.90) == GateDecision::Rejected, "similarity below threshold rejects");
    check(gate_decide(true, 0.90, 0.90) == GateDecision::Activated, "equality activates (>=, not >)");
    const double t = r1::estimated_cosine(80, kDim);
    check(gate_decide(true, r1::estimated_cosine(80, kDim), t) == GateDecision::Activated,
          "threshold at an observed similarity activates that exact value");
    check(gate_decide(false, 0.99, 0.0) == GateDecision::NoExemplar, "missing exemplar -> no_exemplar");
    check(gate_decide(true, std::nan(""), 0.0) == GateDecision::InvalidSimilarity, "NaN similarity rejected");
    check(gate_decide(true, std::numeric_limits<double>::infinity(), 0.0) == GateDecision::InvalidSimilarity,
          "infinite similarity rejected");
    check(gate_decide(true, 1.5, 0.0) == GateDecision::InvalidSimilarity, "similarity > 1 rejected");
    check(gate_decide(true, -0.5, -0.6) == GateDecision::Activated, "negative cosine is a valid similarity");
    check(gate_decide(true, -0.7, -0.6) == GateDecision::Rejected, "negative cosine below a negative threshold");
    check(valid_threshold(-1.0) && valid_threshold(1.0) && valid_threshold(0.0), "thresholds in [-1, 1] accepted");
    check(!valid_threshold(1.01) && !valid_threshold(-1.01) && !valid_threshold(std::nan("")),
          "thresholds outside [-1, 1] or NaN rejected");
    check(std::string(to_string(GateDecision::Rejected)) == "rejected", "decision names");

    std::vector<Record> v = {rec("a", "choice", 70, false, true), rec("b", "choice", 100, true, false)};
    const RecordSet s = ptrs(v);
    check(evaluate(s, Policy::global(1.0)).activated == 0, "threshold 1.0: zero activation (no identical vectors)");
    check(evaluate(s, Policy::global(-1.0)).activated == 2, "threshold -1.0: full activation");
}

static void test_replay()
{
    printf("\n[Offline replay]\n");
    // h: 60 (sim highest) .. 110 (lowest). Outcomes chosen so a threshold helps.
    std::vector<Record> v = {
        rec("r1", "choice", 60, false, true),   // rescue, high sim
        rec("r2", "choice", 70, false, true),   // rescue
        rec("b1", "choice", 80, true, true),    // both correct
        rec("h1", "choice", 90, true, false),   // harm, low sim
        rec("h2", "noul", 100, true, false),    // harm, lowest sim
        rec("n1", "score", 75, false, false),   // both incorrect
    };
    const RecordSet s = ptrs(v);

    const Stats s1 = evaluate(s, Policy::all_s1());
    const Stats r1 = evaluate(s, Policy::all_r1());
    check(s1.r2_correct == s1.s1_correct && s1.activated == 0, "all-S1 endpoint reproduces S1");
    check(r1.r2_correct == r1.r1_correct && r1.activated == r1.n, "all-eligible-R1 endpoint reproduces R1");
    check(s1.s1_correct == 3 && r1.r1_correct == 3, "fixture: S1 3/6, R1 3/6");

    const double t = r1::estimated_cosine(75, kDim);  // activates h <= 75: r1, r2, n1
    const Stats g = evaluate(s, Policy::global(t));
    check(g.activated == 3, "threshold activates exactly the records with similarity >= t");
    check(g.r2_correct == 5, "R2 = R1 where activated, S1 elsewhere");
    check(g.rescue_retained == 2 && g.rescue_lost == 0, "rescues retained / lost");
    check(g.harm_retained == 0 && g.harm_prevented == 2, "harms retained / prevented");
    check(g.r2.rescue == g.rescue_retained && g.r2.harm == g.harm_retained,
          "R2 rescue/harm = retained rescue/harm (rejected queries cannot rescue or harm)");
    check(g.r2_correct - g.s1_correct == g.r2.rescue - g.r2.harm, "R2 - S1 == (rescue - harm)");
    check(std::fabs(g.activation_rate() - 0.5) < 1e-12, "activation rate = activated / n");
    check(r2_prediction(Policy::global(t), v[0]) == v[0].r1_pred &&
          r2_prediction(Policy::global(t), v[3]) == v[3].s1_pred, "per-record R2 prediction selection");
    check(std::fabs(g.est_r2_ms() - (100 + 10 + 300 * 0.5)) < 1e-9,
          "estimated R2 cost = S1 + retrieval + activation x 1-shot pass");
    check(std::fabs(g.est_r1_deployed_ms() - 310) < 1e-9, "deployed R1 cost excludes the S1 pass");

    // Missing R1 prediction: not paired, so never part of the three-way set.
    Record miss = rec("m", "choice", 60, true, false);
    miss.r1_ok = false;
    miss.r1_status = "context_limit";
    check(!miss.paired(), "record without an R1 prediction is not paired");
    check(r2_prediction(Policy::global(-1.0), miss) == miss.s1_pred,
          "activated record without R1 falls back to S1 (no invented R1)");
    Record noex = rec("x", "choice", 0, true, false);
    noex.retrieval_ok = false;
    noex.retrieval_status = "no_eligible_exemplar";
    check(!activates(Policy::all_r1(), noex) && !noex.paired(), "no eligible exemplar never activates");

    // Sweeps.
    const auto exact = sweep_exact(s);
    check(exact.size() == 6, "exact sweep: one row per distinct similarity");
    check(exact.front().stats.activated == 1 && exact.back().stats.activated == 6,
          "exact sweep runs from one activation to all");
    const auto grid = sweep_grid(s, 0.0, 1.0, 0.05);
    check(grid.size() == 21 && grid[20].threshold == 1.0 && grid[1].threshold == 0.05,
          "grid 0.00..1.00 step 0.05 has 21 exact values");
    const auto q = sweep_quantile(s, 4);
    check(q.size() == 5 && q.front().stats.activated == 6, "quantile sweep: q0 activates everything");
    const auto ep = sweep_endpoints(s);
    check(ep.size() == 2 && !ep[0].has_threshold && ep[1].stats.activated == 6, "endpoints are explicit rows");
}

static void test_selection()
{
    printf("\n[Threshold selection]\n");
    std::vector<Record> v = {
        rec("r1", "choice", 60, false, true),
        rec("r2", "choice", 70, false, true),
        rec("b1", "choice", 80, true, true),   // activating this changes nothing (tie)
        rec("h1", "choice", 90, true, false),
    };
    Selection sel = select_threshold(ptrs(v));
    check(!sel.all_s1 && sel.tuning.r2_correct == 4, "selection maximises tuning accuracy");
    check(sel.threshold == r1::estimated_cosine(70, kDim) && sel.tuning.activated == 2,
          "tie between h<=70 and h<=80 goes to the lower activation rate");
    check(sel.candidates == 5, "candidates = never-activate + each distinct similarity");

    // R1 never helps: all-S1 must win, including ties.
    std::vector<Record> w = {rec("b", "choice", 60, true, true), rec("h", "choice", 70, true, false)};
    Selection s2 = select_threshold(ptrs(w));
    check(s2.all_s1 && s2.tuning.activated == 0, "never-activate wins when no threshold beats it");

    // Held-out labels cannot move the selection: only the tuning set is an input.
    std::vector<Record> evalv = {rec("e1", "choice", 65, true, false), rec("e2", "choice", 85, false, true)};
    const Selection before = select_threshold(ptrs(v));
    for (auto& r : evalv) { r.s1_correct = !r.s1_correct; r.r1_correct = !r.r1_correct; }
    const Selection after = select_threshold(ptrs(v));
    check(before.threshold == after.threshold && before.all_s1 == after.all_s1,
          "flipping evaluation labels leaves the selected threshold unchanged");
}

static std::string write_dataset(const std::string& name, const std::vector<json>& rows)
{
    const std::string path = (std::filesystem::temp_directory_path() / name).string();
    std::ofstream f(path, std::ios::binary);
    for (const auto& r : rows) f << r.dump() << "\n";
    return path;
}

static json ds_row(const std::string& id, const std::string& group, const std::string& family,
                   const std::string& split = "validation")
{
    json j;
    j["group_id"] = group;
    j["id"]       = id;
    j["split"]    = split;
    j["question"] = "q " + id;
    j["metadata"] = json::object();
    if (!family.empty()) j["metadata"]["scenario_family"] = family;
    return j;
}

static void test_partition()
{
    printf("\n[Dataset separation]\n");
    std::vector<json> rows;
    std::vector<Record> recs;
    const char* fams[] = {"a/x", "b/y", "c/z", "d/w"};
    int k = 0;
    for (const char* f : fams) {
        for (int g = 0; g < 3; ++g) {
            for (int v = 0; v < 2; ++v) {
                const std::string grp = std::string(f) + "/" + std::to_string(g);
                const std::string id  = grp + "/" + std::to_string(v);
                rows.push_back(ds_row(id, grp, f));
                recs.push_back(rec(id, "choice", 60 + (k++ % 30), true, true));
            }
        }
    }
    for (int g = 0; g < 40; ++g) {
        for (int v = 0; v < 2; ++v) {
            const std::string grp = "nofam/" + std::to_string(g);
            const std::string id  = grp + "/" + std::to_string(v);
            rows.push_back(ds_row(id, grp, ""));
            recs.push_back(rec(id, "noul", 60 + (k++ % 30), true, false));
        }
    }
    const std::string path = write_dataset("pjev_test_r2_ds.jsonl", rows);
    PartitionInfo info;
    std::string err;
    check(apply_partition(path, recs, info, err), "partition builds");
    check(info.matched == static_cast<int>(recs.size()) && info.unmatched == 0, "every record matched");
    check(info.group_overlap == 0, "no group straddles tuning and evaluation");
    check(info.tuning_families == std::vector<std::string>({"a/x", "c/z"}) &&
          info.evaluation_families == std::vector<std::string>({"b/y", "d/w"}),
          "whole families alternate by sorted name");
    int nt = 0, ne = 0, ids_both = 0;
    std::set<std::string> t_ids, e_ids;
    for (const auto& r : recs) {
        (r.partition == Partition::Tuning ? nt : ne)++;
        (r.partition == Partition::Tuning ? t_ids : e_ids).insert(r.query_id);
    }
    for (const auto& id : t_ids) ids_both += e_ids.count(id) > 0;
    check(nt > 0 && ne > 0 && nt + ne == static_cast<int>(recs.size()), "both partitions non-empty, all assigned");
    check(ids_both == 0, "no query ID in both partitions");
    bool variants_together = true;
    for (size_t i = 0; i + 1 < recs.size(); i += 2) variants_together &= recs[i].partition == recs[i + 1].partition;
    check(variants_together, "counterfactual variants of a group stay together");

    // A group spanning two families would straddle: must be detected.
    std::vector<json> bad = {ds_row("g1/0", "g1", "a/x"), ds_row("g1/1", "g1", "b/y")};
    std::vector<Record> bad_recs = {rec("g1/0", "choice", 60, true, true), rec("g1/1", "choice", 61, true, true)};
    PartitionInfo bi;
    apply_partition(write_dataset("pjev_test_r2_bad.jsonl", bad), bad_recs, bi, err);
    check(bi.group_overlap == 1, "straddling group detected");

    std::vector<json> tr = {ds_row("t/0", "t", "", "train")};
    std::vector<Record> tr_recs = {rec("t/0", "choice", 60, true, true)};
    PartitionInfo ti;
    apply_partition(write_dataset("pjev_test_r2_train.jsonl", tr), tr_recs, ti, err);
    check(ti.train_split_records == 1, "query labelled split=train detected");
}

static void test_parse_and_misc()
{
    printf("\n[Parsing, bootstrap, identity]\n");
    json j = {{"query_id", "q1"}, {"kind", "score"}, {"number_of_levels", 5}, {"ground_truth_index", 2},
              {"option_labels", {"A", "B", "C", "D", "E"}},
              {"s1", {{"status", "ok"}, {"prediction_index", 1}, {"correct", false}, {"margin", 0.3}, {"latency_ms", 120}}},
              {"r1", {{"status", "ok"}, {"prediction_index", 2}, {"correct", true}, {"latency_ms", 400}, {"logits", {0.1, 0.2}}}},
              {"retrieval", {{"status", "ok"}, {"hamming_distance", 80.0},
                             {"estimated_cosine", r1::estimated_cosine(80, kDim)}, {"embed_ms", 20}, {"search_ms", 5}}}};
    Record r;
    LoadStats st;
    check(parse_record(j, kDim, r, st) && r.paired(), "R1 prediction record parses as paired");
    check(r.similarity == r1::estimated_cosine(80, kDim) && st.similarity_mismatch == 0,
          "similarity recomputed from Hamming in full precision, matches stored value");
    check(r.s1_pred == 1 && r.r1_pred == 2 && !r.s1_correct && r.r1_correct, "predictions and correctness");
    check(r.retrieval_ms == 25 && r.r1_ms == 400 && r.s1_margin == 0.3, "costs and S1 margin carried");
    json bad = j;
    bad["retrieval"]["estimated_cosine"] = 0.5;
    Record r2;
    parse_record(bad, kDim, r2, st);
    check(st.similarity_mismatch == 1, "stored similarity from another metric is detected");
    json ctx = j;
    ctx["r1"] = {{"status", "context_limit"}};
    Record r3;
    parse_record(ctx, kDim, r3, st);
    check(!r3.r1_ok && !r3.paired(), "context_limit R1 is unavailable, not a wrong answer");
    json gated = j;
    gated["r1"] = {{"status", "gate_rejected"}};
    gated["gate"] = {{"threshold", 0.95}, {"similarity", 0.94}, {"decision", "rejected"}};
    Record r4;
    parse_record(gated, kDim, r4, st);
    check(r4.has_gate && r4.gate_decision == "rejected" && !r4.r1_ok, "online gated record parses");

    std::vector<std::string> cl = {"g1", "g1", "g2", "g3"};
    std::vector<int> a = {1, 1, 0, 1}, b = {0, 1, 0, 0};
    const BootstrapCI ci = paired_bootstrap(cl, a, b, 500, 7);
    const BootstrapCI ci2 = paired_bootstrap(cl, a, b, 500, 7);
    check(std::fabs(ci.diff - 0.5) < 1e-12 && ci.clusters == 3, "bootstrap point estimate and cluster count");
    check(ci.lo <= ci.diff && ci.diff <= ci.hi, "bootstrap interval contains the estimate");
    check(ci.lo == ci2.lo && ci.hi == ci2.hi, "bootstrap deterministic for a fixed seed");

    json c1 = {{"n_ctx", 4096}, {"embedding_fingerprint", "x"}};
    for (const auto& k : identity_keys()) if (!c1.contains(k)) c1[k] = "v";
    json c2 = c1;
    check(config_differences(c1, c2).empty(), "identical configs are compatible");
    c2["n_ctx"] = 2048;
    auto d = config_differences(c1, c2);
    check(d.size() == 1 && d[0] == "n_ctx", "differing inference parameter detected");
    json c3 = c1;
    c3.erase("db_info");
    check(!config_differences(c1, c3).empty(), "missing identity key is not compatible");
}

int main()
{
    printf("=== test_r2 ===\n");
    test_gate();
    test_replay();
    test_selection();
    test_partition();
    test_parse_and_misc();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
