#include "r3_analysis.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include "model/model_identity.h"
#include "r1_metrics.h"
#include "r2_offline.h"

namespace pjev
{
namespace r3
{

using json = nlohmann::ordered_json;

static double num(const json& j, const char* k, double def = 0.0)
{
    return (j.contains(k) && j[k].is_number()) ? j[k].get<double>() : def;
}
static std::string str(const json& j, const char* k)
{
    return (j.contains(k) && j[k].is_string()) ? j[k].get<std::string>() : std::string();
}

// ---------------------------------------------------------------------------
// Report helpers
// ---------------------------------------------------------------------------

static std::string fmt(const char* f, double v)
{
    char b[64];
    snprintf(b, sizeof(b), f, v);
    return b;
}
static std::string pct(double v) { return fmt("%.2f%%", v * 100.0); }
static std::string pp(double v) { return fmt("%+.2f pp", v * 100.0); }
static std::string ci_pp(const ClusterCI& c) { return "[" + fmt("%+.2f", c.lo * 100) + ", " + fmt("%+.2f", c.hi * 100) + "] pp"; }
static bool excludes_zero(const ClusterCI& c) { return c.lo > 0 || c.hi < 0; }

static double mean_of(const std::vector<double>& v) { return r1::mean(v); }
static double median_of(std::vector<double> v) { return r1::percentile(std::move(v), 50); }
static double p95_of(std::vector<double> v) { return r1::percentile(std::move(v), 95); }

struct Opt {
    std::string dir = "results/r3-random-1shot";
    std::string reference;
    std::string split_dataset;
    std::string control;   // R3-L run directory (optional)
    int         bootstrap = 2000;
    uint64_t    seed = 20261019;
};

// Per-query aligned data used by every table.
struct Row {
    const RefRecord*               ref = nullptr;
    std::vector<const SeedRecord*> r3;  // seed order
    double r3_mean = 0.0;
    std::vector<const SeedRecord*> r3l; // R3-L control, its seed order
    double r3l_mean = 0.0;
};

int aggregate_main(int argc, char** argv)
{
    Opt opt;
    for (int i = 1; i < argc; ++i) {
        auto arg  = [&](const char* n) { return strcmp(argv[i], n) == 0; };
        auto next = [&]() -> const char* { if (i + 1 >= argc) { fprintf(stderr, "error: missing value\n"); exit(1); } return argv[++i]; };
        if (arg("--mode")) next();
        else if (arg("--output") || arg("--run-dir")) opt.dir = next();
        else if (arg("--reference")) opt.reference = next();
        else if (arg("--split-dataset")) opt.split_dataset = next();
        else if (arg("--control")) opt.control = next();
        else if (arg("--bootstrap")) opt.bootstrap = atoi(next());
        else if (arg("--seed")) opt.seed = std::strtoull(next(), nullptr, 10);
        else { fprintf(stderr, "error: unknown argument %s\n", argv[i]); return 1; }
    }

    json cfg;
    {
        std::ifstream f(opt.dir + "/config.json");
        if (!f.is_open()) { fprintf(stderr, "error: %s/config.json not found (run --mode run first)\n", opt.dir.c_str()); return 1; }
        cfg = json::parse(f);
    }
    const json& det = cfg["experiment_detail"];
    if (opt.reference.empty()) opt.reference = det.value("reference_predictions", "");
    if (opt.split_dataset.empty()) opt.split_dataset = cfg.value("input_path", "");
    std::vector<uint64_t> seeds = det["seeds"].get<std::vector<uint64_t>>();
    const bool preliminary = cfg.value("preliminary", false);

    // Reference must be byte-identical to the one the run was checked against.
    const std::string ref_sha = sha256_file(opt.reference);
    const bool ref_unchanged = ref_sha == det.value("reference_predictions_sha256", "");
    if (!ref_unchanged) {
        fprintf(stderr, "error: %s changed since the R3 run (sha256 mismatch)\n", opt.reference.c_str());
        return 1;
    }

    // --- Reference S1 / R1 on the held-out partition ---
    std::vector<r2::Record> recs;
    r2::LoadStats ls;
    r2::PartitionInfo pinfo;
    std::string err;
    if (!r2::load_predictions(opt.reference, 768, recs, ls, err) || !r2::apply_partition(opt.split_dataset, recs, pinfo, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    std::map<std::string, RefRecord> ref;
    std::vector<std::string> heldout;
    for (const auto& r : recs) {
        if (r.partition != r2::Partition::Evaluation) continue;
        RefRecord x;
        x.query_id = r.query_id; x.kind = r.kind; x.group_id = r.group_id; x.ground_truth = r.ground_truth;
        x.s1_ok = r.s1_ok; x.r1_ok = r.r1_ok && r.retrieval_ok;
        x.s1_correct = r.s1_correct; x.r1_correct = r.r1_correct;
        x.s1_pred = r.s1_pred; x.r1_pred = r.r1_pred;
        auto label = [&](int p) { return (p >= 0 && p < static_cast<int>(r.option_labels.size())) ? r.option_labels[static_cast<size_t>(p)] : std::string(); };
        x.s1_label = label(r.s1_pred); x.r1_label = label(r.r1_pred); x.demo_label = r.demo_label;
        x.similarity = r.similarity;
        x.s1_tokens = r.s1_tokens; x.r1_tokens = r.r1_tokens;
        x.s1_ms = r.s1_ms; x.retrieval_ms = r.retrieval_ms; x.r1_ms = r.r1_ms;
        heldout.push_back(r.query_id);
        ref.emplace(r.query_id, std::move(x));
    }

    // --- Seeds ---
    int parse_errors = 0;
    auto load_seeds = [&](const std::string& dir, const std::vector<uint64_t>& list,
                          std::map<uint64_t, std::vector<SeedRecord>>& out) {
        for (uint64_t s : list) {
            const std::string path = dir + "/seed-" + std::to_string(s) + "/predictions.jsonl";
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open()) { fprintf(stderr, "error: missing %s\n", path.c_str()); return false; }
            std::string line;
            auto& v = out[s];
            while (std::getline(f, line)) {
                if (line.empty()) continue;
                SeedRecord r;
                try {
                    if (parse_seed_record(json::parse(line), r) && r.seed == s) v.push_back(std::move(r));
                    else parse_errors++;
                } catch (...) { parse_errors++; }
            }
        }
        return true;
    };
    std::map<uint64_t, std::vector<SeedRecord>> seed_recs;
    if (!load_seeds(opt.dir, seeds, seed_recs)) return 1;
    // Expected set: the full held-out partition, or (preliminary) the queries the run evaluated, in order.
    std::vector<std::string> expected;
    if (preliminary) {
        for (const auto& r : seed_recs.begin()->second) expected.push_back(r.query_id);
    } else {
        expected = heldout;
    }
    const Alignment al = align(expected, ref, seed_recs);
    printf("Aligned %zu queries (%zu analysed), missing %d, unexpected %d, duplicates %d, gt mismatch %d\n",
           al.query_ids.size(), al.analysed.size(), al.missing_in_seed, al.unexpected_in_seed,
           al.duplicate_in_seed, al.ground_truth_mismatch);
    if (!al.exact() || parse_errors) {
        fprintf(stderr, "error: seed predictions do not align with the reference (parse errors %d)\n", parse_errors);
        return 1;
    }

    // --- Optional R3-L control: same reference, same queries ---
    const bool has_ctrl = !opt.control.empty();
    json cfg_c;
    std::vector<uint64_t> seeds_c;
    std::map<uint64_t, std::vector<SeedRecord>> ctrl_recs;
    Alignment al_c;
    std::unordered_set<std::string> ctrl_ok;
    if (has_ctrl) {
        std::ifstream f(opt.control + "/config.json");
        if (!f.is_open()) { fprintf(stderr, "error: %s/config.json not found\n", opt.control.c_str()); return 1; }
        cfg_c = json::parse(f);
        const json& dc = cfg_c["experiment_detail"];
        if (dc.value("reference_predictions_sha256", "") != ref_sha || cfg_c.value("preliminary", false) != preliminary) {
            fprintf(stderr, "error: control run used a different reference or subset\n");
            return 1;
        }
        if (r2::config_differences(cfg, cfg_c).size()) {
            fprintf(stderr, "error: control run configuration differs from the R3 run\n");
            return 1;
        }
        seeds_c = dc["seeds"].get<std::vector<uint64_t>>();
        if (!load_seeds(opt.control, seeds_c, ctrl_recs)) return 1;
        al_c = align(expected, ref, ctrl_recs);
        if (!al_c.exact() || parse_errors) {
            fprintf(stderr, "error: control predictions do not align with the reference\n");
            return 1;
        }
        ctrl_ok.insert(al_c.analysed.begin(), al_c.analysed.end());
        printf("Control %s: %zu seeds, %zu analysed\n", opt.control.c_str(), seeds_c.size(), al_c.analysed.size());
    }
    std::vector<std::string> analysed;
    for (const auto& id : al.analysed) if (!has_ctrl || ctrl_ok.count(id)) analysed.push_back(id);
    const int ctrl_excluded = static_cast<int>(al.analysed.size() - analysed.size());

    // --- Rows ---
    std::map<uint64_t, std::unordered_map<std::string, const SeedRecord*>> by_seed;
    for (const auto& [s, v] : seed_recs) for (const auto& r : v) by_seed[s][r.query_id] = &r;
    std::vector<Row> rows;
    std::map<uint64_t, std::unordered_map<std::string, const SeedRecord*>> by_seed_c;
    for (const auto& [s, v] : ctrl_recs) for (const auto& r : v) by_seed_c[s][r.query_id] = &r;
    for (const auto& id : analysed) {
        Row row;
        row.ref = &ref.at(id);
        double c = 0;
        for (uint64_t s : seeds) { row.r3.push_back(by_seed[s].at(id)); c += row.r3.back()->correct; }
        row.r3_mean = c / seeds.size();
        if (has_ctrl) {
            double cl = 0;
            for (uint64_t s : seeds_c) { row.r3l.push_back(by_seed_c[s].at(id)); cl += row.r3l.back()->correct; }
            row.r3l_mean = cl / seeds_c.size();
        }
        rows.push_back(std::move(row));
    }
    const size_t S = seeds.size();
    const std::vector<std::string> kinds = {"choice", "noul", "score", "all"};
    auto in_kind = [](const Row& r, const std::string& k) { return k == "all" || r.ref->kind == k; };

    // Each kind has its own resampler; within a kind every statistic uses the same replicates.
    auto series = [&](const std::string& k, auto get) {
        std::vector<double> v;
        for (const auto& r : rows) if (in_kind(r, k)) v.push_back(get(r));
        return v;
    };
    auto cluster_keys = [&](const std::string& k) {
        std::vector<std::string> c;
        for (const auto& r : rows) if (in_kind(r, k)) c.push_back(r.ref->group_id);
        return c;
    };

    struct KindStats {
        int n = 0;
        double s1 = 0, r1 = 0;
        std::vector<double> r3_seed;      // accuracy per seed
        SeedSummary r3;
        ClusterCI r3_s1, r1_r3, r1_s1;
        std::vector<ClusterCI> r1_r3_seed, r3_s1_seed;
        // Fraction of R1 - S1 attributable to selection: (R1 - R3) / (R1 - S1)
        double share = 0, share_lo = 0, share_hi = 0;
        // R3-L control
        std::vector<double> r3l_seed;
        SeedSummary r3l;
        ClusterCI r1_r3l, r3l_r3, r3l_s1;
    };
    std::map<std::string, KindStats> ks;
    for (const auto& k : kinds) {
        KindStats& st = ks[k];
        const auto s1 = series(k, [](const Row& r) { return r.ref->s1_correct ? 1.0 : 0.0; });
        const auto r1 = series(k, [](const Row& r) { return r.ref->r1_correct ? 1.0 : 0.0; });
        const auto r3m = series(k, [](const Row& r) { return r.r3_mean; });
        st.n = static_cast<int>(s1.size());
        if (!st.n) continue;
        st.s1 = mean_of(s1); st.r1 = mean_of(r1);
        Resampler rs(cluster_keys(k), opt.bootstrap, opt.seed);
        st.r3_s1 = paired_ci(rs, r3m, s1);
        st.r1_r3 = paired_ci(rs, r1, r3m);
        st.r1_s1 = paired_ci(rs, r1, s1);
        for (size_t si = 0; si < S; ++si) {
            const auto r3s = series(k, [&](const Row& r) { return r.r3[si]->correct ? 1.0 : 0.0; });
            st.r3_seed.push_back(mean_of(r3s));
            st.r1_r3_seed.push_back(paired_ci(rs, r1, r3s));
            st.r3_s1_seed.push_back(paired_ci(rs, r3s, s1));
        }
        st.r3 = summarize(st.r3_seed);
        if (has_ctrl) {
            const auto r3lm = series(k, [](const Row& r) { return r.r3l_mean; });
            st.r1_r3l = paired_ci(rs, r1, r3lm);
            st.r3l_r3 = paired_ci(rs, r3lm, r3m);
            st.r3l_s1 = paired_ci(rs, r3lm, s1);
            for (size_t si = 0; si < seeds_c.size(); ++si)
                st.r3l_seed.push_back(mean_of(series(k, [&](const Row& r) { return r.r3l[si]->correct ? 1.0 : 0.0; })));
            st.r3l = summarize(st.r3l_seed);
        }
        {
            std::vector<double> d_sel(s1.size()), d_tot(s1.size());
            for (size_t i = 0; i < s1.size(); ++i) { d_sel[i] = r1[i] - r3m[i]; d_tot[i] = r1[i] - s1[i]; }
            const auto cs = rs.sums(d_sel), ct = rs.sums(d_tot);
            double ts = 0, tt = 0;
            for (size_t i = 0; i < s1.size(); ++i) { ts += d_sel[i]; tt += d_tot[i]; }
            st.share = tt != 0 ? ts / tt : 0.0;
            std::vector<double> reps;
            for (int b = 0; b < rs.reps(); ++b) {
                const double t = rs.weighted(b, ct);
                if (t != 0) reps.push_back(rs.weighted(b, cs) / t);
            }
            st.share_lo = r1::percentile(reps, 2.5);
            st.share_hi = r1::percentile(reps, 97.5);
        }
    }
    const KindStats& all = ks["all"];

    // --- Transitions ---
    struct Trans { r1::TransitionCounts vs_s1; Agreement r1_vs_r3; };
    std::map<std::string, Trans> r1_trans;                // per kind, R1 vs S1
    std::map<std::string, std::vector<Trans>> r3_trans;  // per kind, per seed
    for (const auto& k : kinds) {
        r3_trans[k].resize(S);
        for (const auto& r : rows) {
            if (!in_kind(r, k)) continue;
            r1_trans[k].vs_s1.add(r1::classify(r.ref->s1_correct, r.ref->r1_correct));
            for (size_t si = 0; si < S; ++si) {
                r3_trans[k][si].vs_s1.add(r1::classify(r.ref->s1_correct, r.r3[si]->correct));
                r3_trans[k][si].r1_vs_r3.add(r.ref->r1_correct, r.r3[si]->correct);
            }
        }
    }
    auto mean_over_seeds = [&](const std::string& k, auto get) {
        double t = 0;
        for (const auto& tr : r3_trans[k]) t += get(tr);
        return S ? t / S : 0.0;
    };

    // --- Similarity ---
    std::vector<double> near_sim, rand_sim;
    std::map<std::string, std::vector<double>> near_sim_k, rand_sim_k;
    int rand_sim_missing = 0;
    for (const auto& r : rows) {
        near_sim.push_back(r.ref->similarity);
        near_sim_k[r.ref->kind].push_back(r.ref->similarity);
        for (const auto* x : r.r3) {
            if (!x->has_similarity) { rand_sim_missing++; continue; }
            rand_sim.push_back(x->similarity);
            rand_sim_k[x->kind].push_back(x->similarity);
        }
    }
    // Common bins: sextiles of the pooled (nearest + random) similarities, so both strategies share edges.
    std::vector<double> edges;
    {
        std::vector<double> pooled = near_sim;
        pooled.insert(pooled.end(), rand_sim.begin(), rand_sim.end());
        for (int q = 1; q < 6; ++q) edges.push_back(r1::percentile(pooled, q * 100.0 / 6));
    }
    auto bin_of = [&](double s) { int b = 0; while (b < static_cast<int>(edges.size()) && s >= edges[static_cast<size_t>(b)]) b++; return b; };
    struct SimBin { int n_near = 0, near_correct = 0, near_rescue = 0, near_harm = 0, n_rand = 0, rand_correct = 0, rand_rescue = 0, rand_harm = 0; };
    std::vector<SimBin> sbins(edges.size() + 1);
    for (const auto& r : rows) {
        SimBin& b = sbins[static_cast<size_t>(bin_of(r.ref->similarity))];
        b.n_near++; b.near_correct += r.ref->r1_correct;
        b.near_rescue += !r.ref->s1_correct && r.ref->r1_correct; b.near_harm += r.ref->s1_correct && !r.ref->r1_correct;
        for (const auto* x : r.r3) {
            if (!x->has_similarity) continue;
            SimBin& c = sbins[static_cast<size_t>(bin_of(x->similarity))];
            c.n_rand++; c.rand_correct += x->correct;
            c.rand_rescue += !r.ref->s1_correct && x->correct; c.rand_harm += r.ref->s1_correct && !x->correct;
        }
    }
    // Within-query: does the random draw that happens to be more similar help more? Pearson correlation
    // between (similarity - query mean similarity) and (correct - query mean correctness).
    double wq_num = 0, wq_sx = 0, wq_sy = 0;
    for (const auto& r : rows) {
        double ms = 0, mc = 0; int n = 0;
        for (const auto* x : r.r3) if (x->has_similarity) { ms += x->similarity; mc += x->correct; n++; }
        if (n < 2) continue;
        ms /= n; mc /= n;
        for (const auto* x : r.r3) if (x->has_similarity) {
            const double dx = x->similarity - ms, dy = x->correct - mc;
            wq_num += dx * dy; wq_sx += dx * dx; wq_sy += dy * dy;
        }
    }
    const double within_corr = (wq_sx > 0 && wq_sy > 0) ? wq_num / std::sqrt(wq_sx * wq_sy) : 0.0;

    // --- Labels ---
    struct LabelStats { int n = 0, near_demo_gt = 0, r1_eq_demo = 0, s1_eq_near_demo = 0, rand_n = 0, rand_demo_gt = 0, r3_eq_demo = 0, s1_eq_rand_demo = 0; };
    std::map<std::string, LabelStats> lab;
    for (const auto& k : kinds) {
        LabelStats& L = lab[k];
        for (const auto& r : rows) {
            if (!in_kind(r, k)) continue;
            L.n++;
            const auto& gl = r.ref;
            // Ground-truth label from the seed record (all seeds agree; alignment verified).
            const std::string& gt_label = r.r3[0]->ground_truth_label;
            L.near_demo_gt   += gl->demo_label == gt_label;
            L.r1_eq_demo     += gl->r1_label == gl->demo_label;
            L.s1_eq_near_demo += gl->s1_label == gl->demo_label;
            for (const auto* x : r.r3) {
                L.rand_n++;
                L.rand_demo_gt    += x->demo_label == gt_label;
                L.r3_eq_demo      += x->prediction_label == x->demo_label;
                L.s1_eq_rand_demo += gl->s1_label == x->demo_label;
            }
        }
    }

    // --- Tokens and latency ---
    struct Cost {
        std::vector<double> s1_ms, r1_ms, r1_sel_ms, r1_dec_ms, r3_ms, r3_sel_ms, r3_dec_ms;
        std::vector<double> s1_tok, r1_tok, r3_tok, r1_extra, r3_extra;
        std::vector<double> r3l_ms, r3l_sel_ms, r3l_dec_ms, r3l_tok, r3l_extra, r3l_sim;
        int r3l_draws = 0, r3l_same_as_nearest = 0;
    };
    std::map<std::string, Cost> cost;
    for (const auto& k : kinds) {
        Cost& C = cost[k];
        for (const auto& r : rows) {
            if (!in_kind(r, k)) continue;
            C.s1_ms.push_back(r.ref->s1_ms);
            C.r1_sel_ms.push_back(r.ref->retrieval_ms);
            C.r1_dec_ms.push_back(r.ref->r1_ms);
            C.r1_ms.push_back(r.ref->retrieval_ms + r.ref->r1_ms);
            C.s1_tok.push_back(r.ref->s1_tokens);
            C.r1_tok.push_back(r.ref->r1_tokens);
            C.r1_extra.push_back(r.ref->r1_tokens - r.ref->s1_tokens);
            for (const auto* x : r.r3) {
                C.r3_sel_ms.push_back(x->select_ms + x->fetch_ms + x->decompress_ms);
                C.r3_dec_ms.push_back(x->decision_ms);
                C.r3_ms.push_back(x->total_ms());
                C.r3_tok.push_back(x->prompt_tokens);
                C.r3_extra.push_back(x->prompt_tokens - r.ref->s1_tokens);
            }
            for (const auto* x : r.r3l) {
                C.r3l_sel_ms.push_back(x->select_ms + x->fetch_ms + x->decompress_ms);
                C.r3l_dec_ms.push_back(x->decision_ms);
                C.r3l_ms.push_back(x->total_ms());
                C.r3l_tok.push_back(x->prompt_tokens);
                C.r3l_extra.push_back(x->prompt_tokens - r.ref->s1_tokens);
                if (x->has_similarity) C.r3l_sim.push_back(x->similarity);
                C.r3l_draws++;
                C.r3l_same_as_nearest += x->same_as_nearest;
            }
        }
    }
    // Same-session subset (reference_check.jsonl): S1 and nearest R1 re-run in the R3 process.
    struct Session { int n = 0; std::vector<double> s1, r1_sel, r1_dec, r3_sel, r3_dec; };
    Session sess;
    {
        std::ifstream f(opt.dir + "/reference_check.jsonl", std::ios::binary);
        std::string line;
        std::unordered_map<std::string, const Row*> row_of;
        for (const auto& r : rows) row_of[r.ref->query_id] = &r;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            json j = json::parse(line);
            auto it = row_of.find(str(j, "query_id"));
            if (it == row_of.end()) continue;
            sess.n++;
            sess.s1.push_back(num(j, "s1_ms"));
            sess.r1_sel.push_back(num(j, "embed_ms") + num(j, "search_ms") + num(j, "load_ms"));
            sess.r1_dec.push_back(num(j, "r1_ms"));
            for (const auto* x : it->second->r3) {
                sess.r3_sel.push_back(x->select_ms + x->fetch_ms + x->decompress_ms);
                sess.r3_dec.push_back(x->decision_ms);
            }
        }
    }

    // --- Classification ---
    int seeds_r1_better = 0, seeds_r3_better = 0, seeds_ci_pos = 0, seeds_ci_neg = 0;
    for (size_t si = 0; si < S; ++si) {
        const auto& c = all.r1_r3_seed[si];
        if (c.diff > 0) seeds_r1_better++;
        if (c.diff < 0) seeds_r3_better++;
        if (c.lo > 0) seeds_ci_pos++;
        if (c.hi < 0) seeds_ci_neg++;
    }
    std::string verdict;
    if (all.r1_r3.lo > 0 && seeds_r1_better == static_cast<int>(S)) verdict = "Evidence favoring Nearest 1-shot";
    else if (all.r1_r3.hi < 0 && seeds_r3_better == static_cast<int>(S)) verdict = "Evidence favoring Random 1-shot";
    else verdict = "No clear difference";

    // --- Write outputs ---
    const std::string& D = opt.dir;
    {
        std::ofstream f(D + "/per_seed.csv", std::ios::binary);
        f << "kind,seed,n,r3_accuracy,r3_minus_s1,r1_minus_r3,r1_minus_r3_ci_lo,r1_minus_r3_ci_hi,rescue,harm,net,"
             "r1_correct_r3_incorrect,r1_incorrect_r3_correct,both_correct,both_incorrect\n";
        for (const auto& k : kinds) {
            const auto& st = ks[k];
            for (size_t si = 0; si < S && st.n; ++si) {
                const auto& t = r3_trans[k][si];
                f << k << "," << seeds[si] << "," << st.n << "," << fmt("%.6f", st.r3_seed[si]) << ","
                  << fmt("%.6f", st.r3_s1_seed[si].diff) << "," << fmt("%.6f", st.r1_r3_seed[si].diff) << ","
                  << fmt("%.6f", st.r1_r3_seed[si].lo) << "," << fmt("%.6f", st.r1_r3_seed[si].hi) << ","
                  << t.vs_s1.rescue << "," << t.vs_s1.harm << "," << t.vs_s1.net() << ","
                  << t.r1_vs_r3.a_only << "," << t.r1_vs_r3.b_only << "," << t.r1_vs_r3.both_correct << ","
                  << t.r1_vs_r3.both_incorrect << "\n";
            }
        }
    }
    {
        std::ofstream f(D + "/similarity.csv", std::ios::binary);
        f << "bin,cos_lo,cos_hi,n_nearest,r1_accuracy,r1_rescue,r1_harm,n_random_draws,r3_accuracy,r3_rescue,r3_harm\n";
        for (size_t b = 0; b < sbins.size(); ++b) {
            const auto& x = sbins[b];
            f << b + 1 << "," << (b ? fmt("%.6f", edges[b - 1]) : "-inf") << "," << (b < edges.size() ? fmt("%.6f", edges[b]) : "inf") << ","
              << x.n_near << "," << (x.n_near ? fmt("%.6f", double(x.near_correct) / x.n_near) : "") << "," << x.near_rescue << "," << x.near_harm << ","
              << x.n_rand << "," << (x.n_rand ? fmt("%.6f", double(x.rand_correct) / x.n_rand) : "") << "," << x.rand_rescue << "," << x.rand_harm << "\n";
        }
    }
    {
        std::ofstream f(D + "/latency.csv", std::ios::binary);
        f << "kind,strategy,accuracy,mean_ms,median_ms,p95_ms,selection_mean_ms,decision_mean_ms,mean_prompt_tokens,mean_extra_tokens,source\n";
        for (const auto& k : kinds) {
            const auto& C = cost[k];
            const auto& st = ks[k];
            if (!st.n) continue;
            f << k << ",S1," << fmt("%.6f", st.s1) << "," << fmt("%.3f", mean_of(C.s1_ms)) << "," << fmt("%.3f", median_of(C.s1_ms)) << ","
              << fmt("%.3f", p95_of(C.s1_ms)) << ",0," << fmt("%.3f", mean_of(C.s1_ms)) << "," << fmt("%.1f", mean_of(C.s1_tok)) << ",0,reference R1 run\n";
            f << k << ",R1," << fmt("%.6f", st.r1) << "," << fmt("%.3f", mean_of(C.r1_ms)) << "," << fmt("%.3f", median_of(C.r1_ms)) << ","
              << fmt("%.3f", p95_of(C.r1_ms)) << "," << fmt("%.3f", mean_of(C.r1_sel_ms)) << "," << fmt("%.3f", mean_of(C.r1_dec_ms)) << ","
              << fmt("%.1f", mean_of(C.r1_tok)) << "," << fmt("%.1f", mean_of(C.r1_extra)) << ",reference R1 run\n";
            f << k << ",R3," << fmt("%.6f", st.r3.mean) << "," << fmt("%.3f", mean_of(C.r3_ms)) << "," << fmt("%.3f", median_of(C.r3_ms)) << ","
              << fmt("%.3f", p95_of(C.r3_ms)) << "," << fmt("%.3f", mean_of(C.r3_sel_ms)) << "," << fmt("%.3f", mean_of(C.r3_dec_ms)) << ","
              << fmt("%.1f", mean_of(C.r3_tok)) << "," << fmt("%.1f", mean_of(C.r3_extra)) << ",R3 run (all seeds)\n";
        }
    }

    // Accuracy vs latency plot (SVG): S1, R1, R3 seeds and mean.
    {
        const auto& C = cost["all"];
        struct P { std::string label; double ms, acc; bool small; };
        std::vector<P> pts = {{"S1", mean_of(C.s1_ms), all.s1, false}, {"R1 (nearest)", mean_of(C.r1_ms), all.r1, false},
                              {"R3 mean (random)", mean_of(C.r3_ms), all.r3.mean, false}};
        for (size_t si = 0; si < S; ++si) pts.push_back({"", mean_of(C.r3_ms), all.r3_seed[si], true});
        if (has_ctrl) pts.push_back({"R3-L (length-matched)", mean_of(C.r3l_ms), all.r3l.mean, false});
        double xmax = 0, ymin = 1, ymax = 0;
        for (const auto& p : pts) { xmax = std::max(xmax, p.ms); ymin = std::min(ymin, p.acc); ymax = std::max(ymax, p.acc); }
        xmax *= 1.15; ymin = std::floor(ymin * 50) / 50 - 0.02; ymax = std::ceil(ymax * 50) / 50 + 0.02;
        const double W = 640, H = 420, L = 70, R = 30, T = 30, B = 60;
        auto X = [&](double v) { return L + v / xmax * (W - L - R); };
        auto Y = [&](double v) { return H - B - (v - ymin) / (ymax - ymin) * (H - T - B); };
        std::ofstream f(D + "/accuracy_latency.svg", std::ios::binary);
        f << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << W << "\" height=\"" << H << "\" font-family=\"sans-serif\" font-size=\"12\">\n"
          << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n"
          << "<line x1=\"" << L << "\" y1=\"" << H - B << "\" x2=\"" << W - R << "\" y2=\"" << H - B << "\" stroke=\"black\"/>\n"
          << "<line x1=\"" << L << "\" y1=\"" << T << "\" x2=\"" << L << "\" y2=\"" << H - B << "\" stroke=\"black\"/>\n";
        for (int t = 0; t <= 5; ++t) {
            const double v = xmax * t / 5, y = ymin + (ymax - ymin) * t / 5;
            f << "<text x=\"" << X(v) << "\" y=\"" << H - B + 18 << "\" text-anchor=\"middle\">" << fmt("%.0f", v) << "</text>\n"
              << "<text x=\"" << L - 8 << "\" y=\"" << Y(y) + 4 << "\" text-anchor=\"end\">" << fmt("%.1f%%", y * 100) << "</text>\n";
        }
        f << "<text x=\"" << (L + W - R) / 2 << "\" y=\"" << H - 15 << "\" text-anchor=\"middle\">mean per-query latency, ms (held-out)</text>\n"
          << "<text x=\"18\" y=\"" << (T + H - B) / 2 << "\" text-anchor=\"middle\" transform=\"rotate(-90 18 " << (T + H - B) / 2 << ")\">accuracy</text>\n";
        for (const auto& p : pts) {
            f << "<circle cx=\"" << X(p.ms) << "\" cy=\"" << Y(p.acc) << "\" r=\"" << (p.small ? 3 : 6) << "\" fill=\""
              << (p.small ? "#9ab" : p.label[0] == 'S' ? "#555" : p.label[1] == '1' ? "#c33" : p.label.rfind("R3-L", 0) == 0 ? "#3a3" : "#36c") << "\"/>\n";
            if (!p.label.empty())
                f << "<text x=\"" << X(p.ms) + 10 << "\" y=\"" << Y(p.acc) + 4 << "\">" << p.label << " " << fmt("%.2f%%", p.acc * 100) << "</text>\n";
        }
        f << "</svg>\n";
    }

    json agg;
    agg["verdict"] = verdict;
    agg["n_analysed"] = rows.size();
    agg["seeds"] = seeds;
    for (const auto& k : kinds) {
        const auto& st = ks[k];
        if (!st.n) continue;
        agg["kinds"][k] = {
            {"n", st.n}, {"s1", st.s1}, {"r1", st.r1}, {"r3_per_seed", st.r3_seed},
            {"r3_mean", st.r3.mean}, {"r3_sd", st.r3.sd}, {"r3_min", st.r3.min}, {"r3_max", st.r3.max},
            {"r3_minus_s1", {{"diff", st.r3_s1.diff}, {"lo", st.r3_s1.lo}, {"hi", st.r3_s1.hi}}},
            {"r1_minus_r3", {{"diff", st.r1_r3.diff}, {"lo", st.r1_r3.lo}, {"hi", st.r1_r3.hi}}},
            {"r1_minus_s1", {{"diff", st.r1_s1.diff}, {"lo", st.r1_s1.lo}, {"hi", st.r1_s1.hi}}},
            {"selection_share", {{"point", st.share}, {"lo", st.share_lo}, {"hi", st.share_hi}}},
            {"clusters", st.r1_r3.clusters}};
    }
    agg["within_query_similarity_correctness_correlation"] = within_corr;
    if (has_ctrl) {
        agg["control"] = {{"dir", opt.control}, {"seeds", seeds_c}};
        for (const auto& k : kinds) {
            const auto& st = ks[k];
            if (!st.n) continue;
            agg["control"]["kinds"][k] = {
                {"r3l_per_seed", st.r3l_seed}, {"r3l_mean", st.r3l.mean}, {"r3l_sd", st.r3l.sd},
                {"r1_minus_r3l", {{"diff", st.r1_r3l.diff}, {"lo", st.r1_r3l.lo}, {"hi", st.r1_r3l.hi}}},
                {"r3l_minus_r3", {{"diff", st.r3l_r3.diff}, {"lo", st.r3l_r3.lo}, {"hi", st.r3l_r3.hi}}},
                {"r3l_minus_s1", {{"diff", st.r3l_s1.diff}, {"lo", st.r3l_s1.lo}, {"hi", st.r3l_s1.hi}}},
                {"mean_demo_tokens", mean_of(cost[k].r3l_extra)}};
        }
    }
    {
        std::ofstream f(D + "/aggregate.json", std::ios::binary);
        f << agg.dump(2) << "\n";
    }

    // --- summary.md ---
    std::ofstream o(D + "/summary.md", std::ios::binary);
    o << "# R3 — Random 1-shot vs Nearest 1-shot\n\n";
    if (preliminary) o << "> **Preliminary:** smoke run on a subset of the held-out partition; not a result.\n\n";
    o << "Central question: does nearest-neighbour selection add accuracy over an arbitrary eligible training "
         "demonstration, enough to pay for its retrieval cost?\n\n";

    o << "## Configuration\n\n| Setting | Value |\n|---|---|\n";
    const json& mid = cfg["inference_model_identity"];
    o << "| Inference model | " << cfg.value("inference_model", "") << " (" << mid.value("name", "") << ", sha256 `" << mid.value("sha256", "") << "`) |\n";
    o << "| Embedding model | " << cfg.value("embedding_model", "") << " (fingerprint `" << cfg.value("embedding_fingerprint", "")
      << "`; R1 retrieval and similarity analysis only) |\n";
    o << "| Exemplar DB | " << cfg.value("db_path", "") << " (schema " << cfg["db_info"].value("schema_version", "") << ", read-only) |\n";
    o << "| Training allowlist | " << cfg.value("train_path", "") << " |\n";
    o << "| Queries | " << cfg.value("input_path", "") << " (sha256 `" << cfg.value("input_sha256", "") << "`), held-out partition |\n";
    o << "| Prompt format | " << cfg.value("prompt_format_version", "") << " — " << cfg.value("prompt_format", "") << " |\n";
    o << "| Decision | letters scheme, auto layout, single order, no calibration; candidate-logit argmax, correctness by candidate index |\n";
    o << "| n_ctx | " << cfg.value("n_ctx", 0) << " |\n";
    o << "| Reference S1 / R1 | " << opt.reference << " (sha256 `" << ref_sha.substr(0, 16) << "…`, unchanged) |\n";
    o << "| R3 selection | " << det.value("random_index", "") << " |\n";
    o << "| Seeds (fixed before any R3 result) | ";
    for (size_t si = 0; si < S; ++si) o << (si ? ", " : "") << seeds[si];
    o << " |\n";
    o << "| Run | " << cfg.value("run_timestamp", "") << ", wall " << fmt("%.0f", det.value("wall_seconds", 0.0)) << " s |\n";
    o << "| Bootstrap | " << opt.bootstrap << " cluster replicates over group_id, seed " << opt.seed << " |\n\n";
    o << "All identity keys of the R1 configuration (model hash, embedding fingerprint, DB, prompt format, scheme, layout, "
         "ensemble, calibration, n_ctx, top_k, leakage policy) were checked equal to the reference before the run.\n\n";

    o << "## Data integrity\n\n";
    const json& part = det["partition"];
    const json& rc = det["reference_check"];
    const json& ps = det["pool_stats"];
    o << "- Held-out partition recomputed with the R2 implementation (`r2::apply_partition`) and compared with the saved "
         "assignments in `" << det.value("split_assignments", "") << "`: " << part.value("heldout", 0) << " evaluation queries, "
      << part.value("disagreements", -1) << " disagreements, " << part.value("group_overlap", -1) << " groups straddling "
         "tuning/evaluation, " << part.value("train_split_records", -1) << " queries labelled train.\n";
    o << "- Evaluation families:";
    for (const auto& f : part["evaluation_families"]) o << " `" << f.get<std::string>() << "`";
    o << "; remaining queries split by group_id.\n";
    o << "- Alignment: " << al.query_ids.size() << " queries present in the reference and in every seed; missing "
      << al.missing_in_seed << ", unexpected " << al.unexpected_in_seed << ", duplicates " << al.duplicate_in_seed
      << ", kind/ground-truth disagreements " << al.ground_truth_mismatch << ".\n";
    o << "- Analysed (S1, R1 and all " << S << " R3 seeds ok): **" << rows.size() << "**. Reference failures: "
      << al.reference_failed << "; R3 failures per seed:";
    for (const auto& [s, n] : al.failed_per_seed) o << " " << s << "=" << n;
    o << ". No query is excluded silently: failed queries would be listed here and dropped from every strategy alike.\n";
    o << "- Eligible pool = R1's candidate population: " << ps.value("eligible", 0) << " of " << ps.value("vector_rows", 0)
      << " vector-index rows (" << ps.value("outside_train", 0) << " outside the training split, " << ps.value("no_one_hot", 0)
      << " training rows without a one-hot answer); partitioned by R1's KNN filter (kind, and level count for score):";
    for (const auto& [k, v] : det["pool_sizes"].items()) o << " " << k << "=" << v.get<int64_t>();
    o << ". The query itself is removed before drawing. Queries without an eligible exemplar: " << det.value("no_eligible_exemplar", 0) << ".\n";
    if (has_ctrl)
        o << "- R3-L control `" << opt.control << "`: same reference (sha256 checked), identical identity keys, aligned on the same queries "
             "(missing " << al_c.missing_in_seed << ", duplicates " << al_c.duplicate_in_seed << "); queries dropped because an R3-L "
             "seed failed: " << ctrl_excluded << ".\n";
    o << "- Leakage: every selected exemplar was re-checked at load time (in the allowlist, one-hot, same kind/levels, not the query).\n";
    o << "- Reference reproducibility: S1 and nearest R1 re-run in the R3 process on the first " << rc.value("queries", 0)
      << " held-out queries — S1 logit/token mismatches " << rc.value("s1_mismatch", -1) << ", nearest exemplar mismatches "
      << rc.value("nearest_example_mismatch", -1) << ", R1 logit mismatches " << rc.value("r1_logit_mismatch", -1)
      << ". Similarity probe reproduced the KNN Hamming distance on " << rc.value("hamming_probe_checked", 0) - rc.value("hamming_probe_mismatch", 0)
      << "/" << rc.value("hamming_probe_checked", 0) << " nearest exemplars.\n";
    o << "- Context: R1's policy (no length filter, no truncation); context-limit failures are counted as R3 failures above.\n\n";

    o << "## Overall results (held-out, n=" << all.n << ")\n\n";
    o << "| Strategy | Accuracy |\n|---|---:|\n";
    o << "| S1 — zero-shot | " << pct(all.s1) << " |\n| R1 — nearest 1-shot | " << pct(all.r1) << " |\n";
    o << "| R3 — random 1-shot (mean of " << S << " seeds) | " << pct(all.r3.mean) << " |\n\n";
    o << "| Effect | Difference | 95% cluster bootstrap |\n|---|---:|---|\n";
    o << "| General 1-shot effect (R3 − S1) | " << pp(all.r3_s1.diff) << " | " << ci_pp(all.r3_s1) << " |\n";
    o << "| Nearest-selection effect (R1 − R3) | " << pp(all.r1_r3.diff) << " | " << ci_pp(all.r1_r3) << " |\n";
    o << "| Total (R1 − S1) | " << pp(all.r1_s1.diff) << " | " << ci_pp(all.r1_s1) << " |\n\n";
    o << "Share of the R1 − S1 improvement attributable to nearest selection, (R1 − R3) / (R1 − S1): **"
      << fmt("%.1f%%", all.share * 100) << "** (95% cluster bootstrap [" << fmt("%.1f", all.share_lo * 100) << "%, "
      << fmt("%.1f", all.share_hi * 100) << "%]).\n\n";
    o << "The bootstrap resamples evaluation groups (" << all.r1_r3.clusters << " clusters); each sampled query keeps its S1, R1 and "
         "all seed outcomes, so repeated predictions of one query are never treated as independent.\n\n";

    o << "## Seed variance\n\n| Seed | R3 accuracy | R3 − S1 | R1 − R3 | R1 − R3 95% CI |\n|---:|---:|---:|---:|---|\n";
    for (size_t si = 0; si < S; ++si)
        o << "| " << seeds[si] << " | " << pct(all.r3_seed[si]) << " | " << pp(all.r3_s1_seed[si].diff) << " | "
          << pp(all.r1_r3_seed[si].diff) << " | " << ci_pp(all.r1_r3_seed[si]) << " |\n";
    o << "\nR3 across seeds: mean " << pct(all.r3.mean) << ", SD " << fmt("%.2f pp", all.r3.sd * 100) << ", min " << pct(all.r3.min)
      << ", max " << pct(all.r3.max) << ". R1 beats R3 on " << seeds_r1_better << "/" << S << " seeds (CI above zero on "
      << seeds_ci_pos << "), R3 beats R1 on " << seeds_r3_better << "/" << S << " (CI below zero on " << seeds_ci_neg << ").\n\n";

    o << "## Task-level results\n\n| Kind | n | S1 | R1 | R3 mean (SD) | R3 min–max | R3 − S1 [95% CI] | R1 − R3 [95% CI] | Selection share |\n"
         "|---|---:|---:|---:|---:|---|---:|---:|---:|\n";
    for (const auto& k : kinds) {
        const auto& st = ks[k];
        if (!st.n) continue;
        o << "| " << k << " | " << st.n << " | " << pct(st.s1) << " | " << pct(st.r1) << " | " << pct(st.r3.mean) << " ("
          << fmt("%.2f", st.r3.sd * 100) << ") | " << pct(st.r3.min) << " – " << pct(st.r3.max) << " | " << pp(st.r3_s1.diff)
          << " " << ci_pp(st.r3_s1) << " | " << pp(st.r1_r3.diff) << " " << ci_pp(st.r1_r3) << " | "
          << (std::fabs(st.r1_s1.diff) > 0.005 ? fmt("%.0f%%", st.share * 100) : std::string("n/a")) << " |\n";
    }
    o << "\nPer-seed R1 − R3 by kind:\n\n| Kind |";
    for (uint64_t s : seeds) o << " " << s << " |";
    o << "\n|---|";
    for (size_t si = 0; si < S; ++si) o << "---:|";
    o << "\n";
    for (const auto& k : kinds) {
        if (!ks[k].n) continue;
        o << "| " << k << " |";
        for (size_t si = 0; si < S; ++si) o << " " << pp(ks[k].r1_r3_seed[si].diff) << " |";
        o << "\n";
    }
    o << "\nSelection share is (R1 − R3) / (R1 − S1); shown only where R1 − S1 exceeds 0.5 pp.\n\n";

    o << "## Rescue and harm (relative to S1)\n\n| Kind | Strategy | Rescue | Harm | Net | Net / n |\n|---|---|---:|---:|---:|---:|\n";
    for (const auto& k : kinds) {
        if (!ks[k].n) continue;
        const auto& t = r1_trans[k].vs_s1;
        o << "| " << k << " | R1 | " << t.rescue << " | " << t.harm << " | " << t.net() << " | " << pp(double(t.net()) / t.n()) << " |\n";
        const double mr = mean_over_seeds(k, [](const Trans& x) { return double(x.vs_s1.rescue); });
        const double mh = mean_over_seeds(k, [](const Trans& x) { return double(x.vs_s1.harm); });
        o << "| " << k << " | R3 mean | " << fmt("%.1f", mr) << " | " << fmt("%.1f", mh) << " | " << fmt("%.1f", mr - mh)
          << " | " << pp((mr - mh) / ks[k].n) << " |\n";
    }
    o << "\nR3 per seed (all kinds):\n\n| Seed | Rescue | Harm | Net |\n|---:|---:|---:|---:|\n";
    for (size_t si = 0; si < S; ++si) {
        const auto& t = r3_trans["all"][si].vs_s1;
        o << "| " << seeds[si] << " | " << t.rescue << " | " << t.harm << " | " << t.net() << " |\n";
    }
    o << "\nDirect R1 vs R3 (mean over seeds):\n\n| Kind | R1 correct / R3 incorrect | R1 incorrect / R3 correct | Both correct | Both incorrect |\n"
         "|---|---:|---:|---:|---:|\n";
    for (const auto& k : kinds) {
        if (!ks[k].n) continue;
        o << "| " << k << " | " << fmt("%.1f", mean_over_seeds(k, [](const Trans& x) { return double(x.r1_vs_r3.a_only); }))
          << " | " << fmt("%.1f", mean_over_seeds(k, [](const Trans& x) { return double(x.r1_vs_r3.b_only); }))
          << " | " << fmt("%.1f", mean_over_seeds(k, [](const Trans& x) { return double(x.r1_vs_r3.both_correct); }))
          << " | " << fmt("%.1f", mean_over_seeds(k, [](const Trans& x) { return double(x.r1_vs_r3.both_incorrect); })) << " |\n";
    }
    {
        const auto& t1 = r1_trans["all"].vs_s1;
        const double mr = mean_over_seeds("all", [](const Trans& x) { return double(x.vs_s1.rescue); });
        const double mh = mean_over_seeds("all", [](const Trans& x) { return double(x.vs_s1.harm); });
        o << "\nNearest vs random selection changes rescues by " << fmt("%+.1f", t1.rescue - mr) << " and harms by "
          << fmt("%+.1f", t1.harm - mh) << " (R1 minus R3 mean, all kinds).\n\n";
    }

    if (has_ctrl) {
        const json& dc = cfg_c["experiment_detail"];
        o << "## Length-matched control (R3-L)\n\nRandom demonstrations differ from nearest ones in length (see Cost analysis), so a "
             "secondary control draws uniformly from the " << dc.value("length_window", 0) << " eligible exemplars whose demonstration "
             "length is closest to the nearest exemplar's (same pools, same leakage rules, no similarity used; " << seeds_c.size()
          << " seeds: ";
        for (size_t si = 0; si < seeds_c.size(); ++si) o << (si ? ", " : "") << seeds_c[si];
        o << "). R3 remains the primary baseline.\n\n";
        o << "| Kind | n | R1 | R3 mean | R3-L mean (SD) | R1 − R3-L [95% CI] | R3-L − R3 [95% CI] | Demo tokens R1 / R3 / R3-L |\n"
             "|---|---:|---:|---:|---:|---:|---:|---|\n";
        for (const auto& k : kinds) {
            const auto& st = ks[k];
            if (!st.n) continue;
            o << "| " << k << " | " << st.n << " | " << pct(st.r1) << " | " << pct(st.r3.mean) << " | " << pct(st.r3l.mean) << " ("
              << fmt("%.2f", st.r3l.sd * 100) << ") | " << pp(st.r1_r3l.diff) << " " << ci_pp(st.r1_r3l) << " | " << pp(st.r3l_r3.diff)
              << " " << ci_pp(st.r3l_r3) << " | " << fmt("%.0f", mean_of(cost[k].r1_extra)) << " / " << fmt("%.0f", mean_of(cost[k].r3_extra))
              << " / " << fmt("%.0f", mean_of(cost[k].r3l_extra)) << " |\n";
        }
        o << "\nR3-L per seed (all kinds):";
        for (size_t si = 0; si < seeds_c.size(); ++si) o << " " << seeds_c[si] << " = " << pct(ks["all"].r3l_seed[si]) << (si + 1 < seeds_c.size() ? "," : ".");
        const auto& C = cost["all"];
        o << " R3-L − S1 = " << pp(ks["all"].r3l_s1.diff) << " " << ci_pp(ks["all"].r3l_s1) << ". Draws identical to the nearest exemplar: "
          << C.r3l_same_as_nearest << " / " << C.r3l_draws << ". Mean similarity of R3-L draws: "
          << (C.r3l_sim.empty() ? std::string("n/a") : fmt("%.4f", mean_of(C.r3l_sim))) << ".\n\n";
    }

    o << "## Demonstration label effects\n\nShare of queries whose demonstration answer label equals the query's ground-truth label, "
         "and how often the prediction copies the demonstration label. For noul and score the label also fixes the semantic value.\n\n"
         "| Kind | Nearest demo = GT | Random demo = GT | S1 = nearest demo | R1 = nearest demo | S1 = random demo | R3 = random demo |\n"
         "|---|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& k : kinds) {
        const auto& L = lab[k];
        if (!L.n) continue;
        o << "| " << k << " | " << pct(double(L.near_demo_gt) / L.n) << " | " << pct(double(L.rand_demo_gt) / L.rand_n) << " | "
          << pct(double(L.s1_eq_near_demo) / L.n) << " | " << pct(double(L.r1_eq_demo) / L.n) << " | "
          << pct(double(L.s1_eq_rand_demo) / L.rand_n) << " | " << pct(double(L.r3_eq_demo) / L.rand_n) << " |\n";
    }
    o << "\n";

    o << "## Similarity analysis (exploratory)\n\nEstimated cosine cos(π·h/768) from the Hamming distance between the query's "
         "binary embedding and the exemplar's stored vector (same quantisation as R1's KNN). Similarity was computed after "
         "selection and never influenced it.\n\n| Kind | Nearest mean | Nearest median | Random mean | Random median | Random p90 |\n|---|---:|---:|---:|---:|---:|\n";
    for (const auto& k : std::vector<std::string>{"choice", "noul", "score"}) {
        if (near_sim_k[k].empty()) continue;
        o << "| " << k << " | " << fmt("%.4f", mean_of(near_sim_k[k])) << " | " << fmt("%.4f", median_of(near_sim_k[k])) << " | "
          << fmt("%.4f", mean_of(rand_sim_k[k])) << " | " << fmt("%.4f", median_of(rand_sim_k[k])) << " | "
          << fmt("%.4f", r1::percentile(rand_sim_k[k], 90)) << " |\n";
    }
    o << "| all | " << fmt("%.4f", mean_of(near_sim)) << " | " << fmt("%.4f", median_of(near_sim)) << " | "
      << fmt("%.4f", mean_of(rand_sim)) << " | " << fmt("%.4f", median_of(rand_sim)) << " | " << fmt("%.4f", r1::percentile(rand_sim, 90)) << " |\n\n";
    if (rand_sim_missing) o << "Random draws without a similarity value: " << rand_sim_missing << ".\n\n";
    o << "Outcome by similarity bin (bins = sextiles of the pooled nearest + random similarities; random counts are query×seed draws):\n\n"
         "| Bin | Est. cosine | Nearest n | R1 acc | R1 rescue / harm | Random draws | R3 acc | R3 rescue / harm |\n|---|---|---:|---:|---|---:|---:|---|\n";
    for (size_t b = 0; b < sbins.size(); ++b) {
        const auto& x = sbins[b];
        o << "| " << b + 1 << " | " << (b ? fmt("%.3f", edges[b - 1]) : std::string("min")) << " – " << (b < edges.size() ? fmt("%.3f", edges[b]) : std::string("max"))
          << " | " << x.n_near << " | " << (x.n_near ? pct(double(x.near_correct) / x.n_near) : "–") << " | " << x.near_rescue << " / " << x.near_harm
          << " | " << x.n_rand << " | " << (x.n_rand ? pct(double(x.rand_correct) / x.n_rand) : "–") << " | " << x.rand_rescue << " / " << x.rand_harm << " |\n";
    }
    o << "\nAccuracy across bins mixes query difficulty with similarity (harder queries tend to have less similar neighbours). "
         "A within-query check removes that: across the " << S << " random draws of the same query, the correlation between "
         "draw similarity and correctness is **" << fmt("%+.4f", within_corr) << "**.\n\n";

    o << "## Cost analysis\n\n";
    {
        const auto& C = cost["all"];
        o << "| Strategy | Accuracy | Mean latency | Median | P95 | Selection (mean) | Decision (mean) | Mean prompt tokens | Mean demo tokens |\n"
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        o << "| S1 | " << pct(all.s1) << " | " << fmt("%.1f ms", mean_of(C.s1_ms)) << " | " << fmt("%.1f", median_of(C.s1_ms)) << " | "
          << fmt("%.1f", p95_of(C.s1_ms)) << " | – | " << fmt("%.1f", mean_of(C.s1_ms)) << " | " << fmt("%.1f", mean_of(C.s1_tok)) << " | 0 |\n";
        o << "| R1 | " << pct(all.r1) << " | " << fmt("%.1f ms", mean_of(C.r1_ms)) << " | " << fmt("%.1f", median_of(C.r1_ms)) << " | "
          << fmt("%.1f", p95_of(C.r1_ms)) << " | " << fmt("%.1f", mean_of(C.r1_sel_ms)) << " | " << fmt("%.1f", mean_of(C.r1_dec_ms)) << " | "
          << fmt("%.1f", mean_of(C.r1_tok)) << " | " << fmt("%.1f", mean_of(C.r1_extra)) << " |\n";
        o << "| R3 | " << pct(all.r3.mean) << " | " << fmt("%.1f ms", mean_of(C.r3_ms)) << " | " << fmt("%.1f", median_of(C.r3_ms)) << " | "
          << fmt("%.1f", p95_of(C.r3_ms)) << " | " << fmt("%.3f", mean_of(C.r3_sel_ms)) << " | " << fmt("%.1f", mean_of(C.r3_dec_ms)) << " | "
          << fmt("%.1f", mean_of(C.r3_tok)) << " | " << fmt("%.1f", mean_of(C.r3_extra)) << " |\n";
        if (has_ctrl)
            o << "| R3-L (control) | " << pct(all.r3l.mean) << " | " << fmt("%.1f ms", mean_of(C.r3l_ms)) << " | " << fmt("%.1f", median_of(C.r3l_ms))
              << " | " << fmt("%.1f", p95_of(C.r3l_ms)) << " | " << fmt("%.3f", mean_of(C.r3l_sel_ms)) << " | " << fmt("%.1f", mean_of(C.r3l_dec_ms))
              << " | " << fmt("%.1f", mean_of(C.r3l_tok)) << " | " << fmt("%.1f", mean_of(C.r3l_extra)) << " |\n";
        o << "\n";
        o << "All values are **measured**, per query, on the held-out set. S1 and R1 come from the reference R1 run ("
          << "both models loaded once, untimed warm-up, sequential, full prompt from an empty KV cache); R3 comes from this run under "
             "the same conditions. R1 selection = query embedding + vector search + row fetch + LZ4; R3 selection = hash draw + row "
             "fetch + LZ4 (no embedding, no search). Decision = prompt construction + evaluation + scoring. Hardware: the same local "
             "machine and GPU for all runs; the two runs were on different days.\n\n";
        if (sess.n) {
            o << "Same-session check (first " << sess.n << " held-out queries, S1 and nearest R1 re-run inside the R3 process):\n\n"
                 "| Strategy | Selection mean | Decision mean | Total mean |\n|---|---:|---:|---:|\n";
            o << "| S1 | – | " << fmt("%.1f ms", mean_of(sess.s1)) << " | " << fmt("%.1f ms", mean_of(sess.s1)) << " |\n";
            o << "| R1 | " << fmt("%.1f ms", mean_of(sess.r1_sel)) << " | " << fmt("%.1f ms", mean_of(sess.r1_dec)) << " | "
              << fmt("%.1f ms", mean_of(sess.r1_sel) + mean_of(sess.r1_dec)) << " |\n";
            o << "| R3 | " << fmt("%.3f ms", mean_of(sess.r3_sel)) << " | " << fmt("%.1f ms", mean_of(sess.r3_dec)) << " | "
              << fmt("%.1f ms", mean_of(sess.r3_sel) + mean_of(sess.r3_dec)) << " |\n\n";
        }
        o << "Demonstration length (extra prompt tokens over S1) by kind:\n\n| Kind | R1 mean | R1 median | R3 mean | R3 median | R3 / R1 mean |\n|---|---:|---:|---:|---:|---:|\n";
        for (const auto& k : kinds) {
            const auto& K = cost[k];
            if (K.r1_extra.empty()) continue;
            o << "| " << k << " | " << fmt("%.1f", mean_of(K.r1_extra)) << " | " << fmt("%.0f", median_of(K.r1_extra)) << " | "
              << fmt("%.1f", mean_of(K.r3_extra)) << " | " << fmt("%.0f", median_of(K.r3_extra)) << " | "
              << fmt("%.2f", mean_of(K.r1_extra) > 0 ? mean_of(K.r3_extra) / mean_of(K.r1_extra) : 0.0) << " |\n";
        }
        o << "\nPlot: `accuracy_latency.svg` (accuracy vs mean latency; small points are individual R3 seeds).\n\n";
    }

    o << "## Conclusion\n\n**Classification: " << verdict << ".**\n\n";
    {
        const auto& C = cost["all"];
        o << "1. **Does Random 1-shot significantly outperform Zero-shot?** R3 − S1 = " << pp(all.r3_s1.diff) << " " << ci_pp(all.r3_s1)
          << (excludes_zero(all.r3_s1) ? (all.r3_s1.diff > 0 ? " — yes; the interval excludes zero" : " — no; R3 is worse than S1") : " — the interval includes zero")
          << "; every seed: R3 − S1 between " << pp(*std::min_element(all.r3_seed.begin(), all.r3_seed.end()) - all.s1) << " and "
          << pp(*std::max_element(all.r3_seed.begin(), all.r3_seed.end()) - all.s1) << ".\n";
        o << "2. **Does Nearest 1-shot outperform Random 1-shot?** R1 − R3 = " << pp(all.r1_r3.diff) << " " << ci_pp(all.r1_r3)
          << "; R1 is ahead on " << seeds_r1_better << "/" << S << " seeds. ";
        if (verdict == "No clear difference") o << "This is not evidence of equivalence: the interval bounds the possible selection effect. ";
        std::vector<std::string> helps, unclear, hurts;
        for (const auto& k : std::vector<std::string>{"choice", "noul", "score"}) {
            if (!ks[k].n) continue;
            const auto& c = ks[k].r1_r3;
            (excludes_zero(c) ? (c.diff > 0 ? helps : hurts) : unclear).push_back(k);
        }
        auto join = [](const std::vector<std::string>& v) {
            std::string s;
            for (size_t i = 0; i < v.size(); ++i) s += (i ? (i + 1 == v.size() ? " and " : ", ") : "") + v[i];
            return s;
        };
        o << "By kind:";
        {
            bool first = true;
            for (const auto& k : std::vector<std::string>{"choice", "noul", "score"})
                if (ks[k].n) { o << (first ? " " : "; ") << k << " " << pp(ks[k].r1_r3.diff) << " " << ci_pp(ks[k].r1_r3); first = false; }
        }
        o << ".";
        for (const auto& k : helps) {
            const auto& L = lab[k];
            if (L.n && L.rand_n)
                o << " In " << k << " the nearest demonstration carries the query's ground-truth label in "
                  << pct(double(L.near_demo_gt) / L.n) << " of queries vs " << pct(double(L.rand_demo_gt) / L.rand_n)
                  << " for a random one, and the model copies the demonstration label in " << pct(double(L.r1_eq_demo) / L.n)
                  << " (R1) / " << pct(double(L.r3_eq_demo) / L.rand_n) << " (R3) of predictions, so the selection effect there "
                     "coincides largely with label agreement.";
        }
        if (has_ctrl) {
            o << " Against the length-matched control, R1 − R3-L = " << pp(all.r1_r3l.diff) << " " << ci_pp(all.r1_r3l)
              << " (R3-L − R3 = " << pp(all.r3l_r3.diff) << " " << ci_pp(all.r3l_r3) << ")";
            if (excludes_zero(all.r1_r3) && all.r1_r3.diff > 0)
                o << (excludes_zero(all.r1_r3l) && all.r1_r3l.diff > 0
                          ? ": the nearest advantage survives when demonstration length is matched, so it is not a length artefact."
                          : ": once demonstration length is matched the nearest advantage is no longer detectable, so demonstration "
                            "length may account for part of it.");
            else
                o << ".";
        }
        o << "\n";
        o << "3. **How much of the R1 improvement is due to selection?** " << fmt("%.1f%%", all.share * 100) << " of R1 − S1 ("
          << fmt("%.1f", all.share_lo * 100) << "% – " << fmt("%.1f", all.share_hi * 100) << "%); the remaining "
          << fmt("%.1f%%", (1 - all.share) * 100) << " is obtained with an arbitrary eligible demonstration.\n";
        const double extra_ms = mean_of(C.r1_ms) - mean_of(C.r3_ms);
        const double extra_correct_per_1000 = all.r1_r3.diff * 1000;
        o << "4. **Is the nearest-neighbour latency justified?** R1 costs " << fmt("%+.1f ms", extra_ms) << " per query more than R3 ("
          << fmt("%.1f", mean_of(C.r1_sel_ms)) << " ms retrieval vs " << fmt("%.3f", mean_of(C.r3_sel_ms)) << " ms random selection, "
          << "plus decision-time differences from demonstration length) for " << fmt("%+.1f", extra_correct_per_1000)
          << " correct answers per 1,000 queries.\n";
        o << "5. **Which strategy for deployment?** Measured trade-off: " << pct(all.r1) << " at " << fmt("%.0f ms", mean_of(C.r1_ms))
          << " (R1) vs " << pct(all.r3.mean) << " at " << fmt("%.0f ms", mean_of(C.r3_ms)) << " (R3) vs " << pct(all.s1) << " at "
          << fmt("%.0f ms", mean_of(C.s1_ms)) << " (S1). ";
        if (verdict == "Evidence favoring Nearest 1-shot") {
            o << "Prefer **nearest 1-shot (R1)** wherever its retrieval cost fits the latency budget: it is the most accurate strategy and "
                 "its advantage over random selection is statistically supported.";
            if (!helps.empty())
                o << " The selection benefit is concentrated in " << join(helps) << ".";
            if (!unclear.empty() || !hurts.empty())
                o << " For " << join(unclear.empty() ? hurts : unclear) << " nearest is not detectably better than random, so routing those "
                     "kinds to random selection (no embedding, no search) is a candidate cost saving — to be evaluated on the tuning "
                     "partition, not chosen here from held-out results.";
            o << " Where retrieval cannot be afforded, random 1-shot keeps " << fmt("%.0f%%", (1 - all.share) * 100)
              << " of R1's gain over zero-shot at negligible selection cost and is clearly preferable to zero-shot.";
        } else if (verdict == "Evidence favoring Random 1-shot") {
            o << "Prefer **random 1-shot (R3)**: it is more accurate than nearest selection and avoids the embedding model and vector search.";
        } else {
            o << "Prefer **random 1-shot (R3)** on cost grounds: nearest selection is not detectably better here, and random selection "
                 "avoids the embedding model and vector search. The interval above bounds the accuracy that this choice may give up.";
        }
        o << " This report changes no production default.\n\n";
    }

    o << "## Reproduction\n\n```sh\n"
         "cmake --build build --config Release --target pjev-r3 test_r3\n"
         "./tests/bin/test_r3.exe\n"
         "./bin/pjev-r3.exe --mode run --reference " << opt.reference << " --split-assignments " << det.value("split_assignments", "")
      << " --output " << D << "\n"
         "./bin/pjev-r3.exe --mode run --selection length-matched --output results/r3-length-matched-1shot\n"
         "./bin/pjev-r3.exe --mode aggregate --output " << D << (has_ctrl ? " --control " + opt.control : std::string()) << "\n```\n\n"
         "Outputs: `seed-<seed>/predictions.jsonl` (one record per query: selected exemplar, prediction, logits, tokens, latency, status), "
         "`selections.jsonl` (every (query, seed) draw, written before inference), `reference_check.jsonl`, `config.json`, "
         "`per_seed.csv`, `similarity.csv`, `latency.csv`, `aggregate.json`, `accuracy_latency.svg`, this file.\n";
    o.close();

    printf("S1 %.2f%%  R1 %.2f%%  R3 mean %.2f%% (sd %.2f pp, %zu seeds)  R1-R3 %+.2f pp [%+.2f, %+.2f]  -> %s\n",
           all.s1 * 100, all.r1 * 100, all.r3.mean * 100, all.r3.sd * 100, S, all.r1_r3.diff * 100, all.r1_r3.lo * 100,
           all.r1_r3.hi * 100, verdict.c_str());
    printf("Wrote %s/summary.md\n", D.c_str());
    return 0;
}

} // namespace r3
} // namespace pjev
