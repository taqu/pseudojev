// pjev-r4 stage 1 (tune) and stage 2 (evaluate). Both work offline on the measurement runs written by
// `pjev-r4 --mode run`; nothing here touches a model or the database.

#include "r4_eval.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "model/model_identity.h"
#include "r1_metrics.h"
#include "r3_analysis.h"
#include "r4_core.h"

#include "nlohmann/json.hpp"

namespace pjev
{
namespace r4
{

namespace fs = std::filesystem;

std::string fmt(const char* f, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, f);
    vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

static std::string pct(double x) { return fmt("%.2f%%", x * 100.0); }
static std::string pp(double x) { return fmt("%+.2f pp", x * 100.0); }

static std::string now_iso()
{
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

static const std::vector<std::string> kKinds = {"choice", "noul", "score"};

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

struct RunData {
    std::string dir;
    json config;
    std::vector<QueryRow> rows;
    std::vector<json> raw;
    std::string sha256;   // of queries.jsonl
    int parse_errors = 0;
};

static bool load_run(const std::string& dir, RunData& d, std::string& err)
{
    d = RunData{};
    d.dir = dir;
    {
        std::ifstream cf(dir + "/run_config.json");
        if (!cf.is_open()) { err = "missing " + dir + "/run_config.json (run incomplete?)"; return false; }
        d.config = json::parse(cf);
    }
    const std::string path = dir + "/queries.jsonl";
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { err = "cannot open " + path; return false; }
    std::string line;
    std::set<std::string> ids;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        json j;
        QueryRow r;
        std::string e;
        try { j = json::parse(line); } catch (...) { d.parse_errors++; continue; }
        if (!parse_query_row(j, r, e)) { d.parse_errors++; continue; }
        if (!ids.insert(r.id).second) { err = "duplicate query " + r.id; return false; }
        d.rows.push_back(std::move(r));
        d.raw.push_back(std::move(j));
    }
    d.sha256 = sha256_file(path);
    if (d.parse_errors) { err = std::to_string(d.parse_errors) + " unparseable records in " + path; return false; }
    const size_t expect = d.config["detail"].value("evaluated_queries", size_t{0});
    if (d.rows.size() != expect) {
        err = fmt("%zu records but run_config says %zu", d.rows.size(), expect);
        return false;
    }
    return true;
}

static std::vector<const QueryRow*> pointers(const std::vector<QueryRow>& rows, const std::string& kind = "")
{
    std::vector<const QueryRow*> v;
    for (const auto& r : rows) if (kind.empty() || r.kind == kind) v.push_back(&r);
    return v;
}

static std::vector<const QueryRow*> supported_rows(const std::vector<QueryRow>& rows)
{
    std::vector<const QueryRow*> v;
    for (const auto& r : rows) if (knn_supported(r.kind)) v.push_back(&r);
    return v;
}

static std::vector<Outcome> outcomes_of(const std::vector<const QueryRow*>& rows,
                                        const std::function<Outcome(const QueryRow&)>& f)
{
    std::vector<Outcome> o;
    o.reserve(rows.size());
    for (const auto* r : rows) o.push_back(f(*r));
    return o;
}

static int correct_of(const std::vector<Outcome>& o)
{
    int c = 0;
    for (const auto& x : o) c += x.correct ? 1 : 0;
    return c;
}

static double quantile(std::vector<double> v, double p) { return r1::percentile(std::move(v), p); }

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

static const std::vector<double> kTauGrid = {0.002, 0.005, 0.01, 0.02, 0.05, 0.1};

// Rule statistics on the queries of one kind.
struct RuleStat {
    Rule   rule;
    int    n = 0, correct = 0, failed = 0, fallbacks = 0;
    double ms = 0.0, llm = 0.0, emb = 0.0, tokens = 0.0;   // sums
    std::string key;   // stable sort key
};

static RuleStat rule_stat(const Rule& rule, const std::vector<const QueryRow*>& rows, const KnnConfig& knn)
{
    RuleStat s;
    s.rule = rule;
    s.n = static_cast<int>(rows.size());
    for (const auto* r : rows) {
        const Outcome o = run_rule(*r, rule, knn);
        s.correct += o.correct ? 1 : 0;
        s.failed  += o.failed ? 1 : 0;
        s.fallbacks += o.fallback ? 1 : 0;
        s.ms += o.ms; s.llm += o.llm; s.emb += o.emb; s.tokens += o.tokens;
    }
    s.key = rule.to_json().dump();
    return s;
}

// Non-dominated rules of one kind (more correct, less time), by time.
static std::vector<RuleStat> prune(std::vector<RuleStat> v)
{
    std::sort(v.begin(), v.end(), [](const RuleStat& a, const RuleStat& b) {
        if (a.ms != b.ms) return a.ms < b.ms;
        if (a.correct != b.correct) return a.correct > b.correct;
        return a.key < b.key;
    });
    std::vector<RuleStat> out;
    int best = -1;
    for (auto& s : v) if (s.correct > best) { best = s.correct; out.push_back(s); }
    return out;
}

struct Combo {
    std::vector<size_t> pick;  // per present kind, index into that kind's pruned list
    int    correct = 0, n = 0;
    double ms = 0.0;
    std::string key;
};

static std::vector<Rule> candidate_rules(const std::string& kind, const std::vector<const QueryRow*>& rows, Action two,
                                         bool adaptive)
{
    std::vector<Action> actions = {Action::ZeroShot, Action::Random1, Action::Nearest1, two};
    if (knn_supported(kind)) { actions.push_back(Action::Knn1); actions.push_back(Action::Knn3); actions.push_back(Action::Knn5); }
    std::vector<Rule> rules;
    for (Action a : actions) { Rule r; r.type = Rule::Type::Fixed; r.action = a; rules.push_back(r); }
    if (!adaptive) return rules;

    // Threshold grids: quantiles of the tuning distribution of this kind.
    std::vector<double> sims, margins;
    for (const auto* r : rows) {
        if (r->retrieval_ok()) sims.push_back(r->nb[0].similarity);
        if (r->s1.ok()) margins.push_back(r->s1.margin);
    }
    std::set<double> sim_grid, margin_grid;
    for (int p = 0; p <= 95; p += 5) if (!sims.empty()) sim_grid.insert(quantile(sims, p));
    for (int p = 5; p <= 95; p += 5) if (!margins.empty()) margin_grid.insert(quantile(margins, p));

    if (knn_supported(kind)) {
        for (int k : {1, 3, 5})
            for (double t : sim_grid)
                for (Action fb : {Action::ZeroShot, Action::Nearest1, two}) {
                    Rule r; r.type = Rule::Type::KnnGate; r.gate_k = k; r.threshold = t; r.fallback = fb;
                    rules.push_back(r);
                }
    }
    for (double t : sim_grid) {
        if (!sims.empty() && t <= *std::min_element(sims.begin(), sims.end())) continue;  // = fixed nearest_1shot
        for (Action fb : {Action::ZeroShot, Action::Random1}) {
            Rule r; r.type = Rule::Type::SimGate; r.threshold = t; r.fallback = fb;
            rules.push_back(r);
        }
    }
    for (double m : margin_grid) {
        for (Action fb : actions) {
            if (fb == Action::ZeroShot) continue;
            Rule r; r.type = Rule::Type::MarginGate; r.threshold = m; r.fallback = fb;
            rules.push_back(r);
        }
    }
    return rules;
}

struct Family {
    std::string name;                                  // task_only | adaptive
    std::vector<std::string> kinds;                    // present kinds
    std::vector<std::vector<RuleStat>> per_kind;       // pruned
    std::vector<Combo> combos;
    std::vector<size_t> front;                         // indices into combos
    int n_rules_considered = 0;
};

static Family build_family(const std::string& name, const std::map<std::string, std::vector<const QueryRow*>>& by_kind,
                           Action two, const KnnConfig& knn)
{
    Family f;
    f.name = name;
    for (const auto& k : kKinds) {
        auto it = by_kind.find(k);
        if (it == by_kind.end() || it->second.empty()) continue;
        std::vector<RuleStat> stats;
        for (const auto& rule : candidate_rules(k, it->second, two, name == "adaptive"))
            stats.push_back(rule_stat(rule, it->second, knn));
        f.n_rules_considered += static_cast<int>(stats.size());
        f.kinds.push_back(k);
        f.per_kind.push_back(prune(std::move(stats)));
    }
    // All combinations of per-kind non-dominated rules (a dominated per-kind rule can never be optimal).
    std::vector<size_t> idx(f.kinds.size(), 0);
    std::function<void(size_t)> rec = [&](size_t d) {
        if (d == f.kinds.size()) {
            Combo c;
            c.pick = idx;
            for (size_t i = 0; i < idx.size(); ++i) {
                const auto& s = f.per_kind[i][idx[i]];
                c.correct += s.correct;
                c.n += s.n;
                c.ms += s.ms;
                c.key += s.key + "|";
            }
            f.combos.push_back(std::move(c));
            return;
        }
        for (size_t i = 0; i < f.per_kind[d].size(); ++i) { idx[d] = i; rec(d + 1); }
    };
    rec(0);
    std::vector<std::pair<double, double>> pts;
    for (const auto& c : f.combos) pts.push_back({c.n ? double(c.correct) / c.n : 0.0, c.n ? c.ms / c.n : 0.0});
    f.front = pareto_front(pts);
    return f;
}

static Policy combo_policy(const Family& f, const Combo& c, const std::string& name)
{
    Policy p;
    p.name = name;
    p.family = f.name;
    for (size_t i = 0; i < f.kinds.size(); ++i) p.per_kind[f.kinds[i]] = f.per_kind[i][c.pick[i]].rule;
    return p;
}

// Best combo by a deterministic order: more correct, then less time, then key.
static const Combo* best_combo(const Family& f, const std::function<bool(const Combo&)>& feasible,
                               bool min_time_first = false)
{
    const Combo* best = nullptr;
    for (const auto& c : f.combos) {
        if (!feasible(c)) continue;
        if (!best) { best = &c; continue; }
        bool better;
        if (min_time_first)
            better = c.ms < best->ms || (c.ms == best->ms && (c.correct > best->correct || (c.correct == best->correct && c.key < best->key)));
        else
            better = c.correct > best->correct || (c.correct == best->correct && (c.ms < best->ms || (c.ms == best->ms && c.key < best->key)));
        if (better) best = &c;
    }
    return best;
}

static json stats_json(const PolicyStats& s)
{
    json j{{"n", s.n}, {"accuracy", s.accuracy()}, {"correct", s.correct}, {"failed", s.failed},
           {"fallbacks", s.fallbacks}, {"mean_ms", s.mean_ms}, {"p50_ms", s.p50_ms}, {"p95_ms", s.p95_ms},
           {"p99_ms", s.p99_ms}, {"llm_per_query", s.llm}, {"embeddings_per_query", s.emb},
           {"prompt_tokens_per_query", s.tokens}};
    json routed;
    for (const auto& [k, m] : s.routed) routed[k] = m;
    j["routed"] = routed;
    return j;
}

static PolicyStats policy_stats(const Policy& p, const std::vector<const QueryRow*>& rows, const KnnConfig& knn)
{
    return summarize(outcomes_of(rows, [&](const QueryRow& r) { return run_policy(r, p, knn); }), rows);
}

static Policy fixed_policy(const std::string& name, Action a)
{
    Policy p;
    p.name = name;
    p.family = "static";
    for (const auto& k : kKinds) {
        Rule r;
        r.type = Rule::Type::Fixed;
        r.action = action_applicable(a, k) ? a : Action::Nearest1;
        p.per_kind[k] = r;
    }
    return p;
}

int tune_main(const EvalOptions& o)
{
    if (fs::exists(o.dir + "/evaluation/queries.jsonl")) {
        fprintf(stderr, "error: an evaluation run already exists in %s; the frozen configuration may not be re-tuned "
                        "after held-out data was produced\n", o.dir.c_str());
        return 1;
    }
    RunData t;
    std::string err;
    if (!load_run(o.dir + "/tuning", t, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (t.config.value("partition", "") != "tuning") { fprintf(stderr, "error: not a tuning run\n"); return 1; }
    const bool preliminary = t.config.value("preliminary", true);
    const json& checks = t.config["detail"]["checks"];
    printf("Tuning run: %zu queries%s; reference mismatches s1=%d r1=%d nearest=%d\n", t.rows.size(),
           preliminary ? " [preliminary]" : "", checks.value("s1_reference_mismatch", -1),
           checks.value("r1_reference_mismatch", -1), checks.value("nearest_reference_mismatch", -1));
    const auto all = pointers(t.rows);
    const auto sup = supported_rows(t.rows);
    KnnConfig knn0;

    // --- 2-shot order: higher tuning accuracy over all queries; tie -> A ---
    const int ca = correct_of(outcomes_of(all, [&](const QueryRow& r) { return run_action(r, Action::Nearest2A, knn0); }));
    const int cb = correct_of(outcomes_of(all, [&](const QueryRow& r) { return run_action(r, Action::Nearest2B, knn0); }));
    const std::string order = cb > ca ? "B" : "A";
    const Action two = order == "A" ? Action::Nearest2A : Action::Nearest2B;
    printf("2-shot order: A %d/%zu, B %d/%zu -> %s\n", ca, all.size(), cb, all.size(), order.c_str());
    json order_by_kind;
    for (const auto& k : kKinds) {
        const auto rk = pointers(t.rows, k);
        if (rk.empty()) continue;
        order_by_kind[k] = {
            {"n", rk.size()},
            {"A_correct", correct_of(outcomes_of(rk, [&](const QueryRow& r) { return run_action(r, Action::Nearest2A, knn0); }))},
            {"B_correct", correct_of(outcomes_of(rk, [&](const QueryRow& r) { return run_action(r, Action::Nearest2B, knn0); }))},
            {"nearest_1shot_correct", correct_of(outcomes_of(rk, [&](const QueryRow& r) { return run_action(r, Action::Nearest1, knn0); }))}};
    }

    // --- kNN voting: per k, the variant with the highest tuning accuracy on supported kinds ---
    // Ties -> uniform, then the larger tau (smoother weights).
    json knn_table = json::array();
    KnnConfig knn;
    std::map<int, double> best_weighted_tau;
    for (int k : {3, 5}) {
        auto acc_of = [&](const KnnParams& p) {
            int c = 0;
            for (const auto* r : sup) {
                std::vector<VoteInput> v;
                for (const auto& n : r->nb) v.push_back({n.label, n.similarity});
                const auto res = vote(v, p.k, p.voting, p.tau);
                c += res.ok && res.label == r->gt ? 1 : 0;
            }
            return c;
        };
        KnnParams best{k, Voting::Uniform, 0.0};
        int best_c = acc_of(best);
        knn_table.push_back({{"k", k}, {"voting", "uniform"}, {"correct", best_c}, {"n", sup.size()}});
        int best_w = -1;
        for (double tau : kTauGrid) {
            const KnnParams p{k, Voting::Weighted, tau};
            const int c = acc_of(p);
            knn_table.push_back({{"k", k}, {"voting", "weighted"}, {"tau", tau}, {"correct", c}, {"n", sup.size()}});
            if (c >= best_w) { best_w = c; best_weighted_tau[k] = tau; }   // ties -> larger tau
            if (c > best_c) { best_c = c; best = p; }
            else if (c == best_c && best.voting == Voting::Weighted && tau > best.tau) best = p;
        }
        (k == 3 ? knn.k3 : knn.k5) = best;
        printf("kNN-%d: %s (%d/%zu)\n", k, best.describe().c_str(), best_c, sup.size());
    }

    // --- Routing families on the tuning partition ---
    std::map<std::string, std::vector<const QueryRow*>> by_kind;
    for (const auto& k : kKinds) by_kind[k] = pointers(t.rows, k);
    const auto t0 = std::chrono::steady_clock::now();
    Family task_only = build_family("task_only", by_kind, two, knn);
    Family adaptive  = build_family("adaptive", by_kind, two, knn);
    printf("Routing: task-only %d rules -> %zu combos (%zu on frontier); adaptive %d rules -> %zu combos (%zu on frontier) "
           "in %.1f s\n", task_only.n_rules_considered, task_only.combos.size(), task_only.front.size(),
           adaptive.n_rules_considered, adaptive.combos.size(), adaptive.front.size(),
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    // Reference latencies on tuning: static all-X policies.
    const Policy pa = fixed_policy("A_static_nearest_1shot", Action::Nearest1);
    const PolicyStats sa = policy_stats(pa, all, knn);
    const PolicyStats sr = policy_stats(fixed_policy("all_random_1shot", Action::Random1), all, knn);
    const PolicyStats sz = policy_stats(fixed_policy("all_zero_shot", Action::ZeroShot), all, knn);
    const double n = static_cast<double>(all.size());
    struct Budget { std::string tag; double mean_ms; };
    const std::vector<Budget> budgets = {{"A_latency", sa.mean_ms}, {"random_latency", sr.mean_ms}, {"zero_shot_latency", sz.mean_ms}};

    std::vector<Policy> finalists{pa};
    json finalist_doc = json::array();
    finalist_doc.push_back({{"name", pa.name}, {"selection", "static reference: nearest_1shot for every query"},
                            {"tuning", stats_json(sa)}});
    auto add = [&](const Family& f, const Combo* c, const std::string& name, const std::string& rule) {
        if (!c) return;
        Policy p = combo_policy(f, *c, name);
        const PolicyStats st = policy_stats(p, all, knn);
        finalists.push_back(p);
        finalist_doc.push_back({{"name", name}, {"selection", rule}, {"tuning", stats_json(st)}});
    };
    for (const Family* f : {&task_only, &adaptive}) {
        const std::string pre = f->name == "task_only" ? "B_task_only@" : "C_adaptive@";
        for (const auto& b : budgets) {
            const double cap = b.mean_ms * n * (1.0 + 1e-12);
            add(*f, best_combo(*f, [&](const Combo& c) { return c.ms <= cap; }), pre + b.tag,
                fmt("max tuning accuracy with tuning mean latency <= %.1f ms (%s); ties -> lower latency",
                    b.mean_ms, b.tag.c_str()));
        }
        add(*f, best_combo(*f, [&](const Combo& c) { return c.correct >= sa.correct; }, true), pre + "A_accuracy",
            fmt("min tuning mean latency with tuning accuracy >= policy A's (%d/%d); ties -> more correct",
                sa.correct, sa.n));
    }
    const std::vector<std::string> confirmatory = {"A_static_nearest_1shot", "B_task_only@A_latency", "C_adaptive@A_latency"};
    for (size_t i = 0; i < finalists.size(); ++i) {
        finalist_doc[i]["policy"] = finalists[i].to_json();
        printf("  %-30s acc %s  mean %.1f ms\n", finalists[i].name.c_str(),
               pct(finalist_doc[i]["tuning"]["accuracy"].get<double>()).c_str(), finalist_doc[i]["tuning"]["mean_ms"].get<double>());
    }

    // --- Tuning outputs ---
    fs::create_directories(o.dir + "/routing");
    {
        std::ofstream f(o.dir + "/routing/tuning_policies.csv", std::ios::binary);
        f << "family,kind,rule,n,correct,accuracy,mean_ms,llm_per_query,embeddings_per_query,tokens_per_query,failed,fallbacks,kind_frontier\n";
        for (const Family* fam : {&task_only, &adaptive}) {
            for (size_t ki = 0; ki < fam->kinds.size(); ++ki) {
                const auto& kind = fam->kinds[ki];
                std::set<std::string> on_front;
                for (const auto& s : fam->per_kind[ki]) on_front.insert(s.key);
                std::vector<RuleStat> all_rules;
                for (const auto& rule : candidate_rules(kind, by_kind[kind], two, fam->name == "adaptive"))
                    all_rules.push_back(rule_stat(rule, by_kind[kind], knn));
                for (const auto& s : all_rules) {
                    f << fam->name << "," << kind << ",\"" << s.rule.describe() << "\"," << s.n << "," << s.correct << ","
                      << fmt("%.6f", s.n ? double(s.correct) / s.n : 0.0) << "," << fmt("%.3f", s.n ? s.ms / s.n : 0.0) << ","
                      << fmt("%.4f", s.n ? s.llm / s.n : 0.0) << "," << fmt("%.4f", s.n ? s.emb / s.n : 0.0) << ","
                      << fmt("%.1f", s.n ? s.tokens / s.n : 0.0) << "," << s.failed << "," << s.fallbacks << ","
                      << (on_front.count(s.key) ? 1 : 0) << "\n";
                }
            }
        }
    }
    {
        std::ofstream f(o.dir + "/pareto_frontier_tuning.csv", std::ios::binary);
        f << "family,accuracy,mean_ms,p50_ms,p95_ms,llm_per_query,embeddings_per_query,tokens_per_query";
        for (const auto& k : kKinds) f << ",rule_" << k;
        f << "\n";
        for (const Family* fam : {&task_only, &adaptive}) {
            for (size_t ci : fam->front) {
                const Policy p = combo_policy(*fam, fam->combos[ci], "frontier");
                const PolicyStats s = policy_stats(p, all, knn);
                f << fam->name << "," << fmt("%.6f", s.accuracy()) << "," << fmt("%.3f", s.mean_ms) << ","
                  << fmt("%.3f", s.p50_ms) << "," << fmt("%.3f", s.p95_ms) << "," << fmt("%.4f", s.llm) << ","
                  << fmt("%.4f", s.emb) << "," << fmt("%.1f", s.tokens);
                for (const auto& k : kKinds) {
                    auto it = p.per_kind.find(k);
                    f << ",\"" << (it == p.per_kind.end() ? "" : it->second.describe()) << "\"";
                }
                f << "\n";
            }
        }
    }

    // --- Frozen configuration ---
    json fz;
    fz["stage"]   = "frozen";
    fz["created"] = now_iso();
    fz["preliminary"] = preliminary;
    fz["tuning_run"] = {{"queries", o.dir + "/tuning/queries.jsonl"}, {"queries_sha256", t.sha256},
                        {"run_timestamp", t.config.value("run_timestamp", "")}, {"n", t.rows.size()}};
    json ident;
    for (const auto& key : {"input_path", "input_sha256", "db_path", "train_path", "inference_model", "inference_model_identity",
                            "embedding_model", "embedding_fingerprint", "embedding_dim", "db_info", "query_prefix",
                            "prompt_format_version", "prompt_format", "few_shot_prompt_format", "scheme", "layout",
                            "ensemble", "calibration", "n_ctx", "leakage_policy", "max_fetch_k"})
        if (t.config.contains(key)) ident[key] = t.config[key];
    fz["identity"] = ident;
    fz["data"] = {{"reference_predictions_sha256", t.config["detail"].value("reference_predictions_sha256", "")},
                  {"split_assignments_sha256", t.config["detail"].value("split_assignments_sha256", "")},
                  {"partition_rule", "R2 apply_partition: whole scenario families alternate by sorted name, other queries by "
                                     "group_id hash parity; evaluation families fixed"},
                  {"evaluation_partition", "evaluation"}};
    fz["evaluation"] = {{"bootstrap_replicates", o.bootstrap}, {"bootstrap_seed", o.seed}, {"cluster", "group_id"},
                        {"ci", "95% percentile interval of the paired cluster bootstrap"}};
    fz["actions"] = {
        {"zero_shot", "S1 decision"},
        {"random_1shot", "XXH64(\"r3-random-v1|42|<query_id>\") draw from R1's eligible pool (seed 42), row fetch, LZ4, 1-shot decision"},
        {"nearest_1shot", "query embedding, R1 top-1 search, row fetch, LZ4, 1-shot decision (R1)"},
        {"nearest_2shot", "query embedding, top-10 search, neighbour list (rank 1 = R1's exemplar, ranks 2.. in (distance, rowid) "
                          "order without ID/content duplicates), two row fetches, LZ4, 2-shot decision in the selected order"},
        {"knn_k", "query embedding, search (top-1 for k=1, top-10 otherwise), neighbour fetches, label mapping, vote; no LLM"}};
    fz["fallback_rules"] = {
        {"nearest_2shot", "if fewer than two distinct neighbours exist or the 2-shot prompt exceeds n_ctx: the 1-shot decision "
                          "with the nearest exemplar (the refused attempt's measured cost is kept)"},
        {"knn", "neighbours whose answer cannot be mapped abstain; no mapped neighbour -> failure (counted incorrect)"},
        {"other", "any failed decision counts as incorrect and is reported; nothing is excluded"}};
    fz["nearest_2shot"] = {{"order", order}, {"orders", {{"A", "second-nearest, nearest, query"}, {"B", "nearest, second-nearest, query"}}},
                           {"tuning_correct", {{"A", ca}, {"B", cb}}}, {"tuning_n", all.size()}, {"by_kind", order_by_kind},
                           {"selection_rule", "higher tuning accuracy over all kinds; tie -> A"}};
    fz["knn"] = {{"knn_1", knn.k1.to_json()}, {"knn_3", knn.k3.to_json()}, {"knn_5", knn.k5.to_json()},
                 {"weighted_candidates", {{"knn_3", {{"k", 3}, {"voting", "weighted"}, {"tau", best_weighted_tau[3]}}},
                                          {"knn_5", {{"k", 5}, {"voting", "weighted"}, {"tau", best_weighted_tau[5]}}}}},
                 {"supported_kinds", {"noul", "score"}},
                 {"unsupported", {{"choice", "option identifiers are per-task item names; no candidate identity exists "
                                             "independently of the ground truth"}}},
                 {"mapping", {{"noul", "answer text no/yes -> the query option with the same text"},
                              {"score", "answer level index -> the same level index (level counts must match)"}}},
                 {"weighting", "w_i = exp((s_i - s_max) / tau), s = estimated cosine cos(pi*h/768), s_max = max over voters"},
                 {"tie_break", "highest total weight; equal totals (1e-12 relative) -> label of the best-ranked neighbour"},
                 {"tau_grid", kTauGrid}, {"tuning_table", knn_table},
                 {"selection_rule", "per k, max tuning accuracy on noul+score; ties -> uniform, then larger tau"}};
    fz["routing"] = {
        {"families", {{"task_only", "one fixed action per kind"},
                      {"adaptive", "per kind: a fixed action, a kNN gate (retrieve; unanimous top-k labels and nearest "
                                   "similarity >= t -> kNN, else fallback), a similarity gate (retrieve; similarity >= t -> "
                                   "nearest_1shot, else zero_shot/random_1shot) or an S1-margin gate (S1; margin >= m -> keep, "
                                   "else run the fallback action standalone)"}}},
        {"threshold_grids", "similarity: tuning quantiles 0,5,...,95% of the nearest similarity per kind; margin: tuning "
                            "quantiles 5,...,95% of the S1 winner margin per kind"},
        {"cost_model", "per-query measured stage times of the tuning run composed along each rule's execution plan; "
                       "work needed to decide (retrieval, S1) is always paid"},
        {"budgets_ms", {{"A_latency", sa.mean_ms}, {"random_latency", sr.mean_ms}, {"zero_shot_latency", sz.mean_ms}}},
        {"rules_considered", {{"task_only", task_only.n_rules_considered}, {"adaptive", adaptive.n_rules_considered}}}};
    json fin = json::array();
    for (const auto& p : finalists) fin.push_back(p.to_json());
    fz["finalist_policies"] = fin;
    fz["finalist_selection"] = finalist_doc;
    fz["confirmatory_policies"] = confirmatory;
    fz["confirmatory_comparisons"] = json::array({
        {{"id", 1}, {"a", "nearest_2shot"}, {"b", "nearest_1shot"}, {"scope", "all"}},
        {{"id", 2}, {"a", "nearest_2shot"}, {"b", "zero_shot"}, {"scope", "all"}},
        {{"id", 3}, {"a", "knn_1"}, {"b", "nearest_1shot"}, {"scope", "noul"}},
        {{"id", 4}, {"a", "knn_3"}, {"b", "knn_1"}, {"scope", "noul"}},
        {{"id", 5}, {"a", "knn_5"}, {"b", "knn_1"}, {"scope", "noul"}},
        {{"id", 6}, {"a", "B_task_only@A_latency"}, {"b", "A_static_nearest_1shot"}, {"scope", "all"}},
        {{"id", 7}, {"a", "C_adaptive@A_latency"}, {"b", "A_static_nearest_1shot"}, {"scope", "all"}},
        {{"id", 8}, {"a", "C_adaptive@A_latency"}, {"b", "B_task_only@A_latency"}, {"scope", "all"}}});
    fz["decision_rules"] = {
        {"non_inferiority_margin_pp", 1.0},
        {"min_latency_reduction", 0.10},
        {"rules", json::array({
            "A policy 'qualifies' when either (i) its held-out accuracy difference to A has a 95% CI lower bound >= -1.0 pp and "
            "its held-out mean latency is at least 10% below A's, or (ii) the CI lower bound is > 0 and its mean latency is not "
            "above A's.",
            "Adopt adaptive routing (C) if C qualifies and either C - B has a CI lower bound > 0 or B does not qualify.",
            "Otherwise adopt task-only routing (B) if B qualifies.",
            "Otherwise use direct kNN for noul if kNN-1 - R1 on noul has a CI lower bound >= -1.0 pp.",
            "Otherwise keep nearest 1-shot as the default.",
            "Defer deployment instead if the evaluation run fails its integrity checks (reference mismatches, unexplained failures)."})}};
    {
        std::ofstream f(o.dir + "/frozen_policy.json", std::ios::binary);
        f << fz.dump(2) << "\n";
    }
    const std::string sha = sha256_file(o.dir + "/frozen_policy.json");
    {
        std::ofstream f(o.dir + "/frozen_policy.sha256", std::ios::binary);
        f << sha << "  frozen_policy.json\n";
    }
    {
        json sel{{"frozen_policy_sha256", sha}, {"finalists", finalist_doc}, {"confirmatory_policies", confirmatory}};
        std::ofstream f(o.dir + "/routing/selected_policies.json", std::ios::binary);
        f << sel.dump(2) << "\n";
    }
    FrozenPolicy check;
    if (!load_frozen(o.dir + "/frozen_policy.json", o.dir + "/frozen_policy.sha256", check, err)) {
        fprintf(stderr, "error: the written frozen configuration does not load: %s\n", err.c_str());
        return 1;
    }
    printf("Frozen configuration written: %s/frozen_policy.json (sha256 %s)\n", o.dir.c_str(), sha.c_str());
    return 0;
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

// Measured stage times of one standalone action on one query (means are reported per action).
struct Stages {
    double embed = 0, search = 0, fetch = 0, lz4 = 0, select = 0, prompt = 0, eval = 0, scoring = 0, vote = 0, decision = 0;
    Stages& operator+=(const Stages& o)
    {
        embed += o.embed; search += o.search; fetch += o.fetch; lz4 += o.lz4; select += o.select;
        prompt += o.prompt; eval += o.eval; scoring += o.scoring; vote += o.vote; decision += o.decision;
        return *this;
    }
};

static Stages decision_stages(const json& c)
{
    Stages s;
    s.prompt   = c.value("prompt_ms", 0.0);
    s.eval     = c.value("eval_ms", 0.0);
    s.scoring  = c.value("scoring_ms", 0.0);
    s.decision = c.value("latency_ms", 0.0);
    return s;
}

static Stages stages_of(const json& j, Action a)
{
    Stages s;
    const json& rt = j["retrieval"];
    auto loads = [&](size_t n) {
        const auto& nb = rt["neighbours"];
        if (nb.empty() || n == 0) return;
        const auto& x = nb[std::min(n, nb.size()) - 1];
        s.lz4   = x.value("cum_decompress_ms", 0.0);
        s.fetch = x.value("cum_load_ms", 0.0) - s.lz4;
    };
    switch (a) {
    case Action::ZeroShot: s = decision_stages(j["s1"]); break;
    case Action::Random1: {
        s = decision_stages(j["random_1shot"]);
        s.select = j["random_1shot"].value("select_ms", 0.0);
        s.lz4    = j["random_1shot"].value("decompress_ms", 0.0);
        s.fetch  = j["random_1shot"].value("load_ms", 0.0) - s.lz4;
        break;
    }
    case Action::Nearest1:
        s = decision_stages(j["r1"]);
        s.embed = rt.value("embed_ms", 0.0); s.search = rt.value("search1_ms", 0.0); loads(1);
        break;
    case Action::Nearest2A:
    case Action::Nearest2B:
        s = decision_stages(j[a == Action::Nearest2A ? "two_shot_a" : "two_shot_b"]);
        s.embed = rt.value("embed_ms", 0.0); s.search = rt.value("searchk_ms", 0.0); loads(2);
        break;
    default: {
        const int k = knn_k(a);
        s.embed = rt.value("embed_ms", 0.0);
        s.search = rt.value(k > 1 ? "searchk_ms" : "search1_ms", 0.0);
        loads(static_cast<size_t>(k));
        s.vote = rt.value("vote_ms", 0.0);
    }
    }
    return s;
}

struct Strat {
    std::string name;    // action or policy name
    std::string label;   // for tables
    bool        knn = false;
    bool        policy = false;
    Action      action = Action::ZeroShot;
    std::vector<Outcome> out;   // aligned with rows
};

struct Scope {
    std::string name;
    std::vector<size_t> idx;    // row indices
};

// Paired comparison of two strategies on a scope, with its bootstrap replicates shared per scope.
struct Comparison {
    int id = 0;   // 0 = exploratory
    std::string a, b, scope;
    int n = 0;
    double acc_a = 0, acc_b = 0;
    r3::ClusterCI ci;
    int a_only = 0, b_only = 0;
};

int evaluate_main(const EvalOptions& opt)
{
    FrozenPolicy fz;
    std::string err;
    if (!load_frozen(opt.dir + "/frozen_policy.json", opt.dir + "/frozen_policy.sha256", fz, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    RunData ev, tu;
    if (!load_run(opt.dir + "/evaluation", ev, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!load_run(opt.dir + "/tuning", tu, err)) { fprintf(stderr, "error: tuning run: %s\n", err.c_str()); return 1; }
    if (ev.config.value("partition", "") != "evaluation") { fprintf(stderr, "error: not an evaluation run\n"); return 1; }
    if (ev.config.value("frozen_policy_sha256", "") != fz.sha256) {
        fprintf(stderr, "error: the evaluation run was produced under a different frozen configuration\n");
        return 1;
    }
    if (tu.sha256 != fz.raw["tuning_run"].value("queries_sha256", "")) {
        fprintf(stderr, "error: the tuning run changed after freezing\n");
        return 1;
    }
    const int reps = fz.raw["evaluation"].value("bootstrap_replicates", opt.bootstrap);
    const uint64_t bseed = fz.raw["evaluation"].value("bootstrap_seed", opt.seed);
    const bool preliminary = ev.config.value("preliminary", true) || fz.raw.value("preliminary", true);
    const json& evd = ev.config["detail"];
    const json& checks = evd["checks"];
    const size_t N = ev.rows.size();
    const auto& rows = ev.rows;
    const Action two = fz.two_shot_order == "A" ? Action::Nearest2A : Action::Nearest2B;
    const Action two_other = two == Action::Nearest2A ? Action::Nearest2B : Action::Nearest2A;
    const KnnConfig& knn = fz.knn;
    printf("Evaluation run: %zu queries%s; frozen sha256 %s\n", N, preliminary ? " [preliminary]" : "", fz.sha256.substr(0, 16).c_str());

    // --- Strategies on every held-out query ---
    std::vector<Strat> strats;
    auto add_action = [&](const std::string& name, const std::string& label, Action a, KnnConfig kc) {
        Strat s;
        s.name = name; s.label = label; s.action = a; s.knn = is_knn(a);
        for (const auto& r : rows) s.out.push_back(run_action(r, a, kc));
        strats.push_back(std::move(s));
    };
    add_action("zero_shot", "S1 zero-shot", Action::ZeroShot, knn);
    add_action("random_1shot", "Random 1-shot (seed 42)", Action::Random1, knn);
    add_action("nearest_1shot", "R1 nearest 1-shot", Action::Nearest1, knn);
    add_action("nearest_2shot", "Nearest 2-shot (order " + fz.two_shot_order + ", frozen)", two, knn);
    add_action("nearest_2shot_other", std::string("Nearest 2-shot (order ") + (fz.two_shot_order == "A" ? "B" : "A") + ", not selected)", two_other, knn);
    add_action("knn_1", knn.k1.describe(), Action::Knn1, knn);
    add_action("knn_3", knn.k3.describe() + " (frozen)", Action::Knn3, knn);
    add_action("knn_5", knn.k5.describe() + " (frozen)", Action::Knn5, knn);
    // Voting diagnostics: uniform and the best weighted variant of each k.
    for (int k : {3, 5}) {
        const json& wc = fz.raw["knn"]["weighted_candidates"]["knn_" + std::to_string(k)];
        KnnConfig ku = knn, kw = knn;
        (k == 3 ? ku.k3 : ku.k5) = KnnParams{k, Voting::Uniform, 0.0};
        (k == 3 ? kw.k3 : kw.k5) = KnnParams::from_json(wc);
        const Action a = k == 3 ? Action::Knn3 : Action::Knn5;
        add_action(fmt("knn_%d_uniform", k), fmt("kNN-%d uniform", k), a, ku);
        add_action(fmt("knn_%d_weighted", k), (k == 3 ? kw.k3 : kw.k5).describe(), a, kw);
    }
    for (const auto& p : fz.finalists) {
        Strat s;
        s.name = p.name; s.label = p.name; s.policy = true;
        for (const auto& r : rows) s.out.push_back(run_policy(r, p, knn));
        strats.push_back(std::move(s));
    }
    auto S = [&](const std::string& name) -> const Strat& {
        for (const auto& s : strats) if (s.name == name) return s;
        throw std::runtime_error("unknown strategy " + name);
    };

    // --- Scopes and bootstrap ---
    std::map<std::string, Scope> scopes;
    auto scope_of = [&](const std::string& name) -> const Scope& {
        auto it = scopes.find(name);
        if (it != scopes.end()) return it->second;
        Scope sc;
        sc.name = name;
        for (size_t i = 0; i < N; ++i) {
            const auto& k = rows[i].kind;
            if (name == "all" || name == k || (name == "noul+score" && knn_supported(k))) sc.idx.push_back(i);
        }
        return scopes[name] = sc;
    };
    std::map<std::string, std::unique_ptr<r3::Resampler>> samplers;
    auto sampler = [&](const std::string& scope) -> const r3::Resampler& {
        auto it = samplers.find(scope);
        if (it != samplers.end()) return *it->second;
        std::vector<std::string> cl;
        for (size_t i : scope_of(scope).idx) cl.push_back(rows[i].group_id);
        return *(samplers[scope] = std::make_unique<r3::Resampler>(cl, reps, bseed));
    };
    auto correct_vec = [&](const Strat& s, const Scope& sc) {
        std::vector<double> v;
        for (size_t i : sc.idx) v.push_back(s.out[i].correct ? 1.0 : 0.0);
        return v;
    };
    auto acc = [&](const Strat& s, const std::string& scope) {
        const Scope& sc = scope_of(scope);
        if (sc.idx.empty()) return 0.0;
        int c = 0;
        for (size_t i : sc.idx) c += s.out[i].correct ? 1 : 0;
        return double(c) / sc.idx.size();
    };
    auto compare = [&](const std::string& a, const std::string& b, const std::string& scope, int id) {
        Comparison c;
        c.id = id; c.a = a; c.b = b; c.scope = scope;
        const Scope& sc = scope_of(scope);
        c.n = static_cast<int>(sc.idx.size());
        const auto va = correct_vec(S(a), sc), vb = correct_vec(S(b), sc);
        c.acc_a = acc(S(a), scope);
        c.acc_b = acc(S(b), scope);
        if (c.n) c.ci = r3::paired_ci(sampler(scope), va, vb);
        for (size_t i = 0; i < va.size(); ++i) { if (va[i] > vb[i]) c.a_only++; if (vb[i] > va[i]) c.b_only++; }
        return c;
    };

    std::vector<Comparison> confirmatory, exploratory;
    for (const auto& cj : fz.raw["confirmatory_comparisons"])
        confirmatory.push_back(compare(cj["a"].get<std::string>(), cj["b"].get<std::string>(), cj["scope"].get<std::string>(),
                                       cj["id"].get<int>()));
    // Exploratory: per-kind breakdowns, the unselected order, other scopes, other finalists.
    for (const auto& k : kKinds) {
        if (scope_of(k).idx.empty()) continue;
        exploratory.push_back(compare("nearest_2shot", "nearest_1shot", k, 0));
        exploratory.push_back(compare("random_1shot", "zero_shot", k, 0));
        exploratory.push_back(compare("nearest_1shot", "zero_shot", k, 0));
    }
    exploratory.push_back(compare("nearest_2shot_other", "nearest_1shot", "all", 0));
    exploratory.push_back(compare("nearest_2shot", "nearest_2shot_other", "all", 0));
    exploratory.push_back(compare("nearest_1shot", "zero_shot", "all", 0));
    exploratory.push_back(compare("nearest_1shot", "random_1shot", "all", 0));
    for (const std::string sc : {"score", "noul+score"}) {
        if (scope_of(sc).idx.empty()) continue;
        exploratory.push_back(compare("knn_1", "nearest_1shot", sc, 0));
        exploratory.push_back(compare("knn_3", "knn_1", sc, 0));
        exploratory.push_back(compare("knn_5", "knn_1", sc, 0));
    }
    for (const std::string sc : {"noul", "score", "noul+score"}) {
        if (scope_of(sc).idx.empty()) continue;
        exploratory.push_back(compare("knn_1", "zero_shot", sc, 0));
        exploratory.push_back(compare("knn_3_weighted", "knn_3_uniform", sc, 0));
        exploratory.push_back(compare("knn_5_weighted", "knn_5_uniform", sc, 0));
    }
    for (const auto& p : fz.finalists) {
        if (p.name == "A_static_nearest_1shot") continue;
        exploratory.push_back(compare(p.name, "A_static_nearest_1shot", "all", 0));
        for (const auto& k : kKinds) if (!scope_of(k).idx.empty()) exploratory.push_back(compare(p.name, "A_static_nearest_1shot", k, 0));
    }

    // --- Per-strategy summaries ---
    std::vector<const QueryRow*> all_ptr = pointers(rows);
    std::map<std::string, PolicyStats> stat;     // all queries (kNN: noul+score only)
    for (const auto& s : strats) {
        std::vector<Outcome> o;
        std::vector<const QueryRow*> rp;
        for (size_t i = 0; i < N; ++i) {
            if (s.knn && !knn_supported(rows[i].kind)) continue;
            o.push_back(s.out[i]);
            rp.push_back(&rows[i]);
        }
        stat[s.name] = summarize(o, rp);
    }
    auto mean_stages = [&](Action a, const std::string& kind_scope) {
        Stages t;
        int n = 0;
        for (size_t i = 0; i < N; ++i) {
            if (is_knn(a) && !knn_supported(rows[i].kind)) continue;
            if (kind_scope != "all" && rows[i].kind != kind_scope) continue;
            t += stages_of(ev.raw[i], a);
            n++;
        }
        if (n) {
            const double d = n;
            t.embed /= d; t.search /= d; t.fetch /= d; t.lz4 /= d; t.select /= d; t.prompt /= d; t.eval /= d;
            t.scoring /= d; t.vote /= d; t.decision /= d;
        }
        return t;
    };
    auto rescue_harm = [&](const Strat& s, const std::string& scope) {
        int rescue = 0, harm = 0;
        const Strat& s1 = S("zero_shot");
        for (size_t i : scope_of(scope).idx) {
            if (!s1.out[i].correct && s.out[i].correct) rescue++;
            if (s1.out[i].correct && !s.out[i].correct) harm++;
        }
        return std::pair<int, int>{rescue, harm};
    };

    // --- Policy timing (live end-to-end execution) ---
    struct LiveCheck { int n = 0, agree = 0; double live = 0, composed = 0; std::vector<double> live_v, comp_v; };
    std::map<std::string, LiveCheck> live;
    json live_cfg;
    bool live_ok = false;
    {
        std::ifstream mf(opt.dir + "/policy_timing_config.json");
        if (mf.is_open()) {
            live_cfg = json::parse(mf);
            live_ok = live_cfg.value("frozen_policy_sha256", "") == fz.sha256;
        }
    }
    if (live_ok) {
        std::map<std::string, size_t> row_of;
        for (size_t i = 0; i < N; ++i) row_of[rows[i].id] = i;
        std::ifstream lf(opt.dir + "/policy_timing.jsonl", std::ios::binary);
        std::string line;
        while (std::getline(lf, line)) {
            if (line.empty()) continue;
            const json j = json::parse(line);
            auto it = row_of.find(j["query_id"].get<std::string>());
            if (it == row_of.end()) continue;
            const Strat& s = S(j["policy"].get<std::string>());
            const Outcome& oc = s.out[it->second];
            LiveCheck& c = live[s.name];
            c.n++;
            c.agree += j["prediction_index"].get<int>() == oc.pred ? 1 : 0;
            c.live_v.push_back(j["wall_ms"].get<double>());
            c.comp_v.push_back(oc.ms);
        }
        for (auto& [n, c] : live) { c.live = r1::mean(c.live_v); c.composed = r1::mean(c.comp_v); }
    }

    // --- Decision (pre-registered rules in the frozen configuration) ---
    const double margin = fz.raw["decision_rules"].value("non_inferiority_margin_pp", 1.0) / 100.0;
    const double min_red = fz.raw["decision_rules"].value("min_latency_reduction", 0.10);
    auto find_cmp = [&](int id) -> const Comparison& {
        for (const auto& c : confirmatory) if (c.id == id) return c;
        throw std::runtime_error("missing confirmatory comparison");
    };
    const double lat_a = stat["A_static_nearest_1shot"].mean_ms;
    auto qualifies = [&](const std::string& name, const Comparison& c) {
        const double ms = stat[name].mean_ms;
        return (c.ci.lo >= -margin && ms <= (1.0 - min_red) * lat_a) || (c.ci.lo > 0 && ms <= lat_a);
    };
    const Comparison& c6 = find_cmp(6);
    const Comparison& c7 = find_cmp(7);
    const Comparison& c8 = find_cmp(8);
    const Comparison& c3 = find_cmp(3);
    const bool b_q = qualifies(c6.a, c6), c_q = qualifies(c7.a, c7);
    const bool integrity_ok = checks.value("s1_reference_mismatch", 1) == 0 && checks.value("r1_reference_mismatch", 1) == 0 &&
                              checks.value("nearest_reference_mismatch", 1) == 0 && checks.value("random_r3_mismatch", 1) == 0 &&
                              checks.value("retrieval_failures", 1) == 0;
    std::string recommendation, rec_reason;
    if (!integrity_ok) {
        recommendation = "Defer deployment because improvements are insufficiently supported";
        rec_reason = "the evaluation run failed an integrity check";
    } else if (c_q && (c8.ci.lo > 0 || !b_q)) {
        recommendation = "Adopt an adaptive routing policy";
        rec_reason = c8.ci.lo > 0 ? "C qualifies and is better than B (CI above zero)" : "C qualifies and B does not";
    } else if (b_q) {
        recommendation = "Adopt a simple task-only routing policy";
        rec_reason = c_q ? "B and C both qualify; C is not supported as better than B, so the simpler policy is preferred"
                         : "B qualifies; C does not";
    } else if (c3.ci.lo >= -margin) {
        recommendation = "Use direct kNN classification for selected tasks";
        rec_reason = "kNN-1 is non-inferior to R1 on noul within the margin; neither routing finalist qualifies";
    } else {
        recommendation = "Keep Nearest 1-shot as the default";
        rec_reason = "no routing finalist qualifies and kNN-1 is not non-inferior to R1 on noul";
    }

    // --- Output files ---
    fs::create_directories(opt.dir + "/nearest-2shot");
    fs::create_directories(opt.dir + "/knn");
    fs::create_directories(opt.dir + "/routing");
    auto stage_json = [&](const Stages& st) {
        return json{{"embedding_ms", st.embed}, {"search_ms", st.search}, {"row_fetch_ms", st.fetch}, {"lz4_ms", st.lz4},
                    {"random_select_ms", st.select}, {"prompt_construction_ms", st.prompt}, {"llm_inference_ms", st.eval},
                    {"candidate_scoring_ms", st.scoring}, {"vote_ms", st.vote}, {"decision_wall_ms", st.decision}};
    };
    auto nb_json = [&](const json& raw, size_t n) {
        json a = json::array();
        const auto& nb = raw["retrieval"]["neighbours"];
        for (size_t i = 0; i < std::min(n, nb.size()); ++i)
            a.push_back({{"rank", i + 1}, {"example_id", nb[i]["example_id"]}, {"estimated_cosine", nb[i]["estimated_cosine"]},
                         {"hamming_distance", nb[i]["hamming_distance"]}, {"label", nb[i]["label"]},
                         {"answer_text", nb[i]["answer_text"]}, {"map_status", nb[i]["map_status"]}});
        return a;
    };
    auto pred_record = [&](size_t i, const Strat& s, const json& extra) {
        const auto& r = rows[i];
        const Outcome& o = s.out[i];
        json j{{"query_id", r.id}, {"group_id", r.group_id}, {"kind", r.kind}, {"ground_truth_index", r.gt},
               {"strategy", s.name}, {"prediction_index", o.pred}, {"correct", o.correct},
               {"status", o.failed ? (o.terminal == "unsupported" ? "unsupported" : "failed") : "ok"},
               {"fallback", o.fallback}, {"terminal_action", o.terminal},
               {"llm_evaluations", o.llm}, {"embedding_evaluations", o.emb}, {"prompt_tokens", o.tokens},
               {"total_latency_ms", o.ms}};
        for (auto& [k, v] : extra.items()) j[k] = v;
        return j;
    };
    {
        std::ofstream f(opt.dir + "/nearest-2shot/predictions.jsonl", std::ios::binary);
        for (size_t i = 0; i < N; ++i) {
            for (const char* nm : {"nearest_1shot", "nearest_2shot", "nearest_2shot_other"}) {
                const Strat& s = S(nm);
                const char* cond = s.action == Action::Nearest1 ? "r1" : s.action == Action::Nearest2A ? "two_shot_a" : "two_shot_b";
                json extra{{"order", s.action == Action::Nearest1 ? "-" : s.action == Action::Nearest2A ? "A" : "B"},
                           {"example_ids", ev.raw[i][cond].value("example_ids", json::array())},
                           {"neighbours", nb_json(ev.raw[i], s.action == Action::Nearest1 ? 1 : 2)},
                           {"condition_status", ev.raw[i][cond].value("status", "")},
                           {"stages", stage_json(stages_of(ev.raw[i], s.action))}};
                f << pred_record(i, s, extra).dump() << "\n";
            }
        }
    }
    {
        std::ofstream f(opt.dir + "/knn/predictions.jsonl", std::ios::binary);
        for (size_t i = 0; i < N; ++i) {
            for (const char* nm : {"knn_1", "knn_3", "knn_5"}) {
                const Strat& s = S(nm);
                const int k = knn_k(s.action);
                json extra{{"knn", knn.of(s.action).to_json()}, {"neighbours", nb_json(ev.raw[i], static_cast<size_t>(k))},
                           {"stages", stage_json(stages_of(ev.raw[i], s.action))}};
                f << pred_record(i, s, extra).dump() << "\n";
            }
        }
    }
    {
        std::ofstream f(opt.dir + "/routing/predictions.jsonl", std::ios::binary);
        for (size_t i = 0; i < N; ++i)
            for (const auto& p : fz.finalists) {
                const Rule& rule = p.per_kind.at(rows[i].kind);
                f << pred_record(i, S(p.name), json{{"rule", rule.describe()},
                                                     {"neighbours", nb_json(ev.raw[i], 1)}}).dump() << "\n";
            }
    }
    auto metrics_json = [&](const std::vector<std::string>& names) {
        json m = json::object();
        for (const auto& nm : names) {
            const Strat& s = S(nm);
            json j = stats_json(stat[nm]);
            j["label"] = s.label;
            for (const auto& k : kKinds) {
                if (scope_of(k).idx.empty() || (s.knn && !knn_supported(k))) continue;
                const auto rh = rescue_harm(s, k);
                j["by_kind"][k] = {{"n", scope_of(k).idx.size()}, {"accuracy", acc(s, k)}, {"rescue_vs_s1", rh.first},
                                   {"harm_vs_s1", rh.second}};
            }
            if (!s.policy) j["stages_mean"] = stage_json(mean_stages(s.action, "all"));
            m[nm] = j;
        }
        return m;
    };
    auto cmp_json = [&](const Comparison& c) {
        return json{{"id", c.id}, {"a", c.a}, {"b", c.b}, {"scope", c.scope}, {"n", c.n}, {"accuracy_a", c.acc_a},
                    {"accuracy_b", c.acc_b}, {"diff", c.ci.diff}, {"ci_low", c.ci.lo}, {"ci_high", c.ci.hi},
                    {"a_only_correct", c.a_only}, {"b_only_correct", c.b_only}, {"clusters", c.ci.clusters}, {"replicates", c.ci.reps}};
    };
    {
        json m{{"strategies", metrics_json({"zero_shot", "nearest_1shot", "nearest_2shot", "nearest_2shot_other"})},
               {"order", fz.two_shot_order}, {"context_limit_by_kind", evd.value("context_limit_by_kind", json::object())},
               {"comparisons", json::array({cmp_json(find_cmp(1)), cmp_json(find_cmp(2))})}};
        std::ofstream(opt.dir + "/nearest-2shot/metrics.json", std::ios::binary) << m.dump(2) << "\n";
    }
    {
        json m{{"strategies", metrics_json({"knn_1", "knn_3", "knn_5", "knn_3_uniform", "knn_3_weighted", "knn_5_uniform", "knn_5_weighted"})},
               {"comparisons", json::array({cmp_json(find_cmp(3)), cmp_json(find_cmp(4)), cmp_json(find_cmp(5))})}};
        std::ofstream(opt.dir + "/knn/metrics.json", std::ios::binary) << m.dump(2) << "\n";
    }
    {
        std::vector<std::string> names;
        for (const auto& p : fz.finalists) names.push_back(p.name);
        json m{{"policies", metrics_json(names)},
               {"comparisons", json::array({cmp_json(c6), cmp_json(c7), cmp_json(c8)})},
               {"recommendation", recommendation}, {"reason", rec_reason}};
        std::ofstream(opt.dir + "/routing/metrics.json", std::ios::binary) << m.dump(2) << "\n";
    }
    {
        std::ofstream f(opt.dir + "/pairwise_comparisons.csv", std::ios::binary);
        f << "type,id,a,b,scope,n,accuracy_a,accuracy_b,diff,ci_low,ci_high,a_only_correct,b_only_correct\n";
        for (const auto* v : {&confirmatory, &exploratory})
            for (const auto& c : *v)
                f << (c.id ? "confirmatory" : "exploratory") << "," << c.id << "," << c.a << "," << c.b << "," << c.scope << ","
                  << c.n << "," << fmt("%.6f", c.acc_a) << "," << fmt("%.6f", c.acc_b) << "," << fmt("%.6f", c.ci.diff) << ","
                  << fmt("%.6f", c.ci.lo) << "," << fmt("%.6f", c.ci.hi) << "," << c.a_only << "," << c.b_only << "\n";
    }
    {
        std::ofstream f(opt.dir + "/task_metrics.csv", std::ios::binary);
        f << "strategy,kind,n,correct,accuracy,failed\n";
        for (const auto& s : strats)
            for (const std::string k : {"all", "choice", "noul", "score"}) {
                if (scope_of(k).idx.empty() || (s.knn && (k == "all" || !knn_supported(k)))) continue;
                int c = 0, fl = 0;
                for (size_t i : scope_of(k).idx) { c += s.out[i].correct; fl += s.out[i].failed; }
                f << s.name << "," << k << "," << scope_of(k).idx.size() << "," << c << ","
                  << fmt("%.6f", double(c) / scope_of(k).idx.size()) << "," << fl << "\n";
            }
    }
    {
        std::ofstream f(opt.dir + "/rescue_harm.csv", std::ios::binary);
        f << "strategy,kind,n,rescue_vs_s1,harm_vs_s1,net\n";
        for (const auto& s : strats)
            for (const std::string k : {"all", "choice", "noul", "score"}) {
                if (s.name == "zero_shot" || scope_of(k).idx.empty() || (s.knn && (k == "all" || !knn_supported(k)))) continue;
                const auto rh = rescue_harm(s, k);
                f << s.name << "," << k << "," << scope_of(k).idx.size() << "," << rh.first << "," << rh.second << ","
                  << rh.first - rh.second << "\n";
            }
    }
    {
        std::ofstream f(opt.dir + "/latency.csv", std::ios::binary);
        f << "strategy,n,mean_ms,p50_ms,p95_ms,p99_ms,llm_per_query,embeddings_per_query,prompt_tokens_per_query,"
             "embedding_ms,search_ms,row_fetch_ms,lz4_ms,random_select_ms,prompt_construction_ms,llm_inference_ms,"
             "candidate_scoring_ms,vote_ms,decision_wall_ms\n";
        for (const auto& s : strats) {
            const auto& st = stat[s.name];
            f << s.name << "," << st.n << "," << fmt("%.3f", st.mean_ms) << "," << fmt("%.3f", st.p50_ms) << ","
              << fmt("%.3f", st.p95_ms) << "," << fmt("%.3f", st.p99_ms) << "," << fmt("%.4f", st.llm) << ","
              << fmt("%.4f", st.emb) << "," << fmt("%.1f", st.tokens);
            if (s.policy) f << ",,,,,,,,,,";
            else {
                const Stages m = mean_stages(s.action, "all");
                for (double v : {m.embed, m.search, m.fetch, m.lz4, m.select, m.prompt, m.eval, m.scoring, m.vote, m.decision})
                    f << "," << fmt("%.4f", v);
            }
            f << "\n";
        }
    }
    // Held-out accuracy/latency points (all queries; kNN strategies are not complete policies and are excluded).
    struct Point { std::string name; double acc, ms; bool policy; bool front = false; };
    std::vector<Point> pts;
    for (const auto& s : strats) {
        if (s.knn || s.name.find("knn_") == 0) continue;
        pts.push_back({s.name, stat[s.name].accuracy(), stat[s.name].mean_ms, s.policy});
    }
    {
        std::vector<std::pair<double, double>> am;
        for (const auto& p : pts) am.push_back({p.acc, p.ms});
        for (size_t i : pareto_front(am)) pts[i].front = true;
        std::ofstream f(opt.dir + "/pareto_frontier.csv", std::ios::binary);
        f << "partition,strategy,accuracy,mean_ms,p50_ms,p95_ms,non_dominated\n";
        for (const auto& p : pts)
            f << "evaluation," << p.name << "," << fmt("%.6f", p.acc) << "," << fmt("%.3f", p.ms) << ","
              << fmt("%.3f", stat[p.name].p50_ms) << "," << fmt("%.3f", stat[p.name].p95_ms) << "," << (p.front ? 1 : 0) << "\n";
    }

    // --- Plots ---
    {
        const double W = 760, H = 460, L = 70, R = 230, T = 30, B = 50;
        double x0 = 1e9, x1 = 0, y0 = 1, y1 = 0;
        for (const auto& p : pts) { x0 = std::min(x0, p.ms); x1 = std::max(x1, p.ms); y0 = std::min(y0, p.acc); y1 = std::max(y1, p.acc); }
        x0 = std::max(0.0, x0 - 40); x1 += 40; y0 = std::max(0.0, y0 - 0.02); y1 = std::min(1.0, y1 + 0.02);
        auto X = [&](double v) { return L + (v - x0) / (x1 - x0) * (W - L - R); };
        auto Y = [&](double v) { return H - B - (v - y0) / (y1 - y0) * (H - T - B); };
        std::ofstream f(opt.dir + "/accuracy_latency.svg", std::ios::binary);
        f << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W << "' height='" << H << "' font-family='sans-serif' font-size='11'>\n"
          << "<rect width='100%' height='100%' fill='white'/>\n"
          << "<text x='" << L << "' y='18' font-size='13'>R4 held-out accuracy vs mean latency (n=" << N << ")</text>\n"
          << "<line x1='" << L << "' y1='" << H - B << "' x2='" << W - R << "' y2='" << H - B << "' stroke='black'/>\n"
          << "<line x1='" << L << "' y1='" << T << "' x2='" << L << "' y2='" << H - B << "' stroke='black'/>\n"
          << "<text x='" << (L + W - R) / 2 << "' y='" << H - 12 << "' text-anchor='middle'>mean latency per query (ms)</text>\n"
          << "<text x='16' y='" << (T + H - B) / 2 << "' transform='rotate(-90 16 " << (T + H - B) / 2 << ")' text-anchor='middle'>accuracy</text>\n";
        for (int i = 0; i <= 4; ++i) {
            const double v = x0 + (x1 - x0) * i / 4, a = y0 + (y1 - y0) * i / 4;
            f << "<text x='" << X(v) << "' y='" << H - B + 15 << "' text-anchor='middle'>" << fmt("%.0f", v) << "</text>\n"
              << "<text x='" << L - 6 << "' y='" << Y(a) + 4 << "' text-anchor='end'>" << fmt("%.0f%%", a * 100) << "</text>\n";
        }
        int li = 0;
        for (const auto& p : pts) {
            const std::string col = p.policy ? "#c0392b" : "#2471a3";
            f << "<circle cx='" << X(p.ms) << "' cy='" << Y(p.acc) << "' r='" << (p.front ? 6 : 4) << "' fill='" << col
              << "' fill-opacity='" << (p.front ? 1.0 : 0.5) << "'/>\n";
            f << "<text x='" << W - R + 10 << "' y='" << T + 14 * li << "' fill='" << col << "'>" << li + 1 << ". " << p.name
              << (p.front ? " *" : "") << "</text>\n";
            f << "<text x='" << X(p.ms) + 7 << "' y='" << Y(p.acc) - 5 << "' fill='" << col << "'>" << li + 1 << "</text>\n";
            li++;
        }
        f << "<text x='" << W - R + 10 << "' y='" << T + 14 * li + 10 << "'>* non-dominated; blue = single strategy, red = policy</text>\n";
        f << "</svg>\n";
    }
    {
        const std::vector<std::string> names = {"zero_shot", "random_1shot", "nearest_1shot", "nearest_2shot", "knn_1",
                                                "B_task_only@A_latency", "C_adaptive@A_latency"};
        const std::vector<std::string> cols = {"#7f8c8d", "#f39c12", "#2471a3", "#1abc9c", "#8e44ad", "#c0392b", "#e74c3c"};
        const double W = 760, H = 380, L = 60, T = 40, B = 60, gw = 210;
        std::ofstream f(opt.dir + "/task_accuracy.svg", std::ios::binary);
        f << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W << "' height='" << H << "' font-family='sans-serif' font-size='11'>\n"
          << "<rect width='100%' height='100%' fill='white'/>\n<text x='" << L << "' y='20' font-size='13'>Held-out accuracy by task kind</text>\n";
        auto Y = [&](double a) { return H - B - a * (H - T - B); };
        for (int i = 0; i <= 4; ++i)
            f << "<line x1='" << L << "' y1='" << Y(i / 4.0) << "' x2='" << W - 20 << "' y2='" << Y(i / 4.0) << "' stroke='#ddd'/>"
              << "<text x='" << L - 6 << "' y='" << Y(i / 4.0) + 4 << "' text-anchor='end'>" << i * 25 << "%</text>\n";
        for (size_t ki = 0; ki < kKinds.size(); ++ki) {
            const auto& k = kKinds[ki];
            if (scope_of(k).idx.empty()) continue;
            const double gx = L + 20 + ki * gw;
            f << "<text x='" << gx + gw / 2 - 20 << "' y='" << H - B + 18 << "' text-anchor='middle'>" << k << " (n=" << scope_of(k).idx.size() << ")</text>\n";
            for (size_t si = 0; si < names.size(); ++si) {
                const Strat& s = S(names[si]);
                if (s.knn && !knn_supported(k)) continue;
                const double a = acc(s, k), bw = 22, x = gx + si * (bw + 2);
                f << "<rect x='" << x << "' y='" << Y(a) << "' width='" << bw << "' height='" << Y(0) - Y(a) << "' fill='" << cols[si] << "'/>"
                  << "<text x='" << x + bw / 2 << "' y='" << Y(a) - 3 << "' font-size='8' text-anchor='middle'>" << fmt("%.0f", a * 100) << "</text>\n";
            }
        }
        for (size_t si = 0; si < names.size(); ++si)
            f << "<rect x='" << L + si * 98 << "' y='" << H - 22 << "' width='10' height='10' fill='" << cols[si] << "'/>"
              << "<text x='" << L + si * 98 + 13 << "' y='" << H - 13 << "' font-size='9'>" << names[si] << "</text>\n";
        f << "</svg>\n";
    }

    // --- summary.md ---
    std::ostringstream o;
    auto ci_s = [&](const r3::ClusterCI& c) { return fmt("[%+.2f, %+.2f] pp", c.lo * 100, c.hi * 100); };
    auto supported = [&](const r3::ClusterCI& c) { return c.lo > 0 || c.hi < 0; };
    auto cmp_line = [&](const Comparison& c) {
        return "| " + std::to_string(c.id) + " | " + c.a + " − " + c.b + " | " + c.scope + " | " + std::to_string(c.n) + " | " +
               pct(c.acc_a) + " | " + pct(c.acc_b) + " | " + pp(c.ci.diff) + " | " + ci_s(c.ci) + " | " +
               (supported(c.ci) ? (c.ci.diff > 0 ? "yes (a better)" : "yes (b better)") : "no (CI includes 0)") + " |\n";
    };
    const json& tcfg = tu.config;
    o << "# R4 — Nearest 2-shot, kNN classification and task-aware routing\n\n";
    if (preliminary) o << "> **Preliminary**: a subset run (smoke test). Not a result.\n\n";
    o << "All numbers below are measured on the R2 held-out evaluation partition (n=" << N << ") under a configuration that was frozen on the tuning partition (n="
      << tu.rows.size() << ") before the held-out run. R3 had already reported held-out S1 / random / nearest results, so R4's design is "
         "motivated by known held-out outcomes: the confirmatory tests below are pre-registered in the frozen file, but they are not a "
         "completely independent confirmation of the R3 pattern.\n\n";

    o << "## A. Reproducibility\n\n| Item | Value |\n|---|---|\n";
    o << "| Frozen configuration | `frozen_policy.json`, sha256 `" << fz.sha256 << "` (created " << fz.raw.value("created", "") << ") |\n";
    o << "| Inference model | " << tcfg.value("inference_model", "") << " (sha256 `" << tcfg["inference_model_identity"].value("sha256", "") << "`) |\n";
    o << "| Embedding model | " << tcfg.value("embedding_model", "") << " (fingerprint `" << tcfg.value("embedding_fingerprint", "") << "`) |\n";
    o << "| Exemplar DB | " << tcfg.value("db_path", "") << " (schema " << tcfg["db_info"].value("schema_version", "") << ", read-only) |\n";
    o << "| Queries | " << tcfg.value("input_path", "") << " (sha256 `" << tcfg.value("input_sha256", "") << "`) |\n";
    o << "| Partition | R2 split, recomputed and checked against `" << evd.value("split_assignments", "") << "` (sha256 `"
      << evd.value("split_assignments_sha256", "").substr(0, 16) << "…`): tuning " << evd["partition"].value("saved_tuning", 0)
      << ", evaluation " << evd["partition"].value("saved_evaluation", 0) << ", disagreements " << evd["partition"].value("disagreements", -1)
      << ", group overlap " << evd["partition"].value("group_overlap", -1) << " |\n";
    o << "| Reference | " << evd.value("reference_predictions", "") << " (sha256 `" << evd.value("reference_predictions_sha256", "").substr(0, 16) << "…`, unchanged) |\n";
    o << "| Decision | letters, auto layout, single order, no calibration, candidate-logit argmax; n_ctx " << tcfg.value("n_ctx", 0) << " |\n";
    o << "| Prompt | r1-demo-turn-v1; 2-shot = two demonstration turns then the S1 query |\n";
    o << "| Random 1-shot | seed " << kRandomSeed << " (R3's first seed), XXH64 draw from R1's eligible pool |\n";
    o << "| Bootstrap | " << reps << " cluster replicates over group_id, seed " << bseed << " |\n";
    o << "| Runs | tuning " << tcfg.value("run_timestamp", "") << " (" << fmt("%.0f", tcfg["detail"].value("wall_seconds", 0.0)) << " s), evaluation "
      << ev.config.value("run_timestamp", "") << " (" << fmt("%.0f", evd.value("wall_seconds", 0.0)) << " s) |\n\n";
    o << "**Integrity checks (evaluation run).** S1 reproduced the reference logits and token counts on "
      << N - checks.value("s1_reference_mismatch", 0) << "/" << N << " queries; R1's nearest exemplar on "
      << N - checks.value("nearest_reference_mismatch", 0) << "/" << N << " and its logits on " << N - checks.value("r1_reference_mismatch", 0)
      << "/" << N << " (the 1-shot decision now runs through the generalised few-shot prompt builder); random 1-shot reproduced R3 seed 42 "
      << "(exemplar and logits) on " << checks.value("random_checked", 0) - checks.value("random_r3_mismatch", 0) << "/" << checks.value("random_checked", 0)
      << ". Retrieval failures " << checks.value("retrieval_failures", -1) << "; neighbour lists shorter than five "
      << checks.value("short_neighbour_lists", -1) << "; top-K search ranking a different exemplar first than R1's top-1 search "
      << checks.value("topk_rank1_differs_from_r1", -1) << "; content duplicates skipped " << checks.value("duplicates_skipped", -1)
      << "; neighbours whose content equals the query " << checks.value("query_content_in_bank", -1) << "; queries rejected as training records "
      << evd["partition"].value("queries_rejected_as_training", -1) << ". Every exemplar is re-checked at load (training allowlist, one-hot, same kind and level count, not the query).\n\n";
    o << "Condition status (evaluation run):";
    for (const auto& [cn, m] : evd["condition_status"].items()) {
        o << " " << cn << " {";
        bool first = true;
        for (const auto& [st, c] : m.items()) { o << (first ? "" : ", ") << st << " " << c.get<int>(); first = false; }
        o << "}";
    }
    o << ". Context-limit failures by strategy and kind: " << (evd.value("context_limit_by_kind", json::object()).empty() ? "none" : evd["context_limit_by_kind"].dump()) << ". Nothing is excluded: a failed decision counts as incorrect.\n\n";

    o << "## Overview (held-out)\n\n| Strategy | Accuracy | choice | noul | score | Mean ms | P50 | P95 | P99 | LLM evals | Embeddings | Prompt tokens | Failed |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& s : strats) {
        if (s.name.find("_uniform") != std::string::npos || s.name.find("_weighted") != std::string::npos) continue;
        const auto& st = stat[s.name];
        o << "| " << s.label << " | " << (s.knn ? pct(st.accuracy()) + " (noul+score)" : pct(st.accuracy()));
        for (const auto& k : kKinds) o << " | " << ((s.knn && !knn_supported(k)) || scope_of(k).idx.empty() ? "–" : pct(acc(s, k)));
        o << " | " << fmt("%.1f", st.mean_ms) << " | " << fmt("%.1f", st.p50_ms) << " | " << fmt("%.1f", st.p95_ms) << " | "
          << fmt("%.1f", st.p99_ms) << " | " << fmt("%.2f", st.llm) << " | " << fmt("%.2f", st.emb) << " | " << fmt("%.0f", st.tokens)
          << " | " << st.failed << " |\n";
    }
    o << "\nkNN rows cover noul and score only (choice is unsupported; see C). Latencies are composed from per-query measured stages "
         "along each strategy's execution plan (see Latency). Failed = decisions that did not complete (counted as incorrect); "
         "excluded queries: 0.\n\n";
    o << "Rescue / harm relative to S1 (rescue = S1 wrong and strategy right; net = rescue − harm):\n\n| Strategy | choice | noul | score | all |\n|---|---|---|---|---|\n";
    for (const auto& s : strats) {
        if (s.name == "zero_shot" || s.name.find("_uniform") != std::string::npos || s.name.find("_weighted") != std::string::npos) continue;
        o << "| " << s.label;
        for (const std::string k : {"choice", "noul", "score", "all"}) {
            if (scope_of(k).idx.empty() || (s.knn && (k == "all" || !knn_supported(k)))) { o << " | –"; continue; }
            const auto rh = rescue_harm(s, k);
            o << " | " << rh.first << " / " << rh.second << " (" << fmt("%+d", rh.first - rh.second) << ")";
        }
        o << " |\n";
    }
    o << "\n";

    o << "## Confirmatory comparisons (pre-registered)\n\n| # | Comparison | Scope | n | Acc a | Acc b | Diff | 95% CI | Supported |\n|---|---|---|---:|---:|---:|---:|---|---|\n";
    for (const auto& c : confirmatory) o << cmp_line(c);
    o << "\n\"Supported\" means the 95% cluster-bootstrap interval excludes zero; an interval that includes zero is reported as inconclusive, never as equivalence.\n\n";

    // B. 2-shot
    const auto& s1s = stat["zero_shot"];
    const auto& r1s = stat["nearest_1shot"];
    const auto& t2s = stat["nearest_2shot"];
    const Comparison& c1 = find_cmp(1);
    o << "## B. Nearest 2-shot\n\nOrder " << fz.two_shot_order << " was selected on the tuning partition ("
      << fz.raw["nearest_2shot"]["orders"].value(fz.two_shot_order, "") << "; tuning correct A "
      << fz.raw["nearest_2shot"]["tuning_correct"].value("A", 0) << " vs B " << fz.raw["nearest_2shot"]["tuning_correct"].value("B", 0)
      << " of " << fz.raw["nearest_2shot"].value("tuning_n", 0) << ").\n\n";
    o << "| Kind | n | S1 | R1 1-shot | 2-shot (selected) | 2-shot (other order) | 2-shot − 1-shot [95% CI] |\n|---|---:|---:|---:|---:|---:|---|\n";
    for (const std::string k : {"choice", "noul", "score", "all"}) {
        if (scope_of(k).idx.empty()) continue;
        const Comparison c = k == "all" ? c1 : compare("nearest_2shot", "nearest_1shot", k, 0);
        o << "| " << k << " | " << scope_of(k).idx.size() << " | " << pct(acc(S("zero_shot"), k)) << " | " << pct(acc(S("nearest_1shot"), k))
          << " | " << pct(acc(S("nearest_2shot"), k)) << " | " << pct(acc(S("nearest_2shot_other"), k)) << " | " << pp(c.ci.diff) << " "
          << ci_s(c.ci) << " |\n";
    }
    const Comparison ord = compare("nearest_2shot", "nearest_2shot_other", "all", 0);
    o << "\nOrder effect (exploratory, selected − other): " << pp(ord.ci.diff) << " " << ci_s(ord.ci) << ".\n\n";
    const double extra_ms = t2s.mean_ms - r1s.mean_ms, extra_tok = t2s.tokens - r1s.tokens;
    o << "Cost: 2-shot evaluates " << fmt("%.0f", t2s.tokens) << " prompt tokens per query vs " << fmt("%.0f", r1s.tokens) << " for 1-shot ("
      << fmt("%+.0f", extra_tok) << ", " << fmt("%+.0f%%", r1s.tokens > 0 ? 100.0 * extra_tok / r1s.tokens : 0.0) << ") and takes "
      << fmt("%.1f", t2s.mean_ms) << " ms vs " << fmt("%.1f", r1s.mean_ms) << " ms per query (" << fmt("%+.1f ms", extra_ms)
      << "; its top-10 search is cheaper than R1's top-1 search, see Latency). Accuracy change: " << pp(c1.ci.diff) << " " << ci_s(c1.ci)
      << " = " << fmt("%+.1f", c1.ci.diff * 1000) << " correct answers per 1,000 queries. ";
    if (c1.ci.lo > 0) o << "The improvement over 1-shot is statistically supported.";
    else if (c1.ci.hi < 0) o << "2-shot is worse than 1-shot.";
    else o << "The difference to 1-shot is not statistically supported; the extra prompt cost is therefore not justified by this evidence.";
    o << " Versus zero-shot: " << pp(find_cmp(2).ci.diff) << " " << ci_s(find_cmp(2).ci) << ".\n\n";

    // C. kNN
    o << "## C. kNN classification (no inference LLM)\n\n";
    o << "Label mapping by semantic identity: noul maps the exemplar's answer text (no/yes) to the query option with the same text; score maps the "
         "exemplar's answer level index to the same level (level counts match by R1's filter). **choice is unsupported**: options are per-task item "
         "identifiers, so a letter or position carries no shared meaning. The query's ground truth is never read before scoring.\n\n";
    o << "Frozen voting (selected on tuning): kNN-3 = " << knn.k3.describe() << ", kNN-5 = " << knn.k5.describe()
      << ". Weights w = exp((s − s_max)/τ) on estimated cosine; ties → label of the best-ranked neighbour.\n\n";
    o << "| Kind | n | S1 | R1 1-shot | kNN-1 | kNN-3 | kNN-5 | kNN-3 uniform / weighted | kNN-5 uniform / weighted |\n|---|---:|---:|---:|---:|---:|---:|---|---|\n";
    for (const std::string k : {"noul", "score", "noul+score"}) {
        if (scope_of(k).idx.empty()) continue;
        o << "| " << k << " | " << scope_of(k).idx.size() << " | " << pct(acc(S("zero_shot"), k)) << " | " << pct(acc(S("nearest_1shot"), k))
          << " | " << pct(acc(S("knn_1"), k)) << " | " << pct(acc(S("knn_3"), k)) << " | " << pct(acc(S("knn_5"), k)) << " | "
          << pct(acc(S("knn_3_uniform"), k)) << " / " << pct(acc(S("knn_3_weighted"), k)) << " | " << pct(acc(S("knn_5_uniform"), k))
          << " / " << pct(acc(S("knn_5_weighted"), k)) << " |\n";
    }
    o << "\n";
    // Diagnostics
    o << "Diagnostics (held-out):\n\n| Kind | Mean nearest sim | kNN-1 = R1 prediction | R1 = nearest demo label | kNN-1 ✓ R1 ✗ | R1 ✓ kNN-1 ✗ | Top-3 consistency | Top-3 unanimous | Top-5 consistency | Top-5 unanimous | kNN-1 rescue / harm vs S1 | R1 rescue / harm vs S1 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n";
    for (const std::string k : {"noul", "score"}) {
        const Scope& sc = scope_of(k);
        if (sc.idx.empty()) continue;
        double sim = 0; int agree_r1 = 0, r1_demo = 0, kn_only = 0, r1_only = 0, n = 0;
        double cons3 = 0, cons5 = 0; int un3 = 0, un5 = 0;
        for (size_t i : sc.idx) {
            const auto& r = rows[i];
            if (!r.retrieval_ok()) continue;
            n++;
            sim += r.nb[0].similarity;
            const auto& o1 = S("knn_1").out[i];
            agree_r1 += r.r1.ok() && o1.pred == r.r1.pred;
            r1_demo  += r.r1.ok() && r.r1.pred == r.nb[0].label;
            kn_only  += o1.correct && !r.r1.correct;
            r1_only  += !o1.correct && r.r1.correct;
            std::vector<VoteInput> v;
            for (const auto& x : r.nb) v.push_back({x.label, x.similarity});
            const auto v3 = vote(v, 3, Voting::Uniform, 0), v5 = vote(v, 5, Voting::Uniform, 0);
            if (v3.used) { cons3 += double(v3.agree) / v3.used; un3 += v3.unanimous; }
            if (v5.used) { cons5 += double(v5.agree) / v5.used; un5 += v5.unanimous; }
        }
        const auto rk = rescue_harm(S("knn_1"), k), rr = rescue_harm(S("nearest_1shot"), k);
        auto f = [&](double x) { return n ? pct(x / n) : std::string("–"); };
        o << "| " << k << " | " << (n ? fmt("%.4f", sim / n) : "–") << " | " << f(agree_r1) << " | " << f(r1_demo) << " | " << kn_only
          << " | " << r1_only << " | " << f(cons3) << " | " << f(un3) << " | " << f(cons5) << " | " << f(un5) << " | " << rk.first << " / "
          << rk.second << " | " << rr.first << " / " << rr.second << " |\n";
    }
    o << "\nConsistency = share of the top-k mapped labels that agree with the majority label (mean over queries).\n\n";
    // Label transfer vs LLM decision (noul)
    if (!scope_of("noul").idx.empty()) {
        int copy = 0, copy_ok = 0, dev = 0, dev_ok = 0, dev_label_ok = 0;
        for (size_t i : scope_of("noul").idx) {
            const auto& r = rows[i];
            if (!r.retrieval_ok() || !r.r1.ok()) continue;
            if (r.r1.pred == r.nb[0].label) { copy++; copy_ok += r.r1.correct; }
            else { dev++; dev_ok += r.r1.correct; dev_label_ok += r.nb[0].label == r.gt; }
        }
        o << "**Label transfer vs LLM decision (noul).** R1 predicts the nearest demonstration's label on " << copy << " queries ("
          << pct(copy ? double(copy_ok) / copy : 0) << " correct, which is exactly kNN-1's accuracy on them) and departs from it on " << dev
          << " queries, where R1 is correct " << dev_ok << " times and the demonstration label (= kNN-1) " << dev_label_ok
          << " times. The LLM's contribution beyond label transfer is therefore confined to those " << dev << " departures: net "
          << fmt("%+d", dev_ok - dev_label_ok) << " correct answers for R1 relative to kNN-1.\n\n";
    }
    // Similarity bins (noul + score)
    {
        const Scope& sc = scope_of("noul+score");
        std::vector<double> sims;
        for (size_t i : sc.idx) if (rows[i].retrieval_ok()) sims.push_back(rows[i].nb[0].similarity);
        if (!sims.empty()) {
            std::vector<double> edges;
            for (int p : {20, 40, 60, 80}) edges.push_back(quantile(sims, p));
            o << "Accuracy by nearest-neighbour similarity (exploratory; quintiles of the held-out noul+score similarities):\n\n"
                 "| Bin | Similarity | n | S1 | R1 | kNN-1 | kNN-5 (frozen) |\n|---|---|---:|---:|---:|---:|---:|\n";
            for (int b = 0; b < 5; ++b) {
                const double lo = b ? edges[b - 1] : -1e300, hi = b < 4 ? edges[b] : 1e300;
                int n = 0, a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                for (size_t i : sc.idx) {
                    if (!rows[i].retrieval_ok()) continue;
                    const double s = rows[i].nb[0].similarity;
                    if (s < lo || s >= hi) continue;
                    n++;
                    a0 += S("zero_shot").out[i].correct; a1 += S("nearest_1shot").out[i].correct;
                    a2 += S("knn_1").out[i].correct; a3 += S("knn_5").out[i].correct;
                }
                auto f = [&](int x) { return n ? pct(double(x) / n) : std::string("–"); };
                o << "| " << b + 1 << " | " << (b ? fmt("%.3f", lo) : std::string("min")) << " – " << (b < 4 ? fmt("%.3f", hi) : std::string("max"))
                  << " | " << n << " | " << f(a0) << " | " << f(a1) << " | " << f(a2) << " | " << f(a3) << " |\n";
            }
            o << "\n";
        }
    }
    // choice text-identity coverage
    {
        int n = 0, mapped = 0, mapped_ok = 0;
        for (size_t i : scope_of("choice").idx) {
            const auto& nb = ev.raw[i]["retrieval"]["neighbours"];
            if (nb.empty()) continue;
            n++;
            if (nb[0].value("text_identity_status", "") == "ok") {
                mapped++;
                mapped_ok += nb[0].value("text_identity_label", -1) == rows[i].gt;
            }
        }
        if (n) o << "choice (diagnostic only): the nearest exemplar's answer text occurs among the query's options for " << mapped << " of " << n
                 << " queries (" << pct(double(mapped) / n) << ")" << (mapped ? ", and is the correct option for " + std::to_string(mapped_ok) + " of them" : std::string())
                 << " (" << pct(double(mapped_ok) / n) << " of all choice queries). Text identity "
                    "is therefore not a usable kNN label mapping for choice; this diagnostic was not used by any policy.\n\n";
    }
    {
        std::vector<double> r1_sup;
        for (size_t i : scope_of("noul+score").idx) r1_sup.push_back(S("nearest_1shot").out[i].ms);
        o << "kNN cost on the same noul+score queries: kNN-1 " << fmt("%.1f", stat["knn_1"].mean_ms) << " ms, kNN-3 "
          << fmt("%.1f", stat["knn_3"].mean_ms) << " ms, kNN-5 " << fmt("%.1f", stat["knn_5"].mean_ms) << " ms (no LLM evaluation) vs R1 "
          << fmt("%.1f", r1_sup.empty() ? 0.0 : r1::mean(r1_sup)) << " ms; see Latency for the stages.\n\n";
    }

    // D. Routing
    o << "## D. Task-aware routing\n\nFinalists were chosen on the tuning partition from " << fz.raw["routing"]["rules_considered"].value("task_only", 0)
      << " task-only and " << fz.raw["routing"]["rules_considered"].value("adaptive", 0) << " adaptive per-kind rules (all non-dominated combinations), "
         "using the tuning partition's measured latencies. Tuning frontier: `pareto_frontier_tuning.csv`; every rule: `routing/tuning_policies.csv`.\n\n";
    o << "| Policy | choice | noul | score | Tuning acc / ms | Held-out acc | Mean ms | P50 | P95 | P99 | LLM | Emb | Tokens | Failed |\n|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (size_t pi = 0; pi < fz.finalists.size(); ++pi) {
        const auto& p = fz.finalists[pi];
        const auto& st = stat[p.name];
        const json& tj = fz.raw["finalist_selection"][pi]["tuning"];
        o << "| " << p.name << (std::find(fz.confirmatory_policies.begin(), fz.confirmatory_policies.end(), p.name) != fz.confirmatory_policies.end() ? " (confirmatory)" : "");
        for (const auto& k : kKinds) { auto it = p.per_kind.find(k); o << " | " << (it == p.per_kind.end() ? "–" : it->second.describe()); }
        o << " | " << pct(tj.value("accuracy", 0.0)) << " / " << fmt("%.0f", tj.value("mean_ms", 0.0)) << " | " << pct(st.accuracy()) << " | "
          << fmt("%.1f", st.mean_ms) << " | " << fmt("%.1f", st.p50_ms) << " | " << fmt("%.1f", st.p95_ms) << " | " << fmt("%.1f", st.p99_ms)
          << " | " << fmt("%.2f", st.llm) << " | " << fmt("%.2f", st.emb) << " | " << fmt("%.0f", st.tokens) << " | " << st.failed << " |\n";
    }
    o << "\nRouting frequency (held-out; share of each kind's queries answered by each terminal action):\n\n";
    for (const auto& p : fz.finalists) {
        o << "- **" << p.name << "**:";
        for (const auto& [k, m] : stat[p.name].routed) {
            int tot = 0;
            for (const auto& [t, c] : m) tot += c;
            o << " " << k << " {";
            bool first = true;
            for (const auto& [t, c] : m) { o << (first ? "" : ", ") << t << " " << fmt("%.0f%%", 100.0 * c / tot); first = false; }
            o << "}";
        }
        o << "\n";
    }
    o << "\nHeld-out accuracy/latency points (`pareto_frontier.csv`, `accuracy_latency.svg`); non-dominated:";
    for (const auto& p : pts) if (p.front) o << " " << p.name << " (" << pct(p.acc) << ", " << fmt("%.0f ms", p.ms) << ");";
    o << "\n\nWhich kinds benefit from retrieval (held-out, exploratory per-kind differences):\n\n| Kind | R1 − S1 | random − S1 | R1 − random | kNN-1 − S1 |\n|---|---|---|---|---|\n";
    for (const auto& k : kKinds) {
        if (scope_of(k).idx.empty()) continue;
        const auto a = compare("nearest_1shot", "zero_shot", k, 0), b = compare("random_1shot", "zero_shot", k, 0),
                   c = compare("nearest_1shot", "random_1shot", k, 0);
        o << "| " << k << " | " << pp(a.ci.diff) << " " << ci_s(a.ci) << " | " << pp(b.ci.diff) << " " << ci_s(b.ci) << " | " << pp(c.ci.diff)
          << " " << ci_s(c.ci) << " | ";
        if (knn_supported(k)) { const auto d = compare("knn_1", "zero_shot", k, 0); o << pp(d.ci.diff) << " " << ci_s(d.ci); }
        else o << "unsupported";
        o << " |\n";
    }
    o << "\n**Cost of adaptive decisions.** Every policy is costed along its execution plan with the work needed to decide included: a kNN or "
         "similarity gate always pays the query embedding and the vector search, even when it then answers from S1; an S1-margin gate always "
         "pays the S1 evaluation and, below the threshold, the full fallback action as well (two LLM evaluations). Mean LLM and embedding "
         "evaluations per query are in the table above.\n\n";
    if (live_ok && !live.empty()) {
        o << "**End-to-end check.** The finalists were executed live, plan by plan, on the first " << live_cfg.value("queries", 0)
          << " held-out queries (`policy_timing.jsonl`):\n\n| Policy | n | Live mean ms | Composed mean ms | Live / composed | Same prediction |\n|---|---:|---:|---:|---:|---:|\n";
        for (const auto& p : fz.finalists) {
            auto it = live.find(p.name);
            if (it == live.end()) continue;
            const auto& c = it->second;
            o << "| " << p.name << " | " << c.n << " | " << fmt("%.1f", c.live) << " | " << fmt("%.1f", c.composed) << " | "
              << fmt("%.2f", c.composed > 0 ? c.live / c.composed : 0.0) << " | " << pct(c.n ? double(c.agree) / c.n : 0.0) << " |\n";
        }
        o << "\n";
    } else {
        o << "**End-to-end check.** Not available (run `--mode policy-timing`).\n\n";
    }

    // Latency
    o << "## Latency\n\nMeasured per query in one process (models loaded once, untimed warm-up of every stage, sequential, every LLM "
         "decision from an empty KV cache; all actions of a query back to back). Means in ms over the held-out queries the action applies to:\n\n"
         "| Action | Embedding | Search | Row fetch | LZ4 | Random draw | Prompt construction | LLM inference | Candidate scoring | Vote | Decision wall | Total mean | P50 | P95 | P99 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const auto& s : strats) {
        if (s.policy || s.name.find("_uniform") != std::string::npos || s.name.find("_weighted") != std::string::npos) continue;
        const Stages m = mean_stages(s.action, "all");
        const auto& st = stat[s.name];
        o << "| " << s.name;
        for (double v : {m.embed, m.search, m.fetch, m.lz4, m.select, m.prompt, m.eval, m.scoring, m.vote, m.decision}) o << " | " << fmt(v < 1 ? "%.3f" : "%.1f", v);
        o << " | " << fmt("%.1f", st.mean_ms) << " | " << fmt("%.1f", st.p50_ms) << " | " << fmt("%.1f", st.p95_ms) << " | " << fmt("%.1f", st.p99_ms) << " |\n";
    }
    o << "\nDecision wall = prompt construction + LLM inference + scoring + call overhead. nearest_1shot uses R1's own top-1 search, which "
         "starts the eligibility-filtered KNN at k=1 and doubles until an eligible row appears; the top-10 search used by 2-shot and kNN-3/5 "
         "starts at k=10 and needs fewer rounds, which is why it is cheaper. That is a property of R1's retrieval loop, not of the "
         "strategies, and R1's mechanism was not changed. Offline evaluation reuses each query's single measurement of every stage; a "
         "deployed policy performs only the stages of its plan, which the end-to-end check above executes for real. These are local "
         "measurements on one machine and say nothing about production latency. The tuning run shared the machine with occasional "
         "compilation of the analysis tool; the evaluation run had no concurrent work.\n\n";

    // E. Deployment
    o << "## E. Deployment implications\n\n**Recommendation: " << recommendation << ".** (" << rec_reason << ".)\n\n"
      << "Pre-registered decision rule (frozen file): a policy qualifies when (i) its accuracy difference to A has a CI lower bound ≥ −"
      << fmt("%.1f", margin * 100) << " pp and its mean latency is at least " << fmt("%.0f%%", min_red * 100) << " below A's ("
      << fmt("%.1f", lat_a) << " ms), or (ii) the CI lower bound is above 0 at no more than A's latency. B (" << c6.a << "): " << pp(c6.ci.diff) << " " << ci_s(c6.ci) << ", " << fmt("%.1f", stat[c6.a].mean_ms)
      << " ms → " << (b_q ? "qualifies" : "does not qualify") << ". C (" << c7.a << "): " << pp(c7.ci.diff) << " " << ci_s(c7.ci) << ", "
      << fmt("%.1f", stat[c7.a].mean_ms) << " ms → " << (c_q ? "qualifies" : "does not qualify") << ". C − B: " << pp(c8.ci.diff) << " "
      << ci_s(c8.ci) << ". kNN-1 − R1 on noul: " << pp(c3.ci.diff) << " " << ci_s(c3.ci) << ".\n\n"
      << "No production default was changed; R4 adds an N-demonstration entry point (`decide_with_demonstrations`) used only by this tool.\n\n";

    o << "### Confirmed observations\n\n";
    for (const auto& c : confirmatory)
        o << "- " << c.a << " − " << c.b << " (" << c.scope << "): " << pp(c.ci.diff) << " " << ci_s(c.ci)
          << (supported(c.ci) ? " — statistically supported." : " — inconclusive (CI includes 0).") << "\n";
    o << "\n### Exploratory findings\n\n"
      << "- Per-kind effects, the non-selected 2-shot order, voting variants, similarity bins and the non-confirmatory finalists (all in "
         "`pairwise_comparisons.csv`, type = exploratory) were not pre-registered as tests.\n"
      << "- Order effect (selected − other): " << pp(ord.ci.diff) << " " << ci_s(ord.ci) << ".\n";
    for (const auto& p : fz.finalists) {
        if (std::find(fz.confirmatory_policies.begin(), fz.confirmatory_policies.end(), p.name) != fz.confirmatory_policies.end()) continue;
        const auto c = compare(p.name, "A_static_nearest_1shot", "all", 0);
        o << "- " << p.name << " − A: " << pp(c.ci.diff) << " " << ci_s(c.ci) << " at " << fmt("%.1f", stat[p.name].mean_ms) << " ms.\n";
    }
    o << "\n### Unresolved questions\n\n"
         "- The internal tuning/evaluation split is the only held-out evidence; R3's held-out results were known when R4 was designed.\n"
         "- Whether kNN's label transfer generalises to task families not represented in the exemplar bank (both partitions share the bank).\n"
         "- choice has no kNN mapping; whether a semantic candidate representation could support one was not studied.\n"
         "- Latency is local and single-machine; R1's top-1 retrieval loop cost depends on how many eligible rows sit near the query.\n\n";

    o << "## Commands\n\n```sh\ncmake --build build --config Release --target pjev-r4 test_r4\n./tests/bin/test_r4.exe\n"
         "./bin/pjev-r4.exe --mode run --partition tuning\n./bin/pjev-r4.exe --mode tune\n"
         "./bin/pjev-r4.exe --mode run --partition evaluation\n./bin/pjev-r4.exe --mode policy-timing --limit 300\n"
         "./bin/pjev-r4.exe --mode evaluate\n```\n\n"
         "Files: `frozen_policy.json` / `.sha256`, `tuning/` and `evaluation/` (per-query measurements: `queries.jsonl`, `run_config.json`), "
         "`nearest-2shot/`, `knn/`, `routing/` (`predictions.jsonl`, `metrics.json`; routing also `tuning_policies.csv`, `selected_policies.json`), "
         "`latency.csv`, `pairwise_comparisons.csv`, `task_metrics.csv`, `rescue_harm.csv`, `pareto_frontier.csv`, `pareto_frontier_tuning.csv`, "
         "`policy_timing.jsonl`, `accuracy_latency.svg`, `task_accuracy.svg`.\n";
    std::ofstream(opt.dir + "/summary.md", std::ios::binary) << o.str();

    printf("2-shot - 1-shot %s %s | kNN-1 - R1 (noul) %s %s | B - A %s %s | C - A %s %s\n", pp(c1.ci.diff).c_str(), ci_s(c1.ci).c_str(),
           pp(c3.ci.diff).c_str(), ci_s(c3.ci).c_str(), pp(c6.ci.diff).c_str(), ci_s(c6.ci).c_str(), pp(c7.ci.diff).c_str(), ci_s(c7.ci).c_str());
    printf("Recommendation: %s (%s)\nWrote %s/summary.md\n", recommendation.c_str(), rec_reason.c_str(), opt.dir.c_str());
    return 0;
}

} // namespace r4
} // namespace pjev
