// Unit tests for R3 random 1-shot (no model needed).
//
//  Selection:   determinism, independence from query order and from other seeds, seed independence,
//               uniformity, self exclusion, empty pools.
//  Eligibility: training allowlist, one-hot filter, kind / score-level pools, choice option counts
//               pooled, duplicate IDs, sorted order.
//  Prompt:      the selected exemplar enters the prompt only through the R1 demonstration path; the
//               query part is identical to S1 whichever exemplar is drawn.
//  Analysis:    record parsing / correctness, alignment (missing, unexpected, duplicate, mismatched),
//               seed summaries, cluster bootstrap grouping.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "r1_task.h"
#include "r3_analysis.h"
#include "r3_select.h"

#include "nlohmann/json.hpp"

using namespace pjev;
using namespace pjev::r3;
using json = nlohmann::ordered_json;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const char* desc)
{
    if (cond) { printf("  PASS: %s\n", desc); g_pass++; }
    else      { printf("  FAIL: %s\n", desc); g_fail++; }
}

static std::vector<int64_t> ids(int64_t from, int64_t to)
{
    std::vector<int64_t> v;
    for (int64_t i = from; i < to; ++i) v.push_back(i);
    return v;
}

// ---------------------------------------------------------------------------

static void test_selection()
{
    printf("\n[Random selection]\n");
    const auto pool = ids(100, 200);

    const auto a = select_random(pool, -1, 42, "q/1");
    const auto b = select_random(pool, -1, 42, "q/1");
    check(a.ok && b.ok && a.id == b.id && a.index == b.index, "same seed and query -> same exemplar");
    check(a.pool_size == 100, "pool size without the query");

    // Order independence: selecting other queries or seeds first changes nothing.
    std::vector<int64_t> forward, backward;
    for (int q = 0; q < 50; ++q) forward.push_back(select_random(pool, -1, 7, "q/" + std::to_string(q)).id);
    for (int q = 49; q >= 0; --q) {
        select_random(pool, -1, 999, "noise");
        backward.insert(backward.begin(), select_random(pool, -1, 7, "q/" + std::to_string(q)).id);
    }
    check(forward == backward, "selection independent of query order and of other draws");

    // Seed independence: different seeds give different draws for most queries.
    int same = 0;
    for (int q = 0; q < 500; ++q)
        same += select_random(pool, -1, 42, "q/" + std::to_string(q)).id == select_random(pool, -1, 123, "q/" + std::to_string(q)).id;
    check(same < 25, "seeds 42 and 123 agree on few queries (expected ~1%)");

    // Different query IDs in one seed are not all the same exemplar.
    std::set<int64_t> distinct;
    for (int q = 0; q < 200; ++q) distinct.insert(select_random(pool, -1, 42, "q/" + std::to_string(q)).id);
    check(distinct.size() > 80, "draws spread over the pool");

    // Uniformity: chi-square over 10 buckets of a 10-element pool, 20000 draws.
    const auto small = ids(0, 10);
    std::vector<int> hist(10, 0);
    for (int q = 0; q < 20000; ++q) hist[static_cast<size_t>(select_random(small, -1, 2026, "u" + std::to_string(q)).id)]++;
    double chi = 0;
    for (int h : hist) chi += (h - 2000.0) * (h - 2000.0) / 2000.0;
    check(chi < 27.9, "uniform draws (chi-square, 9 df, p > 0.001)");

    // Self exclusion.
    bool never_self = true;
    for (int q = 0; q < 2000; ++q) never_self &= select_random(small, 3, 42, "s" + std::to_string(q)).id != 3;
    check(never_self, "the query itself is never drawn");
    const auto s = select_random(small, 3, 42, "s0");
    check(s.pool_size == 9, "self removed from the pool size");
    std::vector<int> seen(10, 0);
    for (int q = 0; q < 5000; ++q) seen[static_cast<size_t>(select_random(small, 3, 1, "t" + std::to_string(q)).id)]++;
    bool all_others = true;
    for (int i = 0; i < 10; ++i) if (i != 3) all_others &= seen[static_cast<size_t>(i)] > 0;
    check(all_others && seen[3] == 0, "every other candidate stays reachable");

    const auto only_self = select_random({5}, 5, 42, "x");
    check(!only_self.ok && only_self.pool_size == 0, "pool of only the query -> no eligible exemplar");
    check(!select_random({}, -1, 42, "x").ok, "empty pool -> no eligible exemplar");

    check(random_index(42, "q", 1) == 0, "single candidate -> index 0");
    check(random_index(42, "q", 1000) == random_index(42, "q", 1000) && random_index(42, "q", 1000) != random_index(43, "q", 1000),
          "random_index deterministic and seed dependent");
    check(std::string(to_string(ExampleSelection::Nearest)) == "nearest" && std::string(to_string(ExampleSelection::Random)) == "random",
          "strategy names");
}

static void test_eligibility()
{
    printf("\n[Eligible pool]\n");
    std::vector<PoolRow> rows = {
        {1, "choice", 0, 0}, {2, "choice", 0, 2},   // two choice rows (different option counts are pooled)
        {3, "choice", 0, -1},                       // train, no one-hot -> excluded
        {4, "choice", 0, 1},                        // not in train -> excluded
        {5, "noul", 0, 1},
        {6, "score", 5, 3}, {7, "score", 3, 0}, {8, "score", 5, 4},
        {2, "choice", 0, 2},                        // duplicate row
    };
    std::unordered_set<int64_t> train = {1, 2, 3, 5, 6, 7, 8};
    PoolStats st;
    const auto pool = ExemplarPool::build(rows, train, &st);
    check(pool.candidates("choice") == std::vector<int64_t>({1, 2}), "choice pool = train + one-hot rows, sorted");
    check(!pool.contains("choice", 3), "soft-target training row excluded");
    check(!pool.contains("choice", 4), "non-training row excluded");
    check(pool.candidates("noul") == std::vector<int64_t>({5}), "noul pool");
    check(pool.candidates("score/5") == std::vector<int64_t>({6, 8}) && pool.candidates("score/3") == std::vector<int64_t>({7}),
          "score pools split by level count (R1 KNN filter)");
    check(pool.candidates("score").empty() && pool.candidates("unknown").empty(), "unknown keys are empty");
    check(st.rows == 9 && st.not_in_train == 1 && st.no_one_hot == 1 && st.duplicate_ids == 1 && st.eligible == 6, "pool statistics");
    check(pool_key("score", 5) == "score/5" && pool_key("choice", 0) == "choice" && pool_key("noul", 0) == "noul", "pool keys");

    // Row order in the DB does not matter.
    std::vector<PoolRow> rev(rows.rbegin(), rows.rend());
    const auto pool2 = ExemplarPool::build(rev, train);
    check(pool2.by_key() == pool.by_key(), "pool independent of database row order");
    check(select_random(pool.candidates("score/5"), 6, 42, "q").id == 8, "score query excluded from its own pool");

    // Leakage: every draw from a pool is an allowlisted, one-hot, same-pool exemplar.
    bool ok = true;
    for (int q = 0; q < 200; ++q) {
        const auto p = select_random(pool.candidates("choice"), 999, 42, "q" + std::to_string(q));
        ok &= p.ok && train.count(p.id) && p.id != 3 && p.id != 4;
    }
    check(ok, "draws never leave the training allowlist");
}

static void test_length_matched()
{
    printf("\n[Length-matched control]\n");
    // Lengths 0, 10, 20, ..., 990 for IDs 1000..1099, plus a tie at 500.
    std::vector<std::pair<int64_t, int>> id_tok;
    for (int i = 0; i < 100; ++i) id_tok.emplace_back(1000 + i, i * 10);
    id_tok.emplace_back(5000, 500);
    const auto pool = LengthPool::build(id_tok);
    check(pool.by_length.size() == 101 && std::is_sorted(pool.by_length.begin(), pool.by_length.end()), "pool sorted by (tokens, id)");

    const auto p = select_length_matched(pool, -1, 503, 5, 42, "q");
    check(p.ok && p.window_size == 5 && p.window_lo == 490 && p.window_hi == 520, "window = 5 closest lengths (500, 500, 510, 490, 520)");
    check(std::abs(p.tokens - 503) <= 23, "pick inside the window");
    const auto p2 = select_length_matched(pool, -1, 503, 5, 42, "q");
    check(p.id == p2.id, "deterministic for seed and query");

    // Ties at the boundary resolve by lower ID: target 505, window 2 -> {500 (id 1050), 500 (id 5000)}... both at
    // distance 5, plus 510 at distance 5 -> lowest two IDs among the three tied: 1050 and 1051.
    std::set<int64_t> seen;
    for (int q = 0; q < 200; ++q) seen.insert(select_length_matched(pool, -1, 505, 2, 1, "t" + std::to_string(q)).id);
    check(seen == std::set<int64_t>({1050, 1051}), "boundary ties broken by lower ID");

    // Self exclusion and window filling.
    bool never_self = true;
    for (int q = 0; q < 300; ++q) never_self &= select_length_matched(pool, 1050, 500, 3, 7, "s" + std::to_string(q)).id != 1050;
    check(never_self, "query itself never drawn");
    check(select_length_matched(pool, 1050, 500, 3, 7, "s0").window_size == 3, "window refilled after removing the query");

    // Extremes and small pools.
    const auto lo = select_length_matched(pool, -1, -100, 3, 1, "x");
    check(lo.ok && lo.window_lo == 0 && lo.window_hi == 20, "target below every length -> shortest exemplars");
    const auto hi = select_length_matched(pool, -1, 99999, 3, 1, "x");
    check(hi.ok && hi.window_lo == 970 && hi.window_hi == 990, "target above every length -> longest exemplars");
    const auto all = select_length_matched(pool, -1, 500, 1000, 1, "x");
    check(all.ok && all.window_size == 101, "window larger than the pool -> whole pool");
    check(!select_length_matched(LengthPool{}, -1, 10, 5, 1, "x").ok, "empty pool -> no pick");

    // Seeds differ.
    int same = 0;
    for (int q = 0; q < 300; ++q)
        same += select_length_matched(pool, -1, 500, 50, 42, "u" + std::to_string(q)).id ==
                select_length_matched(pool, -1, 500, 50, 123, "u" + std::to_string(q)).id;
    check(same < 30, "different seeds draw different exemplars inside the window");
    // The random and length-matched derivations are independent.
    check(random_index(42, "q", 1000, kRandomSelectionVersion) != random_index(42, "q", 1000, kLengthMatchedVersion),
          "R3 and R3-L use distinct hash versions");
}

static void test_prompt_equivalence()
{
    printf("\n[Prompt equivalence]\n");
    PromptConfig cfg;
    cfg.layout = Layout::AUTO;
    cfg.scheme = Scheme::LETTERS;
    PromptStrategy strategy(cfg);
    std::string err;
    r1::TaskMapping q;
    r1::build_task("choice", "Which?", json{{"k", "v"}}, {"P1", "P2", "P3"}, q, err);

    auto payload = [](int64_t id, const std::vector<std::string>& opts, int ans, const std::string& question) {
        r0::ExemplarPayload p;
        p.example_id = id; p.kind = "choice"; p.question = question; p.state = json{{"f", 1}}; p.options = opts; p.answer_index = ans;
        return p;
    };
    const auto near = payload(11, {"a", "b"}, 0, "nearest question");
    const auto rnd  = payload(22, {"x", "y", "z", "w"}, 3, "random question");

    // The same exemplar gives the same prompt whichever strategy chose it: selection never reaches the prompt.
    Demonstration d1, d1b, d2;
    r1::build_demonstration(near, d1, err);
    r1::build_demonstration(near, d1b, err);
    r1::build_demonstration(rnd, d2, err);
    const std::string p_near = r1::render_r1_prompt(strategy, q.input, d1);
    check(p_near == r1::render_r1_prompt(strategy, q.input, d1b), "same exemplar -> identical prompt");

    const std::string p_rand = r1::render_r1_prompt(strategy, q.input, d2);
    const std::string s1 = r1::render_s1_prompt(strategy, q.input);
    const std::string query_part = s1.substr(std::string("<|im_start|>user\n").size());
    auto ends_with = [](const std::string& s, const std::string& t) { return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0; };
    check(ends_with(p_near, query_part) && ends_with(p_rand, query_part), "query part identical to S1 for nearest and random");
    check(p_rand.find("random question") != std::string::npos && p_rand.find("D: w") != std::string::npos &&
          p_rand.find("assistant\n\nD<|im_end|>") != std::string::npos, "random exemplar formatted like R1 (own options, label answer)");
    // Removing the demonstration turn leaves the same remainder for both.
    check(p_near.substr(p_near.size() - query_part.size()) == p_rand.substr(p_rand.size() - query_part.size()),
          "prompts differ only in the demonstration turn");
}

static json seed_line(const std::string& id, uint64_t seed, int gt, int pred, const std::string& status = "ok",
                      const std::string& kind = "choice")
{
    json j;
    j["query_id"] = id;
    j["kind"] = kind;
    j["ground_truth_index"] = gt;
    j["ground_truth_label"] = std::string(1, static_cast<char>('A' + gt));
    j["strategy"] = "random";
    j["seed"] = seed;
    j["exemplar"] = {{"status", "ok"}, {"example_id", "00000000000000aa"}, {"answer_label", "B"},
                     {"hamming_distance", 120.0}, {"estimated_cosine", 0.86}, {"select_ms", 0.01},
                     {"fetch_ms", 0.02}, {"decompress_ms", 0.03}};
    json c = {{"status", status}};
    if (status == "ok") {
        c["prediction_index"] = pred;
        c["prediction_label"] = std::string(1, static_cast<char>('A' + pred));
        c["correct"] = pred == gt;
    }
    c["prompt_tokens"] = 500;
    c["latency_ms"] = 100.0;
    j["r3"] = c;
    return j;
}

static void test_records_and_alignment()
{
    printf("\n[Records and alignment]\n");
    SeedRecord r;
    check(parse_seed_record(seed_line("q1", 42, 1, 1), r) && r.ok() && r.correct, "correct prediction parsed");
    check(parse_seed_record(seed_line("q1", 42, 1, 2), r) && r.ok() && !r.correct, "incorrect prediction parsed");
    check(std::fabs(r.total_ms() - 100.06) < 1e-9 && r.prompt_tokens == 500 && r.has_similarity && r.similarity == 0.86,
          "latency, tokens and similarity parsed");
    check(parse_seed_record(seed_line("q1", 42, 1, 0, "context_limit"), r) && !r.ok() && !r.correct,
          "failed inference is not ok and never correct");
    check(!parse_seed_record(json{{"query_id", "x"}}, r), "incomplete record rejected");

    std::map<std::string, RefRecord> ref;
    for (const std::string id : {"a", "b", "c"}) {
        RefRecord x;
        x.query_id = id; x.kind = "choice"; x.group_id = "g"; x.ground_truth = 1; x.s1_ok = x.r1_ok = true;
        ref[id] = x;
    }
    auto recs = [&](std::vector<std::string> qs, uint64_t seed) {
        std::vector<SeedRecord> v;
        for (const auto& q : qs) { SeedRecord s; parse_seed_record(seed_line(q, seed, 1, 1), s); v.push_back(s); }
        return v;
    };
    const std::vector<std::string> expected = {"a", "b", "c"};
    {
        std::map<uint64_t, std::vector<SeedRecord>> s = {{1, recs({"c", "b", "a"}, 1)}, {2, recs({"a", "b", "c"}, 2)}};
        const auto al = align(expected, ref, s);
        check(al.exact() && al.query_ids == expected && al.analysed.size() == 3, "aligned regardless of file order");
    }
    {
        std::map<uint64_t, std::vector<SeedRecord>> s = {{1, recs({"a", "b"}, 1)}, {2, recs({"a", "b", "c", "z"}, 2)}};
        const auto al = align(expected, ref, s);
        check(al.missing_in_seed == 1 && al.unexpected_in_seed == 1 && !al.exact(), "missing and unexpected queries detected");
        check(al.query_ids == std::vector<std::string>({"a", "b"}), "only fully present queries aligned");
    }
    {
        std::map<uint64_t, std::vector<SeedRecord>> s = {{1, recs({"a", "a", "b", "c"}, 1)}};
        check(align(expected, ref, s).duplicate_in_seed == 1, "duplicate query detected");
    }
    {
        auto v = recs({"a", "b", "c"}, 1);
        v[1].ground_truth = 0;
        std::map<uint64_t, std::vector<SeedRecord>> s = {{1, v}};
        check(align(expected, ref, s).ground_truth_mismatch == 1, "ground-truth disagreement detected");
    }
    {
        auto v = recs({"a", "b", "c"}, 1);
        parse_seed_record(seed_line("b", 1, 1, 0, "context_limit"), v[1]);
        auto ref2 = ref;
        ref2["c"].r1_ok = false;
        std::map<uint64_t, std::vector<SeedRecord>> s = {{1, v}};
        const auto al = align(expected, ref2, s);
        check(al.exact() && al.failed_per_seed.at(1) == 1 && al.reference_failed == 1 &&
              al.analysed == std::vector<std::string>({"a"}), "failures are counted, not silently dropped");
    }

    Agreement ag;
    ag.add(true, true); ag.add(true, false); ag.add(false, true); ag.add(false, false); ag.add(true, false);
    check(ag.both_correct == 1 && ag.a_only == 2 && ag.b_only == 1 && ag.both_incorrect == 1 && ag.n() == 5, "R1 vs R3 agreement table");
}

static void test_statistics()
{
    printf("\n[Seed summary and bootstrap]\n");
    const auto s = summarize({0.50, 0.52, 0.54});
    check(std::fabs(s.mean - 0.52) < 1e-12 && std::fabs(s.sd - 0.02) < 1e-12 && s.min == 0.50 && s.max == 0.54, "mean / sample SD / min / max");
    check(summarize({0.4}).sd == 0.0, "single seed has zero SD");

    // Two clusters with opposite effects: a cluster-level bootstrap must see only two units.
    std::vector<std::string> cl;
    std::vector<double> a, b;
    for (int i = 0; i < 100; ++i) { cl.push_back("g1"); a.push_back(1); b.push_back(0); }
    for (int i = 0; i < 100; ++i) { cl.push_back("g2"); a.push_back(0); b.push_back(0); }
    const auto ci = cluster_bootstrap(cl, a, b, 2000, 1);
    check(ci.clusters == 2 && std::fabs(ci.diff - 0.5) < 1e-12, "difference and cluster count");
    check(ci.lo < 0.05 && ci.hi > 0.95, "clustered queries move together (interval spans the cluster values)");

    // Same data with every query its own cluster gives a far narrower interval.
    std::vector<std::string> own;
    for (int i = 0; i < 200; ++i) own.push_back("q" + std::to_string(i));
    const auto ci2 = cluster_bootstrap(own, a, b, 2000, 1);
    check(ci2.hi - ci2.lo < 0.2, "independent queries -> narrow interval");

    // Fractional (seed-averaged) values are allowed.
    const auto ci3 = cluster_bootstrap({"x", "y"}, {0.6, 0.6}, {0.2, 0.2}, 500, 3);
    check(std::fabs(ci3.diff - 0.4) < 1e-12 && std::fabs(ci3.lo - 0.4) < 1e-12 && std::fabs(ci3.hi - 0.4) < 1e-12,
          "seed-averaged values: constant difference -> degenerate interval");

    const auto r1 = cluster_bootstrap(cl, a, b, 500, 9), r2 = cluster_bootstrap(cl, a, b, 500, 9);
    check(r1.lo == r2.lo && r1.hi == r2.hi, "bootstrap deterministic for a seed");
    check(cluster_bootstrap({}, {}, {}, 10, 1).clusters == 0, "empty input");
}

int main()
{
    printf("=== R3 random 1-shot unit tests ===\n");
    test_selection();
    test_eligibility();
    test_length_matched();
    test_prompt_equivalence();
    test_records_and_alignment();
    test_statistics();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
