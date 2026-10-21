// Unit tests for R4 (no model needed).
//
//  Neighbours:  deterministic top-2 list (rank 1 = R1's exemplar), ID / content duplicate exclusion, query
//               self / content leakage, training allowlist, score-level compatibility.
//  Prompt:      2-shot ordering, 1-demo few-shot == R1 segments, 0-demo == S1 segments.
//  kNN:         semantic label mapping (noul text, score level), unsupported mappings, uniform vote counting,
//               weighted vote, tie-breaking.
//  Routing:     action and rule cost composition (decision work always paid), context-limit fallback,
//               determinism, rule / policy JSON round trip, Pareto front, frozen-configuration loading.
//  Records:     query-record parsing and alignment, statistics (percentiles, cluster bootstrap).

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "model/model_identity.h"
#include "r1_task.h"
#include "r3_analysis.h"
#include "r4_core.h"

#include "nlohmann/json.hpp"

using namespace pjev;
using namespace pjev::r4;
using json = nlohmann::ordered_json;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const char* desc)
{
    if (cond) { printf("  PASS: %s\n", desc); g_pass++; }
    else      { printf("  FAIL: %s\n", desc); g_fail++; }
}

static bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

// ---------------------------------------------------------------------------

static void test_neighbours()
{
    printf("\n[Neighbour list]\n");
    // candidates[0] = R1's top-1 result; then the top-K search (which repeats it first).
    std::vector<NeighbourCandidate> c = {{7, 70}, {7, 70}, {3, 30}, {9, 90}, {4, 40}, {5, 50}, {6, 60}};
    auto a = merge_neighbours(c, 1, 10, 5);
    auto b = merge_neighbours(c, 1, 10, 5);
    check(a.picked == b.picked, "deterministic");
    check(a.picked.size() == 5 && c[a.picked[0]].id == 7 && c[a.picked[1]].id == 3, "rank 1 = R1 nearest, rank 2 = next distinct");
    {
        std::vector<int64_t> ids;
        for (size_t i : a.picked) ids.push_back(c[i].id);
        check(ids == std::vector<int64_t>({7, 3, 9, 4, 5}), "top-K order kept after the nearest");
    }
    auto t2 = merge_neighbours(c, 1, 10, 2);
    check(t2.picked.size() == 2 && c[t2.picked[0]].id != c[t2.picked[1]].id, "top-2 distinct");

    // R1's nearest differs from the top-K rank 1 (tie at the boundary): still rank 1.
    std::vector<NeighbourCandidate> tie = {{8, 80}, {3, 30}, {8, 80}, {9, 90}};
    auto tt = merge_neighbours(tie, 1, 10, 2);
    check(tie[tt.picked[0]].id == 8 && tie[tt.picked[1]].id == 3, "R1 exemplar stays rank 1 when top-K ranks another first");

    // Content duplicate of an earlier pick is skipped.
    std::vector<NeighbourCandidate> dup = {{7, 70}, {7, 70}, {3, 70}, {9, 90}};
    auto d = merge_neighbours(dup, 1, 10, 2);
    check(d.duplicates_skipped == 1 && dup[d.picked[1]].id == 9, "content duplicate excluded");

    // Self by ID and by content.
    std::vector<NeighbourCandidate> self = {{7, 70}, {1, 11}, {5, 10}, {6, 60}};
    auto sf = merge_neighbours(self, 1, 10, 3);
    check(sf.self_skipped == 2 && sf.picked.size() == 2 && self[sf.picked[1]].id == 6, "query itself excluded (ID and content)");
    check(merge_neighbours({}, 1, 10, 2).picked.empty(), "empty candidate list");

    check(content_key("noul", "q", "s", {"no", "yes"}) == content_key("noul", "q", "s", {"no", "yes"}), "content key stable");
    check(content_key("noul", "q", "s", {"no", "yes"}) != content_key("noul", "q", "s", {"yes", "no"}), "content key order-sensitive");
    check(content_key("noul", "ab", "c", {}) != content_key("noul", "a", "bc", {}), "content key field boundaries");
}

static void test_eligibility()
{
    printf("\n[Eligibility / leakage]\n");
    std::unordered_set<int64_t> allow = {10, 11, 12};
    check(exemplar_eligible(10, "noul", 0, 99, "noul", 0, allow), "allowlisted same-kind exemplar");
    check(!exemplar_eligible(13, "noul", 0, 99, "noul", 0, allow), "outside the training allowlist rejected");
    check(!exemplar_eligible(10, "noul", 0, 10, "noul", 0, allow), "the query itself rejected");
    check(!exemplar_eligible(10, "choice", 0, 99, "noul", 0, allow), "different kind rejected");
    check(exemplar_eligible(11, "score", 5, 99, "score", 5, allow), "score with the same level count");
    check(!exemplar_eligible(11, "score", 3, 99, "score", 5, allow), "score level count mismatch rejected");
}

static void test_prompt_order()
{
    printf("\n[Two-shot prompt ordering]\n");
    PromptConfig cfg;
    cfg.layout = Layout::AUTO;
    cfg.scheme = Scheme::LETTERS;
    PromptStrategy strategy(cfg);
    std::string err;
    r1::TaskMapping q;
    r1::build_task("noul", "Is it late?", json{{"k", "v"}}, {"no", "yes"}, q, err);
    auto demo = [&](const std::string& question, int ans) {
        r0::ExemplarPayload p;
        p.kind = "noul"; p.question = question; p.state = json{{"f", 1}}; p.options = {"no", "yes"}; p.answer_index = ans;
        Demonstration d;
        r1::build_demonstration(p, d, err);
        return d;
    };
    const Demonstration nearest = demo("NEAREST?", 1), second = demo("SECOND?", 0);
    auto text = [](const std::vector<PromptSegment>& s) { std::string t; for (const auto& x : s) t += x.text; return t; };

    const auto a = r1::few_shot_prompt_segments(strategy, q.input, {second, nearest});   // order A
    const auto b = r1::few_shot_prompt_segments(strategy, q.input, {nearest, second});   // order B
    const std::string ta = text(a), tb = text(b);
    check(ta.find("SECOND?") < ta.find("NEAREST?") && ta.find("NEAREST?") < ta.find("Is it late?"), "order A: second, nearest, query");
    check(tb.find("NEAREST?") < tb.find("SECOND?") && tb.find("SECOND?") < tb.find("Is it late?"), "order B: nearest, second, query");
    check(a.size() == 11 && b.size() == 11, "2-shot segment count (open + 2 x 4 + query 2)");
    // Trust flags: answer labels and contents untrusted, template parts trusted.
    bool flags = true;
    for (size_t i = 0; i < a.size(); ++i) {
        const bool template_part = a[i].text == "<|im_start|>user\n" || a[i].text == PromptStrategy::ANSWER_ANCHOR ||
                                   a[i].text == PromptStrategy::DEMO_TURN_SEPARATOR;
        flags = flags && a[i].trusted == template_part;
    }
    check(flags, "only template segments are trusted");
    const std::string s1 = r1::render_s1_prompt(strategy, q.input);
    const std::string query_part = s1.substr(std::string("<|im_start|>user\n").size());
    check(ta.size() > query_part.size() && ta.compare(ta.size() - query_part.size(), query_part.size(), query_part) == 0,
          "query part identical to S1");
    check(text(r1::few_shot_prompt_segments(strategy, q.input, {nearest})) == r1::render_r1_prompt(strategy, q.input, nearest),
          "one demonstration == R1 prompt");
    const auto one = r1::few_shot_prompt_segments(strategy, q.input, {nearest});
    const auto r1s = r1::r1_prompt_segments(strategy, q.input, nearest);
    bool same = one.size() == r1s.size();
    for (size_t i = 0; same && i < one.size(); ++i) same = one[i].text == r1s[i].text && one[i].trusted == r1s[i].trusted;
    check(same, "one demonstration == R1 segments (text and trust)");
    const auto zero = r1::few_shot_prompt_segments(strategy, q.input, {});
    const auto s1s = r1::s1_prompt_segments(strategy, q.input);
    same = zero.size() == s1s.size();
    for (size_t i = 0; same && i < zero.size(); ++i) same = zero[i].text == s1s[i].text && zero[i].trusted == s1s[i].trusted;
    check(same, "no demonstration == S1 segments");
    check(ta.find("assistant\n\nB<|im_end|>") != std::string::npos && ta.find("assistant\n\nA<|im_end|>") != std::string::npos,
          "each demonstration carries its own answer label");
}

static void test_mapping()
{
    printf("\n[Label mapping]\n");
    // Query noul with options in dataset order (yes, no): candidates are (false=no, true=yes).
    r1::TaskMapping q;
    std::string err;
    r1::build_task("noul", "Q", json("s"), {"yes", "no"}, q, err);
    auto m_yes = map_label("noul", {"yes", "no"}, q.option_to_candidate, "noul", {"no", "yes"}, 1);
    auto m_no  = map_label("noul", {"yes", "no"}, q.option_to_candidate, "noul", {"no", "yes"}, 0);
    check(m_yes.status == MapStatus::Ok && m_yes.candidate == 1, "noul 'yes' -> query 'true' candidate whatever the option order");
    check(m_no.status == MapStatus::Ok && m_no.candidate == 0, "noul 'no' -> query 'false' candidate");
    check(map_label("noul", {"yes", "no"}, q.option_to_candidate, "noul", {"maybe", "yes"}, 0).status == MapStatus::NoMatch,
          "noul answer outside no/yes -> no match");
    r1::TaskMapping s;
    r1::build_task("score", "Q", json("s"), {"bad", "ok", "good"}, s, err);
    auto ms = map_label("score", {"bad", "ok", "good"}, s.option_to_candidate, "score", {"low", "mid", "high"}, 2);
    check(ms.status == MapStatus::Ok && ms.candidate == 2, "score maps by level index, not by text");
    check(map_label("score", {"bad", "ok", "good"}, s.option_to_candidate, "score", {"1", "2", "3", "4", "5"}, 4).status ==
              MapStatus::NoMatch, "score level count mismatch -> no match");
    r1::TaskMapping c;
    r1::build_task("choice", "Q", json("s"), {"item_1", "item_2"}, c, err);
    check(map_label("choice", {"item_1", "item_2"}, c.option_to_candidate, "choice", {"item_1", "item_2"}, 0).status ==
              MapStatus::Unsupported, "choice is unsupported even with identical texts");
    check(map_label("noul", {"yes", "no"}, q.option_to_candidate, "score", {"no", "yes"}, 0).status == MapStatus::NoMatch,
          "kind mismatch -> no match");
    check(map_label("noul", {"yes", "no"}, q.option_to_candidate, "noul", {"no", "yes"}, -1).status == MapStatus::NoMatch,
          "no one-hot answer -> no match");
    auto ti = map_label_text_identity({"item_1", "item_2"}, c.option_to_candidate, {"item_9", "item_2"}, 1);
    check(ti.status == MapStatus::Ok && ti.candidate == 1, "text-identity diagnostic finds the same option text");
    check(map_label_text_identity({"item_1"}, {0}, {"item_9"}, 0).status == MapStatus::NoMatch, "text-identity no match");
    check(knn_supported("noul") && knn_supported("score") && !knn_supported("choice"), "supported kinds");
}

static void test_vote()
{
    printf("\n[Voting]\n");
    std::vector<VoteInput> v = {{1, 0.95}, {0, 0.94}, {0, 0.93}, {1, 0.80}, {1, 0.70}};
    auto u1 = vote(v, 1, Voting::Uniform, 0);
    check(u1.ok && u1.label == 1 && u1.used == 1 && u1.unanimous, "k=1 = nearest label");
    auto u3 = vote(v, 3, Voting::Uniform, 0);
    check(u3.label == 0 && u3.agree == 2 && u3.used == 3 && !u3.unanimous, "k=3 uniform majority");
    auto u5 = vote(v, 5, Voting::Uniform, 0);
    check(u5.label == 1 && u5.agree == 3, "k=5 uniform majority");
    auto w5 = vote(v, 5, Voting::Weighted, 0.01);
    // weights: exp(0)=1, exp(-1), exp(-2), exp(-15), exp(-25): label 1 = 1 + tiny, label 0 = 0.368 + 0.135
    check(w5.label == 1 && w5.win_weight > 1.0 && w5.win_weight < 1.0001, "weighted vote favours the close neighbour");
    auto w3 = vote(v, 3, Voting::Weighted, 0.1);
    // label 1 = 1, label 0 = exp(-0.1) + exp(-0.2) = 1.72
    check(w3.label == 0, "weighted with a wide tau behaves like counting");
    // Ties: equal totals -> best-ranked label.
    std::vector<VoteInput> t = {{2, 0.9}, {0, 0.9}, {0, 0.9}, {2, 0.9}};
    check(vote(t, 4, Voting::Uniform, 0).label == 2, "uniform tie -> best-ranked label");
    check(vote(t, 4, Voting::Weighted, 0.05).label == 2, "weighted tie -> best-ranked label");
    std::vector<VoteInput> t2 = {{0, 0.9}, {2, 0.9}};
    check(vote(t2, 2, Voting::Uniform, 0).label == 0, "tie resolved by rank, not by label value");
    // Unmapped neighbours abstain; none mapped -> failure.
    std::vector<VoteInput> ab = {{-1, 0.99}, {1, 0.5}, {-1, 0.4}};
    auto va = vote(ab, 3, Voting::Weighted, 0.01);
    check(va.ok && va.label == 1 && va.used == 1, "unmapped neighbours abstain (s_max over voters)");
    check(!vote({{-1, 0.9}}, 1, Voting::Uniform, 0).ok, "no mapped neighbour -> abstention");
    check(!vote({}, 3, Voting::Uniform, 0).ok, "empty list -> abstention");
    // Numerical stability: tiny tau, large similarity gaps.
    auto st = vote({{0, 1.0}, {1, -1.0}}, 2, Voting::Weighted, 1e-6);
    check(st.ok && st.label == 0 && std::isfinite(st.total_weight), "weighted vote stable for tiny tau");
    // Determinism
    bool same = true;
    for (int i = 0; i < 5; ++i) same = same && vote(v, 5, Voting::Weighted, 0.02).label == w5.label;
    check(same, "vote deterministic");
    check(KnnParams::from_json(KnnParams{3, Voting::Weighted, 0.02}.to_json()).tau == 0.02, "kNN params round trip");
}

// A query record with fixed stage costs.
static QueryRow row(const std::string& kind, int gt)
{
    QueryRow r;
    r.id = "q"; r.kind = kind; r.group_id = "g"; r.gt = gt; r.n_options = 2;
    r.retrieval_status = "ok";
    r.embed_ms = 5; r.search1_ms = 100; r.searchk_ms = 60; r.vote_ms = 0.01;
    for (int i = 0; i < 5; ++i) {
        NeighbourRow n;
        n.id = "e" + std::to_string(i);
        n.similarity = 0.95 - 0.01 * i;
        n.label = i < 3 ? 1 : 0;
        n.cum_load_ms = 0.1 * (i + 1);
        r.nb.push_back(n);
    }
    auto dec = [](int pred, int gt, double ms, int tok, double margin) {
        Decision d;
        d.status = "ok"; d.pred = pred; d.correct = pred == gt; d.ms = ms; d.tokens = tok; d.margin = margin;
        return d;
    };
    r.s1 = dec(0, gt, 200, 300, 0.5);
    r.r1 = dec(1, gt, 300, 600, 3.0);
    r.two_a = dec(1, gt, 400, 900, 3.5);
    r.two_b = dec(0, gt, 410, 900, 1.0);
    r.rnd = dec(1, gt, 290, 550, 1.0);
    r.rnd_select_ms = 0.1; r.rnd_load_ms = 0.2;
    return r;
}

static void test_costs()
{
    printf("\n[Action and rule costs]\n");
    const QueryRow r = row("noul", 1);
    KnnConfig knn;
    auto z = run_action(r, Action::ZeroShot, knn);
    check(near(z.ms, 200) && z.llm == 1 && z.emb == 0 && !z.correct, "zero_shot = S1 decision");
    auto rn = run_action(r, Action::Random1, knn);
    check(near(rn.ms, 290.3) && rn.llm == 1 && rn.emb == 0 && rn.correct, "random_1shot = draw + load + decision");
    auto n1 = run_action(r, Action::Nearest1, knn);
    check(near(n1.ms, 5 + 100 + 0.1 + 300) && n1.emb == 1 && n1.llm == 1 && n1.tokens == 600, "nearest_1shot = embed + top-1 search + fetch + decision");
    auto n2 = run_action(r, Action::Nearest2A, knn);
    check(near(n2.ms, 5 + 60 + 0.2 + 400) && n2.correct && !n2.fallback, "nearest_2shot = embed + top-K search + 2 fetches + decision");
    auto k1 = run_action(r, Action::Knn1, knn);
    check(near(k1.ms, 5 + 100 + 0.1 + 0.01) && k1.llm == 0 && k1.correct, "knn_1: top-1 search, no LLM");
    auto k5 = run_action(r, Action::Knn5, knn);
    check(near(k5.ms, 5 + 60 + 0.5 + 0.01) && k5.pred == 1, "knn_5: top-K search, five fetches, vote");
    auto kc = run_action(row("choice", 1), Action::Knn1, knn);
    check(kc.failed && kc.terminal == "unsupported", "kNN on choice -> unsupported");

    // Context-limit fallback for 2-shot.
    QueryRow cl = r;
    cl.two_a.status = "context_limit"; cl.two_a.ms = 2.0; cl.two_a.pred = -1;
    auto f = run_action(cl, Action::Nearest2A, knn);
    check(f.fallback && f.correct && near(f.ms, 5 + 60 + 0.2 + 2.0 + 300) && f.llm == 1 && f.terminal == "nearest_1shot",
          "2-shot context limit -> 1-shot fallback, refused attempt paid");
    QueryRow one = r;
    one.nb.resize(1);
    check(run_action(one, Action::Nearest2B, knn).fallback, "2-shot without a second neighbour -> 1-shot fallback");
    QueryRow fail = r;
    fail.r1.status = "inference_error";
    auto fo = run_action(fail, Action::Nearest1, knn);
    check(fo.failed && !fo.correct && fo.pred == -1, "failed decision counts as incorrect");
    {
        DecisionOutput o;
        o.error = "prompt too long: 5000 tokens > n_ctx 4096";
        r1::ConditionResult c;
        r1::fill_condition(o, {"A", "B"}, {"no", "yes"}, 0, 1.0, c);
        check(c.status == "context_limit" && !c.correct, "engine refusal recorded as context_limit");
    }

    // Rules: deciding work is always paid.
    Rule g; g.type = Rule::Type::KnnGate; g.gate_k = 3; g.threshold = 0.9; g.fallback = Action::ZeroShot;
    auto ga = run_rule(r, g, knn);   // top-3 labels 1,1,1 unanimous, sim .95 >= .9 -> kNN
    check(ga.terminal == "knn_3" && ga.llm == 0 && near(ga.ms, 5 + 60 + 0.3 + 0.01), "kNN gate accepted: retrieval only");
    g.threshold = 0.99;
    auto gr = run_rule(r, g, knn);
    check(gr.fallback && gr.terminal == "zero_shot" && near(gr.ms, 5 + 60 + 0.3 + 0.01 + 200) && gr.emb == 1 && gr.llm == 1,
          "kNN gate rejected: retrieval paid, then S1");
    g.fallback = Action::Nearest2A;
    auto g2 = run_rule(r, g, knn);
    check(near(g2.ms, 5 + 60 + 0.3 + 0.01 + 400), "kNN gate -> 2-shot reuses the retrieval");
    Rule m; m.type = Rule::Type::MarginGate; m.threshold = 0.4; m.fallback = Action::Nearest1;
    auto ma = run_rule(r, m, knn);
    check(ma.terminal == "zero_shot" && near(ma.ms, 200) && ma.llm == 1, "margin gate kept S1");
    m.threshold = 0.6;
    auto mr = run_rule(r, m, knn);
    check(mr.fallback && near(mr.ms, 200 + 5 + 100 + 0.1 + 300) && mr.llm == 2 && mr.emb == 1, "margin gate fallback pays S1 + full action");
    Rule sg; sg.type = Rule::Type::SimGate; sg.threshold = 0.99; sg.fallback = Action::ZeroShot;
    auto sr = run_rule(r, sg, knn);
    check(sr.fallback && near(sr.ms, 5 + 100 + 200) && sr.emb == 1, "similarity gate rejected: embed + search + S1");
    sg.threshold = 0.5;
    check(near(run_rule(r, sg, knn).ms, 5 + 100 + 0.1 + 300), "similarity gate accepted: nearest_1shot");
    check(rule_valid(g, "noul") && !rule_valid(g, "choice"), "kNN gate valid only for supported kinds");
    Rule bad; bad.type = Rule::Type::SimGate; bad.fallback = Action::Nearest1;
    check(!rule_valid(bad, "noul"), "similarity gate fallback must not need retrieval");
    Rule badm; badm.type = Rule::Type::MarginGate; badm.fallback = Action::ZeroShot;
    check(!rule_valid(badm, "noul"), "margin gate fallback cannot be S1");
}

static void test_routing()
{
    printf("\n[Routing]\n");
    KnnConfig knn;
    Policy p;
    p.name = "P"; p.family = "adaptive";
    Rule a; a.type = Rule::Type::Fixed; a.action = Action::ZeroShot;
    Rule b; b.type = Rule::Type::KnnGate; b.gate_k = 5; b.threshold = 0.93; b.fallback = Action::Nearest1;
    Rule c; c.type = Rule::Type::MarginGate; c.threshold = 1.25; c.fallback = Action::Knn3;
    p.per_kind = {{"choice", a}, {"noul", b}, {"score", c}};
    std::vector<QueryRow> rows = {row("choice", 0), row("noul", 1), row("score", 1)};
    bool same = true;
    for (int i = 0; i < 3; ++i)
        for (const auto& r : rows) {
            const auto x = run_policy(r, p, knn), y = run_policy(r, p, knn);
            same = same && x.pred == y.pred && x.ms == y.ms && x.terminal == y.terminal;
        }
    check(same, "routing deterministic");
    const Policy back = Policy::from_json(p.to_json());
    check(back.to_json().dump() == p.to_json().dump(), "policy JSON round trip");
    check(Rule::from_json(b.to_json()).threshold == 0.93 && Rule::from_json(b.to_json()).gate_k == 5, "rule round trip keeps parameters");
    auto nr = run_policy(rows[1], p, knn);   // top-5 labels 1,1,1,0,0 not unanimous -> fallback nearest_1shot
    check(nr.fallback && nr.terminal == "nearest_1shot", "kNN-5 gate needs unanimity");
    QueryRow other = row("noul", 1);
    other.kind = "unknown";
    check(run_policy(other, p, knn).failed, "kind without a rule -> failure");

    const auto front = pareto_front({{0.5, 100}, {0.6, 200}, {0.55, 250}, {0.6, 150}, {0.4, 50}, {0.6, 150}});
    check(front == std::vector<size_t>({4, 0, 3}), "Pareto front by latency; dominated and tied points dropped");

    std::vector<Outcome> outs;
    std::vector<const QueryRow*> rp;
    for (const auto& r : rows) { outs.push_back(run_policy(r, p, knn)); rp.push_back(&r); }
    const auto st = summarize(outs, rp);
    check(st.n == 3 && st.routed.at("choice").at("zero_shot") == 1 && st.routed.size() == 3, "routing frequency by kind");
    check(near(st.llm, (1 + 1 + 1) / 3.0), "mean LLM evaluations");
}

static void test_frozen()
{
    printf("\n[Frozen configuration]\n");
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "pjev_test_r4";
    fs::create_directories(dir);
    Policy pa;
    pa.name = "A"; pa.family = "static";
    for (const char* k : {"choice", "noul", "score"}) { Rule r; r.action = Action::Nearest1; pa.per_kind[k] = r; }
    Policy pb = pa;
    pb.name = "B";
    Rule kn; kn.action = Action::Knn1;
    pb.per_kind["noul"] = kn;
    auto write = [&](const json& j) {
        std::ofstream f(dir / "frozen.json", std::ios::binary);
        f << j.dump(2) << "\n";
        f.close();
        std::ifstream in(dir / "frozen.json", std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        std::ofstream(dir / "frozen.sha256") << sha256_data(bytes) << "  frozen.json\n";
    };
    json fz{{"stage", "frozen"}, {"nearest_2shot", {{"order", "B"}}},
            {"knn", {{"knn_1", KnnParams{1, Voting::Uniform, 0}.to_json()}, {"knn_3", KnnParams{3, Voting::Weighted, 0.02}.to_json()},
                     {"knn_5", KnnParams{5, Voting::Uniform, 0}.to_json()}}},
            {"finalist_policies", json::array({pa.to_json(), pb.to_json()})}, {"confirmatory_policies", json::array({"A", "B"})}};
    write(fz);
    FrozenPolicy f;
    std::string err;
    const std::string p = (dir / "frozen.json").string(), sp = (dir / "frozen.sha256").string();
    check(load_frozen(p, sp, f, err), "valid frozen configuration loads");
    check(f.two_shot_order == "B" && f.knn.k3.voting == Voting::Weighted && f.knn.k3.tau == 0.02 && f.finalists.size() == 2,
          "frozen parameters restored");
    check(f.finalists[1].per_kind["noul"].action == Action::Knn1, "frozen rules restored");
    {
        std::ofstream(dir / "frozen.json", std::ios::app) << " ";
        FrozenPolicy g;
        check(!load_frozen(p, sp, g, err) && err.find("sha256") != std::string::npos, "modified after freezing -> refused");
    }
    json bad = fz;
    bad["stage"] = "tuning";
    write(bad);
    FrozenPolicy g1;
    check(!load_frozen(p, sp, g1, err), "not marked frozen -> refused");
    bad = fz;
    Policy pc = pa;
    pc.name = "C";
    Rule ch; ch.action = Action::Knn3;
    pc.per_kind["choice"] = ch;
    bad["finalist_policies"].push_back(pc.to_json());
    write(bad);
    FrozenPolicy g2;
    check(!load_frozen(p, sp, g2, err), "kNN for choice -> refused");
    bad = fz;
    Policy pd = pa;
    pd.name = "D";
    Rule two; two.action = Action::Nearest2A;   // order B was selected
    pd.per_kind["noul"] = two;
    bad["finalist_policies"].push_back(pd.to_json());
    write(bad);
    FrozenPolicy g3;
    check(!load_frozen(p, sp, g3, err), "unselected 2-shot order -> refused");
    bad = fz;
    bad["confirmatory_policies"].push_back("Z");
    write(bad);
    FrozenPolicy g4;
    check(!load_frozen(p, sp, g4, err), "confirmatory policy not among finalists -> refused");
    FrozenPolicy g5;
    check(!load_frozen((dir / "missing.json").string(), "", g5, err), "missing file -> refused");
    fs::remove_all(dir);
}

static void test_records()
{
    printf("\n[Records and statistics]\n");
    json j = json::parse(R"({"query_id":"x","group_id":"g1","kind":"noul","options_count":2,"ground_truth_index":1,
        "retrieval":{"status":"ok","embed_ms":5,"search1_ms":100,"searchk_ms":60,"vote_ms":0.01,
          "neighbours":[{"example_id":"00ab","hamming_distance":90,"estimated_cosine":0.93,"label":1,"map_status":"ok","cum_load_ms":0.1}]},
        "s1":{"status":"ok","prediction_index":0,"correct":false,"prompt_tokens":300,"latency_ms":200,"margin":0.4},
        "r1":{"status":"ok","prediction_index":1,"correct":true,"prompt_tokens":600,"latency_ms":300,"margin":2},
        "two_shot_a":{"status":"ok","prediction_index":1,"correct":true,"prompt_tokens":900,"latency_ms":400},
        "two_shot_b":{"status":"context_limit","prompt_tokens":0,"latency_ms":1},
        "random_1shot":{"status":"ok","prediction_index":1,"correct":true,"prompt_tokens":500,"latency_ms":290,"select_ms":0.1,"load_ms":0.2}})");
    QueryRow r;
    std::string err;
    check(parse_query_row(j, r, err), "record parses");
    check(r.group_id == "g1" && r.gt == 1 && r.nb.size() == 1 && r.nb[0].label == 1 && near(r.nb[0].similarity, 0.93),
          "fields restored");
    check(r.s1.margin == 0.4 && r.two_b.status == "context_limit" && r.rnd_load_ms == 0.2, "decision fields restored");
    KnnConfig knn;
    check(run_action(r, Action::Nearest2B, knn).fallback, "parsed context-limit record falls back");
    json bad = j;
    bad.erase("ground_truth_index");
    QueryRow r2;
    check(!parse_query_row(bad, r2, err), "missing ground truth -> parse error");
    json noret = j;
    noret.erase("retrieval");
    check(!parse_query_row(noret, r2, err), "missing retrieval block -> parse error");

    // Statistics: summarize percentiles; cluster bootstrap keeps clusters together.
    std::vector<Outcome> o(4);
    for (int i = 0; i < 4; ++i) { o[i].ms = 100.0 * (i + 1); o[i].correct = i % 2 == 0; }
    const auto st = summarize(o, {});
    check(st.correct == 2 && near(st.mean_ms, 250) && near(st.p50_ms, 250) && st.p99_ms <= 400 && st.p99_ms > 390, "summary statistics");
    std::vector<std::string> cl = {"a", "a", "b", "b", "c", "c"};
    std::vector<double> x = {1, 1, 0, 0, 1, 1}, y = {0, 0, 0, 0, 1, 1};
    r3::Resampler rs(cl, 500, 7);
    const auto ci = r3::paired_ci(rs, x, y);
    check(rs.clusters() == 3 && near(ci.diff, 1.0 / 3.0) && ci.lo >= 0.0 && ci.hi <= 1.0, "paired cluster bootstrap");
    const auto ci0 = r3::paired_ci(rs, x, x);
    check(ci0.diff == 0 && ci0.lo == 0 && ci0.hi == 0, "identical strategies -> zero interval");
    r3::Resampler rs2(cl, 500, 7);
    check(r3::paired_ci(rs2, x, y).lo == ci.lo, "bootstrap deterministic for a fixed seed");
}

int main()
{
    test_neighbours();
    test_eligibility();
    test_prompt_order();
    test_mapping();
    test_vote();
    test_costs();
    test_routing();
    test_frozen();
    test_records();
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
