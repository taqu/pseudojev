// Unit tests for task-aware routing (no model, no database): kNN voting and label mapping, task
// mapping, the neighbour rule, per-kind dispatch, fallbacks and error codes, and the API path.

#include "decision/decision_engine.h"
#include "decision/decision_service.h"
#include "mock_backend.h"
#include "retrieval/exemplar_store.h"
#include "routing/knn.h"
#include "routing/task_mapping.h"
#include "routing/task_policy.h"
#include "routing/task_router.h"
#include "server/jev_api.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pjev;
using namespace pjev::routing;
using retrieval::ExemplarPayload;
using retrieval::SearchHit;
using json = nlohmann::ordered_json;

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); g_failures++; } \
    else         { printf("pass: %s\n", msg); } \
} while (0)

static bool near(double a, double b, double eps = 1e-12) { return std::fabs(a - b) <= eps; }

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

struct FakeEmbedder : public ITextEmbedder {
    std::vector<std::string> texts;
    bool fail = false;
    bool encode(const std::string& text, std::vector<float>& out) override {
        texts.push_back(text);
        if (fail) return false;
        out.assign(8, 0.5f);
        return true;
    }
    int dimension() const override { return 768; }
};

struct FakeIndex : public retrieval::IExemplarIndex {
    std::map<int64_t, ExemplarPayload> rows;
    std::set<int64_t> allowed;
    std::vector<SearchHit> top1, topk;  // returned for top_k == 1 / top_k > 1
    bool fail_search = false;
    mutable int searches_top1 = 0, searches_topk = 0, loads = 0;
    mutable std::vector<std::pair<std::string, int>> filters;  // (kind, levels) per search

    bool search(const std::string& kind, int levels, const std::vector<float>&, int top_k,
                std::vector<SearchHit>& out, std::string& err) const override {
        filters.push_back({kind, levels});
        if (fail_search) { err = "exemplar search failed: disk I/O error"; return false; }
        (top_k == 1 ? searches_top1 : searches_topk)++;
        out.clear();
        for (const auto& h : top_k == 1 ? top1 : topk) {
            auto it = rows.find(h.id);
            const bool kind_ok = it == rows.end() ||
                (it->second.kind == kind && (kind != "score" || it->second.number_of_levels == levels));
            if (kind_ok && static_cast<int>(out.size()) < top_k) out.push_back(h);
        }
        return true;
    }
    bool load(int64_t id, ExemplarPayload& out, std::string& err) const override {
        loads++;
        auto it = rows.find(id);
        if (it == rows.end()) { err = "exemplar not found"; return false; }
        out = it->second;
        return true;
    }
    bool is_allowed(int64_t id) const override { return allowed.count(id) > 0; }

    void add(const ExemplarPayload& p, bool allow = true) {
        rows[p.id] = p;
        if (allow) allowed.insert(p.id);
    }
};

// Counts evaluations, records the prompt of each, and refuses prompts longer than max_tokens.
class RecordingBackend : public MockBackend {
public:
    int evals = 0;
    size_t max_tokens = 1u << 30;
    std::vector<std::string> prompts;
    const float* eval_tokens(const std::vector<int>& tokens) override {
        if (tokens.size() > max_tokens) throw std::runtime_error("prompt too long (" + std::to_string(tokens.size()) + " tokens)");
        evals++;
        std::string text;
        for (int t : tokens) text += t >= 1000 ? std::string(1, (char)(t - 1000)) : token_to_piece(t);
        prompts.push_back(text);
        return MockBackend::eval_tokens(tokens);
    }
};

static ExemplarPayload payload(int64_t id, const std::string& kind, const std::string& question,
                               std::vector<std::string> options, int answer, json state = json::object())
{
    ExemplarPayload p;
    p.id = id;
    p.kind = kind;
    p.question = question;
    p.options = std::move(options);
    p.answer_index = answer;
    p.number_of_levels = kind == "score" ? static_cast<int>(p.options.size()) : 0;
    p.state = state.is_object() && state.empty() ? json{{"policy_name", "policy " + question}} : state;
    return p;
}

static DecisionInput choice_input()
{
    DecisionInput in;
    in.type = Type::Choice;
    in.state = R"({"policy_name":"query policy"})";
    in.question = "QUERY-CHOICE?";
    in.options = {{"item_1", "item_1"}, {"item_2", "item_2"}, {"abstain", "abstain"}};
    return in;
}

static DecisionInput noul_input(bool true_first)
{
    DecisionInput in;
    in.type = Type::Noul;
    in.state = "plain text state";
    in.question = "QUERY-NOUL?";
    in.options = true_first ? std::vector<std::pair<std::string, std::string>>{{"true", ""}, {"false", ""}}
                            : std::vector<std::pair<std::string, std::string>>{{"false", ""}, {"true", ""}};
    return in;
}

static DecisionInput score_input(int levels)
{
    DecisionInput in;
    in.type = Type::Score;
    in.state = R"({"policy_name":"rubric"})";
    in.question = "QUERY-SCORE?";
    for (int i = 0; i < levels; ++i) in.options.push_back({std::to_string(i), "level " + std::to_string(i)});
    return in;
}

// ---------------------------------------------------------------------------
// kNN pieces
// ---------------------------------------------------------------------------

static void test_similarity()
{
    CHECK(near(estimated_cosine(0, 768), 1.0), "similarity: hamming 0 -> 1");
    CHECK(near(estimated_cosine(384, 768), 0.0, 1e-15), "similarity: hamming dim/2 -> 0");
    CHECK(estimated_cosine(84, 768) == 0.9415440651830208, "similarity: hamming 84 matches the R4 record");
    CHECK(estimated_cosine(10, 0) == 0.0, "similarity: dim 0 -> 0");
}

static void test_vote()
{
    // Weighted: two close votes for label 1 beat one slightly closer vote for label 0.
    std::vector<VoteInput> v = {{0, 0.950}, {1, 0.948}, {1, 0.947}, {2, 0.900}, {0, 0.890}};
    VoteResult r = vote(v, kKnnK, kKnnTau);
    const double w0 = 1.0 + std::exp((0.890 - 0.950) / kKnnTau);
    const double w1 = std::exp((0.948 - 0.950) / kKnnTau) + std::exp((0.947 - 0.950) / kKnnTau);
    CHECK(r.ok && r.used == 5, "vote: all five neighbours vote");
    CHECK(r.label == (w1 > w0 ? 1 : 0), "vote: weighted winner");
    CHECK(near(r.weight[0], w0) && near(r.weight[1], w1), "vote: weights exp((s - s_max) / tau)");
    CHECK(near(r.total_weight, w0 + w1 + std::exp((0.900 - 0.950) / kKnnTau)), "vote: total weight");

    // Uniform majority would pick 0 here; the weights pick the nearer label 1.
    std::vector<VoteInput> u = {{1, 0.97}, {0, 0.95}, {0, 0.95}};
    CHECK(vote(u, 3, 0.0).label == 0, "vote: tau 0 is uniform majority");
    CHECK(vote(u, 3, kKnnTau).label == 1, "vote: weighting overrides the uniform majority");

    // Exact tie -> best-ranked neighbour's label.
    std::vector<VoteInput> t = {{2, 0.95}, {1, 0.95}};
    CHECK(vote(t, 5, kKnnTau).label == 2, "vote: tie goes to the best-ranked label");
    std::vector<VoteInput> t2 = {{1, 0.95}, {2, 0.95}};
    CHECK(vote(t2, 5, kKnnTau).label == 1, "vote: tie order follows rank, not label value");

    // Unmapped neighbours abstain; s_max is taken over voters only.
    std::vector<VoteInput> a = {{-1, 0.99}, {3, 0.90}};
    VoteResult ra = vote(a, 5, kKnnTau);
    CHECK(ra.ok && ra.label == 3 && ra.used == 1 && near(ra.weight[3], 1.0), "vote: unmapped abstains");
    CHECK(!vote({{-1, 0.9}}, 5, kKnnTau).ok, "vote: no mapped neighbour -> not ok");
    CHECK(!vote({}, 5, kKnnTau).ok, "vote: empty -> not ok");
    // Only the first k vote.
    std::vector<VoteInput> k = {{0, 0.9}, {1, 0.95}, {1, 0.95}};
    CHECK(vote(k, 1, kKnnTau).label == 0, "vote: k limits the voters");
}

static void test_map_label()
{
    const std::vector<std::string> noul_opts = {"no", "yes"};
    LabelMap m = map_label("noul", noul_opts, {0, 1}, "noul", {"no", "yes"}, 1);
    CHECK(m.status == MapStatus::Ok && m.candidate == 1, "map: noul yes -> candidate 1");
    m = map_label("noul", noul_opts, {0, 1}, "noul", {"yes", "no"}, 1);
    CHECK(m.status == MapStatus::Ok && m.candidate == 0, "map: noul by answer text, not position");
    m = map_label("score", {"a", "b", "c"}, {0, 1, 2}, "score", {"x", "y", "z"}, 2);
    CHECK(m.status == MapStatus::Ok && m.candidate == 2, "map: score by level index");
    m = map_label("score", {"a", "b", "c"}, {0, 1, 2}, "score", {"x", "y"}, 1);
    CHECK(m.status == MapStatus::NoMatch, "map: score level count mismatch");
    m = map_label("choice", {"a", "b"}, {0, 1}, "choice", {"a", "b"}, 0);
    CHECK(m.status == MapStatus::Unsupported, "map: choice unsupported");
    m = map_label("score", {"a", "b"}, {0, 1}, "score", {"a", "b"}, -1);
    CHECK(m.status == MapStatus::NoMatch, "map: no one-hot answer");
}

// ---------------------------------------------------------------------------
// Task mapping
// ---------------------------------------------------------------------------

static void test_route_task()
{
    RoutedTask t;
    std::string err;
    CHECK(route_task(choice_input(), t, err), "route: choice ok");
    CHECK(t.retrieval.kind == "choice" && t.retrieval.options == std::vector<std::string>({"item_1", "item_2", "abstain"}),
          "route: choice options are the keys");
    CHECK(t.to_caller == std::vector<int>({0, 1, 2}) && t.decision.options == choice_input().options,
          "route: choice keeps the caller order");
    CHECK(t.retrieval.state.is_object() && t.retrieval.state["policy_name"] == "query policy",
          "route: JSON object state is structured");
    CHECK(t.decision.collect_corrected_logits && !t.decision.prior_correction, "route: raw logits, no prior correction");

    CHECK(route_task(score_input(5), t, err), "route: score ok");
    CHECK(t.retrieval.number_of_levels == 5 && t.retrieval.options[3] == "level 3", "route: score options are the level texts");

    CHECK(route_task(noul_input(true), t, err), "route: noul ok");
    CHECK(t.decision.options[0].first == "false" && t.decision.options[1].first == "true",
          "route: noul candidates reordered to (false, true)");
    CHECK(t.to_caller == std::vector<int>({1, 0}), "route: noul maps back to the caller order");
    CHECK(t.retrieval.options == std::vector<std::string>({"no", "yes"}), "route: noul options no/yes");
    CHECK(t.retrieval.state.is_string() && t.retrieval.state.get<std::string>() == "plain text state",
          "route: plain-text state stays text");

    DecisionInput arr = noul_input(false);
    arr.state = "[1, 2]";
    CHECK(route_task(arr, t, err) && t.retrieval.state.is_string(), "route: JSON array state stays text");

    DecisionInput bad = noul_input(false);
    bad.options = {{"yes", ""}, {"no", ""}};
    CHECK(!route_task(bad, t, err), "route: noul requires false/true keys");
    DecisionInput one = choice_input();
    one.options.resize(1);
    CHECK(!route_task(one, t, err), "route: at least 2 candidates");
    DecisionInput unk = choice_input();
    unk.type = Type::Unknown;
    CHECK(!route_task(unk, t, err), "route: unknown type rejected");
}

static void test_content_and_text()
{
    const uint64_t a = content_key("noul", "Q", "S", {"no", "yes"});
    CHECK(a == content_key("noul", "Q", "S", {"no", "yes"}), "content: deterministic");
    CHECK(a != content_key("noul", "Q", "S", {"yes", "no"}), "content: option order matters");
    CHECK(content_key("choice", "ab", "c", {}) != content_key("choice", "a", "bc", {}), "content: length-prefixed fields");
    CHECK(state_text(json("x")) == "x", "state_text: string verbatim");
    CHECK(state_text(json::parse(R"({"b":1,"a":"é"})")) == "{\"b\":1,\"a\":\"\xc3\xa9\"}",
          "state_text: compact JSON, key order kept, UTF-8 unescaped");

    RoutedTask t;
    std::string err;
    DecisionInput in = noul_input(false);
    in.domain = "support";
    in.family = "policy";
    route_task(in, t, err);
    const std::string text = embedding_query_text(t.retrieval);
    CHECK(text.rfind("search_query: Task: noul\nDomain: support\nFamily: policy\n", 0) == 0, "embedding text: prefix and header");
    CHECK(text.find("QUERY-NOUL?") != std::string::npos && text.find("Context:\nplain text state") != std::string::npos,
          "embedding text: question and context");
}

static void test_build_demonstration()
{
    Demonstration d;
    std::string err;
    CHECK(build_demonstration(payload(1, "noul", "q", {"yes", "no"}, 0), d, err), "demo: noul ok");
    CHECK(d.options.size() == 2 && d.options[0].first == "false" && d.answer_index == 1,
          "demo: noul answer 'yes' -> candidate true");
    CHECK(d.state == "{\"policy_name\":\"policy q\"}", "demo: structured state as compact JSON");
    CHECK(build_demonstration(payload(2, "score", "q", {"lo", "mid", "hi"}, 2), d, err) &&
          d.options[2].first == "2" && d.options[2].second == "hi" && d.answer_index == 2, "demo: score");
    CHECK(build_demonstration(payload(3, "choice", "q", {"x", "y"}, 1), d, err) &&
          d.options[1].first == "y" && d.options[1].second == "y", "demo: choice option is key and text");
    CHECK(!build_demonstration(payload(4, "choice", "q", {"x", "y"}, -1), d, err), "demo: no one-hot answer fails");
    CHECK(!build_demonstration(payload(5, "noul", "q", {"a", "b"}, 0), d, err), "demo: noul without no/yes fails");
}

// ---------------------------------------------------------------------------
// Neighbour rule
// ---------------------------------------------------------------------------

static void test_neighbour_list()
{
    RoutedTask t;
    std::string err;
    route_task(choice_input(), t, err);
    FakeIndex idx;
    idx.add(payload(1, "choice", "q1", {"a", "b"}, 0));
    idx.add(payload(2, "choice", "q1", {"a", "b"}, 1));  // same content as 1 (answer differs)
    idx.add(payload(3, "choice", "q3", {"a", "b"}, 0));
    ExemplarPayload self = payload(4, "choice", t.retrieval.question, t.retrieval.options, 0,
                                   json::parse(t.retrieval.state_text));
    idx.add(self);
    idx.add(payload(5, "choice", "q5", {"a", "b"}, 0));

    NeighbourList nl(t.retrieval, idx, 768);
    CHECK(nl.extend({{1, 80}}, 3, err) && nl.neighbours().size() == 1, "neighbours: rank 1 from the top-1 search");
    CHECK(nl.extend({{1, 80}, {4, 81}, {2, 82}, {3, 83}, {5, 84}}, 3, err), "neighbours: extend with top-K");
    const auto& n = nl.neighbours();
    CHECK(n.size() == 3 && n[0].id == 1 && n[1].id == 3 && n[2].id == 5,
          "neighbours: repeated ID, query content and duplicate content skipped");
    CHECK(near(n[1].similarity, estimated_cosine(83, 768)) && n[1].hamming == 83, "neighbours: similarity from hamming");
    CHECK(idx.loads == 5, "neighbours: stops loading once enough are picked");

    FakeIndex bad;
    bad.add(payload(7, "noul", "q", {"no", "yes"}, 0));  // wrong kind for a choice query
    NeighbourList nl2(t.retrieval, bad, 768);
    CHECK(!nl2.extend({{7, 10}}, 1, err) && err.find("not eligible") != std::string::npos, "neighbours: eligibility re-checked");
    FakeIndex notallowed;
    notallowed.add(payload(8, "choice", "q", {"a", "b"}, 0), false);
    NeighbourList nl3(t.retrieval, notallowed, 768);
    CHECK(!nl3.extend({{8, 10}}, 1, err), "neighbours: allowlist re-checked");
    FakeIndex missing;
    NeighbourList nl4(t.retrieval, missing, 768);
    CHECK(!nl4.extend({{9, 10}}, 1, err) && err.find("not found") != std::string::npos, "neighbours: payload error propagates");

    RoutedTask s;
    route_task(score_input(5), s, err);
    FakeIndex lv;
    lv.add(payload(10, "score", "q", {"a", "b", "c"}, 0));  // 3 levels for a 5-level query
    NeighbourList nl5(s.retrieval, lv, 768);
    CHECK(!nl5.extend({{10, 10}}, 1, err), "neighbours: score level count re-checked");
}

// ---------------------------------------------------------------------------
// Router dispatch
// ---------------------------------------------------------------------------

struct Fixture {
    RecordingBackend backend;
    DecisionEngine engine;
    FakeEmbedder embedder;
    FakeIndex index;
    TaskRouter router;
    Fixture()
        : engine(backend, PromptConfig{Layout::AUTO, Scheme::LETTERS}, CalibrationConfig{},
                 EnsembleConfig{NoulEnsembleMode::NONE, ChoiceEnsembleMode::NONE, ScoreEnsembleMode::NONE})
        , router(engine, embedder, index) {}
};

static void test_choice_route()
{
    Fixture f;
    f.index.add(payload(11, "choice", "NEAREST-Q", {"a", "b"}, 1));
    f.index.add(payload(12, "choice", "SECOND-Q", {"a", "b", "c"}, 2));
    f.index.add(payload(13, "choice", "THIRD-Q", {"a", "b"}, 0));
    f.index.top1 = {{11, 80}};
    f.index.topk = {{12, 80}, {11, 80}, {13, 90}};  // the top-K search ranks another tied exemplar first
    f.backend.set_winner('C');

    DecisionOutput out = f.router.decide(choice_input());
    CHECK(out.ok, "choice: ok");
    CHECK(out.routing && out.routing->strategy == kNearest2Shot && !out.routing->fallback, "choice: nearest 2-shot");
    CHECK(out.routing->demonstration_ids == std::vector<int64_t>({12, 11}), "choice: order A = second-nearest, nearest");
    CHECK(out.routing->neighbour_ids == std::vector<int64_t>({11, 12}), "choice: rank 1 is the top-1 search result");
    CHECK(f.index.searches_top1 == 1 && f.index.searches_topk == 1, "choice: top-1 and top-K searches");
    CHECK(f.index.filters[0] == std::make_pair(std::string("choice"), 0), "choice: kind filter");
    CHECK(f.backend.evals == 1, "choice: one inference evaluation");
    const std::string& p = f.backend.prompts.back();
    const size_t second = p.find("SECOND-Q"), nearest = p.find("NEAREST-Q"), query = p.find("QUERY-CHOICE?");
    CHECK(second != std::string::npos && second < nearest && nearest < query, "choice: prompt order second, nearest, query");
    CHECK(p.find("THIRD-Q") == std::string::npos, "choice: only two demonstrations");
    CHECK(out.selected == 2 && out.keys == std::vector<std::string>({"item_1", "item_2", "abstain"}),
          "choice: candidate-logit argmax in caller order");
    CHECK(out.raw_logits.size() == 3 && out.raw_logits[2] == 100.0f, "choice: raw candidate logits reported");
    CHECK(f.embedder.texts.size() == 1 && f.embedder.texts[0].rfind("search_query: Task: choice", 0) == 0,
          "choice: one query embedding");

    // Deterministic
    DecisionOutput again = f.router.decide(choice_input());
    CHECK(again.selected == out.selected && again.raw_logits == out.raw_logits &&
          again.routing->demonstration_ids == out.routing->demonstration_ids && f.backend.prompts[0] == f.backend.prompts[1],
          "choice: deterministic");
}

static void test_choice_fallbacks()
{
    {   // Only one neighbour -> 1-shot with the nearest
        Fixture f;
        f.index.add(payload(21, "choice", "ONLY-Q", {"a", "b"}, 0));
        f.index.top1 = {{21, 70}};
        f.index.topk = {{21, 70}};
        DecisionOutput out = f.router.decide(choice_input());
        CHECK(out.ok && out.routing->fallback && out.routing->demonstration_ids == std::vector<int64_t>({21}),
              "choice fallback: fewer than two neighbours -> nearest 1-shot");
        CHECK(f.backend.evals == 1, "choice fallback: one evaluation");
    }
    {   // 2-shot exceeds the context -> 1-shot
        Fixture f;
        f.index.add(payload(31, "choice", "NEAREST-Q", {"a", "b"}, 0));
        f.index.add(payload(32, "choice", "SECOND-Q", {"a", "b"}, 1));
        f.index.top1 = {{31, 70}};
        f.index.topk = {{31, 70}, {32, 71}};
        RecordingBackend probe;  // measure the 1-shot prompt size
        {
            DecisionEngine e(probe, PromptConfig{Layout::AUTO, Scheme::LETTERS}, CalibrationConfig{},
                             EnsembleConfig{NoulEnsembleMode::NONE, ChoiceEnsembleMode::NONE, ScoreEnsembleMode::NONE});
            Demonstration d;
            std::string err;
            build_demonstration(f.index.rows[31], d, err);
            RoutedTask t;
            route_task(choice_input(), t, err);
            f.backend.max_tokens = static_cast<size_t>(e.decide_with_demonstration(t.decision, d).prompt_token_count);
        }
        DecisionOutput out = f.router.decide(choice_input());
        CHECK(out.ok && out.routing->fallback && out.routing->demonstration_ids == std::vector<int64_t>({31}),
              "choice fallback: 2-shot over the context -> nearest 1-shot");
        f.backend.max_tokens = 10;
        out = f.router.decide(choice_input());
        CHECK(!out.ok && out.error_code == ErrorCode::CONTEXT_TOO_LONG, "choice: 1-shot over the context -> context_too_long");
    }
}

static void test_noul_route()
{
    Fixture f;
    f.index.add(payload(41, "noul", "NEAREST-Q", {"no", "yes"}, 1));
    f.index.add(payload(42, "noul", "OTHER-Q", {"no", "yes"}, 0));
    f.index.top1 = {{41, 60}};
    f.index.topk = {{42, 50}, {41, 60}};
    f.backend.set_winner('B');  // candidate B = true in the (false, true) order

    DecisionOutput out = f.router.decide(noul_input(true));  // caller order (true, false)
    CHECK(out.ok && out.routing->strategy == kNearest1Shot, "noul: nearest 1-shot");
    CHECK(out.routing->demonstration_ids == std::vector<int64_t>({41}), "noul: the top-1 search result");
    CHECK(f.index.searches_top1 == 1 && f.index.searches_topk == 0, "noul: no top-K search");
    CHECK(f.backend.evals == 1, "noul: one inference evaluation");
    const std::string& p = f.backend.prompts.back();
    CHECK(p.find("NEAREST-Q") < p.find("QUERY-NOUL?") && p.find("OTHER-Q") == std::string::npos, "noul: one demonstration");
    CHECK(p.find("A: No\nB: Yes") != std::string::npos, "noul: query candidates in (false, true) order");
    CHECK(out.keys == std::vector<std::string>({"true", "false"}) && out.selected == 0,
          "noul: output in the caller's candidate order");
    CHECK(out.probs.size() == 2 && out.probs[0] > 0.99 && near(out.p_true, out.probs[0]), "noul: p_true is the true candidate");

    // Nearest exemplar identical to the query -> skipped, next from the top-K search.
    Fixture g;
    RoutedTask t;
    std::string err;
    route_task(noul_input(false), t, err);
    g.index.add(payload(51, "noul", t.retrieval.question, {"no", "yes"}, 1, json(t.retrieval.state_text)));
    g.index.add(payload(52, "noul", "NEXT-Q", {"no", "yes"}, 0));
    g.index.top1 = {{51, 0}};
    g.index.topk = {{51, 0}, {52, 40}};
    DecisionOutput o2 = g.router.decide(noul_input(false));
    CHECK(o2.ok && o2.routing->demonstration_ids == std::vector<int64_t>({52}) && g.index.searches_topk == 1,
          "noul: query content skipped, top-K consulted");
}

static void test_score_route()
{
    Fixture f;
    // Labels per neighbour (level index): 4, 1, 1, 4, 0 with similarities chosen so weights decide.
    const int labels[5] = {4, 1, 1, 4, 0};
    const double ham[5] = {70, 71, 71, 75, 76};
    for (int i = 0; i < 5; ++i) {
        f.index.add(payload(60 + i, "score", "S" + std::to_string(i), {"l0", "l1", "l2", "l3", "l4"}, labels[i]));
        f.index.topk.push_back({60 + i, ham[i]});
    }
    f.index.add(payload(70, "score", "S5", {"l0", "l1", "l2", "l3", "l4"}, 2));
    f.index.topk.push_back({70, 77});  // sixth: must not vote
    f.index.top1 = {{60, 70}};

    DecisionOutput out = f.router.decide(score_input(5));
    CHECK(out.ok && out.routing->strategy == kKnn5, "score: weighted kNN-5");
    CHECK(f.backend.evals == 0, "score: no inference-model evaluation");
    CHECK(f.index.filters[0] == std::make_pair(std::string("score"), 5), "score: level-count filter");
    CHECK(out.routing->neighbour_ids == std::vector<int64_t>({60, 61, 62, 63, 64}), "score: top-5 neighbours");
    CHECK(out.routing->neighbour_labels == std::vector<int32_t>({4, 1, 1, 4, 0}), "score: semantic level labels");

    std::vector<VoteInput> v;
    for (int i = 0; i < 5; ++i) v.push_back({labels[i], estimated_cosine(ham[i], 768)});
    const VoteResult expect = vote(v, 5, kKnnTau);
    CHECK(out.selected == expect.label, "score: prediction = vote winner");
    double sum = 0, ev = 0;
    for (size_t i = 0; i < out.probs.size(); ++i) { sum += out.probs[i]; ev += i * out.probs[i]; }
    CHECK(out.probs.size() == 5 && near(sum, 1.0) && near(out.probs[4], expect.weight[4] / expect.total_weight),
          "score: probabilities are normalised vote weights");
    CHECK(near(out.expected_score, ev) && out.probs[2] == 0.0, "score: expected level; level 2 (6th neighbour) has no vote");
    CHECK(out.keys.size() == 5 && out.keys[0] == "0", "score: level keys");

    Fixture none;
    DecisionOutput miss = none.router.decide(score_input(7));
    CHECK(!miss.ok && miss.error_code == ErrorCode::NO_ELIGIBLE_EXEMPLAR &&
          miss.error.find("score with 7 levels") != std::string::npos, "score: no eligible exemplar -> no_eligible_exemplar");
    CHECK(none.backend.evals == 0, "score: no fallback inference");
}

static void test_router_errors()
{
    {
        Fixture f;
        f.embedder.fail = true;
        DecisionOutput out = f.router.decide(noul_input(false));
        CHECK(!out.ok && out.error_code == ErrorCode::CONTEXT_TOO_LONG, "errors: embedding failure -> context_too_long");
    }
    {
        Fixture f;
        f.index.fail_search = true;
        DecisionOutput out = f.router.decide(choice_input());
        CHECK(!out.ok && out.error_code == ErrorCode::INFERENCE_ERROR && out.error.find("disk I/O") != std::string::npos,
              "errors: database failure -> inference_error");
    }
    {
        Fixture f;
        f.index.top1 = {{99, 10}};  // not loadable
        f.index.allowed.insert(99);
        DecisionOutput out = f.router.decide(noul_input(false));
        CHECK(!out.ok && out.error_code == ErrorCode::INFERENCE_ERROR, "errors: payload failure -> inference_error");
        CHECK(f.backend.evals == 0, "errors: no answer after a payload failure");
    }
    {
        Fixture f;
        DecisionInput in = choice_input();
        in.type = Type::Unknown;
        DecisionOutput out = f.router.decide(in);
        CHECK(!out.ok && out.error_code == ErrorCode::UNSUPPORTED_PRIMITIVE, "errors: unsupported kind");
        DecisionInput bad = noul_input(false);
        bad.options = {{"maybe", ""}, {"true", ""}};
        out = f.router.decide(bad);
        CHECK(!out.ok && out.error_code == ErrorCode::INVALID_REQUEST, "errors: invalid candidate mapping");
        CHECK(f.embedder.texts.empty(), "errors: nothing embedded for invalid requests");
    }
    {
        Fixture f;  // empty bank
        DecisionOutput out = f.router.decide(choice_input());
        CHECK(!out.ok && out.error_code == ErrorCode::NO_ELIGIBLE_EXEMPLAR && f.backend.evals == 0,
              "errors: empty retrieval -> no_eligible_exemplar, no answer");
    }
}

// ---------------------------------------------------------------------------
// HTTP API through the router
// ---------------------------------------------------------------------------

static void test_api()
{
    Fixture f;
    f.index.add(payload(81, "noul", "N", {"no", "yes"}, 1));
    f.index.top1 = {{81, 50}};
    f.backend.set_winner('B');
    JevApiHandler handler(f.router, "test");
    int32_t status = 0;
    std::string body;
    handler.handle(R"({"state":"s","questions":{"q":{"type":"noul","instructions":"Is it?","domain":"support","family":"policy"}}})",
                   status, body);
    json r = json::parse(body);
    CHECK(status == 200 && r["answers"]["q"]["noul"].get<double>() > 0.99, "api: noul routed");
    CHECK(f.embedder.texts.back().find("Domain: support\nFamily: policy") != std::string::npos,
          "api: domain/family reach retrieval");

    handler.handle(R"({"state":"s","questions":{"q":{"type":"score","instructions":"Rate","criteria":["a","b","c","d","e","f"]}}})",
                   status, body);
    r = json::parse(body);
    CHECK(status == 422 && r["error"]["code"] == "no_eligible_exemplar", "api: no eligible exemplar -> 422");

    f.embedder.fail = true;
    handler.handle(R"({"state":"s","questions":{"q":{"type":"noul","instructions":"Is it?"}}})", status, body);
    r = json::parse(body);
    CHECK(status == 400 && r["error"]["code"] == "context_too_long", "api: context limit -> 400 context_too_long");
    f.embedder.fail = false;

    handler.handle(R"({"state":"s","questions":{"q":{"type":"noul","instructions":"Is it?","domain":3}}})", status, body);
    r = json::parse(body);
    CHECK(status == 400 && r["error"]["code"] == "invalid_request", "api: non-string domain rejected");

    handler.handle(R"({"state":"s","questions":{"q":{"type":"ranking","instructions":"?"}}})", status, body);
    r = json::parse(body);
    CHECK(status == 422 && r["error"]["code"] == "unsupported_primitive", "api: unsupported kind -> 422");
}

int main()
{
    test_similarity();
    test_vote();
    test_map_label();
    test_route_task();
    test_content_and_text();
    test_build_demonstration();
    test_neighbour_list();
    test_choice_route();
    test_choice_fallbacks();
    test_noul_route();
    test_score_route();
    test_router_errors();
    test_api();

    if (g_failures == 0) {
        printf("\nPASS (0 failure(s))\n");
        return 0;
    }
    fprintf(stderr, "\nFAIL (%d failure(s))\n", g_failures);
    return 1;
}
