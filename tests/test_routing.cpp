// Unit tests for task-aware routing (no model, no database): kNN voting and label mapping, task
// mapping, the neighbour rule, per-kind dispatch, the zero-shot fallback and error codes, and the API path.

#include "calibration/calibration.h"
#include "decision/decision_engine.h"
#include "decision/decision_service.h"
#include "mock_backend.h"
#include "retrieval/exemplar_store.h"
#include "routing/knn.h"
#include "routing/task_mapping.h"
#include "routing/task_policy.h"
#include "routing/task_router.h"
#include "server/jev_api.h"

#include <algorithm>
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
using retrieval::LoadResult;
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
    std::set<int64_t> undecodable;      // load() -> Invalid (corrupt payload)
    bool fail_search = false;
    bool fail_load = false;             // load() -> Failed (database error)
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
    LoadResult load(int64_t id, ExemplarPayload& out, std::string& err) const override {
        loads++;
        if (fail_load) { err = "exemplar read failed: database disk image is malformed"; return LoadResult::Failed; }
        if (undecodable.count(id)) { err = "exemplar state_lz4: corrupt frame"; return LoadResult::Invalid; }
        auto it = rows.find(id);
        if (it == rows.end()) { err = "exemplar not found"; return LoadResult::Invalid; }
        out = it->second;
        return LoadResult::Ok;
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
    bool fail = false;    // every evaluation throws a backend error unrelated to the context
    bool varied = false;  // letter logits depend on the prompt length (instead of one fixed winner)
    std::vector<std::string> prompts;
    const float* eval_tokens(const std::vector<int>& tokens) override {
        if (fail) throw std::runtime_error("llama_decode failed");
        if (tokens.size() > max_tokens) throw std::runtime_error("prompt too long (" + std::to_string(tokens.size()) + " tokens)");
        evals++;
        std::string text;
        for (int t : tokens) text += t >= 1000 ? std::string(1, (char)(t - 1000)) : token_to_piece(t);
        prompts.push_back(text);
        if (varied) {
            varied_logits_.assign(600, 0.0f);
            for (size_t i = 0; i < 20; ++i) varied_logits_[65 + i] = static_cast<float>((tokens.size() * (i + 3)) % 11) * 0.75f;
            return varied_logits_.data();
        }
        return MockBackend::eval_tokens(tokens);
    }

private:
    std::vector<float> varied_logits_;
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

    CHECK(nl.offered() == 5 && nl.rejected() == 2, "neighbours: offered / rejected counts");

    // Invalid candidates are skipped (never counted); the next valid one is taken.
    auto skipped = [&](FakeIndex& idx2, const RetrievalTask& q, int64_t bad_id, const char* msg) {
        idx2.add(payload(500, q.kind, "valid", q.kind == "noul" ? std::vector<std::string>{"no", "yes"}
                                               : q.kind == "score" ? q.options : std::vector<std::string>{"a", "b"}, 0));
        NeighbourList l(q, idx2, 768);
        std::string e;
        const bool ok = l.extend({{bad_id, 10}, {500, 11}}, 2, e);
        CHECK(ok && l.neighbours().size() == 1 && l.neighbours()[0].id == 500 && l.rejected() == 1, msg);
    };
    FakeIndex bad;
    bad.add(payload(7, "noul", "q", {"no", "yes"}, 0));  // wrong kind for a choice query
    skipped(bad, t.retrieval, 7, "neighbours: wrong kind skipped");
    FakeIndex notallowed;
    notallowed.add(payload(8, "choice", "q", {"a", "b"}, 0), false);
    skipped(notallowed, t.retrieval, 8, "neighbours: not in the allowlist skipped");
    FakeIndex missing;
    skipped(missing, t.retrieval, 9, "neighbours: missing record skipped");
    FakeIndex corrupt;
    corrupt.add(payload(6, "choice", "q", {"a", "b"}, 0));
    corrupt.undecodable.insert(6);
    skipped(corrupt, t.retrieval, 6, "neighbours: undecodable record skipped");
    FakeIndex soft;
    soft.add(payload(5, "choice", "q", {"a", "b"}, -1));
    skipped(soft, t.retrieval, 5, "neighbours: no one-hot answer skipped");

    RoutedTask s;
    route_task(score_input(5), s, err);
    FakeIndex lv;
    lv.add(payload(10, "score", "q", {"a", "b", "c"}, 0));  // 3 levels for a 5-level query
    skipped(lv, s.retrieval, 10, "neighbours: score level count re-checked");
    FakeIndex unmapped;
    ExemplarPayload odd = payload(11, "score", "q", {"a", "b", "c", "d", "e"}, 0);
    odd.options.pop_back();  // 5 levels stored with 4 options: the answer cannot map onto the query
    unmapped.add(odd);
    skipped(unmapped, s.retrieval, 11, "neighbours: score answer without a level mapping skipped");
    NeighbourList sl(s.retrieval, lv, 768);
    CHECK(sl.extend({{500, 10}}, 1, err) && sl.neighbours()[0].label == 0, "neighbours: score label mapped");

    FakeIndex dberr;
    dberr.add(payload(12, "choice", "q", {"a", "b"}, 0));
    dberr.fail_load = true;
    NeighbourList nl6(t.retrieval, dberr, 768);
    CHECK(!nl6.extend({{12, 10}}, 1, err) && err.find("malformed") != std::string::npos,
          "neighbours: database failure propagates");
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
    CHECK(out.routing->requested_strategy == kNearest2Shot && out.routing->fallback_reason.empty() &&
          out.routing->required_exemplars == 2 && out.routing->valid_exemplars == 2, "choice: routing diagnostics");
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

// The S1 zero-shot answer for `in`, computed by a separate engine with the production configuration.
static DecisionOutput s1_reference(const DecisionInput& in, std::string& prompt, char winner)
{
    RecordingBackend b;
    b.set_winner(winner);
    DecisionEngine e(b, PromptConfig{Layout::AUTO, Scheme::LETTERS}, CalibrationConfig{},
                     EnsembleConfig{NoulEnsembleMode::NONE, ChoiceEnsembleMode::NONE, ScoreEnsembleMode::NONE});
    RoutedTask t;
    std::string err;
    route_task(in, t, err);
    DecisionOutput out = e.decide(t.decision);
    prompt = b.prompts.empty() ? "" : b.prompts.back();
    // Back to the caller's candidate order, as the router reports it.
    DecisionOutput r = out;
    for (size_t c = 0; c < t.to_caller.size(); ++c) {
        const size_t to = static_cast<size_t>(t.to_caller[c]);
        r.probs[to] = out.probs[c];
        r.raw_logits[to] = out.raw_logits[c];
    }
    r.selected = t.to_caller[static_cast<size_t>(out.selected)];
    return r;
}

// out is a zero-shot fallback from `requested`, equal to S1 for `in`.
static void check_zero_shot(const DecisionOutput& out, const RecordingBackend& backend, const DecisionInput& in,
                            const char* requested, const char* reason, int valid, int required, char winner,
                            const std::string& what)
{
    CHECK(out.ok && out.routing && out.routing->strategy == kZeroShot && out.routing->fallback,
          (what + ": S1 zero-shot").c_str());
    CHECK(out.routing && out.routing->requested_strategy == requested && out.routing->fallback_reason == reason &&
          out.routing->valid_exemplars == valid && out.routing->required_exemplars == required &&
          out.routing->demonstration_ids.empty(), (what + ": routing diagnostics").c_str());
    std::string s1_prompt;
    const DecisionOutput s1 = s1_reference(in, s1_prompt, winner);
    CHECK(backend.evals == 1 && !backend.prompts.empty() && backend.prompts.back() == s1_prompt,
          (what + ": one evaluation of the S1 prompt").c_str());
    CHECK(out.selected == s1.selected && out.probs == s1.probs && out.raw_logits == s1.raw_logits &&
          out.prompt_token_count == s1.prompt_token_count, (what + ": prediction equals S1").c_str());
}

static void test_choice_fallbacks()
{
    for (int valid = 0; valid < 2; ++valid) {  // 0 or 1 valid exemplar -> zero-shot, never 1-shot
        Fixture f;
        f.backend.set_winner('B');
        f.index.add(payload(21, "choice", "ONLY-Q", {"a", "b"}, 0));
        if (valid == 1) {
            f.index.top1 = {{21, 70}};
            f.index.topk = {{21, 70}};
        }
        DecisionOutput out = f.router.decide(choice_input());
        check_zero_shot(out, f.backend, choice_input(), kNearest2Shot,
                        valid == 0 ? kReasonNoExemplar : kReasonInsufficientExemplars, valid, 2, 'B',
                        "choice with " + std::to_string(valid) + " valid exemplar(s)");
        CHECK(f.backend.prompts.back().find("ONLY-Q") == std::string::npos, "choice fallback: no demonstration in the prompt");
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
        CHECK(out.ok && out.routing->fallback && out.routing->demonstration_ids == std::vector<int64_t>({31}) &&
              out.routing->strategy == kNearest1Shot && out.routing->fallback_reason == kReasonContextLimit,
              "choice fallback: 2-shot over the context -> nearest 1-shot (R4)");
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

    // No valid exemplar -> zero-shot.
    Fixture z;
    z.backend.set_winner('B');
    check_zero_shot(z.router.decide(noul_input(true)), z.backend, noul_input(true), kNearest1Shot, kReasonNoExemplar,
                    0, 1, 'B', "noul with 0 valid exemplars");
    // Only the query itself in the bank -> zero-shot.
    Fixture q;
    q.backend.set_winner('A');
    q.index.add(payload(51, "noul", t.retrieval.question, {"no", "yes"}, 1, json(t.retrieval.state_text)));
    q.index.top1 = {{51, 0}};
    q.index.topk = {{51, 0}};
    check_zero_shot(q.router.decide(noul_input(false)), q.backend, noul_input(false), kNearest1Shot,
                    kReasonInsufficientExemplars, 0, 1, 'A', "noul with only the query in the bank");
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

    CHECK(out.routing->valid_exemplars == 5 && out.routing->required_exemplars == 5 && !out.routing->fallback,
          "score: routing diagnostics");

    // 0-4 valid exemplars -> zero-shot, never kNN with k < 5.
    for (int valid = 0; valid < 5; ++valid) {
        Fixture g;
        g.backend.set_winner('C');
        for (int i = 0; i < valid; ++i) {
            g.index.add(payload(60 + i, "score", "S" + std::to_string(i), {"l0", "l1", "l2", "l3", "l4"}, labels[i]));
            g.index.topk.push_back({60 + i, ham[i]});
        }
        if (valid > 0) g.index.top1 = {{60, 70}};
        check_zero_shot(g.router.decide(score_input(5)), g.backend, score_input(5), kKnn5,
                        valid == 0 ? kReasonNoExemplar : kReasonInsufficientExemplars, valid, 5, 'C',
                        "score with " + std::to_string(valid) + " valid exemplar(s)");
    }
    Fixture none;  // no 7-level scale in the bank
    none.backend.set_winner('G');
    check_zero_shot(none.router.decide(score_input(7)), none.backend, score_input(7), kKnn5, kReasonNoExemplar, 0, 5,
                    'G', "score with an unknown level count");
}

// Invalid or duplicated exemplars never make up the required count.
static void test_invalid_exemplars()
{
    {   // choice: the second hit duplicates the first's content -> one valid exemplar -> zero-shot
        Fixture f;
        f.index.add(payload(101, "choice", "SAME-Q", {"a", "b"}, 0));
        f.index.add(payload(102, "choice", "SAME-Q", {"a", "b"}, 1));
        f.index.top1 = {{101, 50}};
        f.index.topk = {{101, 50}, {102, 51}};
        DecisionOutput out = f.router.decide(choice_input());
        check_zero_shot(out, f.backend, choice_input(), kNearest2Shot, kReasonInsufficientExemplars, 1, 2, 'A',
                        "choice with a duplicated exemplar");
        CHECK(out.routing->rejected_exemplars == 1, "choice duplicate: counted as rejected");
    }
    {   // choice: the same ID twice -> zero-shot
        Fixture f;
        f.index.add(payload(103, "choice", "Q", {"a", "b"}, 0));
        f.index.top1 = {{103, 50}};
        f.index.topk = {{103, 50}, {103, 50}};
        check_zero_shot(f.router.decide(choice_input()), f.backend, choice_input(), kNearest2Shot,
                        kReasonInsufficientExemplars, 1, 2, 'A', "choice with a repeated ID");
    }
    {   // score: one of five is undecodable, one has no one-hot answer, one is not allowlisted -> zero-shot
        Fixture f;
        for (int i = 0; i < 5; ++i) {
            f.index.add(payload(110 + i, "score", "S" + std::to_string(i), {"l0", "l1", "l2", "l3", "l4"}, i),
                        i != 4);
            f.index.topk.push_back({110 + i, 70.0 + i});
        }
        f.index.undecodable.insert(111);
        f.index.rows[112].answer_index = -1;
        f.index.top1 = {{110, 70}};
        DecisionOutput out = f.router.decide(score_input(5));
        check_zero_shot(out, f.backend, score_input(5), kKnn5, kReasonInsufficientExemplars, 2, 5, 'A',
                        "score with invalid exemplars");
        CHECK(out.routing->rejected_exemplars == 3 && out.routing->neighbour_ids == std::vector<int64_t>({110, 113}),
              "score invalid: only valid exemplars listed");
    }
    {   // score: an invalid exemplar is replaced by the next valid one in rank order -> kNN-5
        Fixture f;
        for (int i = 0; i < 6; ++i) {
            f.index.add(payload(120 + i, "score", "S" + std::to_string(i), {"l0", "l1", "l2", "l3", "l4"}, i % 5));
            f.index.topk.push_back({120 + i, 70.0 + i});
        }
        f.index.undecodable.insert(122);
        f.index.top1 = {{120, 70}};
        DecisionOutput out = f.router.decide(score_input(5));
        CHECK(out.ok && out.routing->strategy == kKnn5 && !out.routing->fallback && f.backend.evals == 0 &&
              out.routing->neighbour_ids == std::vector<int64_t>({120, 121, 123, 124, 125}) &&
              out.routing->rejected_exemplars == 1, "score: invalid exemplar skipped, next valid one votes");
    }
    {   // noul: the only hit is returned by the search but not in the allowlist -> zero-shot
        Fixture f;
        f.index.add(payload(130, "noul", "N", {"no", "yes"}, 0), false);
        f.index.top1 = {{130, 10}};
        f.index.topk = {{130, 10}};
        check_zero_shot(f.router.decide(noul_input(false)), f.backend, noul_input(false), kNearest1Shot,
                        kReasonInsufficientExemplars, 0, 1, 'A', "noul with an ineligible exemplar");
    }
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
        f.index.add(payload(99, "noul", "N", {"no", "yes"}, 0));
        f.index.top1 = {{99, 10}};
        f.index.fail_load = true;
        DecisionOutput out = f.router.decide(noul_input(false));
        CHECK(!out.ok && out.error_code == ErrorCode::INFERENCE_ERROR && out.error.find("malformed") != std::string::npos,
              "errors: database read failure -> inference_error");
        CHECK(f.backend.evals == 0, "errors: no zero-shot answer after a database failure");
    }
    {
        Fixture f;
        f.index.fail_search = true;
        DecisionOutput out = f.router.decide(score_input(5));
        CHECK(!out.ok && out.error_code == ErrorCode::INFERENCE_ERROR && f.backend.evals == 0,
              "errors: score search failure -> inference_error, no zero-shot answer");
    }
    {   // Backend failure on the normal path is not turned into a fallback.
        Fixture f;
        f.index.add(payload(97, "choice", "A-Q", {"a", "b"}, 0));
        f.index.add(payload(98, "choice", "B-Q", {"a", "b"}, 1));
        f.index.top1 = {{97, 10}};
        f.index.topk = {{97, 10}, {98, 11}};
        f.backend.fail = true;
        DecisionOutput out = f.router.decide(choice_input());
        CHECK(!out.ok && out.error_code == ErrorCode::INFERENCE_ERROR && out.routing &&
              out.routing->strategy == kNearest2Shot && !out.routing->fallback,
              "errors: inference failure on 2-shot -> inference_error, no fallback");
    }
    {   // A failing fallback reports through the usual error codes.
        Fixture f;
        f.backend.fail = true;
        DecisionOutput out = f.router.decide(noul_input(false));
        CHECK(!out.ok && out.error_code == ErrorCode::INFERENCE_ERROR && out.routing &&
              out.routing->strategy == kZeroShot, "errors: zero-shot inference failure -> inference_error");
        f.backend.fail = false;
        f.backend.max_tokens = 3;
        out = f.router.decide(noul_input(false));
        CHECK(!out.ok && out.error_code == ErrorCode::CONTEXT_TOO_LONG, "errors: zero-shot over the context -> context_too_long");
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
        Fixture f;  // empty bank: not an error
        DecisionOutput out = f.router.decide(choice_input());
        CHECK(out.ok && out.routing->strategy == kZeroShot, "errors: empty retrieval -> zero-shot, not an error");
    }
}

// ---------------------------------------------------------------------------
// Calibration (R5)
// ---------------------------------------------------------------------------

// The supplied parameters, for the production (letters) formulation.
static CalibrationArtifact letters_calibration(bool prior_correction)
{
    CalibrationArtifact a;
    a.formulation        = {"auto", "letters", prior_correction};
    a.choice_temperature = 2.046567219106836;
    a.choice_prior_alpha = 0.9999999999999999;
    a.noul_temperature   = 99.99999952490862;
    a.noul_prior_alpha   = 0.0;
    a.score_temperature  = 11.960309669703673;
    a.score_prior_alpha  = 1.0999999999999999;
    return a;
}

static const EnsembleConfig kSingleOrder{NoulEnsembleMode::NONE, ChoiceEnsembleMode::NONE, ScoreEnsembleMode::NONE};

struct CalibratedFixture {
    RecordingBackend backend;
    DecisionEngine engine;
    FakeEmbedder embedder;
    FakeIndex index;
    TaskRouter router;
    explicit CalibratedFixture(const CalibrationLoad& c)
        : engine(backend, PromptConfig{Layout::AUTO, Scheme::LETTERS}, c.config(), kSingleOrder)
        , router(engine, embedder, index, c.prior_correction()) { backend.varied = true; }
};

static CalibrationLoad enabled(bool prior_correction)
{
    CalibrationLoad c;
    c.state    = CalibrationState::Enabled;
    c.artifact = letters_calibration(prior_correction);
    return c;
}

// Blank-prompt prior of the existing implementation (a separate engine on an identical backend).
static std::vector<float> reference_prior(Type type, int n)
{
    RecordingBackend b;
    b.varied = true;
    DecisionEngine e(b, PromptConfig{Layout::AUTO, Scheme::LETTERS}, CalibrationConfig{}, kSingleOrder);
    return e.compute_blank_logits(type, n);
}

// out (candidates in decision order) follows the existing calibration: prior correction with alpha,
// then the temperature softmax with T.
static void check_calibrated(const DecisionOutput& out, const std::vector<float>& prior, double alpha, double T,
                             const std::string& what)
{
    CHECK(out.ok && out.raw_logits.size() == prior.size() && out.corrected_logits.size() == prior.size(),
          (what + ": logits reported").c_str());
    if (!out.ok || out.raw_logits.size() != prior.size()) return;
    std::vector<float> corrected = out.raw_logits;
    for (size_t i = 0; i < corrected.size(); ++i) corrected[i] -= (float)(alpha * (double)prior[i]);
    std::vector<double> probs;
    temperature_softmax(corrected, T, probs);
    CHECK(out.corrected_logits == corrected, (what + ": prior correction = raw - alpha * prior").c_str());
    CHECK(out.probs == probs, (what + ": probabilities = temperature softmax of the corrected logits").c_str());
    const auto best = std::max_element(probs.begin(), probs.end()) - probs.begin();
    CHECK(out.selected == best, (what + ": prediction = calibrated argmax").c_str());
}

static void test_calibration_llm_paths()
{
    const CalibrationArtifact a = letters_calibration(true);
    {   // choice: nearest 2-shot with the choice parameters
        CalibratedFixture f(enabled(true));
        f.index.add(payload(11, "choice", "NEAREST-Q", {"a", "b"}, 1));
        f.index.add(payload(12, "choice", "SECOND-Q", {"a", "b", "c"}, 2));
        f.index.top1 = {{11, 80}};
        f.index.topk = {{11, 80}, {12, 81}};
        DecisionOutput out = f.router.decide(choice_input());
        CHECK(out.routing && out.routing->strategy == kNearest2Shot, "calibration: choice stays nearest 2-shot");
        check_calibrated(out, reference_prior(Type::Choice, 3), a.choice_prior_alpha, a.choice_temperature,
                         "calibration: choice 2-shot");
    }
    {   // noul: nearest 1-shot with the noul parameters (caller order = decision order)
        CalibratedFixture f(enabled(true));
        f.index.add(payload(41, "noul", "NEAREST-Q", {"no", "yes"}, 1));
        f.index.top1 = {{41, 60}};
        DecisionOutput out = f.router.decide(noul_input(false));
        CHECK(out.routing && out.routing->strategy == kNearest1Shot, "calibration: noul stays nearest 1-shot");
        check_calibrated(out, reference_prior(Type::Noul, 2), a.noul_prior_alpha, a.noul_temperature,
                         "calibration: noul 1-shot");
        CHECK(out.p_true == out.probs[1], "calibration: noul p_true is the calibrated true probability");
    }
    // Zero-shot fallback: each kind with its own parameters; score with the score LLM parameters.
    struct ZeroShot { DecisionInput in; Type type; double alpha, T; const char* what; };
    const ZeroShot cases[] = {
        {choice_input(), Type::Choice, a.choice_prior_alpha, a.choice_temperature, "calibration: choice zero-shot"},
        {noul_input(false), Type::Noul, a.noul_prior_alpha, a.noul_temperature, "calibration: noul zero-shot"},
        {score_input(5), Type::Score, a.score_prior_alpha, a.score_temperature, "calibration: score zero-shot"},
    };
    for (const auto& c : cases) {
        CalibratedFixture f(enabled(true));
        DecisionOutput out = f.router.decide(c.in);
        CHECK(out.routing && out.routing->strategy == kZeroShot, (std::string(c.what) + " is the fallback").c_str());
        check_calibrated(out, reference_prior(c.type, static_cast<int>(c.in.options.size())), c.alpha, c.T, c.what);
    }
    {   // score zero-shot: expected level from the calibrated probabilities
        CalibratedFixture f(enabled(true));
        DecisionOutput out = f.router.decide(score_input(4));
        double ev = 0;
        for (size_t i = 0; i < out.probs.size(); ++i) ev += i * out.probs[i];
        CHECK(out.ok && near(out.expected_score, ev), "calibration: score zero-shot expected level from calibrated probabilities");
    }
    {   // without prior correction in the artifact: temperature only
        CalibratedFixture f(enabled(false));
        DecisionOutput out = f.router.decide(choice_input());
        check_calibrated(out, std::vector<float>(3, 0.0f), 0.0, a.choice_temperature, "calibration: temperature only");
        CHECK(f.backend.evals == 1, "calibration: no blank-prompt evaluation without prior correction");
    }
}

static void test_calibration_knn_unchanged()
{
    auto fill = [](FakeIndex& idx) {
        const int labels[5] = {4, 1, 1, 4, 0};
        const double ham[5] = {70, 71, 71, 75, 76};
        for (int i = 0; i < 5; ++i) {
            idx.add(payload(60 + i, "score", "S" + std::to_string(i), {"l0", "l1", "l2", "l3", "l4"}, labels[i]));
            idx.topk.push_back({60 + i, ham[i]});
        }
        idx.top1 = {{60, 70}};
    };
    Fixture plain;
    fill(plain.index);
    CalibratedFixture cal(enabled(true));
    fill(cal.index);
    const DecisionOutput a = plain.router.decide(score_input(5));
    const DecisionOutput b = cal.router.decide(score_input(5));
    CHECK(b.ok && b.routing->strategy == kKnn5, "calibration: score stays weighted kNN-5");
    CHECK(a.selected == b.selected && a.probs == b.probs && a.raw_probs == b.raw_probs && a.expected_score == b.expected_score,
          "calibration: kNN-5 votes and probabilities unchanged");
    CHECK(cal.backend.evals == 0, "calibration: kNN-5 evaluates nothing (no prior, no temperature)");
}

static void test_calibration_reuse()
{
    CalibratedFixture f(enabled(true));
    f.router.decide(choice_input());
    CHECK(f.backend.evals == 2, "calibration reuse: first choice decision computes the prior once");
    f.router.decide(choice_input());
    f.router.decide(choice_input());
    CHECK(f.backend.evals == 4, "calibration reuse: later decisions reuse the cached prior");
    DecisionInput two = choice_input();
    two.options.resize(2);
    f.router.decide(two);
    CHECK(f.backend.evals == 6, "calibration reuse: one prior per candidate count");
    f.router.decide(noul_input(false));
    f.router.decide(noul_input(true));
    CHECK(f.backend.evals == 9, "calibration reuse: one prior per kind");
}

static void test_calibration_disabled()
{
    // Disabled, missing and incompatible calibration all give the uncalibrated (R4) decision.
    for (CalibrationState st : {CalibrationState::Disabled, CalibrationState::Missing, CalibrationState::Incompatible}) {
        CalibrationLoad c;
        c.state    = st;
        c.artifact = letters_calibration(true);
        CalibratedFixture f(c);
        f.backend.varied = false;
        Fixture plain;
        plain.index.add(payload(11, "choice", "NEAREST-Q", {"a", "b"}, 1));
        plain.index.add(payload(12, "choice", "SECOND-Q", {"a", "b"}, 0));
        plain.index.top1 = {{11, 80}};
        plain.index.topk = {{11, 80}, {12, 81}};
        f.index = plain.index;
        const DecisionOutput a = plain.router.decide(choice_input());
        const DecisionOutput b = f.router.decide(choice_input());
        const std::string what = std::string("calibration ") + to_string(st) + ": uncalibrated decision";
        CHECK(b.ok && a.probs == b.probs && a.selected == b.selected && a.corrected_logits == b.corrected_logits &&
              f.backend.prompts == plain.backend.prompts && f.backend.evals == 1, what.c_str());
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
    CHECK(status == 200 && r["answers"]["q"]["probabilities"].size() == 6, "api: no eligible exemplar -> zero-shot answer");
    CHECK(r.dump().find("zero_shot") == std::string::npos && r.dump().find("fallback") == std::string::npos,
          "api: routing diagnostics are not exposed");

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
    test_invalid_exemplars();
    test_router_errors();
    test_calibration_llm_paths();
    test_calibration_knn_unchanged();
    test_calibration_reuse();
    test_calibration_disabled();
    test_api();

    if (g_failures == 0) {
        printf("\nPASS (0 failure(s))\n");
        return 0;
    }
    fprintf(stderr, "\nFAIL (%d failure(s))\n", g_failures);
    return 1;
}
