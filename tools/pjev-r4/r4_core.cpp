#include "r4_core.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_set>

#include "lz4/xxhash.h"
#include "model/model_identity.h"
#include "r1_metrics.h"

namespace pjev
{
namespace r4
{

// ---------------------------------------------------------------------------
// Neighbour list
// ---------------------------------------------------------------------------

uint64_t content_key(const std::string& kind, const std::string& question, const std::string& state_text,
                     const std::vector<std::string>& options)
{
    // Length-prefixed fields, so no separator can make two different tasks collide by construction.
    std::string buf;
    auto put = [&](const std::string& s) {
        buf += std::to_string(s.size());
        buf += ':';
        buf += s;
    };
    put(kind);
    put(question);
    put(state_text);
    for (const auto& o : options) put(o);
    return XXH64(buf.data(), buf.size(), 0x5234u /* "R4" */);
}

NeighbourList merge_neighbours(const std::vector<NeighbourCandidate>& candidates, int64_t self_id,
                               uint64_t self_content, size_t max_n)
{
    NeighbourList out;
    std::unordered_set<int64_t>  ids;
    std::unordered_set<uint64_t> contents;
    for (size_t i = 0; i < candidates.size() && out.picked.size() < max_n; ++i) {
        const auto& c = candidates[i];
        if (!ids.insert(c.id).second) continue;           // R1's nearest repeated by the top-K search
        if (c.id == self_id || c.content == self_content) { out.self_skipped++; continue; }
        if (!contents.insert(c.content).second) { out.duplicates_skipped++; continue; }
        out.picked.push_back(i);
    }
    return out;
}

bool exemplar_eligible(int64_t ex_id, const std::string& ex_kind, int ex_levels, int64_t query_id,
                       const std::string& query_kind, int query_levels, const std::unordered_set<int64_t>& allow)
{
    if (ex_id == query_id || !allow.count(ex_id) || ex_kind != query_kind) return false;
    return query_kind != "score" || ex_levels == query_levels;
}

// ---------------------------------------------------------------------------
// Label mapping
// ---------------------------------------------------------------------------

const char* to_string(MapStatus s)
{
    switch (s) {
    case MapStatus::Ok:          return "ok";
    case MapStatus::Unsupported: return "unsupported";
    case MapStatus::NoMatch:     return "no_match";
    }
    return "?";
}

bool knn_supported(const std::string& kind) { return kind == "noul" || kind == "score"; }

LabelMap map_label(const std::string& query_kind, const std::vector<std::string>& query_options,
                   const std::vector<int>& query_option_to_candidate, const std::string& ex_kind,
                   const std::vector<std::string>& ex_options, int ex_answer_index)
{
    LabelMap m;
    if (!knn_supported(query_kind)) return m;  // choice: unsupported
    if (ex_kind != query_kind || ex_answer_index < 0 || ex_answer_index >= static_cast<int>(ex_options.size()) ||
        query_option_to_candidate.size() != query_options.size()) {
        m.status = MapStatus::NoMatch;
        return m;
    }
    if (query_kind == "noul") {
        const std::string& ans = ex_options[static_cast<size_t>(ex_answer_index)];
        if (ans != "no" && ans != "yes") { m.status = MapStatus::NoMatch; return m; }
        for (size_t i = 0; i < query_options.size(); ++i) {
            if (query_options[i] == ans) {
                m.status    = MapStatus::Ok;
                m.candidate = query_option_to_candidate[i];
                return m;
            }
        }
        m.status = MapStatus::NoMatch;
        return m;
    }
    // score: ordinal level index on a scale with the same number of levels.
    if (ex_options.size() != query_options.size()) { m.status = MapStatus::NoMatch; return m; }
    m.status    = MapStatus::Ok;
    m.candidate = query_option_to_candidate[static_cast<size_t>(ex_answer_index)];
    return m;
}

LabelMap map_label_text_identity(const std::vector<std::string>& query_options,
                                 const std::vector<int>& query_option_to_candidate,
                                 const std::vector<std::string>& ex_options, int ex_answer_index)
{
    LabelMap m;
    m.status = MapStatus::NoMatch;
    if (ex_answer_index < 0 || ex_answer_index >= static_cast<int>(ex_options.size())) return m;
    const std::string& ans = ex_options[static_cast<size_t>(ex_answer_index)];
    for (size_t i = 0; i < query_options.size() && i < query_option_to_candidate.size(); ++i) {
        if (query_options[i] == ans) {
            m.status    = MapStatus::Ok;
            m.candidate = query_option_to_candidate[i];
            return m;
        }
    }
    return m;
}

// ---------------------------------------------------------------------------
// Voting
// ---------------------------------------------------------------------------

const char* to_string(Voting v) { return v == Voting::Uniform ? "uniform" : "weighted"; }

VoteResult vote(const std::vector<VoteInput>& neighbours, int k, Voting voting, double tau)
{
    VoteResult r;
    const size_t n = std::min(neighbours.size(), static_cast<size_t>(std::max(k, 0)));
    double s_max = -1e300;
    for (size_t i = 0; i < n; ++i)
        if (neighbours[i].label >= 0) s_max = std::max(s_max, neighbours[i].similarity);
    std::map<int, double> weight;     // label -> total weight
    std::map<int, size_t> first_rank; // label -> best rank carrying it
    for (size_t i = 0; i < n; ++i) {
        const auto& nb = neighbours[i];
        if (nb.label < 0) continue;
        double w = 1.0;
        if (voting == Voting::Weighted && tau > 0.0) w = std::exp((nb.similarity - s_max) / tau);
        weight[nb.label] += w;
        first_rank.emplace(nb.label, i);
        r.used++;
        r.total_weight += w;
    }
    if (r.used == 0) return r;
    double best_w = 0.0;
    for (const auto& [label, w] : weight) {
        if (r.label < 0) { r.label = label; best_w = w; continue; }
        const double tol = 1e-12 * std::max(w, best_w);
        if (w > best_w + tol || (std::fabs(w - best_w) <= tol && first_rank[label] < first_rank[r.label])) {
            r.label = label;
            best_w  = w;
        }
    }
    r.ok         = true;
    r.win_weight = weight[r.label];
    for (size_t i = 0; i < n; ++i) if (neighbours[i].label == r.label) r.agree++;
    r.unanimous = r.agree == r.used;
    return r;
}

json KnnParams::to_json() const
{
    json j{{"k", k}, {"voting", to_string(voting)}};
    if (voting == Voting::Weighted) j["tau"] = tau;
    return j;
}

KnnParams KnnParams::from_json(const json& j)
{
    KnnParams p;
    p.k      = j.at("k").get<int>();
    p.voting = j.at("voting").get<std::string>() == "weighted" ? Voting::Weighted : Voting::Uniform;
    p.tau    = j.value("tau", 0.0);
    return p;
}

std::string KnnParams::describe() const
{
    std::ostringstream o;
    o << "kNN-" << k << " " << to_string(voting);
    if (voting == Voting::Weighted) o << " (tau " << tau << ")";
    return o.str();
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

static Decision parse_decision(const json& j)
{
    Decision d;
    if (!j.is_object()) return d;
    d.status  = j.value("status", "not_run");
    d.pred    = j.value("prediction_index", -1);
    d.correct = j.value("correct", false);
    d.tokens  = j.value("prompt_tokens", 0);
    d.ms      = j.value("latency_ms", 0.0);
    d.margin  = j.value("margin", 0.0);
    return d;
}

bool parse_query_row(const json& j, QueryRow& r, std::string& err)
{
    try {
        r = QueryRow{};
        r.id        = j.at("query_id").get<std::string>();
        r.kind      = j.at("kind").get<std::string>();
        r.group_id  = j.value("group_id", "");
        r.levels    = j.value("number_of_levels", 0);
        r.n_options = j.value("options_count", 0);
        r.gt        = j.at("ground_truth_index").get<int>();
        const json& rt = j.at("retrieval");
        r.retrieval_status = rt.value("status", "not_run");
        r.embed_ms   = rt.value("embed_ms", 0.0);
        r.search1_ms = rt.value("search1_ms", 0.0);
        r.searchk_ms = rt.value("searchk_ms", 0.0);
        r.vote_ms    = rt.value("vote_ms", 0.0);
        if (rt.contains("neighbours")) {
            for (const auto& n : rt["neighbours"]) {
                NeighbourRow nb;
                nb.id          = n.value("example_id", "");
                nb.hamming     = n.value("hamming_distance", 0.0);
                nb.similarity  = n.value("estimated_cosine", 0.0);
                nb.label       = n.value("label", -1);
                nb.map_status  = n.value("map_status", "");
                nb.cum_load_ms = n.value("cum_load_ms", 0.0);
                r.nb.push_back(nb);
            }
        }
        r.s1    = parse_decision(j.value("s1", json()));
        r.r1    = parse_decision(j.value("r1", json()));
        r.two_a = parse_decision(j.value("two_shot_a", json()));
        r.two_b = parse_decision(j.value("two_shot_b", json()));
        r.rnd   = parse_decision(j.value("random_1shot", json()));
        if (j.contains("random_1shot")) {
            r.rnd_select_ms = j["random_1shot"].value("select_ms", 0.0);
            r.rnd_load_ms   = j["random_1shot"].value("load_ms", 0.0);
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

static const std::vector<std::pair<Action, const char*>> kActionNames = {
    {Action::ZeroShot, "zero_shot"}, {Action::Random1, "random_1shot"}, {Action::Nearest1, "nearest_1shot"},
    {Action::Nearest2A, "nearest_2shot_a"}, {Action::Nearest2B, "nearest_2shot_b"},
    {Action::Knn1, "knn_1"}, {Action::Knn3, "knn_3"}, {Action::Knn5, "knn_5"}};

const char* to_string(Action a)
{
    for (const auto& [x, s] : kActionNames) if (x == a) return s;
    return "?";
}

std::optional<Action> action_from_string(const std::string& s)
{
    for (const auto& [x, n] : kActionNames) if (s == n) return x;
    return std::nullopt;
}

bool is_knn(Action a) { return a == Action::Knn1 || a == Action::Knn3 || a == Action::Knn5; }
int  knn_k(Action a) { return a == Action::Knn1 ? 1 : a == Action::Knn3 ? 3 : a == Action::Knn5 ? 5 : 0; }

bool action_applicable(Action a, const std::string& kind) { return !is_knn(a) || knn_supported(kind); }

const KnnParams& KnnConfig::of(Action a) const { return a == Action::Knn1 ? k1 : a == Action::Knn3 ? k3 : k5; }

static void llm(Outcome& o, const Decision& d)
{
    o.ms += d.ms;
    o.llm++;
    o.tokens += d.tokens;
}

static void answer(Outcome& o, const Decision& d, int gt)
{
    o.failed  = !d.ok();
    o.pred    = d.ok() ? d.pred : -1;
    o.correct = d.ok() && d.pred == gt;
}

// Fetch cost of the payloads needed for the first n neighbours.
static double load_upto(const QueryRow& r, size_t n)
{
    if (r.nb.empty() || n == 0) return 0.0;
    return r.nb[std::min(n, r.nb.size()) - 1].cum_load_ms;
}

static std::vector<VoteInput> vote_inputs(const QueryRow& r)
{
    std::vector<VoteInput> v;
    for (const auto& n : r.nb) v.push_back({n.label, n.similarity});
    return v;
}

// Retrieval stage: embedding + search (+ the payloads of the first `need` neighbours).
static void retrieval(Outcome& o, const QueryRow& r, bool top_k, size_t need)
{
    o.emb++;
    o.ms += r.embed_ms + (top_k ? r.searchk_ms : r.search1_ms) + load_upto(r, need);
}

// The 2-shot decision once retrieval is paid: fallback to the 1-shot decision when there is no second
// neighbour or the 2-shot prompt does not fit.
static void two_shot_decision(Outcome& o, const QueryRow& r, const Decision& two)
{
    if (r.nb.size() < 2 || two.status == "not_run") {
        o.fallback = true;
        llm(o, r.r1);
        answer(o, r.r1, r.gt);
        o.terminal = "nearest_1shot";
        return;
    }
    if (two.status == "context_limit") {
        o.fallback = true;
        o.ms += two.ms;  // the refused attempt (tokenization only; not an evaluation)
        llm(o, r.r1);
        answer(o, r.r1, r.gt);
        o.terminal = "nearest_1shot";
        return;
    }
    llm(o, two);
    answer(o, two, r.gt);
}

static void knn_answer(Outcome& o, const QueryRow& r, const KnnParams& p)
{
    const VoteResult v = vote(vote_inputs(r), p.k, p.voting, p.tau);
    o.ms     += r.vote_ms;
    o.failed  = !v.ok;
    o.pred    = v.ok ? v.label : -1;
    o.correct = v.ok && v.label == r.gt;
}

Outcome run_action(const QueryRow& r, Action a, const KnnConfig& knn)
{
    Outcome o;
    o.terminal = to_string(a);
    switch (a) {
    case Action::ZeroShot:
        llm(o, r.s1);
        answer(o, r.s1, r.gt);
        return o;
    case Action::Random1:
        o.ms += r.rnd_select_ms + r.rnd_load_ms;
        llm(o, r.rnd);
        answer(o, r.rnd, r.gt);
        return o;
    case Action::Nearest1:
        retrieval(o, r, false, 1);
        if (!r.retrieval_ok()) { o.failed = true; return o; }
        llm(o, r.r1);
        answer(o, r.r1, r.gt);
        return o;
    case Action::Nearest2A:
    case Action::Nearest2B:
        retrieval(o, r, true, 2);
        if (!r.retrieval_ok()) { o.failed = true; return o; }
        two_shot_decision(o, r, a == Action::Nearest2A ? r.two_a : r.two_b);
        return o;
    case Action::Knn1:
    case Action::Knn3:
    case Action::Knn5: {
        if (!knn_supported(r.kind)) { o.failed = true; o.terminal = "unsupported"; return o; }
        const int k = knn_k(a);
        retrieval(o, r, k > 1, static_cast<size_t>(k));
        if (!r.retrieval_ok()) { o.failed = true; return o; }
        knn_answer(o, r, knn.of(a));
        return o;
    }
    }
    return o;
}

// ---------------------------------------------------------------------------
// Rules and policies
// ---------------------------------------------------------------------------

static const char* rule_type_name(Rule::Type t)
{
    switch (t) {
    case Rule::Type::Fixed:      return "fixed";
    case Rule::Type::KnnGate:    return "knn_gate";
    case Rule::Type::SimGate:    return "similarity_gate";
    case Rule::Type::MarginGate: return "s1_margin_gate";
    }
    return "?";
}

json Rule::to_json() const
{
    json j{{"type", rule_type_name(type)}};
    if (type == Type::Fixed) {
        j["action"] = to_string(action);
        return j;
    }
    if (type == Type::KnnGate) j["gate_k"] = gate_k;
    j["threshold"] = threshold;
    j["fallback"]  = to_string(fallback);
    return j;
}

Rule Rule::from_json(const json& j)
{
    Rule r;
    const std::string t = j.at("type").get<std::string>();
    auto act = [](const json& v) {
        auto a = action_from_string(v.get<std::string>());
        if (!a) throw std::runtime_error("unknown action: " + v.get<std::string>());
        return *a;
    };
    if (t == "fixed") { r.type = Type::Fixed; r.action = act(j.at("action")); return r; }
    if (t == "knn_gate") r.type = Type::KnnGate;
    else if (t == "similarity_gate") r.type = Type::SimGate;
    else if (t == "s1_margin_gate") r.type = Type::MarginGate;
    else throw std::runtime_error("unknown rule type: " + t);
    r.gate_k    = j.value("gate_k", 1);
    r.threshold = j.at("threshold").get<double>();
    r.fallback  = act(j.at("fallback"));
    return r;
}

std::string Rule::describe() const
{
    std::ostringstream o;
    o.precision(6);
    switch (type) {
    case Type::Fixed:
        o << to_string(action);
        break;
    case Type::KnnGate:
        o << "retrieve; if nearest sim >= " << threshold;
        if (gate_k > 1) o << " and top-" << gate_k << " labels unanimous";
        o << " -> knn_" << gate_k << ", else " << to_string(fallback);
        break;
    case Type::SimGate:
        o << "retrieve; if nearest sim >= " << threshold << " -> nearest_1shot, else " << to_string(fallback);
        break;
    case Type::MarginGate:
        o << "S1; if S1 margin >= " << threshold << " keep S1, else " << to_string(fallback);
        break;
    }
    return o.str();
}

json Policy::to_json() const
{
    json j{{"name", name}, {"family", family}, {"rules", json::object()}};
    for (const auto& [k, r] : per_kind) {
        j["rules"][k] = r.to_json();
        j["rules"][k]["description"] = r.describe();
    }
    return j;
}

Policy Policy::from_json(const json& j)
{
    Policy p;
    p.name   = j.at("name").get<std::string>();
    p.family = j.value("family", "");
    for (const auto& [k, v] : j.at("rules").items()) p.per_kind[k] = Rule::from_json(v);
    return p;
}

Outcome run_rule(const QueryRow& r, const Rule& rule, const KnnConfig& knn)
{
    switch (rule.type) {
    case Rule::Type::Fixed:
        return run_action(r, rule.action, knn);

    case Rule::Type::KnnGate: {
        Outcome o;
        const int k = std::max(1, rule.gate_k);
        // A 2-shot fallback needs the second payload too; fetch everything once.
        const size_t need = std::max<size_t>(static_cast<size_t>(k),
            rule.fallback == Action::Nearest2A || rule.fallback == Action::Nearest2B ? 2 : 1);
        const bool top_k = k > 1 || need > 1;
        retrieval(o, r, top_k, need);
        if (!r.retrieval_ok()) { o.failed = true; o.terminal = "retrieval_failed"; return o; }
        o.ms += r.vote_ms;
        const VoteResult v = vote(vote_inputs(r), k, Voting::Uniform, 0.0);
        if (v.ok && v.unanimous && r.nb[0].similarity >= rule.threshold) {
            o.pred     = v.label;
            o.correct  = v.label == r.gt;
            o.terminal = "knn_" + std::to_string(k);
            return o;
        }
        o.fallback = true;
        o.terminal = to_string(rule.fallback);
        switch (rule.fallback) {
        case Action::ZeroShot: llm(o, r.s1); answer(o, r.s1, r.gt); break;
        case Action::Nearest1: llm(o, r.r1); answer(o, r.r1, r.gt); break;
        case Action::Nearest2A: two_shot_decision(o, r, r.two_a); break;
        case Action::Nearest2B: two_shot_decision(o, r, r.two_b); break;
        default:  // not generated by the tuner; refused by load_frozen
            o.failed   = true;
            o.terminal = "invalid_rule";
        }
        return o;
    }

    case Rule::Type::SimGate: {
        Outcome o;
        retrieval(o, r, false, 0);
        if (!r.retrieval_ok()) { o.failed = true; o.terminal = "retrieval_failed"; return o; }
        if (r.nb[0].similarity >= rule.threshold) {
            o.ms += load_upto(r, 1);
            llm(o, r.r1);
            answer(o, r.r1, r.gt);
            o.terminal = "nearest_1shot";
            return o;
        }
        o.fallback = true;
        if (rule.fallback != Action::ZeroShot && rule.fallback != Action::Random1) {
            o.failed   = true;  // not generated by the tuner; refused by load_frozen
            o.terminal = "invalid_rule";
            return o;
        }
        const Outcome f = run_action(r, rule.fallback, knn);  // needs no retrieval
        o.ms += f.ms; o.llm += f.llm; o.tokens += f.tokens;
        o.pred = f.pred; o.correct = f.correct; o.failed = f.failed;
        o.terminal = to_string(rule.fallback);
        return o;
    }

    case Rule::Type::MarginGate: {
        Outcome o;
        llm(o, r.s1);
        if (r.s1.ok() && r.s1.margin >= rule.threshold) {
            answer(o, r.s1, r.gt);
            o.terminal = "zero_shot";
            return o;
        }
        o.fallback = true;
        const Outcome f = run_action(r, rule.fallback, knn);
        o.ms += f.ms; o.llm += f.llm; o.emb += f.emb; o.tokens += f.tokens;
        o.pred = f.pred; o.correct = f.correct; o.failed = f.failed;
        o.terminal = f.terminal;
        return o;
    }
    }
    return Outcome{};
}

Outcome run_policy(const QueryRow& r, const Policy& p, const KnnConfig& knn)
{
    auto it = p.per_kind.find(r.kind);
    if (it == p.per_kind.end()) {
        Outcome o;
        o.failed   = true;
        o.terminal = "no_rule";
        return o;
    }
    return run_rule(r, it->second, knn);
}

PolicyStats summarize(const std::vector<Outcome>& outcomes, const std::vector<const QueryRow*>& rows)
{
    PolicyStats s;
    s.n = static_cast<int>(outcomes.size());
    std::vector<double> ms;
    ms.reserve(outcomes.size());
    double llm = 0, emb = 0, tok = 0;
    for (size_t i = 0; i < outcomes.size(); ++i) {
        const auto& o = outcomes[i];
        s.correct   += o.correct ? 1 : 0;
        s.failed    += o.failed ? 1 : 0;
        s.fallbacks += o.fallback ? 1 : 0;
        ms.push_back(o.ms);
        llm += o.llm; emb += o.emb; tok += o.tokens;
        if (i < rows.size()) s.routed[rows[i]->kind][o.terminal]++;
    }
    if (s.n) {
        s.mean_ms = r1::mean(ms);
        s.p50_ms  = r1::percentile(ms, 50);
        s.p95_ms  = r1::percentile(ms, 95);
        s.p99_ms  = r1::percentile(ms, 99);
        s.llm = llm / s.n; s.emb = emb / s.n; s.tokens = tok / s.n;
    }
    return s;
}

std::vector<size_t> pareto_front(const std::vector<std::pair<double, double>>& acc_ms)
{
    std::vector<size_t> idx(acc_ms.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    // By latency ascending, then accuracy descending; keep points that raise the best accuracy so far.
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (acc_ms[a].second != acc_ms[b].second) return acc_ms[a].second < acc_ms[b].second;
        if (acc_ms[a].first != acc_ms[b].first) return acc_ms[a].first > acc_ms[b].first;
        return a < b;
    });
    std::vector<size_t> front;
    double best = -1.0;
    for (size_t i : idx) {
        if (acc_ms[i].first > best) {
            front.push_back(i);
            best = acc_ms[i].first;
        }
    }
    return front;
}

// ---------------------------------------------------------------------------
// Frozen configuration
// ---------------------------------------------------------------------------

bool rule_valid(const Rule& r, const std::string& kind)
{
    switch (r.type) {
    case Rule::Type::Fixed:
        return action_applicable(r.action, kind);
    case Rule::Type::KnnGate:
        return knn_supported(kind) && (r.gate_k == 1 || r.gate_k == 3 || r.gate_k == 5) &&
               (r.fallback == Action::ZeroShot || r.fallback == Action::Nearest1 ||
                r.fallback == Action::Nearest2A || r.fallback == Action::Nearest2B);
    case Rule::Type::SimGate:
        return r.fallback == Action::ZeroShot || r.fallback == Action::Random1;
    case Rule::Type::MarginGate:
        return r.fallback != Action::ZeroShot && action_applicable(r.fallback, kind);
    }
    return false;
}

bool load_frozen(const std::string& path, const std::string& sha_path, FrozenPolicy& out, std::string& err)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { err = "cannot open frozen configuration " + path; return false; }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string bytes = ss.str();
    out = FrozenPolicy{};
    out.sha256 = sha256_data(bytes);
    if (!sha_path.empty()) {
        std::ifstream sf(sha_path);
        std::string recorded;
        if (!(sf >> recorded)) { err = "cannot read " + sha_path; return false; }
        if (recorded != out.sha256) { err = "frozen configuration changed after freezing (sha256 mismatch)"; return false; }
    }
    try {
        out.raw = json::parse(bytes);
        if (out.raw.value("stage", "") != "frozen") { err = "not a frozen configuration"; return false; }
        out.two_shot_order = out.raw.at("nearest_2shot").at("order").get<std::string>();
        if (out.two_shot_order != "A" && out.two_shot_order != "B") { err = "invalid 2-shot order"; return false; }
        const json& k = out.raw.at("knn");
        out.knn.k1 = KnnParams::from_json(k.at("knn_1"));
        out.knn.k3 = KnnParams::from_json(k.at("knn_3"));
        out.knn.k5 = KnnParams::from_json(k.at("knn_5"));
        if (out.knn.k1.k != 1 || out.knn.k3.k != 3 || out.knn.k5.k != 5) { err = "invalid kNN k"; return false; }
        std::set<std::string> names;
        for (const auto& p : out.raw.at("finalist_policies")) {
            out.finalists.push_back(Policy::from_json(p));
            if (!names.insert(out.finalists.back().name).second) { err = "duplicate policy name"; return false; }
            for (const auto& [kind, rule] : out.finalists.back().per_kind) {
                if (!rule_valid(rule, kind)) { err = "policy " + out.finalists.back().name + ": invalid rule for " + kind; return false; }
                const Action used = rule.type == Rule::Type::Fixed ? rule.action : rule.fallback;
                if (!action_applicable(used, kind)) { err = "policy " + out.finalists.back().name + " uses an inapplicable action"; return false; }
                const Action other = out.two_shot_order == "A" ? Action::Nearest2B : Action::Nearest2A;
                if (used == other) { err = "policy uses the 2-shot order that was not selected"; return false; }
            }
        }
        for (const auto& n : out.raw.at("confirmatory_policies")) {
            const std::string s = n.get<std::string>();
            if (!names.count(s)) { err = "confirmatory policy not among finalists: " + s; return false; }
            out.confirmatory_policies.push_back(s);
        }
    } catch (const std::exception& e) {
        err = std::string("malformed frozen configuration: ") + e.what();
        return false;
    }
    return true;
}

} // namespace r4
} // namespace pjev
