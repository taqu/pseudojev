#pragma once

// R4 core logic shared by the runner, the tuner, the evaluator and the tests. No model access here:
// everything operates on stable identifiers and on measured per-query records.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"

namespace pjev
{
namespace r4
{

using json = nlohmann::ordered_json;

// Seed of the deployable random 1-shot action: the first predefined R3 seed.
static constexpr uint64_t kRandomSeed = 42;
// Neighbours kept per query (kNN-5 needs five) and the top-K search size that supplies them.
static constexpr int kMaxNeighbours = 5;
static constexpr int kSearchK       = 10;

// ---------------------------------------------------------------------------
// Neighbour list
// ---------------------------------------------------------------------------

// Content identity of a task (kind, question, state, options in dataset order), independent of its ID
// and its answer. Two exemplars with the same key are duplicated demonstrations.
uint64_t content_key(const std::string& kind, const std::string& question, const std::string& state_text,
                     const std::vector<std::string>& options);

struct NeighbourCandidate {
    int64_t  id      = 0;
    uint64_t content = 0;
};

struct NeighbourList {
    std::vector<size_t> picked;          // indices into the candidate vector, rank order
    int  duplicates_skipped = 0;         // candidates dropped as content duplicates of an earlier pick
    int  self_skipped       = 0;         // candidates equal to the query (by ID or by content)
};

// `candidates` = R1's top-1 result followed by the top-K search in (distance, rowid) order, so rank 1 is
// exactly R1's exemplar (2-shot and kNN-1 share it). Picks candidates in order, skipping repeated IDs, the
// query itself (by ID or by content) and content duplicates of an earlier pick. At most max_n picks.
NeighbourList merge_neighbours(const std::vector<NeighbourCandidate>& candidates, int64_t self_id,
                               uint64_t self_content, size_t max_n);

// Leakage / eligibility check applied to every exemplar R4 loads (same rules as R1's filter): in the
// training allowlist (which already excludes rows without a one-hot answer), not the query itself, same kind,
// and for score the same number of levels.
bool exemplar_eligible(int64_t ex_id, const std::string& ex_kind, int ex_levels, int64_t query_id,
                       const std::string& query_kind, int query_levels, const std::unordered_set<int64_t>& allow);

// ---------------------------------------------------------------------------
// Label mapping
// ---------------------------------------------------------------------------

enum class MapStatus { Ok, Unsupported, NoMatch };
const char* to_string(MapStatus s);

struct LabelMap {
    MapStatus status    = MapStatus::Unsupported;
    int       candidate = -1;   // query candidate index
};

// Translate an exemplar's stored answer into the query's candidate through the answer's semantic
// identity, never through the letter and never through the query's target:
//   noul   — the answer option text ("no" / "yes") selects the query option with the same text;
//   score  — the answer level index selects the same level of the query (level counts must match);
//   choice — unsupported (option identifiers are per-task item names; a letter carries no meaning).
// `query_option_to_candidate` maps the query's dataset option index to its candidate index.
LabelMap map_label(const std::string& query_kind, const std::vector<std::string>& query_options,
                   const std::vector<int>& query_option_to_candidate, const std::string& ex_kind,
                   const std::vector<std::string>& ex_options, int ex_answer_index);

// Diagnostic only (never a deployable action): exact option-text identity for choice.
LabelMap map_label_text_identity(const std::vector<std::string>& query_options,
                                 const std::vector<int>& query_option_to_candidate,
                                 const std::vector<std::string>& ex_options, int ex_answer_index);

bool knn_supported(const std::string& kind);

// ---------------------------------------------------------------------------
// Voting
// ---------------------------------------------------------------------------

struct Vote {
    int    label     = -1;
    double weight    = 0.0;
};

struct VoteInput {
    int    label      = -1;    // -1 = unmapped (abstains)
    double similarity = 0.0;   // estimated cosine
};

enum class Voting { Uniform, Weighted };
const char* to_string(Voting v);

struct VoteResult {
    bool   ok        = false;  // false: no mapped neighbour among the first k
    int    label     = -1;
    int    used      = 0;      // mapped neighbours that voted
    int    agree     = 0;      // voters with the winning label
    bool   unanimous = false;  // used > 0 && agree == used
    double win_weight = 0.0, total_weight = 0.0;
};

// Vote over the first min(k, size) neighbours (rank order). Weighted: w_i = exp((s_i - s_max) / tau)
// with s_max the largest similarity among the voters, so every weight is in (0, 1] and the formula
// cannot overflow. Ties in total weight (within 1e-12 relative) go to the label of the best-ranked
// neighbour carrying one of the tied labels.
VoteResult vote(const std::vector<VoteInput>& neighbours, int k, Voting voting, double tau);

struct KnnParams {
    int    k      = 1;
    Voting voting = Voting::Uniform;
    double tau    = 0.0;      // Weighted only
    json to_json() const;
    static KnnParams from_json(const json& j);
    std::string describe() const;
};

// ---------------------------------------------------------------------------
// Per-query measured record (one line of <partition>/queries.jsonl, reduced)
// ---------------------------------------------------------------------------

struct Decision {
    std::string status = "not_run";
    int    pred    = -1;
    bool   correct = false;
    int    tokens  = 0;
    double ms      = 0.0;      // decision wall time: prompt construction + evaluation + scoring
    double margin  = 0.0;      // top-1 minus top-2 candidate logit
    bool ok() const { return status == "ok"; }
};

struct NeighbourRow {
    std::string id;            // hex exemplar ID
    double hamming     = 0.0;
    double similarity  = 0.0;  // estimated cosine
    int    label       = -1;   // mapped query candidate, -1 when unmapped
    std::string map_status;
    double cum_load_ms = 0.0;  // payload fetch + LZ4 of every candidate inspected up to this pick
};

struct QueryRow {
    std::string id, kind, group_id;
    int levels = 0, n_options = 0, gt = -1;
    std::string retrieval_status;   // ok | embedding_error | no_eligible_exemplar | ...
    double embed_ms = 0.0, search1_ms = 0.0, searchk_ms = 0.0, vote_ms = 0.0;
    std::vector<NeighbourRow> nb;   // rank order; nb[0] = R1's nearest
    Decision s1, r1, two_a, two_b, rnd;
    double rnd_select_ms = 0.0, rnd_load_ms = 0.0;
    int s1_tokens() const { return s1.tokens; }
    bool retrieval_ok() const { return retrieval_status == "ok" && !nb.empty(); }
};

bool parse_query_row(const json& j, QueryRow& r, std::string& err);

// ---------------------------------------------------------------------------
// Actions and their deployable cost
// ---------------------------------------------------------------------------

enum class Action { ZeroShot, Random1, Nearest1, Nearest2A, Nearest2B, Knn1, Knn3, Knn5 };
const char* to_string(Action a);
std::optional<Action> action_from_string(const std::string& s);
bool action_applicable(Action a, const std::string& kind);
bool is_knn(Action a);
int  knn_k(Action a);

struct KnnConfig {
    KnnParams k1{1, Voting::Uniform, 0.0}, k3{3, Voting::Uniform, 0.0}, k5{5, Voting::Uniform, 0.0};
    const KnnParams& of(Action a) const;
};

// Outcome of one action, or one policy, on one query, with the cost of the work it really needs.
struct Outcome {
    int    pred     = -1;
    bool   correct  = false;
    bool   failed   = false;
    double ms       = 0.0;
    int    llm      = 0;      // inference-model evaluations
    int    emb      = 0;      // embedding evaluations
    int    tokens   = 0;      // prompt tokens evaluated (all LLM passes)
    bool   fallback = false;  // the action's fallback rule fired
    std::string terminal;     // what produced the answer (action or "knn_gate:<k>" etc.)
};

// Standalone execution plan of each action (see summary for the stage list):
//   zero_shot      S1 decision
//   random_1shot   hash draw + row fetch + LZ4 + 1-shot decision
//   nearest_1shot  embedding + top-1 search + row fetch + LZ4 + 1-shot decision
//   nearest_2shot  embedding + top-K search + two row fetches + LZ4 + 2-shot decision;
//                  fallback to the 1-shot decision when no second neighbour exists or the prompt does
//                  not fit n_ctx (the refused attempt's cost is kept)
//   knn_k          embedding + search (top-1 for k=1, top-K otherwise) + neighbour fetches + vote; no LLM
Outcome run_action(const QueryRow& r, Action a, const KnnConfig& knn);

// ---------------------------------------------------------------------------
// Routing rules (one per task kind) and policies
// ---------------------------------------------------------------------------

struct Rule {
    enum class Type { Fixed, KnnGate, SimGate, MarginGate };
    Type   type     = Type::Fixed;
    Action action   = Action::Nearest1;   // Fixed
    // KnnGate: retrieval first; answer by kNN-gate_k when its first gate_k mapped neighbours are unanimous
    // and the nearest similarity >= threshold, otherwise run `fallback` (retrieval already paid).
    // SimGate: retrieval (top-1) first; similarity >= threshold -> nearest_1shot, else `fallback`.
    // MarginGate: S1 first; S1 margin >= threshold -> keep S1, else run `fallback` standalone.
    int    gate_k    = 1;
    double threshold = 0.0;
    Action fallback  = Action::Nearest1;

    json to_json() const;
    static Rule from_json(const json& j);
    std::string describe() const;
    bool adaptive() const { return type != Type::Fixed; }
};

struct Policy {
    std::string name;
    std::string family;                   // static | task_only | adaptive
    std::map<std::string, Rule> per_kind; // choice / noul / score
    json to_json() const;
    static Policy from_json(const json& j);
};

// Whether a rule is executable for a kind (fallback types per rule as documented above).
bool rule_valid(const Rule& r, const std::string& kind);

Outcome run_rule(const QueryRow& r, const Rule& rule, const KnnConfig& knn);
Outcome run_policy(const QueryRow& r, const Policy& p, const KnnConfig& knn);

// Aggregate of a policy (or action) over a set of queries.
struct PolicyStats {
    int    n = 0, correct = 0, failed = 0, fallbacks = 0;
    double mean_ms = 0.0, p50_ms = 0.0, p95_ms = 0.0, p99_ms = 0.0;
    double llm = 0.0, emb = 0.0, tokens = 0.0;    // means per query
    std::map<std::string, std::map<std::string, int>> routed;  // kind -> terminal -> count
    double accuracy() const { return n ? static_cast<double>(correct) / n : 0.0; }
};
PolicyStats summarize(const std::vector<Outcome>& outcomes, const std::vector<const QueryRow*>& rows);

// Pareto frontier over (accuracy up, mean latency down): indices of non-dominated points, by latency.
std::vector<size_t> pareto_front(const std::vector<std::pair<double, double>>& acc_ms);

// ---------------------------------------------------------------------------
// Frozen configuration
// ---------------------------------------------------------------------------

struct FrozenPolicy {
    json        raw;                 // the whole file
    std::string sha256;
    std::string two_shot_order;      // "A" | "B"
    KnnConfig   knn;
    std::vector<Policy> finalists;
    std::vector<std::string> confirmatory_policies;  // names, subset of finalists
};

// Loads and validates frozen_policy.json (and its recorded sha256 when `sha_path` is non-empty).
bool load_frozen(const std::string& path, const std::string& sha_path, FrozenPolicy& out, std::string& err);

} // namespace r4
} // namespace pjev
