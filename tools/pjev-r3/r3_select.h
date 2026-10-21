#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace pjev
{
namespace r3
{

// How the single demonstration of a 1-shot prompt is chosen. Everything after selection (payload
// load, demonstration construction, prompt, inference, scoring) is the R1 path for both.
enum class ExampleSelection { Nearest, Random, LengthMatched };
const char* to_string(ExampleSelection s);

// Version tags of the random-index derivations, recorded in config.json and selections.jsonl.
static constexpr const char* kRandomSelectionVersion = "r3-random-v1";
static constexpr const char* kLengthMatchedVersion   = "r3-length-v1";

// One row of the exemplar vector index (exemplar_vectors joined with examples by rowid).
struct PoolRow {
    int64_t     id               = 0;   // examples.id (hashed textual ID)
    std::string kind;                   // exemplar_vectors.kind (KNN filter column)
    int         number_of_levels = 0;   // exemplar_vectors.number_of_levels (KNN filter column)
    int         answer_index     = -1;  // -1 / NULL = no one-hot answer
};

// Partition of the eligible exemplars by the KNN filter R1 applies: kind, plus the level count for
// score ("choice", "noul", "score/<levels>"). Only kind and level count enter the key; option count
// does not (R1's KNN does not filter on it).
std::string pool_key(const std::string& kind, int number_of_levels);

struct PoolStats {
    int64_t rows            = 0;
    int64_t not_in_train    = 0;   // outside the training allowlist
    int64_t no_one_hot      = 0;   // in the allowlist but without a one-hot answer
    int64_t eligible        = 0;
    int64_t duplicate_ids   = 0;
};

// The candidate population of R1's retriever: every vector-index row whose ID is in the training
// allowlist and that has a one-hot answer, grouped by pool_key and sorted by ID so that selection
// never depends on database iteration order.
class ExemplarPool
{
public:
    // `train_ids`: IDs whose split is "train" in the training file (before the one-hot filter).
    static ExemplarPool build(const std::vector<PoolRow>& rows, const std::unordered_set<int64_t>& train_ids,
                              PoolStats* stats = nullptr);

    // Sorted candidate IDs for `key` (empty when the key is unknown).
    const std::vector<int64_t>& candidates(const std::string& key) const;
    bool contains(const std::string& key, int64_t id) const;

    const std::map<std::string, std::vector<int64_t>>& by_key() const { return by_key_; }

private:
    std::map<std::string, std::vector<int64_t>> by_key_;
};

// Deterministic uniform index in [0, n): XXH64 of "<version>|<seed>|<query_id>". Depends only on the
// seed, the textual query ID and n; never on global RNG state or the order queries are processed.
uint64_t random_index(uint64_t seed, const std::string& query_id, uint64_t n,
                      const char* version = kRandomSelectionVersion);

struct RandomPick {
    bool    ok        = false;
    int64_t id        = 0;
    int64_t pool_size = 0;   // candidates after removing the query itself
    int64_t index     = -1;  // position within those candidates
};

// Uniform choice from `candidates` (sorted) with `self_id` removed if present.
RandomPick select_random(const std::vector<int64_t>& candidates, int64_t self_id, uint64_t seed,
                         const std::string& query_id);

// R3-L control: one pool's candidates with their demonstration lengths (prompt tokens the
// demonstration turn adds), sorted by (tokens, id).
struct LengthPool {
    std::vector<std::pair<int, int64_t>> by_length;  // (demo tokens, id)
    static LengthPool build(const std::vector<std::pair<int64_t, int>>& id_tokens);
};

struct LengthPick {
    bool    ok          = false;
    int64_t id          = 0;
    int     tokens      = 0;     // demonstration tokens of the pick
    int64_t window_size = 0;
    int     window_lo   = 0;     // demonstration-token range of the window
    int     window_hi   = 0;
    int64_t index       = -1;
};

// Uniform choice among the `window` candidates whose demonstration length is closest to `target`
// (|tokens - target|, ties by lower ID), the query itself excluded. Similarity is never consulted.
LengthPick select_length_matched(const LengthPool& pool, int64_t self_id, int target, int window, uint64_t seed,
                                 const std::string& query_id);

} // namespace r3
} // namespace pjev
