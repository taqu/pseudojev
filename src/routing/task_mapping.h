#ifndef INC_PJEV_TASK_MAPPING_H_
#define INC_PJEV_TASK_MAPPING_H_
#include <cstdint>
#include <string>
#include <vector>

#include "decision/decision_engine.h"
#include "nlohmann/json.hpp"
#include "retrieval/exemplar_store.h"

namespace pjev
{
namespace routing
{

// A decision request seen as an Open Jev task: what is embedded and matched against the exemplar bank.
struct RetrievalTask {
    std::string kind;              // "choice" | "noul" | "score"
    std::string domain;
    std::string family;
    std::string question;
    std::string state_text;        // the state exactly as shown to the model
    nlohmann::ordered_json state;  // the parsed JSON object when state_text is one, otherwise state_text
    std::vector<std::string> options;      // Open Jev option list (see route_task)
    std::vector<int> option_to_candidate;  // Open Jev option index -> decision candidate index
    int number_of_levels = 0;      // score: number of levels; 0 otherwise
    uint64_t content = 0;          // content_key() of the task
};

// A request prepared for routing.
struct RoutedTask {
    RetrievalTask retrieval;
    DecisionInput decision;        // candidates in the validated order (noul: false, true)
    std::vector<int> to_caller;    // decision candidate index -> index in the caller's options
};

// Builds the routed task. Candidate mapping per kind:
//   choice - candidates in the caller's order; Open Jev options are the candidate keys;
//   score  - levels in the caller's order (lowest first); Open Jev options are the level descriptions;
//   noul   - candidates reordered to (false, true); Open Jev options are ("no", "yes").
// Fails on an unsupported type or an invalid candidate list.
bool route_task(const DecisionInput& in, RoutedTask& out, std::string& err);

// State as a pjev client sends it: string states verbatim, structured states as compact JSON (key
// order preserved).
std::string state_text(const nlohmann::ordered_json& state);

// Content identity of a task (kind, question, state text, options in dataset order), independent of
// its ID and its answer. An exemplar with the query's content is never used as its neighbour, and two
// exemplars with the same key are duplicate demonstrations.
uint64_t content_key(const std::string& kind, const std::string& question, const std::string& state_text,
                     const std::vector<std::string>& options);

// "search_query: " + the canonical embedding text of the task (same format as the stored documents).
std::string embedding_query_text(const RetrievalTask& task);

// The demonstration for a retrieved exemplar, in the exemplar's own option order with its stored
// answer. Fails when the exemplar has no one-hot answer or cannot be expressed as a decision.
bool build_demonstration(const retrieval::ExemplarPayload& payload, Demonstration& demo, std::string& err);

} // namespace routing
} // namespace pjev
#endif // INC_PJEV_TASK_MAPPING_H_
