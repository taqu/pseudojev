#ifndef INC_PJEV_TASK_POLICY_H_
#define INC_PJEV_TASK_POLICY_H_

// The frozen R4 task-aware routing policy B_task_only@A_latency (results/r4/frozen_policy.json,
// sha256 22556532...) as deployed. Changing any value here changes validated behaviour.
//
//   choice -> nearest 2-shot, order A (second-nearest, nearest, query)
//   noul   -> nearest 1-shot
//   score  -> similarity-weighted kNN-5, no inference-model evaluation
//
// A strategy runs only with its full number of valid exemplars (2 / 1 / 5). With fewer, the task is
// decided by S1 zero-shot instead; the strategies are never run with fewer exemplars (no 1-shot
// choice, no kNN-k with k < 5) and missing exemplars are never filled with arbitrary ones.

namespace pjev
{
namespace routing
{

constexpr const char* kPolicyName = "B_task_only@A_latency";
constexpr const char* kPromptFormatVersion = "r1-demo-turn-v1";

// Candidate-logit formulation of every LLM decision (layout AUTO, scheme LETTERS), in the names a
// calibration artifact uses. A calibration fitted for another formulation is not applied.
constexpr const char* kFormulationLayout          = "auto";
constexpr const char* kFormulationCandidateScheme = "letters";

// Prefix of query embedding texts (exemplar documents use "search_document: ").
constexpr const char* kQueryPrefix = "search_query: ";

// Size of the top-K search that supplies neighbours after rank 1.
constexpr int kSearchK = 10;

// kNN classifier for score.
constexpr int    kKnnK   = 5;
constexpr double kKnnTau = 0.005;  // weights w_i = exp((s_i - s_max) / tau)

// Valid exemplars each strategy requires.
constexpr int kNearest2ShotExemplars = 2;
constexpr int kNearest1ShotExemplars = 1;
constexpr int kKnn5Exemplars         = kKnnK;

// Strategy names reported in routing diagnostics.
constexpr const char* kNearest2Shot = "nearest_2shot";
constexpr const char* kNearest1Shot = "nearest_1shot";
constexpr const char* kKnn5         = "knn_5";
constexpr const char* kZeroShot     = "zero_shot";  // S1: single-order prompt without demonstrations

// Fallback reasons reported in routing diagnostics.
constexpr const char* kReasonNoExemplar            = "no_eligible_exemplar";   // the search found none
constexpr const char* kReasonInsufficientExemplars = "insufficient_exemplars"; // fewer valid than required
constexpr const char* kReasonContextLimit          = "context_limit";          // choice: 2-shot over the context

} // namespace routing
} // namespace pjev
#endif // INC_PJEV_TASK_POLICY_H_
