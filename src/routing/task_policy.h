#ifndef INC_PJEV_TASK_POLICY_H_
#define INC_PJEV_TASK_POLICY_H_

// The frozen R4 task-aware routing policy B_task_only@A_latency (results/r4/frozen_policy.json,
// sha256 22556532...) as deployed. Changing any value here changes validated behaviour.
//
//   choice -> nearest 2-shot, order A (second-nearest, nearest, query)
//   noul   -> nearest 1-shot
//   score  -> similarity-weighted kNN-5, no inference-model evaluation

namespace pjev
{
namespace routing
{

constexpr const char* kPolicyName = "B_task_only@A_latency";
constexpr const char* kPromptFormatVersion = "r1-demo-turn-v1";

// Prefix of query embedding texts (exemplar documents use "search_document: ").
constexpr const char* kQueryPrefix = "search_query: ";

// Size of the top-K search that supplies neighbours after rank 1.
constexpr int kSearchK = 10;

// kNN classifier for score.
constexpr int    kKnnK   = 5;
constexpr double kKnnTau = 0.005;  // weights w_i = exp((s_i - s_max) / tau)

// Strategy names reported in routing diagnostics.
constexpr const char* kNearest2Shot = "nearest_2shot";
constexpr const char* kNearest1Shot = "nearest_1shot";
constexpr const char* kKnn5         = "knn_5";

} // namespace routing
} // namespace pjev
#endif // INC_PJEV_TASK_POLICY_H_
