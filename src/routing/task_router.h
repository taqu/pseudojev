#ifndef INC_PJEV_TASK_ROUTER_H_
#define INC_PJEV_TASK_ROUTER_H_
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "decision/decision_engine.h"
#include "decision/decision_service.h"
#include "retrieval/exemplar_store.h"
#include "task_mapping.h"

namespace pjev
{
namespace routing
{

// Query embedding (an interface so routing can be tested without an embedding model).
class ITextEmbedder
{
public:
    virtual ~ITextEmbedder() = default;
    // Unit-norm embedding of `text`; false when the text cannot be embedded (empty, or longer than the
    // embedding model's context).
    virtual bool encode(const std::string& text, std::vector<float>& out) = 0;
    virtual int  dimension() const = 0;
};

// One neighbour of a query.
struct Neighbour {
    int64_t                    id         = 0;
    double                     hamming    = 0.0;
    double                     similarity = 0.0;  // estimated cosine
    int32_t                    label      = -1;   // score: the answer's query candidate (map_label)
    retrieval::ExemplarPayload payload;
    Demonstration              demo;
};

// The R4 neighbour rule. Candidates are considered in the order they are offered: rank 1 is the nearest
// eligible exemplar of the top-1 search (R1's exemplar), later ranks come from the top-K search in
// (distance, rowid) order. A candidate is skipped when its ID was already offered, when its content
// equals the query's, or when its content duplicates an earlier neighbour. Every offered candidate is
// loaded and validated; it is skipped (it never counts as a neighbour) when it is missing or cannot be
// decoded, has no one-hot answer or no demonstration form, is not eligible (allowlist, kind, score
// level count), or (score) its answer does not map onto the query's levels.
class NeighbourList
{
public:
    NeighbourList(const RetrievalTask& task, const retrieval::IExemplarIndex& index, int embedding_dim);

    // Offers `hits` in order until `need` neighbours are picked. False only on a database failure.
    bool extend(const std::vector<retrieval::SearchHit>& hits, size_t need, std::string& err);

    const std::vector<Neighbour>& neighbours() const { return picked_; }
    // Candidates offered so far, and those of them not picked.
    size_t offered() const { return offered_ids_.size(); }
    size_t rejected() const { return offered_ids_.size() - picked_.size(); }

private:
    const RetrievalTask&              task_;
    const retrieval::IExemplarIndex&  index_;
    int                               dim_;
    std::vector<Neighbour>            picked_;
    std::vector<int64_t>              offered_ids_;
    std::vector<uint64_t>             contents_;
};

// The task-aware decision path (policy B_task_only@A_latency, see task_policy.h):
//   choice -> decide_nearest_2shot, noul -> decide_nearest_1shot, score -> decide_weighted_knn5,
// each run only with its required number of valid neighbours; with fewer, decide_zero_shot (S1).
// Outputs are in the caller's candidate order; out.routing records the requested and executed
// strategy. Calibration is the engine's: with prior_correction, every LLM decision (2-shot, 1-shot,
// zero-shot) subtracts the engine's prior alpha times the blank-prompt candidate logits
// (compute_blank_logits, once per kind and candidate count) before its temperature; kNN-5 votes are
// never calibrated. Failures carry an ErrorCode in error_code:
//   context_too_long     - the task does not fit the embedding or inference context (after the 2-shot
//                          -> 1-shot fallback for choice);
//   invalid_request / unsupported_primitive - the request cannot be routed;
//   inference_error      - embedding, database or inference-backend failure.
// Not thread-safe; callers serialize access.
class TaskRouter : public DecisionService
{
public:
    TaskRouter(DecisionEngine& engine, ITextEmbedder& embedder, const retrieval::IExemplarIndex& index,
               bool prior_correction = false);

    DecisionOutput decide(const DecisionInput& input) override;

private:
    DecisionEngine&                  engine_;
    ITextEmbedder&                   embedder_;
    const retrieval::IExemplarIndex& index_;
    const bool                       prior_correction_;
    std::map<std::pair<Type, size_t>, std::vector<float>> priors_;  // blank-prompt logits

    // With prior correction, attaches the blank-prompt prior to the task's decision input (LLM paths
    // only). False (out filled with inference_error) when the prior cannot be computed.
    bool attach_prior(RoutedTask& task, DecisionOutput& out);

    // Strategies, given exactly their required neighbours (nb) in rank order.
    DecisionOutput decide_nearest_2shot(const RoutedTask& task, const std::vector<Neighbour>& nb,
                                        DecisionOutput::RoutingDiag& diag);
    DecisionOutput decide_nearest_1shot(const RoutedTask& task, const std::vector<Neighbour>& nb,
                                        DecisionOutput::RoutingDiag& diag);
    DecisionOutput decide_weighted_knn5(const RoutedTask& task, const std::vector<Neighbour>& nb,
                                        DecisionOutput::RoutingDiag& diag);
    // S1 zero-shot, when fewer than the required neighbours are found.
    DecisionOutput decide_zero_shot(const RoutedTask& task);

    // Embedding + searches, filling `list` with up to `need` neighbours; fewer (or none) is not a
    // failure. On an embedding or database failure fills out.error / out.error_code and returns false;
    // otherwise sets out.retrieval_us.
    bool retrieve(const RoutedTask& task, size_t need, NeighbourList& list, DecisionOutput& out);
};

} // namespace routing
} // namespace pjev
#endif // INC_PJEV_TASK_ROUTER_H_
