#ifndef INC_PJEV_TASK_ROUTER_H_
#define INC_PJEV_TASK_ROUTER_H_
#include <cstdint>
#include <string>
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
    retrieval::ExemplarPayload payload;
    Demonstration              demo;
};

// The R4 neighbour rule. Candidates are considered in the order they are offered: rank 1 is the nearest
// eligible exemplar of the top-1 search (R1's exemplar), later ranks come from the top-K search in
// (distance, rowid) order. A candidate is skipped when its ID was already offered, when its content
// equals the query's, or when its content duplicates an earlier neighbour. Every offered candidate is
// loaded and re-checked for eligibility (allowlist, kind, score level count).
class NeighbourList
{
public:
    NeighbourList(const RetrievalTask& task, const retrieval::IExemplarIndex& index, int embedding_dim);

    // Offers `hits` in order until `need` neighbours are picked. False on a payload or eligibility error.
    bool extend(const std::vector<retrieval::SearchHit>& hits, size_t need, std::string& err);

    const std::vector<Neighbour>& neighbours() const { return picked_; }

private:
    const RetrievalTask&              task_;
    const retrieval::IExemplarIndex&  index_;
    int                               dim_;
    std::vector<Neighbour>            picked_;
    std::vector<int64_t>              offered_ids_;
    std::vector<uint64_t>             contents_;
};

// The task-aware decision path (policy B_task_only@A_latency, see task_policy.h):
//   choice -> decide_nearest_2shot, noul -> decide_nearest_1shot, score -> decide_weighted_knn5.
// Outputs are in the caller's candidate order. Failures carry an ErrorCode in error_code:
//   no_eligible_exemplar - the bank has no eligible exemplar for the task;
//   context_too_long     - the task does not fit the embedding or inference context (after the 2-shot
//                          -> 1-shot fallback for choice);
//   invalid_request / unsupported_primitive - the request cannot be routed;
//   inference_error      - embedding, database, payload or inference-backend failure.
// Not thread-safe; callers serialize access.
class TaskRouter : public DecisionService
{
public:
    TaskRouter(DecisionEngine& engine, ITextEmbedder& embedder, const retrieval::IExemplarIndex& index);

    DecisionOutput decide(const DecisionInput& input) override;

private:
    DecisionEngine&                  engine_;
    ITextEmbedder&                   embedder_;
    const retrieval::IExemplarIndex& index_;

    DecisionOutput decide_nearest_2shot(const RoutedTask& task);
    DecisionOutput decide_nearest_1shot(const RoutedTask& task);
    DecisionOutput decide_weighted_knn5(const RoutedTask& task);

    // Embedding + searches + neighbour list with at least `need` neighbours when the bank has them.
    // On failure fills out.error / out.error_code and returns false.
    bool retrieve(const RoutedTask& task, size_t need, std::vector<Neighbour>& neighbours, DecisionOutput& out);
};

} // namespace routing
} // namespace pjev
#endif // INC_PJEV_TASK_ROUTER_H_
