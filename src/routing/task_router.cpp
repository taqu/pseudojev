#include "task_router.h"

#include <algorithm>
#include <chrono>

#include "knn.h"
#include "server/jev_api.h"
#include "spdlog/spdlog.h"
#include "task_policy.h"

namespace pjev
{
namespace routing
{

namespace
{
int64_t now_us()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

DecisionOutput failure(const char* code, std::string message)
{
    DecisionOutput out;
    out.ok         = false;
    out.error_code = code;
    out.error      = std::move(message);
    return out;
}

bool is_context_limit(const DecisionOutput& out)
{
    // LlamaBackend refuses prompts that do not fit the context window.
    return out.error.rfind("prompt too long", 0) == 0;
}

// Classifies an inference-model failure.
void classify_engine_error(DecisionOutput& out)
{
    if (out.ok || !out.error_code.empty()) return;
    out.error_code = is_context_limit(out) ? ErrorCode::CONTEXT_TOO_LONG : ErrorCode::INFERENCE_ERROR;
}

// Reorders per-candidate vectors from the routed (validated) candidate order to the caller's order.
template <class T>
void permute(std::vector<T>& v, const std::vector<int>& to_caller)
{
    if (v.size() != to_caller.size()) return;
    std::vector<T> r(v.size());
    for (size_t c = 0; c < v.size(); ++c) r[static_cast<size_t>(to_caller[c])] = v[c];
    v.swap(r);
}

void to_caller_order(DecisionOutput& out, const RoutedTask& task, const DecisionInput& caller)
{
    out.keys.clear();
    for (const auto& kv : caller.options) out.keys.push_back(kv.first);
    if (!out.ok) return;
    permute(out.probs, task.to_caller);
    permute(out.raw_probs, task.to_caller);
    permute(out.raw_logits, task.to_caller);
    permute(out.corrected_logits, task.to_caller);
    if (out.selected >= 0 && out.selected < static_cast<int32_t>(task.to_caller.size()))
        out.selected = task.to_caller[static_cast<size_t>(out.selected)];
}

void record_neighbours(DecisionOutput::RoutingDiag& diag, const std::vector<Neighbour>& nb)
{
    for (const auto& n : nb) {
        diag.neighbour_ids.push_back(n.id);
        diag.neighbour_hamming.push_back(n.hamming);
        diag.neighbour_similarity.push_back(n.similarity);
    }
}
} // namespace

// ---------------------------------------------------------------------------
// Neighbour list
// ---------------------------------------------------------------------------

NeighbourList::NeighbourList(const RetrievalTask& task, const retrieval::IExemplarIndex& index, int embedding_dim)
    : task_(task), index_(index), dim_(embedding_dim)
{}

bool NeighbourList::extend(const std::vector<retrieval::SearchHit>& hits, size_t need, std::string& err)
{
    for (const auto& h : hits) {
        if (picked_.size() >= need) break;
        if (std::find(offered_ids_.begin(), offered_ids_.end(), h.id) != offered_ids_.end()) continue;
        offered_ids_.push_back(h.id);

        Neighbour n;
        if (!index_.load(h.id, n.payload, err)) return false;
        if (!build_demonstration(n.payload, n.demo, err)) {
            err = "exemplar " + retrieval::format_exemplar_id(h.id) + ": " + err;
            return false;
        }
        const bool eligible = index_.is_allowed(n.payload.id) && n.payload.kind == task_.kind &&
                              (task_.kind != "score" || n.payload.number_of_levels == task_.number_of_levels);
        if (!eligible) {
            err = "exemplar " + retrieval::format_exemplar_id(h.id) + " returned by the search is not eligible";
            return false;
        }
        const uint64_t ck = content_key(n.payload.kind, n.payload.question, state_text(n.payload.state),
                                        n.payload.options);
        if (ck == task_.content) continue;  // the query itself is in the bank
        if (std::find(contents_.begin(), contents_.end(), ck) != contents_.end()) continue;
        contents_.push_back(ck);
        n.id         = h.id;
        n.hamming    = h.distance;
        n.similarity = estimated_cosine(h.distance, dim_);
        picked_.push_back(std::move(n));
    }
    return true;
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

TaskRouter::TaskRouter(DecisionEngine& engine, ITextEmbedder& embedder, const retrieval::IExemplarIndex& index)
    : engine_(engine), embedder_(embedder), index_(index)
{}

DecisionOutput TaskRouter::decide(const DecisionInput& input)
{
    RoutedTask task;
    std::string err;
    if (!route_task(input, task, err)) {
        const bool unsupported = input.type != Type::Choice && input.type != Type::Noul && input.type != Type::Score;
        DecisionOutput out = failure(unsupported ? ErrorCode::UNSUPPORTED_PRIMITIVE : ErrorCode::INVALID_REQUEST, err);
        for (const auto& kv : input.options) out.keys.push_back(kv.first);
        return out;
    }

    DecisionOutput out;
    switch (task.decision.type) {
    case Type::Choice: out = decide_nearest_2shot(task); break;
    case Type::Noul:   out = decide_nearest_1shot(task); break;
    case Type::Score:  out = decide_weighted_knn5(task); break;
    default:           out = failure(ErrorCode::UNSUPPORTED_PRIMITIVE, "unsupported decision type"); break;
    }
    to_caller_order(out, task, input);
    if (out.ok && out.routing) {
        spdlog::debug("routing: {} -> {}{} (selected {})", task.retrieval.kind, out.routing->strategy,
                      out.routing->fallback ? " [1-shot fallback]" : "", out.selected);
    }
    return out;
}

bool TaskRouter::retrieve(const RoutedTask& task, size_t need, std::vector<Neighbour>& neighbours, DecisionOutput& out)
{
    const int64_t t0 = now_us();
    const RetrievalTask& q = task.retrieval;
    auto fail = [&](const char* code, std::string msg) {
        out = failure(code, std::move(msg));
        out.retrieval_us = now_us() - t0;
        return false;
    };

    std::vector<float> embedding;
    if (!embedder_.encode(embedding_query_text(q), embedding) || embedding.empty()) {
        return fail(ErrorCode::CONTEXT_TOO_LONG,
                    "query embedding failed: the task text is empty or exceeds the embedding model's context");
    }

    std::string err;
    NeighbourList list(q, index_, embedder_.dimension());
    // Rank 1: R1's top-1 search.
    std::vector<retrieval::SearchHit> hits;
    if (!index_.search(q.kind, q.number_of_levels, embedding, 1, hits, err)) return fail(ErrorCode::INFERENCE_ERROR, err);
    if (!list.extend(hits, need, err)) return fail(ErrorCode::INFERENCE_ERROR, err);
    // Further ranks: the top-K search, only when rank 1 alone does not satisfy the strategy.
    if (!hits.empty() && list.neighbours().size() < need) {
        if (!index_.search(q.kind, q.number_of_levels, embedding, kSearchK, hits, err))
            return fail(ErrorCode::INFERENCE_ERROR, err);
        if (!list.extend(hits, need, err)) return fail(ErrorCode::INFERENCE_ERROR, err);
    }
    if (list.neighbours().empty()) {
        std::string what = q.kind;
        if (q.kind == "score") what += " with " + std::to_string(q.number_of_levels) + " levels";
        return fail(ErrorCode::NO_ELIGIBLE_EXEMPLAR, "no eligible exemplar for " + what);
    }
    neighbours = list.neighbours();
    out.retrieval_us = now_us() - t0;
    return true;
}

DecisionOutput TaskRouter::decide_nearest_2shot(const RoutedTask& task)
{
    DecisionOutput r;
    std::vector<Neighbour> nb;
    if (!retrieve(task, 2, nb, r)) return r;

    DecisionOutput::RoutingDiag diag;
    diag.strategy = kNearest2Shot;
    record_neighbours(diag, nb);

    DecisionOutput out;
    if (nb.size() >= 2) {
        // Order A: second-nearest, nearest, query.
        out = engine_.decide_with_demonstrations(task.decision, {nb[1].demo, nb[0].demo});
        diag.demonstration_ids = {nb[1].id, nb[0].id};
    }
    // Validated fallback: fewer than two neighbours, or the 2-shot prompt exceeds the context.
    if (nb.size() < 2 || (!out.ok && is_context_limit(out))) {
        out = engine_.decide_with_demonstration(task.decision, nb[0].demo);
        diag.fallback = true;
        diag.demonstration_ids = {nb[0].id};
    }
    classify_engine_error(out);
    out.retrieval_us = r.retrieval_us;
    out.routing = std::move(diag);
    return out;
}

DecisionOutput TaskRouter::decide_nearest_1shot(const RoutedTask& task)
{
    DecisionOutput r;
    std::vector<Neighbour> nb;
    if (!retrieve(task, 1, nb, r)) return r;

    DecisionOutput out = engine_.decide_with_demonstration(task.decision, nb[0].demo);
    classify_engine_error(out);
    out.retrieval_us = r.retrieval_us;
    DecisionOutput::RoutingDiag diag;
    diag.strategy = kNearest1Shot;
    diag.demonstration_ids = {nb[0].id};
    record_neighbours(diag, nb);
    out.routing = std::move(diag);
    return out;
}

DecisionOutput TaskRouter::decide_weighted_knn5(const RoutedTask& task)
{
    DecisionOutput out;
    std::vector<Neighbour> nb;
    if (!retrieve(task, kKnnK, nb, out)) return out;

    const int64_t t0 = now_us();
    const RetrievalTask& q = task.retrieval;
    DecisionOutput::RoutingDiag diag;
    diag.strategy = kKnn5;
    record_neighbours(diag, nb);
    std::vector<VoteInput> votes;
    for (const auto& n : nb) {
        const LabelMap m = map_label(q.kind, q.options, q.option_to_candidate, n.payload.kind, n.payload.options,
                                     n.payload.answer_index);
        const int label = m.status == MapStatus::Ok ? m.candidate : -1;
        votes.push_back({label, n.similarity});
        diag.neighbour_labels.push_back(label);
    }
    const VoteResult v = vote(votes, kKnnK, kKnnTau);
    const int64_t retrieval_us = out.retrieval_us + (now_us() - t0);
    if (!v.ok) {
        out = failure(ErrorCode::NO_ELIGIBLE_EXEMPLAR, "no neighbour answer maps onto the query's levels");
        out.retrieval_us = retrieval_us;
        out.routing = std::move(diag);
        return out;
    }

    // Probabilities are the normalised vote weights; the prediction is the vote winner.
    const size_t n = task.decision.options.size();
    out = DecisionOutput{};
    out.ok = true;
    out.selected = v.label;
    out.probs.assign(n, 0.0);
    for (size_t i = 0; i < n && i < v.weight.size(); ++i) out.probs[i] = v.weight[i] / v.total_weight;
    out.raw_probs = out.probs;
    for (size_t i = 0; i < n; ++i) out.expected_score += static_cast<double>(i) * out.probs[i];
    out.retrieval_us = retrieval_us;
    out.routing = std::move(diag);
    return out;
}

} // namespace routing
} // namespace pjev
