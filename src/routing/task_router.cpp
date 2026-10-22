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
        // An unusable exemplar is skipped: it does not count towards the strategy's neighbours.
        auto reject = [&](const std::string& why) {
            spdlog::debug("routing: exemplar {} skipped: {}", retrieval::format_exemplar_id(h.id), why);
        };

        Neighbour n;
        std::string why;
        const retrieval::LoadResult loaded = index_.load(h.id, n.payload, why);
        if (loaded == retrieval::LoadResult::Failed) {
            err = std::move(why);
            return false;
        }
        if (loaded != retrieval::LoadResult::Ok) { reject(why); continue; }
        if (!build_demonstration(n.payload, n.demo, why)) { reject(why); continue; }
        const bool eligible = index_.is_allowed(n.payload.id) && n.payload.kind == task_.kind &&
                              (task_.kind != "score" || n.payload.number_of_levels == task_.number_of_levels);
        if (!eligible) { reject("not eligible for the task"); continue; }
        if (task_.kind == "score") {
            const LabelMap m = map_label(task_.kind, task_.options, task_.option_to_candidate, n.payload.kind,
                                         n.payload.options, n.payload.answer_index);
            if (m.status != MapStatus::Ok) { reject("answer does not map onto the query's levels"); continue; }
            n.label = m.candidate;
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

TaskRouter::TaskRouter(DecisionEngine& engine, ITextEmbedder& embedder, const retrieval::IExemplarIndex& index,
                       bool prior_correction)
    : engine_(engine), embedder_(embedder), index_(index), prior_correction_(prior_correction)
{}

bool TaskRouter::attach_prior(RoutedTask& task, DecisionOutput& out)
{
    if (!prior_correction_) return true;
    DecisionInput& in = task.decision;
    const auto key = std::make_pair(in.type, in.options.size());
    auto it = priors_.find(key);
    if (it == priors_.end()) {
        std::vector<float> prior = engine_.compute_blank_logits(in.type, static_cast<int>(in.options.size()));
        if (prior.size() != in.options.size()) {
            out = failure(ErrorCode::INFERENCE_ERROR, "calibration prior: blank-prompt logits could not be computed");
            return false;
        }
        it = priors_.emplace(key, std::move(prior)).first;
    }
    in.prior_correction = true;
    in.prior_logits     = it->second;
    return true;
}

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

    DecisionOutput::RoutingDiag diag;
    switch (task.decision.type) {
    case Type::Choice: diag.requested_strategy = kNearest2Shot; diag.required_exemplars = kNearest2ShotExemplars; break;
    case Type::Noul:   diag.requested_strategy = kNearest1Shot; diag.required_exemplars = kNearest1ShotExemplars; break;
    case Type::Score:  diag.requested_strategy = kKnn5;         diag.required_exemplars = kKnn5Exemplars;         break;
    default: {
        DecisionOutput out = failure(ErrorCode::UNSUPPORTED_PRIMITIVE, "unsupported decision type");
        to_caller_order(out, task, input);
        return out;
    }
    }
    const size_t required = static_cast<size_t>(diag.required_exemplars);

    DecisionOutput r;
    NeighbourList list(task.retrieval, index_, embedder_.dimension());
    if (!retrieve(task, required, list, r)) {
        to_caller_order(r, task, input);
        return r;
    }
    const std::vector<Neighbour>& nb = list.neighbours();
    diag.strategy           = diag.requested_strategy;
    diag.valid_exemplars    = static_cast<int32_t>(nb.size());
    diag.rejected_exemplars = static_cast<int32_t>(list.rejected());
    record_neighbours(diag, nb);

    DecisionOutput out;
    const bool zero_shot = nb.size() < required;
    if (zero_shot) {
        diag.strategy        = kZeroShot;
        diag.fallback        = true;
        diag.fallback_reason = list.offered() == 0 ? kReasonNoExemplar : kReasonInsufficientExemplars;
    }
    // Calibration prior for LLM decisions only: kNN-5 votes are never calibrated.
    const bool llm = zero_shot || task.decision.type != Type::Score;
    if (!llm || attach_prior(task, out)) {
        if (zero_shot) {
            out = decide_zero_shot(task);
        } else {
            switch (task.decision.type) {
            case Type::Choice: out = decide_nearest_2shot(task, nb, diag); break;
            case Type::Noul:   out = decide_nearest_1shot(task, nb, diag); break;
            default:           out = decide_weighted_knn5(task, nb, diag); break;
            }
        }
    }
    out.retrieval_us += r.retrieval_us;
    out.routing = std::move(diag);
    to_caller_order(out, task, input);
    const auto& d = *out.routing;
    if (spdlog::should_log(spdlog::level::debug))
        spdlog::debug("routing: {} -> {}{} (valid exemplars {}/{}, rejected {}; {})", task.retrieval.kind, d.strategy,
                      d.fallback ? " [fallback from " + d.requested_strategy + ": " + d.fallback_reason + "]" : "",
                      d.valid_exemplars, d.required_exemplars, d.rejected_exemplars,
                      out.ok ? "selected " + std::to_string(out.selected) : "failed: " + out.error_code);
    return out;
}

bool TaskRouter::retrieve(const RoutedTask& task, size_t need, NeighbourList& list, DecisionOutput& out)
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
    out.retrieval_us = now_us() - t0;
    return true;
}

DecisionOutput TaskRouter::decide_nearest_2shot(const RoutedTask& task, const std::vector<Neighbour>& nb,
                                                DecisionOutput::RoutingDiag& diag)
{
    // Order A: second-nearest, nearest, query.
    DecisionOutput out = engine_.decide_with_demonstrations(task.decision, {nb[1].demo, nb[0].demo});
    diag.demonstration_ids = {nb[1].id, nb[0].id};
    // Validated R4 fallback: the 2-shot prompt exceeds the context -> the nearest exemplar alone.
    if (!out.ok && is_context_limit(out)) {
        out = engine_.decide_with_demonstration(task.decision, nb[0].demo);
        diag.strategy          = kNearest1Shot;
        diag.fallback          = true;
        diag.fallback_reason   = kReasonContextLimit;
        diag.demonstration_ids = {nb[0].id};
    }
    classify_engine_error(out);
    return out;
}

DecisionOutput TaskRouter::decide_nearest_1shot(const RoutedTask& task, const std::vector<Neighbour>& nb,
                                                DecisionOutput::RoutingDiag& diag)
{
    DecisionOutput out = engine_.decide_with_demonstration(task.decision, nb[0].demo);
    diag.demonstration_ids = {nb[0].id};
    classify_engine_error(out);
    return out;
}

DecisionOutput TaskRouter::decide_weighted_knn5(const RoutedTask& task, const std::vector<Neighbour>& nb,
                                                DecisionOutput::RoutingDiag& diag)
{
    const int64_t t0 = now_us();
    std::vector<VoteInput> votes;
    for (const auto& n : nb) {
        votes.push_back({n.label, n.similarity});
        diag.neighbour_labels.push_back(n.label);
    }
    const VoteResult v = vote(votes, kKnnK, kKnnTau);
    if (!v.ok) {  // not expected: NeighbourList validated every neighbour's label
        DecisionOutput out = failure(ErrorCode::INFERENCE_ERROR, "kNN vote failed");
        out.retrieval_us = now_us() - t0;
        return out;
    }

    // Probabilities are the normalised vote weights; the prediction is the vote winner.
    const size_t n = task.decision.options.size();
    DecisionOutput out;
    out.ok = true;
    out.selected = v.label;
    out.probs.assign(n, 0.0);
    for (size_t i = 0; i < n && i < v.weight.size(); ++i) out.probs[i] = v.weight[i] / v.total_weight;
    out.raw_probs = out.probs;
    for (size_t i = 0; i < n; ++i) out.expected_score += static_cast<double>(i) * out.probs[i];
    out.retrieval_us = now_us() - t0;
    return out;
}

// The existing S1 path (prompt, candidate mapping and candidate-logit scoring of decide_zero_shot).
DecisionOutput TaskRouter::decide_zero_shot(const RoutedTask& task)
{
    DecisionOutput out = engine_.decide_zero_shot(task.decision);
    classify_engine_error(out);
    return out;
}

} // namespace routing
} // namespace pjev
