#include "task_mapping.h"

#include "lz4/xxhash.h"
#include "retrieval/embedding_text.h"
#include "task_policy.h"

namespace pjev
{
namespace routing
{

namespace
{
// An Open Jev task (kind, options in dataset order) as decision candidates. Mirrors the mapping the
// R1-R4 evaluations used: choice options are both key and text, score levels are keyed by level index,
// noul maps to pjev's (false, true) candidates.
bool openjev_candidates(const std::string& kind, const std::vector<std::string>& options,
                        std::vector<std::pair<std::string, std::string>>& candidates,
                        std::vector<int>& option_to_candidate, std::string& err)
{
    candidates.clear();
    option_to_candidate.clear();
    const size_t n = options.size();
    if (n < 2) { err = "need at least 2 options"; return false; }
    if (n > static_cast<size_t>(PromptStrategy::MAX_CANDIDATES)) {
        err = "too many options (max " + std::to_string(PromptStrategy::MAX_CANDIDATES) + ")";
        return false;
    }
    if (kind == "choice" || kind == "score") {
        for (size_t i = 0; i < n; ++i) {
            candidates.push_back({kind == "choice" ? options[i] : std::to_string(i), options[i]});
            option_to_candidate.push_back(static_cast<int>(i));
        }
        return true;
    }
    if (kind == "noul") {
        if (n != 2) { err = "noul requires exactly 2 options"; return false; }
        int no_idx = -1, yes_idx = -1;
        for (size_t i = 0; i < n; ++i) {
            if (options[i] == "no")  no_idx  = static_cast<int>(i);
            if (options[i] == "yes") yes_idx = static_cast<int>(i);
        }
        if (no_idx < 0 || yes_idx < 0) { err = "noul options must be \"no\" and \"yes\""; return false; }
        candidates = {{"false", ""}, {"true", ""}};
        option_to_candidate.assign(2, -1);
        option_to_candidate[static_cast<size_t>(no_idx)]  = 0;
        option_to_candidate[static_cast<size_t>(yes_idx)] = 1;
        return true;
    }
    err = "unsupported kind: " + kind;
    return false;
}
} // namespace

std::string state_text(const nlohmann::ordered_json& state)
{
    if (state.is_string()) return state.get<std::string>();
    return state.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

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
    return XXH64(buf.data(), buf.size(), 0x5234u);
}

std::string embedding_query_text(const RetrievalTask& task)
{
    return std::string(kQueryPrefix) + retrieval::build_embedding_text(task.kind, task.domain, task.family,
                                                                        task.question, task.state, task.options);
}

bool route_task(const DecisionInput& in, RoutedTask& out, std::string& err)
{
    out = RoutedTask{};
    if (in.type != Type::Choice && in.type != Type::Noul && in.type != Type::Score) {
        err = "unsupported decision type: " + std::string(to_string(in.type));
        return false;
    }
    const size_t n = in.options.size();
    if (n < 2) { err = "need at least 2 candidates"; return false; }
    if (n > static_cast<size_t>(PromptStrategy::MAX_CANDIDATES)) {
        err = "too many candidates (max " + std::to_string(PromptStrategy::MAX_CANDIDATES) + ")";
        return false;
    }

    RetrievalTask& t = out.retrieval;
    t.kind       = to_string(in.type);
    t.domain     = in.domain;
    t.family     = in.family;
    t.question   = in.question;
    t.state_text = in.state;
    // A JSON object state is embedded field by field, exactly like the structured states of the stored
    // exemplars; any other text is embedded as free-text context.
    nlohmann::ordered_json parsed = nlohmann::ordered_json::parse(in.state, nullptr, false);
    t.state = (!parsed.is_discarded() && parsed.is_object()) ? std::move(parsed) : nlohmann::ordered_json(in.state);

    out.decision = in;
    out.decision.prior_correction = false;
    out.decision.prior_logits.clear();
    out.decision.collect_corrected_logits = true;  // raw candidate logits are reported in diagnostics

    switch (in.type) {
    case Type::Choice:
        for (size_t i = 0; i < n; ++i) {
            t.options.push_back(in.options[i].first);
            t.option_to_candidate.push_back(static_cast<int>(i));
            out.to_caller.push_back(static_cast<int>(i));
        }
        break;
    case Type::Score:
        t.number_of_levels = static_cast<int>(n);
        for (size_t i = 0; i < n; ++i) {
            t.options.push_back(in.options[i].second);
            t.option_to_candidate.push_back(static_cast<int>(i));
            out.to_caller.push_back(static_cast<int>(i));
        }
        break;
    case Type::Noul: {
        int false_idx = -1, true_idx = -1;
        for (size_t i = 0; i < n; ++i) {
            if (in.options[i].first == "false") false_idx = static_cast<int>(i);
            if (in.options[i].first == "true")  true_idx  = static_cast<int>(i);
        }
        if (n != 2 || false_idx < 0 || true_idx < 0) {
            err = "noul candidates must be exactly \"false\" and \"true\"";
            return false;
        }
        out.decision.options = {in.options[static_cast<size_t>(false_idx)], in.options[static_cast<size_t>(true_idx)]};
        out.to_caller = {false_idx, true_idx};
        t.options = {"no", "yes"};
        t.option_to_candidate = {0, 1};
        break;
    }
    default:
        break;
    }
    t.content = content_key(t.kind, t.question, t.state_text, t.options);
    return true;
}

bool build_demonstration(const retrieval::ExemplarPayload& payload, Demonstration& demo, std::string& err)
{
    if (payload.answer_index < 0 || payload.answer_index >= static_cast<int>(payload.options.size())) {
        err = "exemplar has no one-hot answer";
        return false;
    }
    std::vector<std::pair<std::string, std::string>> candidates;
    std::vector<int> option_to_candidate;
    if (!openjev_candidates(payload.kind, payload.options, candidates, option_to_candidate, err)) return false;
    demo              = Demonstration{};
    demo.type         = to_type(payload.kind);
    demo.state        = state_text(payload.state);
    demo.question     = payload.question;
    demo.options      = std::move(candidates);
    demo.answer_index = option_to_candidate[static_cast<size_t>(payload.answer_index)];
    return true;
}

} // namespace routing
} // namespace pjev
