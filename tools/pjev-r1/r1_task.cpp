#include "r1_task.h"

namespace pjev
{
namespace r1
{

std::string state_text(const nlohmann::ordered_json& state)
{
    if (state.is_string()) return state.get<std::string>();
    return state.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

bool build_task(const std::string&              kind,
                const std::string&              question,
                const nlohmann::ordered_json&   state,
                const std::vector<std::string>& options,
                TaskMapping&                    out,
                std::string&                    err)
{
    out = TaskMapping{};
    out.input.type     = to_type(kind);
    out.input.question = question;
    out.input.state    = state_text(state);
    // Logits are recorded for margins; no calibration or prior correction is applied.
    out.input.collect_corrected_logits = true;

    const size_t n = options.size();
    if (n < 2) { err = "need at least 2 options"; return false; }
    if (n > static_cast<size_t>(PromptStrategy::MAX_CANDIDATES)) {
        err = "too many options (max " + std::to_string(PromptStrategy::MAX_CANDIDATES) + ")";
        return false;
    }

    switch (out.input.type) {
    case Type::Choice:
        // The option identifier is both the semantic key and the text shown to the model.
        for (const auto& o : options) out.input.options.push_back({o, o});
        for (size_t i = 0; i < n; ++i) {
            out.option_to_candidate.push_back(static_cast<int>(i));
            out.candidate_to_option.push_back(static_cast<int>(i));
        }
        return true;
    case Type::Score:
        // Levels in dataset order (lowest to highest), keyed by level index as in the API.
        for (size_t i = 0; i < n; ++i) {
            out.input.options.push_back({std::to_string(i), options[i]});
            out.option_to_candidate.push_back(static_cast<int>(i));
            out.candidate_to_option.push_back(static_cast<int>(i));
        }
        return true;
    case Type::Noul: {
        if (n != 2) { err = "noul requires exactly 2 options"; return false; }
        int no_idx = -1, yes_idx = -1;
        for (size_t i = 0; i < n; ++i) {
            if (options[i] == "no")  no_idx  = static_cast<int>(i);
            if (options[i] == "yes") yes_idx = static_cast<int>(i);
        }
        if (no_idx < 0 || yes_idx < 0) { err = "noul options must be \"no\" and \"yes\""; return false; }
        // Same candidates the API builds for a noul question without criteria.
        out.input.options = {{"false", ""}, {"true", ""}};
        out.option_to_candidate.assign(2, -1);
        out.option_to_candidate[static_cast<size_t>(no_idx)]  = 0;
        out.option_to_candidate[static_cast<size_t>(yes_idx)] = 1;
        out.candidate_to_option = {no_idx, yes_idx};
        return true;
    }
    default:
        err = "unsupported kind: " + kind;
        return false;
    }
}

bool build_demonstration(const r0::ExemplarPayload& payload,
                         Demonstration&             demo,
                         std::string&               err)
{
    if (payload.answer_index < 0 ||
        payload.answer_index >= static_cast<int>(payload.options.size())) {
        err = "exemplar has no one-hot answer";
        return false;
    }
    TaskMapping m;
    if (!build_task(payload.kind, payload.question, payload.state, payload.options, m, err)) {
        return false;
    }
    demo              = Demonstration{};
    demo.type         = m.input.type;
    demo.state        = std::move(m.input.state);
    demo.question     = std::move(m.input.question);
    demo.options      = std::move(m.input.options);
    demo.answer_index = m.option_to_candidate[static_cast<size_t>(payload.answer_index)];
    return true;
}

void fill_condition(const DecisionOutput&           out,
                    const std::vector<std::string>& labels,
                    const std::vector<std::string>& texts,
                    int                             ground_truth,
                    double                          wall_ms,
                    ConditionResult&                res)
{
    res = ConditionResult{};
    res.prompt_tokens = out.prompt_token_count;
    res.prompt_ms     = out.tokenize_us / 1000.0;
    res.eval_ms       = out.eval_us / 1000.0;
    res.scoring_ms    = out.decision_us / 1000.0;
    res.total_ms      = wall_ms;
    if (!out.ok) {
        // LlamaBackend refuses prompts that do not fit the context window.
        res.status = (out.error.rfind("prompt too long", 0) == 0) ? "context_limit" : "inference_error";
        res.error  = out.error;
        return;
    }
    res.status        = "ok";
    res.prediction    = out.selected;
    if (out.selected >= 0 && out.selected < static_cast<int>(labels.size())) {
        res.prediction_label = labels[static_cast<size_t>(out.selected)];
        res.prediction_text  = texts[static_cast<size_t>(out.selected)];
    }
    res.correct       = (out.selected == ground_truth);
    res.logits        = out.raw_logits;
    res.probs         = out.probs;
    res.margin        = winner_margin(out.raw_logits);
    res.signed_margin = signed_margin(out.raw_logits, ground_truth);
}

std::vector<std::string> candidate_labels(const PromptStrategy& strategy, Type type, size_t n)
{
    std::vector<Candidate> c(n);
    strategy.assign_labels(type, c);
    std::vector<std::string> labels;
    for (const auto& x : c) labels.push_back(x.internal);
    return labels;
}

static std::vector<Candidate> make_candidates(const PromptStrategy& strategy, Type type,
    const std::vector<std::pair<std::string, std::string>>& options)
{
    std::vector<Candidate> c;
    for (const auto& kv : options) {
        Candidate x;
        x.key         = kv.first;
        x.description = kv.second;
        c.push_back(x);
    }
    strategy.assign_labels(type, c);
    return c;
}

static std::string join(const std::vector<PromptSegment>& segs)
{
    std::string s;
    for (const auto& seg : segs) s += seg.text;
    return s;
}

std::string render_s1_prompt(const PromptStrategy& strategy, const DecisionInput& input)
{
    return join(strategy.build_prompt_segments(
        input.type, input.state, input.question,
        make_candidates(strategy, input.type, input.options)));
}

std::string render_r1_prompt(const PromptStrategy& strategy, const DecisionInput& input,
                             const Demonstration& demo)
{
    return join(strategy.build_demonstration_prompt_segments(
        input.type, demo.state, demo.question,
        make_candidates(strategy, demo.type, demo.options),
        static_cast<size_t>(demo.answer_index),
        input.state, input.question,
        make_candidates(strategy, input.type, input.options)));
}

} // namespace r1
} // namespace pjev
