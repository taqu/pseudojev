#include "decision_engine.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>

namespace pjev
{
namespace
{
    int64_t now_us()
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
} // namespace

DecisionEngine::DecisionEngine(ILlamaBackend& backend, const PromptConfig& cfg,
                               const CalibrationConfig& calib)
    : backend_(backend)
    , strategy_(cfg)
    , calib_cfg_(calib)
{
    // Verify all candidate tokens for the configured scheme
    std::set<int32_t> seen;
    bool ok = true;
    for(const auto& s: strategy_.all_internal_texts()) {
        std::vector<int32_t> toks = backend_.tokenize(s, false, false);
        if(toks.size() != 1) {
            ok = false;
            fprintf(stderr, "candidate %s: NOT a single token (%zu tokens)\n",
                    s.c_str(), toks.size());
            continue;
        }
        if(seen.count(toks[0])) {
            ok = false;
            fprintf(stderr, "candidate %s: duplicate token id %d\n", s.c_str(), toks[0]);
            continue;
        }
        seen.insert(toks[0]);
        cand_token_map_[s] = toks[0];
    }
    if(!ok) {
        throw std::runtime_error("candidate token verification failed — see stderr for details");
    }
}

std::string DecisionEngine::restricted_softmax(const std::vector<float>& logits,
                                               std::vector<double>& probs)
{
    probs.clear();
    if(logits.empty()) {
        return "empty candidate set";
    }
    for(float l: logits) {
        if(!std::isfinite(l)) {
            return "non-finite candidate logit";
        }
    }
    const double m = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for(float l: logits) {
        probs.push_back(std::exp((double)l - m));
        sum += probs.back();
    }
    double check = 0.0;
    for(double& p: probs) {
        p /= sum;
        if(!std::isfinite(p) || p < 0.0 || p > 1.0) {
            return "probability out of range";
        }
        check += p;
    }
    if(std::fabs(check - 1.0) > 1e-9) {
        return "probabilities do not sum to 1";
    }
    return "";
}

double DecisionEngine::expected_level(const std::vector<double>& probs)
{
    double s = 0.0;
    for(size_t i = 0; i < probs.size(); i++) {
        s += (double)i * probs[i];
    }
    return s;
}

int32_t DecisionEngine::argmax(const std::vector<double>& v)
{
    return (int32_t)(std::max_element(v.begin(), v.end()) - v.begin());
}

std::vector<int32_t> DecisionEngine::make_prompt_tokens(
    const DecisionInput& input, const std::vector<Candidate>& candidates)
{
    std::vector<PromptSegment> segments = strategy_.build_prompt_segments(
        input.type, input.state, input.question, candidates);

    std::vector<int32_t> prompt_tokens;
    bool first = true;
    for(const auto& seg: segments) {
        auto toks = backend_.tokenize(seg.text, first && seg.trusted, seg.trusted);
        first = false;
        prompt_tokens.insert(prompt_tokens.end(), toks.begin(), toks.end());
    }
    return prompt_tokens;
}

DecisionOutput DecisionEngine::finish_from_logits(
    const DecisionInput& input,
    const std::vector<std::string>& keys,
    const std::vector<Candidate>& candidates,
    const std::vector<int32_t>& cand_ids,
    const float* logits,
    int32_t token_count,
    int64_t tokenize_us_val,
    int64_t eval_us_val)
{
    DecisionOutput out;
    out.keys = keys;
    out.prompt_token_count = token_count;
    out.tokenize_us = tokenize_us_val;
    out.eval_us = eval_us_val;

    int64_t td0 = now_us();

    std::vector<float> cand_logits;
    for(int32_t id: cand_ids) cand_logits.push_back(logits[id]);

    if(input.prior_correction && input.prior_logits.size() == cand_ids.size()) {
        for(size_t i = 0; i < cand_logits.size(); i++) {
            cand_logits[i] -= input.prior_logits[i];
        }
    }

    if(input.collect_corrected_logits) {
        out.corrected_logits = cand_logits;
    }

    {
        std::string err = restricted_softmax(cand_logits, out.raw_probs);
        if(!err.empty()) {
            out.error = err;
            out.decision_us = now_us() - td0;
            return out;
        }
    }

    double T = calib_cfg_.temperature_for(input.type);
    std::vector<double> probs;
    if(T != 1.0) {
        std::string err = temperature_softmax(cand_logits, T, probs);
        if(!err.empty()) {
            out.error = err;
            out.decision_us = now_us() - td0;
            return out;
        }
    } else {
        probs = out.raw_probs;
    }

    out.probs = probs;
    out.selected = argmax(probs);
    if(input.type == "score")
        out.expected_score = expected_level(probs);
    if(input.type == "noul") {
        for(size_t i = 0; i < candidates.size(); ++i) {
            if(candidates[i].noul_value.has_value() && *candidates[i].noul_value == NoulValue::True) {
                out.p_true = probs[i];
                break;
            }
        }
    }
    out.ok = true;
    out.decision_us = now_us() - td0;
    return out;
}

DecisionOutput DecisionEngine::decide(const DecisionInput& input)
{
    DecisionOutput out;
    out.keys.reserve(input.options.size());
    for(const auto& kv: input.options) out.keys.push_back(kv.first);

    if(input.type != "noul" && input.type != "choice" && input.type != "score") {
        out.error = "unknown decision type: " + input.type;
        return out;
    }
    if(input.options.size() < 2) {
        out.error = "need at least 2 candidates";
        return out;
    }
    if((int32_t)input.options.size() > PromptStrategy::MAX_CANDIDATES) {
        out.error = "too many candidates (max " + std::to_string(PromptStrategy::MAX_CANDIDATES) + ")";
        return out;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(input.options.size());
    for(const auto& kv: input.options) {
        Candidate c;
        c.key = kv.first;
        c.description = kv.second;
        c.internal = "";
        candidates.push_back(c);
    }
    strategy_.assign_labels(input.type, candidates);

    std::set<int32_t> id_set;
    std::vector<int32_t> cand_ids;
    for(const auto& c: candidates) {
        auto it = cand_token_map_.find(c.internal);
        if(it == cand_token_map_.end()) {
            out.error = "candidate " + c.internal + " is not a verified single token";
            return out;
        }
        if(!id_set.insert(it->second).second) {
            out.error = "duplicate candidate token id for " + c.internal;
            return out;
        }
        cand_ids.push_back(it->second);
    }

    auto t0 = now_us();
    auto prompt_tokens = make_prompt_tokens(input, candidates);
    auto t1 = now_us();
    out.prompt_token_count = (int32_t)prompt_tokens.size();
    out.tokenize_us = t1 - t0;

    const float* logits = nullptr;
    try {
        auto t2 = now_us();
        logits = backend_.eval_tokens(prompt_tokens);
        auto t3 = now_us();
        out.eval_us = t3 - t2;
    } catch(const std::exception& e) {
        out.error = e.what();
        return out;
    }

    return finish_from_logits(input, out.keys, candidates, cand_ids, logits,
                              out.prompt_token_count, out.tokenize_us, out.eval_us);
}

std::vector<DecisionOutput> DecisionEngine::decide_batch(
    const std::vector<DecisionInput>& inputs, const BatchConfig& cfg)
{
    if(inputs.empty())
        return {};
    if(inputs.size() == 1 || !cfg.use_kv_reuse) {
        std::vector<DecisionOutput> r;
        r.reserve(inputs.size());
        for(const auto& inp: inputs) r.push_back(decide(inp));
        return r;
    }

    const std::string& shared_state = inputs[0].state;
    bool can_reuse = true;
    for(const auto& inp: inputs) {
        if(inp.state != shared_state) {
            can_reuse = false;
            break;
        }
        if(!strategy_.build_prefix_info(inp.type, inp.state).valid) {
            can_reuse = false;
            break;
        }
    }

    if(!can_reuse) {
        std::vector<DecisionOutput> r;
        r.reserve(inputs.size());
        for(const auto& inp: inputs) r.push_back(decide(inp));
        return r;
    }

    PromptStrategy::PrefixInfo pinfo = strategy_.build_prefix_info(inputs[0].type, shared_state);
    std::vector<int32_t> seg0_toks = backend_.tokenize(pinfo.seg0_text, true, true);
    std::vector<int32_t> partial_toks = backend_.tokenize(pinfo.partial_content, false, false);
    int32_t prefix_len = (int32_t)(seg0_toks.size() + partial_toks.size());

    std::vector<DecisionOutput> results;
    results.reserve(inputs.size());

    for(size_t i = 0; i < inputs.size(); i++) {
        const auto& input = inputs[i];
        DecisionOutput out;
        out.keys.reserve(input.options.size());
        for(const auto& kv: input.options) out.keys.push_back(kv.first);

        if(input.type != "noul" && input.type != "choice" && input.type != "score") {
            out.error = "unknown decision type: " + input.type;
            results.push_back(out);
            continue;
        }
        if(input.options.size() < 2) {
            out.error = "need at least 2 candidates";
            results.push_back(out);
            continue;
        }
        if((int32_t)input.options.size() > PromptStrategy::MAX_CANDIDATES) {
            out.error = "too many candidates (max " + std::to_string(PromptStrategy::MAX_CANDIDATES) + ")";
            results.push_back(out);
            continue;
        }

        std::vector<Candidate> candidates;
        candidates.reserve(input.options.size());
        for(const auto& kv: input.options) {
            Candidate c;
            c.key = kv.first;
            c.description = kv.second;
            c.internal = "";
            candidates.push_back(c);
        }
        strategy_.assign_labels(input.type, candidates);

        std::set<int32_t> id_set;
        std::vector<int32_t> cand_ids;
        bool cand_ok = true;
        for(const auto& c: candidates) {
            auto it = cand_token_map_.find(c.internal);
            if(it == cand_token_map_.end()) {
                out.error = "candidate " + c.internal + " is not a verified single token";
                cand_ok = false;
                break;
            }
            if(!id_set.insert(it->second).second) {
                out.error = "duplicate candidate token id for " + c.internal;
                cand_ok = false;
                break;
            }
            cand_ids.push_back(it->second);
        }
        if(!cand_ok) {
            results.push_back(out);
            continue;
        }

        auto t0 = now_us();
        auto full_tokens = make_prompt_tokens(input, candidates);
        auto t1 = now_us();
        out.tokenize_us = t1 - t0;
        out.prompt_token_count = (int32_t)full_tokens.size();

        const float* logits = nullptr;
        try {
            auto t2 = now_us();
            if(i == 0) {
                logits = backend_.eval_tokens(full_tokens);
            } else {
                logits = backend_.eval_tokens_reuse_prefix(full_tokens, prefix_len);
            }
            auto t3 = now_us();
            out.eval_us = t3 - t2;
        } catch(const std::exception& e) {
            out.error = e.what();
            results.push_back(out);
            continue;
        }

        results.push_back(finish_from_logits(input, out.keys, candidates, cand_ids, logits,
                                             out.prompt_token_count, out.tokenize_us, out.eval_us));
    }
    return results;
}

std::vector<float> DecisionEngine::compute_blank_logits(const std::string& type, int n_options)
{
    if(n_options < 2 || n_options > PromptStrategy::MAX_CANDIDATES)
        return {};

    // Synthetic candidates with empty descriptions
    std::vector<Candidate> candidates;
    candidates.reserve(n_options);
    for(int32_t i = 0; i < n_options; i++) {
        Candidate c;
        c.key = std::to_string(i);
        c.description = "";
        c.internal = "";
        candidates.push_back(c);
    }
    strategy_.assign_labels(type, candidates);

    // Resolve token IDs
    std::vector<int32_t> cand_ids;
    for(const auto& c: candidates) {
        auto it = cand_token_map_.find(c.internal);
        if(it == cand_token_map_.end())
            return {};
        cand_ids.push_back(it->second);
    }

    // Build blank prompt
    auto segments = strategy_.build_prompt_segments(type, "", "", candidates);
    std::vector<int32_t> prompt_tokens;
    bool first = true;
    for(const auto& seg: segments) {
        std::vector<int32_t> toks = backend_.tokenize(seg.text, first && seg.trusted, seg.trusted);
        first = false;
        prompt_tokens.insert(prompt_tokens.end(), toks.begin(), toks.end());
    }

    const float* logits = nullptr;
    try {
        logits = backend_.eval_tokens(prompt_tokens);
    } catch(...) {
        return {};
    }

    std::vector<float> result;
    for(int32_t id: cand_ids) {
        result.push_back(logits[id]);
    }
    return result;
}
} // namespace pjev
