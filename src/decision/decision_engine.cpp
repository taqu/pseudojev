#include "decision_engine.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <set>
#include <stdexcept>

namespace pjev
{
DecisionEngine::DecisionEngine(ILlamaBackend& backend, const PromptConfig& cfg,
                               const CalibrationConfig& calib)
    : backend_(backend), strategy_(cfg), calib_cfg_(calib)
{
    // Verify all candidate tokens for the configured scheme
    std::set<int32_t> seen;
    bool ok = true;
    for (const auto& s : strategy_.all_internal_texts()) {
        auto toks = backend_.tokenize(s, false, false);
        if (toks.size() != 1) {
            ok = false;
            fprintf(stderr, "candidate %s: NOT a single token (%zu tokens)\n",
                    s.c_str(), toks.size());
            continue;
        }
        if (seen.count(toks[0])) {
            ok = false;
            fprintf(stderr, "candidate %s: duplicate token id %d\n", s.c_str(), toks[0]);
            continue;
        }
        seen.insert(toks[0]);
        cand_token_map_[s] = toks[0];
    }
    if (!ok) {
        throw std::runtime_error("candidate token verification failed — see stderr for details");
    }
}

std::string DecisionEngine::restricted_softmax(const std::vector<float>& logits,
                                                std::vector<double>& probs) {
    probs.clear();
    if (logits.empty()){
        return "empty candidate set";
    }
    for (float l : logits) {
        if (!std::isfinite(l)){
            return "non-finite candidate logit";
        }
    }
    const double m = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (float l : logits) {
        probs.push_back(std::exp((double)l - m));
        sum += probs.back();
    }
    double check = 0.0;
    for (double& p : probs) {
        p /= sum;
        if (!std::isfinite(p) || p < 0.0 || p > 1.0){
            return "probability out of range";
        }
        check += p;
    }
    if (std::fabs(check - 1.0) > 1e-9){
        return "probabilities do not sum to 1";
    }
    return "";
}

double DecisionEngine::expected_level(const std::vector<double>& probs) {
    double s = 0.0;
    for (size_t i = 0; i < probs.size(); i++){
        s += (double)i * probs[i];
    }
    return s;
}

int32_t DecisionEngine::argmax(const std::vector<double>& v) {
    return (int32_t)(std::max_element(v.begin(), v.end()) - v.begin());
}

DecisionOutput DecisionEngine::decide(const DecisionInput& input) {
    DecisionOutput out;
    out.keys.reserve(input.options.size());
    for (const auto& kv : input.options) out.keys.push_back(kv.first);

    if (input.type != "noul" && input.type != "choice" && input.type != "score") {
        out.error = "unknown decision type: " + input.type;
        return out;
    }
    if (input.options.size() < 2) {
        out.error = "need at least 2 candidates";
        return out;
    }
    if ((int32_t)input.options.size() > PromptStrategy::MAX_CANDIDATES) {
        out.error = "too many candidates (max " +
                    std::to_string(PromptStrategy::MAX_CANDIDATES) + ")";
        return out;
    }

    // Build candidates with descriptions
    std::vector<Candidate> candidates;
    candidates.reserve(input.options.size());
    for (const auto& kv : input.options) {
        Candidate c;
        c.key         = kv.first;
        c.description = kv.second;
        c.internal    = "";
        candidates.push_back(c);
    }
    strategy_.assign_labels(input.type, candidates);

    // Resolve candidate tokens
    std::set<int32_t> id_set;
    std::vector<int32_t> cand_ids;
    for (const auto& c : candidates) {
        auto it = cand_token_map_.find(c.internal);
        if (it == cand_token_map_.end()) {
            out.error = "candidate " + c.internal + " is not a verified single token";
            return out;
        }
        if (!id_set.insert(it->second).second) {
            out.error = "duplicate candidate token id for " + c.internal;
            return out;
        }
        cand_ids.push_back(it->second);
    }

    // Build prompt token sequence with proper parse_special flags for special-token safety
    auto segments = strategy_.build_prompt_segments(
        input.type, input.state, input.question, candidates);

    std::vector<int32_t> prompt_tokens;
    bool first = true;
    for (const auto& seg : segments) {
        // add_special=true only for the first segment so BOS is added if the model expects it.
        // parse_special=seg.trusted so special-token strings in user content are never parsed.
        auto toks = backend_.tokenize(seg.text, first && seg.trusted, seg.trusted);
        first = false;
        prompt_tokens.insert(prompt_tokens.end(), toks.begin(), toks.end());
    }

    out.prompt_token_count = (int32_t)prompt_tokens.size();

    // Evaluate prompt
    const float* logits = nullptr;
    try {
        logits = backend_.eval_tokens(prompt_tokens);
    } catch (const std::exception& e) {
        out.error = e.what();
        return out;
    }

    // Extract candidate logits
    std::vector<float> cand_logits;
    for (int32_t id : cand_ids) cand_logits.push_back(logits[id]);

    // Apply prior correction if requested
    if (input.prior_correction && input.prior_logits.size() == cand_ids.size()) {
        for (size_t i = 0; i < cand_logits.size(); i++) {
            cand_logits[i] -= input.prior_logits[i];
        }
    }

    // Store corrected logits before temperature if requested
    if (input.collect_corrected_logits) {
        out.corrected_logits = cand_logits;
    }

    // Compute raw probabilities (T=1) for before/after comparison
    {
        std::string err = restricted_softmax(cand_logits, out.raw_probs);
        if (!err.empty()) {
            out.error = err;
            return out;
        }
    }

    // Apply temperature scaling if calibration is enabled
    double T = calib_cfg_.temperature_for(input.type);
    std::vector<double> probs;
    if (T != 1.0) {
        std::string err = temperature_softmax(cand_logits, T, probs);
        if (!err.empty()) {
            out.error = err;
            return out;
        }
    } else {
        probs = out.raw_probs;
    }

    out.probs    = probs;
    out.selected = argmax(probs);
    if (input.type == "score") out.expected_score = expected_level(probs);
    if (input.type == "noul")  out.p_true = probs[1];  // index 1 = "true"
    out.ok = true;
    return out;
}

std::vector<float> DecisionEngine::compute_blank_logits(const std::string& type, int n_options) {
    if (n_options < 2 || n_options > PromptStrategy::MAX_CANDIDATES) return {};

    // Synthetic candidates with empty descriptions
    std::vector<Candidate> candidates;
    candidates.reserve(n_options);
    for (int32_t i = 0; i < n_options; i++) {
        Candidate c;
        c.key = std::to_string(i);
        c.description = "";
        c.internal = "";
        candidates.push_back(c);
    }
    strategy_.assign_labels(type, candidates);

    // Resolve token IDs
    std::vector<int32_t> cand_ids;
    for (const auto& c : candidates) {
        auto it = cand_token_map_.find(c.internal);
        if (it == cand_token_map_.end()) return {};
        cand_ids.push_back(it->second);
    }

    // Build blank prompt
    auto segments = strategy_.build_prompt_segments(type, "", "", candidates);
    std::vector<int32_t> prompt_tokens;
    bool first = true;
    for (const auto& seg : segments) {
        auto toks = backend_.tokenize(seg.text, first && seg.trusted, seg.trusted);
        first = false;
        prompt_tokens.insert(prompt_tokens.end(), toks.begin(), toks.end());
    }

    const float* logits = nullptr;
    try {
        logits = backend_.eval_tokens(prompt_tokens);
    } catch (...) {
        return {};
    }

    std::vector<float> result;
    for (int32_t id : cand_ids){
        result.push_back(logits[id]);
    }
    return result;
}
}

