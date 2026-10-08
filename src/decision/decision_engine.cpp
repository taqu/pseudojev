#include "decision_engine.h"
#include "rotation.h"
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
                               const CalibrationConfig& calib,
                               const EnsembleConfig& ensemble)
    : backend_(backend)
    , strategy_(cfg)
    , calib_cfg_(calib)
    , ensemble_cfg_(ensemble)
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

    if(input.collect_corrected_logits) {
        out.raw_logits = cand_logits;
    }

    if(input.prior_correction && input.prior_logits.size() == cand_ids.size()) {
        double alpha = calib_cfg_.prior_alpha_for(input.type);
        for(size_t i = 0; i < cand_logits.size(); i++) {
            cand_logits[i] -= (float)(alpha * (double)input.prior_logits[i]);
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
    if(input.type == Type::Score)
        out.expected_score = expected_level(probs);
    if(input.type == Type::Noul) {
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

DecisionOutput DecisionEngine::decide_single(const DecisionInput& input)
{
    return decide_single_impl(input, nullptr, nullptr, nullptr);
}

DecisionOutput DecisionEngine::decide_single_impl(const DecisionInput& input,
                                                  const std::vector<int32_t>* prev_tokens,
                                                  std::vector<int32_t>* tokens_out,
                                                  int32_t* reused_out)
{
    if(reused_out)
        *reused_out = 0;
    DecisionOutput out;
    out.keys.reserve(input.options.size());
    for(const auto& kv: input.options) out.keys.push_back(kv.first);

    if(input.type != Type::Noul && input.type != Type::Choice && input.type != Type::Score) {
        out.error = "unknown decision type: " + std::string(to_string(input.type));
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

    // Shared token prefix with the previously evaluated sequence. Keep at least the last
    // token for evaluation so logits are produced for the final position.
    int32_t prefix_len = 0;
    if(prev_tokens) {
        size_t limit = prompt_tokens.empty() ? 0 : std::min(prev_tokens->size(), prompt_tokens.size() - 1);
        while((size_t)prefix_len < limit && (*prev_tokens)[prefix_len] == prompt_tokens[prefix_len])
            prefix_len++;
    }

    const float* logits = nullptr;
    try {
        auto t2 = now_us();
        if(prefix_len > 0)
            logits = backend_.eval_tokens_reuse_prefix(prompt_tokens, prefix_len);
        else
            logits = backend_.eval_tokens(prompt_tokens);
        auto t3 = now_us();
        out.eval_us = t3 - t2;
    } catch(const std::exception& e) {
        out.error = e.what();
        return out;
    }
    if(reused_out)
        *reused_out = prefix_len;
    if(tokens_out)
        *tokens_out = std::move(prompt_tokens);

    return finish_from_logits(input, out.keys, candidates, cand_ids, logits,
                              out.prompt_token_count, out.tokenize_us, out.eval_us);
}

DecisionOutput DecisionEngine::decide(const DecisionInput& input)
{
    if(input.type == Type::Noul && ensemble_cfg_.noul_mode == NoulEnsembleMode::BINARY_ORDER)
        return decide_noul_ensemble(input);
    if(input.type == Type::Choice && ensemble_cfg_.choice_mode == ChoiceEnsembleMode::CYCLIC_ROTATION)
        return decide_choice_ensemble(input);
    return decide_single(input);
}

DecisionOutput DecisionEngine::decide_choice_ensemble(const DecisionInput& input)
{
    const int32_t n = (int32_t)input.options.size();
    if(input.type != Type::Choice || n < 2) {
        DecisionOutput out;
        out.error = "choice ensemble: requires choice with at least 2 options";
        return out;
    }

    DecisionOutput::ChoiceEnsembleDiag diag;
    std::vector<double> sum_sem(n, 0.0), sum_raw_sem(n, 0.0);
    int32_t total_tokens = 0;
    int64_t total_tokenize_us = 0, total_eval_us = 0;
    std::vector<int32_t> prev_tokens;

    for(int32_t r = 0; r < n; r++) {
        RotationMapping map = make_cyclic_rotation(n, r);

        // Same state / question / descriptions; only the option-to-candidate binding changes.
        // prior_logits stay in candidate order: the prior belongs to the label position.
        DecisionInput inp = input;
        inp.options = to_candidate_order(input.options, map);
        inp.collect_corrected_logits = true;

        const bool reuse = ensemble_cfg_.choice_prefix_reuse && r > 0;
        std::vector<int32_t> tokens;
        int32_t reused = 0;
        DecisionOutput o = decide_single_impl(inp, reuse ? &prev_tokens : nullptr,
                                              ensemble_cfg_.choice_prefix_reuse ? &tokens : nullptr, &reused);
        if(!o.ok)
            return o;
        if((int32_t)o.corrected_logits.size() != n || (int32_t)o.raw_logits.size() != n) {
            DecisionOutput out;
            out.error = "choice ensemble: missing candidate logits";
            return out;
        }
        if(ensemble_cfg_.choice_prefix_reuse)
            prev_tokens = std::move(tokens);

        DecisionOutput::RotationDiag rd;
        rd.rotation = r;
        rd.cand_to_sem = map.cand_to_sem;
        rd.raw_logits = o.raw_logits;
        rd.corrected_logits = o.corrected_logits;
        rd.semantic_logits = to_semantic_order(o.corrected_logits, map);
        rd.prefix_reused = reused;
        std::vector<float> raw_sem = to_semantic_order(o.raw_logits, map);
        rd.semantic_prediction = (int32_t)(std::max_element(rd.semantic_logits.begin(), rd.semantic_logits.end()) -
                                           rd.semantic_logits.begin());
        for(int32_t i = 0; i < n; i++) {
            sum_sem[i] += rd.semantic_logits[i];
            sum_raw_sem[i] += raw_sem[i];
        }
        total_tokens += o.prompt_token_count;
        total_tokenize_us += o.tokenize_us;
        total_eval_us += o.eval_us;
        diag.rotations.push_back(std::move(rd));
    }

    int64_t td0 = now_us();
    DecisionOutput out;
    out.keys.reserve(n);
    for(const auto& kv: input.options) out.keys.push_back(kv.first);
    out.prompt_token_count = total_tokens;
    out.tokenize_us = total_tokenize_us;
    out.eval_us = total_eval_us;

    std::vector<float> mean_sem(n), mean_raw_sem(n);
    for(int32_t i = 0; i < n; i++) {
        diag.mean_semantic_logits.push_back(sum_sem[i] / n);
        diag.mean_raw_semantic_logits.push_back(sum_raw_sem[i] / n);
        mean_sem[i] = (float)diag.mean_semantic_logits[i];
        mean_raw_sem[i] = (float)diag.mean_raw_semantic_logits[i];
    }
    if(input.collect_corrected_logits) {
        out.corrected_logits = mean_sem;
        out.raw_logits = mean_raw_sem;
    }

    // Temperature is applied once, after semantic aggregation.
    std::string err = restricted_softmax(mean_sem, out.raw_probs);
    if(err.empty()) {
        double T = calib_cfg_.temperature_for(Type::Choice);
        if(T != 1.0)
            err = temperature_softmax(mean_sem, T, out.probs);
        else
            out.probs = out.raw_probs;
    }
    if(!err.empty()) {
        out.error = err;
        return out;
    }
    out.selected = argmax(out.probs);
    out.choice_ensemble_diag = std::move(diag);
    out.ok = true;
    out.decision_us = now_us() - td0;
    return out;
}

DecisionOutput DecisionEngine::decide_noul_ensemble(const DecisionInput& input)
{
    // Validate
    if(input.type != Type::Noul || input.options.size() != 2) {
        DecisionOutput out;
        out.error = "ensemble: requires noul with exactly 2 options";
        return out;
    }

    // Ordering 1: original option order; always collect corrected logits
    DecisionInput inp1 = input;
    inp1.collect_corrected_logits = true;

    // Ordering 2: reversed options; prior_logits unchanged (same tokens A/B regardless of semantic binding)
    DecisionInput inp2 = input;
    inp2.options = {input.options[1], input.options[0]};
    inp2.collect_corrected_logits = true;

    DecisionOutput out1 = decide_single(inp1);
    DecisionOutput out2 = decide_single(inp2);

    if(!out1.ok) return out1;
    if(!out2.ok) return out2;

    // Find semantic true/false indices by key
    auto key_idx = [](const std::vector<std::string>& keys, const std::string& k) -> int {
        for(int i = 0; i < (int)keys.size(); i++) if(keys[i] == k) return i;
        return -1;
    };

    int ti1 = key_idx(out1.keys, "true"),  fi1 = key_idx(out1.keys, "false");
    int ti2 = key_idx(out2.keys, "true"),  fi2 = key_idx(out2.keys, "false");

    if(ti1 < 0 || fi1 < 0 || ti2 < 0 || fi2 < 0 ||
       ti1 >= (int)out1.corrected_logits.size() || fi1 >= (int)out1.corrected_logits.size() ||
       ti2 >= (int)out2.corrected_logits.size() || fi2 >= (int)out2.corrected_logits.size()) {
        DecisionOutput out;
        out.error = "ensemble: cannot locate true/false corrected logits";
        return out;
    }

    double m1 = (double)out1.corrected_logits[ti1] - (double)out1.corrected_logits[fi1];
    double m2 = (double)out2.corrected_logits[ti2] - (double)out2.corrected_logits[fi2];
    double m_ensemble = (m1 + m2) / 2.0;

    double T      = calib_cfg_.temperature_for(Type::Noul);
    double p_true = 1.0 / (1.0 + std::exp(-m_ensemble / T));

    // Build output in the original option order
    DecisionOutput out;
    out.ok   = true;
    out.keys = out1.keys;
    out.probs.resize(input.options.size());
    out.raw_probs.resize(input.options.size());
    for(size_t i = 0; i < input.options.size(); i++) {
        bool is_true = (input.options[i].first == "true");
        out.probs[i]     = is_true ? p_true : 1.0 - p_true;
        double p_raw     = 1.0 / (1.0 + std::exp(-m_ensemble));  // T=1
        out.raw_probs[i] = is_true ? p_raw : 1.0 - p_raw;
    }
    out.p_true  = p_true;
    out.selected = (p_true >= 0.5) ? ti1 : fi1;
    out.prompt_token_count = out1.prompt_token_count + out2.prompt_token_count;
    out.eval_us = out1.eval_us + out2.eval_us;

    // Diagnostics
    DecisionOutput::EnsembleDiag diag;
    diag.m1       = m1;
    diag.m2       = m2;
    diag.m_ensemble = m_ensemble;
    diag.p_true_ord1 = 1.0 / (1.0 + std::exp(-m1));   // T=1
    diag.p_true_ord2 = 1.0 / (1.0 + std::exp(-m2));
    diag.ord1_corrected_logits = out1.corrected_logits;
    diag.ord2_corrected_logits = out2.corrected_logits;
    diag.ord1_raw_logits = out1.raw_logits;
    diag.ord2_raw_logits = out2.raw_logits;
    diag.ord1_keys = out1.keys;
    diag.ord2_keys = out2.keys;
    out.ensemble_diag = diag;

    return out;
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
        // E2 evaluates several prompts per question; route through decide().
        if(inp.type == Type::Choice && ensemble_cfg_.choice_mode == ChoiceEnsembleMode::CYCLIC_ROTATION) {
            can_reuse = false;
            break;
        }
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

        if(input.type != Type::Noul && input.type != Type::Choice && input.type != Type::Score) {
            out.error = "unknown decision type: " + std::string(to_string(input.type));
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

std::vector<float> DecisionEngine::compute_blank_logits(Type type, int n_options)
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
