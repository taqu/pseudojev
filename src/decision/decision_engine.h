#ifndef INC_PJEV_DECISION_ENGINE_H_
#define INC_PJEV_DECISION_ENGINE_H_
#include "../calibration/calibration.h"
#include "../inference/backend.h"
#include "prompt_strategy.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pjev
{
struct DecisionInput
{
    std::string type; // "noul" | "choice" | "score"
    std::string state;
    std::string question; // from "instructions"
    // ordered list of (key, description) — order matters for candidate assignment
    std::vector<std::pair<std::string, std::string>> options;
    bool prior_correction = false;
    std::vector<float> prior_logits; // per-candidate prior logits (size == options.size()); used when prior_correction=true
    bool collect_corrected_logits = false; // populate corrected_logits in output
};

struct DecisionOutput
{
    bool ok = false;
    std::string error;
    int32_t selected = -1;         // index into options
    double p_true = 0.0;           // noul: probability of "true"
    double expected_score = 0.0;   // score: expected value
    std::vector<double> probs;     // per-candidate calibrated probabilities (or raw if calibration disabled)
    std::vector<double> raw_probs; // per-candidate probabilities before temperature scaling
    std::vector<float>  corrected_logits; // post prior-correction, pre temperature (when collect_corrected_logits=true)
    std::vector<std::string> keys; // candidate keys in order
    int32_t prompt_token_count = 0; // number of tokens in the prompt
    int64_t tokenize_us = 0;  // microseconds for prompt tokenization
    int64_t eval_us     = 0;  // microseconds for llama_decode
    int64_t decision_us = 0;  // microseconds for logit extraction + softmax
};

struct BatchConfig {
    bool use_kv_reuse = true;
};

class DecisionEngine
{
public:
    // Throws std::runtime_error if candidate token verification fails.
    DecisionEngine(ILlamaBackend& backend, const PromptConfig& cfg = {},
                   const CalibrationConfig& calib = {});

    // Not thread-safe. External callers must serialize access.
    DecisionOutput decide(const DecisionInput& input);

    // Process multiple questions, reusing the shared-state KV prefix when possible.
    // Falls back to sequential decide() calls if prefix reuse is not applicable.
    // Not thread-safe. Callers must serialize access.
    std::vector<DecisionOutput> decide_batch(const std::vector<DecisionInput>& inputs,
                                              const BatchConfig& cfg = {});

    // Compute candidate logits for a blank (empty state/question) prompt of the given type and n_options.
    // Returns per-candidate raw logits (size = n_options), or empty vector on failure.
    // Used by the experiment framework for prior correction.
    std::vector<float> compute_blank_logits(const std::string& type, int n_options);

private:
    ILlamaBackend&   backend_;
    PromptStrategy   strategy_;
    CalibrationConfig calib_cfg_;
    std::map<std::string, int> cand_token_map_; // internal_text -> token_id

    static std::string restricted_softmax(const std::vector<float>& logits,
                                          std::vector<double>& probs);
    static double expected_level(const std::vector<double>& probs);
    static int32_t argmax(const std::vector<double>& v);

    // Build full prompt tokens for one question.
    std::vector<int32_t> make_prompt_tokens(const DecisionInput& input,
                                             const std::vector<Candidate>& candidates);
    // Complete a DecisionOutput from already-computed logits pointer.
    // cand_ids: token ids of candidates in order.
    // candidates: full candidate metadata (noul_value used for p_true).
    DecisionOutput finish_from_logits(const DecisionInput& input,
                                       const std::vector<std::string>& keys,
                                       const std::vector<Candidate>& candidates,
                                       const std::vector<int32_t>& cand_ids,
                                       const float* logits,
                                       int32_t token_count,
                                       int64_t tokenize_us_val,
                                       int64_t eval_us_val);
};
} // namespace pjev
#endif // INC_PJEV_DECISION_ENGINE_H_
