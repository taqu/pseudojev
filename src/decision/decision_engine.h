#ifndef INC_PJEV_DECISION_ENGINE_H_
#define INC_PJEV_DECISION_ENGINE_H_
#include "../calibration/calibration.h"
#include "../inference/backend.h"
#include "prompt_strategy.h"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace pjev
{

enum class NoulEnsembleMode { NONE, BINARY_ORDER };
enum class ChoiceEnsembleMode { NONE, CYCLIC_ROTATION };

struct EnsembleConfig
{
    NoulEnsembleMode noul_mode = NoulEnsembleMode::NONE;
    ChoiceEnsembleMode choice_mode = ChoiceEnsembleMode::NONE;
    // E2 optimized path: reuse the KV prefix shared with the previous rotation.
    // false = sequential reference path (every rotation evaluated from scratch).
    bool choice_prefix_reuse = false;
};

struct DecisionInput
{
    //std::string type; // "noul" | "choice" | "score"
    Type type;
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
    std::vector<float>  raw_logits;       // pre prior-correction candidate logits (when collect_corrected_logits=true)
    std::vector<std::string> keys; // candidate keys in order
    int32_t prompt_token_count = 0; // number of tokens in the prompt
    int64_t tokenize_us = 0;  // microseconds for prompt tokenization
    int64_t eval_us     = 0;  // microseconds for llama_decode
    int64_t decision_us = 0;  // microseconds for logit extraction + softmax

    // E1 ensemble diagnostics (only populated when NoulEnsembleMode::BINARY_ORDER is active)
    struct EnsembleDiag {
        double m1 = 0.0;               // semantic margin for ordering 1
        double m2 = 0.0;               // semantic margin for ordering 2
        double m_ensemble = 0.0;       // combined semantic margin
        double p_true_ord1 = 0.0;      // P(true) from ordering 1 alone (T=1)
        double p_true_ord2 = 0.0;      // P(true) from ordering 2 alone (T=1)
        std::vector<float> ord1_corrected_logits;
        std::vector<float> ord2_corrected_logits;
        std::vector<float> ord1_raw_logits;    // pre prior-correction
        std::vector<float> ord2_raw_logits;
        std::vector<std::string> ord1_keys;    // candidate key per position for each ordering
        std::vector<std::string> ord2_keys;
    };
    std::optional<EnsembleDiag> ensemble_diag;

    // E2 cyclic rotation diagnostics (only populated when ChoiceEnsembleMode::CYCLIC_ROTATION is active).
    // All semantic vectors are in the original option order (same as keys).
    struct RotationDiag {
        int32_t rotation = 0;
        std::vector<int32_t> cand_to_sem;     // candidate position -> semantic option index
        std::vector<float> raw_logits;        // candidate order, pre prior-correction
        std::vector<float> corrected_logits;  // candidate order, post prior-correction
        std::vector<float> semantic_logits;   // semantic order (corrected)
        int32_t semantic_prediction = -1;     // argmax of semantic_logits
        int32_t prefix_reused = 0;            // KV prefix tokens reused (0 = full evaluation)
    };
    struct ChoiceEnsembleDiag {
        std::vector<RotationDiag> rotations;
        std::vector<double> mean_semantic_logits;     // corrected, pre temperature
        std::vector<double> mean_raw_semantic_logits; // pre prior-correction
    };
    std::optional<ChoiceEnsembleDiag> choice_ensemble_diag;
};

struct BatchConfig {
    bool use_kv_reuse = true;
};

class DecisionEngine
{
public:
    // Throws std::runtime_error if candidate token verification fails.
    DecisionEngine(ILlamaBackend& backend, const PromptConfig& cfg = {},
                   const CalibrationConfig& calib = {},
                   const EnsembleConfig& ensemble = {});

    // Not thread-safe. External callers must serialize access.
    // Dispatches to decide_noul_ensemble when ensemble mode is BINARY_ORDER and type=="noul".
    DecisionOutput decide(const DecisionInput& input);

    // Process multiple questions, reusing the shared-state KV prefix when possible.
    // Falls back to sequential decide() calls if prefix reuse is not applicable.
    // Not thread-safe. Callers must serialize access.
    std::vector<DecisionOutput> decide_batch(const std::vector<DecisionInput>& inputs,
                                              const BatchConfig& cfg = {});

    // Compute candidate logits for a blank (empty state/question) prompt of the given type and n_options.
    // Returns per-candidate raw logits (size = n_options), or empty vector on failure.
    // Used by the experiment framework for prior correction.
    std::vector<float> compute_blank_logits(Type type, int n_options);

private:
    ILlamaBackend&   backend_;
    PromptStrategy   strategy_;
    CalibrationConfig calib_cfg_;
    EnsembleConfig    ensemble_cfg_;
    std::map<std::string, int> cand_token_map_; // internal_text -> token_id

    static std::string restricted_softmax(const std::vector<float>& logits,
                                          std::vector<double>& probs);
    static double expected_level(const std::vector<double>& probs);
    static int32_t argmax(const std::vector<double>& v);

    // Build full prompt tokens for one question.
    std::vector<int32_t> make_prompt_tokens(const DecisionInput& input,
                                             const std::vector<Candidate>& candidates);
    // Complete a DecisionOutput from already-computed logits pointer.
    DecisionOutput finish_from_logits(const DecisionInput& input,
                                       const std::vector<std::string>& keys,
                                       const std::vector<Candidate>& candidates,
                                       const std::vector<int32_t>& cand_ids,
                                       const float* logits,
                                       int32_t token_count,
                                       int64_t tokenize_us_val,
                                       int64_t eval_us_val);

    // Single-ordering inference path (no ensemble dispatch).
    DecisionOutput decide_single(const DecisionInput& input);
    // decide_single with optional KV prefix reuse: when prev_tokens is non-null, the token prefix
    // shared with prev_tokens (the most recently evaluated sequence) is reused. The evaluated
    // prompt tokens are written to tokens_out and the reused length to reused_out when non-null.
    DecisionOutput decide_single_impl(const DecisionInput& input,
                                      const std::vector<int32_t>* prev_tokens,
                                      std::vector<int32_t>* tokens_out,
                                      int32_t* reused_out);

    // Cyclic option-rotation ensemble for choice (E2): evaluates N rotations, remaps candidate
    // logits to semantic option order, averages them, then applies temperature and softmax.
    DecisionOutput decide_choice_ensemble(const DecisionInput& input);

    // Binary option-order ensemble for noul: evaluates both candidate orderings and combines
    // semantic margins. Input must be type=="noul" with exactly 2 options (false/true).
    DecisionOutput decide_noul_ensemble(const DecisionInput& input);
};
} // namespace pjev
#endif // INC_PJEV_DECISION_ENGINE_H_
