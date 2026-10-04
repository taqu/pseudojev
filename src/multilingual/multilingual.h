#ifndef INC_PJEV_MULTILINGUAL_H_
#define INC_PJEV_MULTILINGUAL_H_
#include "../calibration/calibration_metrics.h"
#include "../experiment/runner.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace pjev
{

enum class ErrorTransition
{
    BOTH_CORRECT,
    REF_ONLY_CORRECT,   // EN correct → JA wrong (primary degradation case)
    TARGET_ONLY_CORRECT,
    BOTH_WRONG
};

std::string error_transition_str(ErrorTransition t);

struct PairedRecord
{
    std::string pair_id;
    std::string type;
    std::string difficulty;
    bool ref_correct    = false;
    bool target_correct = false;
    bool semantically_consistent = false; // same semantic key prediction
    ErrorTransition transition = ErrorTransition::BOTH_WRONG;
    double ref_confidence    = 0.0; // probs[selected] for reference
    double target_confidence = 0.0;
    double ref_target_prob    = 0.0; // probs[correct_index] for reference (if labeled)
    double target_target_prob = 0.0;
    int32_t ref_prompt_tokens    = 0;
    int32_t target_prompt_tokens = 0;
    int64_t ref_eval_ms    = 0;
    int64_t target_eval_ms = 0;
};

struct PairedPrimitiveStats
{
    int n = 0;
    int both_correct     = 0;
    int ref_only_correct = 0;
    int target_only_correct = 0;
    int both_wrong       = 0;
    int n_consistent     = 0;

    double ref_accuracy    = -1.0;
    double target_accuracy = -1.0;

    // Calibration metrics (populated when corrected_logits collected)
    PrimitiveCalibrationMetrics ref_cal;
    PrimitiveCalibrationMetrics target_cal;

    double semantic_consistency_rate = -1.0; // n_consistent / n
    double mean_conf_shift           = 0.0;  // mean(target_conf - ref_conf)
    double mean_target_prob_shift    = 0.0;  // mean(target_target_prob - ref_target_prob)

    // Tokenization / latency
    double ref_mean_tokens    = 0.0;
    double target_mean_tokens = 0.0;
    double ref_mean_eval_ms   = 0.0;
    double target_mean_eval_ms = 0.0;

    nlohmann::json to_json() const;
};

struct MultilingualResult
{
    std::string reference_language; // "en"
    std::string target_language;    // "ja"
    std::string model_path;

    PairedPrimitiveStats noul;
    PairedPrimitiveStats choice;
    PairedPrimitiveStats score;

    std::vector<PairedRecord> records; // per-pair detailed records

    nlohmann::json to_json() const;
};

// Build a MultilingualResult by pairing reference and target RunResults on pair_id.
// Items without a pair_id are matched on id as fallback.
// Items with no match in the other run are excluded from paired stats but counted.
MultilingualResult compare_runs(const RunResult& reference,
                                 const RunResult& target,
                                 const std::string& ref_lang = "en",
                                 const std::string& target_lang = "ja",
                                 const std::string& model_path = "");

// Validate that pair semantics are consistent: same type, same option count.
// Returns list of violation descriptions (empty = valid).
std::vector<std::string> validate_pairs(
    const std::vector<DatasetRow>& ref_rows,
    const std::vector<DatasetRow>& target_rows);

} // namespace pjev
#endif // INC_PJEV_MULTILINGUAL_H_
