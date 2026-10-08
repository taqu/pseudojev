#ifndef INC_PJEV_CHOICE_METRICS_H_
#define INC_PJEV_CHOICE_METRICS_H_
// Confidence, calibration, winner-margin and rotation-stability metrics for choice.
//
// Pipeline:  DecisionOutput -> semantic normalization (ChoiceSample) -> compute_choice_metrics
//
// NLL / Brier / ECE come from the existing compute_primitive_metrics (calibration_metrics.h),
// so choice keeps a single definition (summed multi-class Brier). Samples are serialized as
// "choice_items" so metrics can be recomputed from stored results. See doc/metrics.md.
#include "../calibration/calibration_metrics.h"
#include "noul_metrics.h"
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace pjev
{
// S_gt - max_{i != gt} S_i. NaN if gt is out of range or there is no other option.
double signed_winner_margin(const std::vector<double>& scores, int32_t gt);

struct ChoiceRotation
{
    int32_t rotation = 0;
    std::vector<int32_t> cand_to_sem;     // candidate position -> semantic option index
    std::vector<double> raw_logits;       // candidate order, pre prior-correction
    std::vector<double> corrected_logits; // candidate order, post prior-correction
    std::vector<double> semantic_logits;  // semantic order (corrected)
    int32_t prediction = -1;              // semantic argmax of this rotation
    int32_t prefix_reused = 0;            // KV prefix tokens reused for this rotation
};

struct ChoiceSample
{
    std::string source_id;
    std::vector<std::string> keys; // semantic options in the order the logits use
    int32_t ground_truth = -1;     // index into keys; -1 = unlabeled
    int32_t prediction = -1;       // index into keys
    double temperature = 1.0;      // temperature the run applied to logits
    // Pre-temperature semantic logits. "logits" is what the decision used (post prior-correction,
    // the mean over rotations for E2); "raw_logits" is pre prior-correction.
    std::vector<double> logits;
    std::vector<double> raw_logits;
    std::vector<double> probs; // probabilities the run reported
    int32_t prompt_tokens = 0; // tokens evaluated (summed over rotations)
    std::vector<ChoiceRotation> rotations; // E2 only

    bool labeled() const { return ground_truth >= 0; }
    bool correct() const { return labeled() && prediction == ground_truth; }
    double signed_margin() const { return signed_winner_margin(logits, ground_truth); }
    double raw_signed_margin() const { return signed_winner_margin(raw_logits, ground_truth); }

    // Rotation stability (meaningful only when rotations is non-empty).
    int32_t distinct_winners() const;
    bool rotation_disagreement() const { return distinct_winners() > 1; }
    std::vector<int32_t> winner_counts() const;  // per semantic option, rotations it won
    double winner_agreement() const;             // fraction of rotations whose winner == prediction
    // Population variance Var_r(S_r(option_i)) per option, summarized over options.
    std::vector<double> semantic_logit_variance() const;
    double mean_semantic_logit_variance() const;
    double max_semantic_logit_variance() const;

    nlohmann::json to_json() const;
    static ChoiceSample from_json(const nlohmann::json& j);
};

struct ChoiceMetrics
{
    int32_t n = 0;
    int32_t labeled = 0;
    double accuracy = METRIC_UNDEFINED;
    PrimitiveCalibrationMetrics calibrated; // at the run's temperature
    PrimitiveCalibrationMetrics raw;        // at T = 1
    double temperature = 1.0;
    MarginSummary signed_margin;     // signed winner margin (corrected logits)
    MarginSummary raw_signed_margin; // signed winner margin (pre prior-correction)
    int64_t prompt_tokens = 0;
    // E2 only (labels not required)
    bool has_rotations = false;
    int32_t rotation_n = 0;
    double mean_rotations = METRIC_UNDEFINED; // evaluations per item
    int32_t samples_with_any_rotation_disagreement = 0;
    double mean_distinct_winners = METRIC_UNDEFINED;
    double mean_winner_agreement = METRIC_UNDEFINED;
    double mean_semantic_logit_variance = METRIC_UNDEFINED;     // mean over samples of per-sample mean
    double mean_max_semantic_logit_variance = METRIC_UNDEFINED; // mean over samples of per-sample max
    double max_semantic_logit_variance = METRIC_UNDEFINED;      // max over samples

    double rotation_disagreement_rate() const
    {
        return rotation_n > 0 ? (double)samples_with_any_rotation_disagreement / rotation_n : METRIC_UNDEFINED;
    }
    // Merges the new fields into an existing per-primitive JSON object
    // (never overwrites n / labeled / accuracy / eval_ms).
    void merge_into(nlohmann::json& j) const;
};

// All samples are expected to come from one run (one temperature; the first sample's is used).
ChoiceMetrics compute_choice_metrics(const std::vector<ChoiceSample>& samples,
                                     int32_t n_bins = DEFAULT_ECE_BINS);

MarginGain compute_margin_gain(const std::vector<ChoiceSample>& baseline,
                               const std::vector<ChoiceSample>& target,
                               double tol = MARGIN_GAIN_TOL);

std::string format_choice_comparison(const std::string& title,
                                     const std::string& baseline_label, const ChoiceMetrics& baseline, int64_t baseline_eval_ms,
                                     const std::string& target_label, const ChoiceMetrics& target, int64_t target_eval_ms,
                                     const MarginGain& gain);
} // namespace pjev
#endif // INC_PJEV_CHOICE_METRICS_H_
