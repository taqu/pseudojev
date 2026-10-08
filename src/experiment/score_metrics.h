#ifndef INC_PJEV_SCORE_METRICS_H_
#define INC_PJEV_SCORE_METRICS_H_
// Ordinal metrics for score (S0 digit baseline and later S1/S2 comparisons).
//
// Pipeline:  DecisionOutput -> semantic normalization (ScoreSample) -> compute_score_metrics
//
// NLL / Brier / ECE use the shared compute_primitive_metrics (summed multi-class Brier),
// QWK uses TypeMetrics::qwk, margins use signed_winner_margin / summarize_margins.
// Samples are serialized as "score_items". See doc/metrics.md and doc/s0.md.
#include "../calibration/calibration_metrics.h"
#include "choice_metrics.h"
#include "noul_metrics.h"
#include <cmath>
#include <cstdint>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace pjev
{
// |predicted_level - ground_truth| >= this counts as a large error.
constexpr int32_t SCORE_LARGE_ERROR_THRESHOLD = 2;

// Per-rotation diagnostics for S2 label-rotation ensemble.
struct ScoreRotation
{
    int32_t rotation = 0;
    std::vector<int32_t> level_to_label; // level i -> letter index (0=A,...) = (i+rotation)%n
    std::vector<std::string> labels;     // per level: actual letter string ("A".."Z")
    std::vector<int32_t> token_ids;      // per level
    std::vector<double> raw_logits;      // per level (semantic order, pre prior-correction)
    std::vector<double> corrected_logits; // per level (semantic order, post prior-correction)
    int32_t prediction = -1;             // semantic level of argmax
    double expected_score_r = 0.0;       // per-rotation expected score (T=1 softmax of corrected)
};

// sum_i level_i * p_i. levels[i] is the semantic level of position i. NaN on size mismatch.
double expected_score(const std::vector<double>& probs, const std::vector<int32_t>& levels);

// Counts of |pred - gt| = 0, 1, 2, ... (vector sized to the largest distance + 1).
std::vector<int32_t> error_distance_histogram(const std::vector<int32_t>& predicted,
                                              const std::vector<int32_t>& ground_truth);

struct ScoreSample
{
    std::string source_id;
    // Per candidate position (the order logits / probs use):
    std::vector<int32_t> levels;      // semantic score level
    std::vector<std::string> labels;  // candidate label shown in the prompt ("0", "1", ... for S0)
    std::vector<int32_t> token_ids;   // candidate token id whose logit is read
    int32_t ground_truth = -1;        // semantic level; -1 = unlabeled
    int32_t prediction = -1;          // semantic level of the argmax
    double temperature = 1.0;
    std::vector<double> logits;       // pre-temperature, post prior-correction
    std::vector<double> raw_logits;   // pre prior-correction
    std::vector<double> probs;        // reported (calibrated when an artifact is loaded)
    std::vector<double> raw_probs;    // T = 1
    int32_t prompt_tokens = 0;
    int32_t evaluations = 1;
    std::vector<ScoreRotation> rotations; // S2 only

    bool labeled() const { return ground_truth >= 0; }
    bool correct() const { return labeled() && prediction == ground_truth; }
    int32_t gt_position() const; // position whose level == ground_truth, -1 if none
    double expected_score() const { return pjev::expected_score(probs, levels); }
    double raw_expected_score() const { return pjev::expected_score(raw_probs, levels); }
    double ground_truth_probability() const;
    double signed_margin() const { return signed_winner_margin(logits, gt_position()); }
    double raw_signed_margin() const { return signed_winner_margin(raw_logits, gt_position()); }
    int32_t abs_error() const; // -1 if unlabeled
    double expected_abs_error() const;
    double raw_expected_abs_error() const;
    double expected_argmax_distance() const { return std::abs(expected_score() - prediction); }
    double raw_expected_argmax_distance() const { return std::abs(raw_expected_score() - prediction); }
    std::string selected_label() const; // candidate label at the predicted level ("" if none)

    // Rotation stability (meaningful only when rotations is non-empty).
    int32_t distinct_winners() const;
    bool rotation_disagreement() const { return distinct_winners() > 1; }
    std::vector<double> semantic_logit_variance() const; // per level: Var_r(corrected_logit_r(level_i))
    double mean_semantic_logit_variance() const;
    double max_semantic_logit_variance() const;
    double rotation_expected_score_stddev() const; // stddev of per-rotation expected_score_r

    nlohmann::json to_json() const;
    static ScoreSample from_json(const nlohmann::json& j);
};

struct ScoreMetrics
{
    int32_t n = 0;
    int32_t labeled = 0;
    int32_t n_levels = 0; // largest level count seen (QWK)
    double accuracy = METRIC_UNDEFINED;
    double mae = METRIC_UNDEFINED;                // discrete |argmax level - gt|
    double expected_mae = METRIC_UNDEFINED;       // |expected score - gt|, reported probabilities
    double raw_expected_mae = METRIC_UNDEFINED;   // same at T = 1
    double qwk = METRIC_UNDEFINED;
    double temperature = 1.0;
    PrimitiveCalibrationMetrics calibrated; // at the run's temperature
    PrimitiveCalibrationMetrics raw;        // at T = 1
    MarginSummary signed_margin;
    MarginSummary raw_signed_margin;
    double mean_expected_argmax_distance = METRIC_UNDEFINED;     // reported probabilities, all samples
    double raw_mean_expected_argmax_distance = METRIC_UNDEFINED; // T = 1
    std::vector<int32_t> error_histogram; // labeled samples by |pred - gt|
    int32_t large_error_threshold = SCORE_LARGE_ERROR_THRESHOLD;
    double large_error_rate = METRIC_UNDEFINED;
    int64_t prompt_tokens = 0;
    double evaluations_per_item = METRIC_UNDEFINED;
    // Candidate-token bias diagnostics: how often each label / level is selected (all samples)
    // vs how often each level is the ground truth (labeled samples).
    std::map<std::string, int32_t> selection_by_label;
    std::map<int32_t, int32_t> prediction_by_level;
    std::map<int32_t, int32_t> ground_truth_by_level;
    // S2 rotation diagnostics (only populated when rotations are present in samples).
    bool has_rotations = false;
    int32_t rotation_n = 0;
    double mean_rotations = METRIC_UNDEFINED;
    int32_t samples_with_rotation_disagreement = 0;
    double mean_distinct_winners = METRIC_UNDEFINED;
    double mean_semantic_logit_variance = METRIC_UNDEFINED;
    double max_semantic_logit_variance = METRIC_UNDEFINED;
    double mean_rotation_expected_score_stddev = METRIC_UNDEFINED;

    double rotation_disagreement_rate() const
    {
        return rotation_n > 0 ? (double)samples_with_rotation_disagreement / rotation_n : METRIC_UNDEFINED;
    }

    void merge_into(nlohmann::json& j) const;
};

ScoreMetrics compute_score_metrics(const std::vector<ScoreSample>& samples,
                                   int32_t n_bins = DEFAULT_ECE_BINS);

MarginGain compute_margin_gain(const std::vector<ScoreSample>& baseline,
                               const std::vector<ScoreSample>& target,
                               double tol = MARGIN_GAIN_TOL);

std::string format_score_comparison(const std::string& title,
                                    const std::string& baseline_label, const ScoreMetrics& baseline, int64_t baseline_eval_ms,
                                    const std::string& target_label, const ScoreMetrics& target, int64_t target_eval_ms,
                                    const MarginGain& gain);
} // namespace pjev
#endif // INC_PJEV_SCORE_METRICS_H_
