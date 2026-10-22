#ifndef INC_PJEV_CALIBRATION_METRICS_H_
#define INC_PJEV_CALIBRATION_METRICS_H_
#include "calibration.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace pjev
{

// One sample used for calibration fitting and evaluation.
struct CalibrationSample
{
    std::vector<float> logits;        // working logits (raw_logits after prior correction at the current alpha)
    std::vector<float> raw_logits;    // pre prior-correction candidate logits
    std::vector<float> prior_logits;  // blank-prompt logits of the same kind and candidate count
    int correct_index = -1;
    std::string type;
    double expected_score = 0.0;      // score: the correct level (for MAE)
};

struct ReliabilityBin
{
    double confidence_min  = 0.0;
    double confidence_max  = 0.0;
    int    count           = 0;
    double mean_confidence = 0.0;
    double accuracy        = 0.0;
};

struct PrimitiveCalibrationMetrics
{
    int    n        = 0;
    double accuracy = -1.0;
    double nll      = -1.0;
    double brier    = -1.0;  // multi-class: sum over candidates of (p_i - y_i)^2, averaged over samples
    double ece      = -1.0;  // max-probability confidence, n_bins equal-width bins
    double mae      = -1.0;  // score only; -1 when not applicable
    double mean_max_prob          = -1.0;
    double mean_conf_correct      = -1.0;  // mean max probability of correct predictions (-1: none)
    double mean_conf_incorrect    = -1.0;  // mean max probability of incorrect predictions (-1: none)
    int    n_bins   = 15;
    std::vector<ReliabilityBin> reliability_bins;  // non-empty bins only

    nlohmann::json to_json() const;
};

// -log softmax(logits / T)[correct], computed with log-sum-exp in double precision. Returns NaN for an
// invalid sample (empty or non-finite logits, correct index out of range, T <= 0).
double sample_nll(const std::vector<float>& logits, int correct_index, double T);

// Calibration metrics for a set of same-type samples (their `logits`) at temperature T. T = 1 on raw
// logits gives the uncalibrated metrics. Accuracy compares the argmax with the exact candidate index.
// Samples that cannot be scored are not counted (n reports the scored ones).
PrimitiveCalibrationMetrics compute_primitive_metrics(
    const std::vector<CalibrationSample>& samples,
    double temperature,
    int n_bins = 15);

// The samples with logits = raw_logits - alpha * prior_logits, in the decision engine's float arithmetic
// (DecisionEngine::finish_from_logits).
std::vector<CalibrationSample> apply_prior_correction(const std::vector<CalibrationSample>& samples, double alpha);

} // namespace pjev
#endif // INC_PJEV_CALIBRATION_METRICS_H_
