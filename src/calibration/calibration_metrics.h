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
    std::vector<float> logits; // corrected logits (post prior-correction, pre temperature)
    int correct_index = -1;    // ground-truth candidate index
    std::string type;          // "noul" | "choice" | "score"
    double expected_score = 0.0; // ground-truth level as float (score: same as correct_index)
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
    double brier    = -1.0;
    double ece      = -1.0;
    double mae      = -1.0; // score only; -1 when not applicable
    int    n_bins   = 10;
    std::vector<ReliabilityBin> reliability_bins;

    nlohmann::json to_json() const;
};

struct CalibrationReport
{
    PrimitiveCalibrationMetrics noul_raw,   noul_calibrated;
    PrimitiveCalibrationMetrics choice_raw, choice_calibrated;
    PrimitiveCalibrationMetrics score_raw,  score_calibrated;

    nlohmann::json to_json() const;
};

// Compute calibration metrics for a set of same-type samples at a given temperature.
// T = 1.0 produces uncalibrated metrics.
PrimitiveCalibrationMetrics compute_primitive_metrics(
    const std::vector<CalibrationSample>& samples,
    double temperature,
    int n_bins = 10);

// Build a full before/after report for all three primitive types.
CalibrationReport build_calibration_report(
    const std::vector<CalibrationSample>& noul_samples,
    const std::vector<CalibrationSample>& choice_samples,
    const std::vector<CalibrationSample>& score_samples,
    double T_noul, double T_choice, double T_score,
    int n_bins = 10);

} // namespace pjev
#endif // INC_PJEV_CALIBRATION_METRICS_H_
