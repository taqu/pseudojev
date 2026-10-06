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
    std::vector<float> logits;        // working logits (at current alpha)
    std::vector<float> raw_logits;    // pre prior-correction (for alpha grid search)
    std::vector<float> prior_logits;  // blank-prompt logits (for alpha grid search)
    int correct_index = -1;
    std::string type;
    double expected_score = 0.0;
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
    int    n_bins   = 15;
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

struct CalibrationComparisonRow
{
    std::string label;        // "raw", "temperature", "prior", "prior+temperature"
    double alpha        = 0.0;
    double temperature  = 1.0;
    PrimitiveCalibrationMetrics metrics;
    nlohmann::json to_json() const;
};

struct CalibrationComparisonReport
{
    std::vector<CalibrationComparisonRow> noul_rows;
    std::vector<CalibrationComparisonRow> choice_rows;
    std::vector<CalibrationComparisonRow> score_rows;
    nlohmann::json to_json() const;
};

// Compute calibration metrics for a set of same-type samples at a given temperature.
// T = 1.0 produces uncalibrated metrics.
PrimitiveCalibrationMetrics compute_primitive_metrics(
    const std::vector<CalibrationSample>& samples,
    double temperature,
    int n_bins = 15);

// Build a full before/after report for all three primitive types.
CalibrationReport build_calibration_report(
    const std::vector<CalibrationSample>& noul_samples,
    const std::vector<CalibrationSample>& choice_samples,
    const std::vector<CalibrationSample>& score_samples,
    double T_noul, double T_choice, double T_score,
    int n_bins = 15);

// Build a 4-way comparison: raw / temperature-only / prior-only / prior+temperature.
// Requires CalibrationSamples with raw_logits and prior_logits populated.
CalibrationComparisonReport build_comparison_report(
    const std::vector<CalibrationSample>& noul_samples,
    const std::vector<CalibrationSample>& choice_samples,
    const std::vector<CalibrationSample>& score_samples,
    double T_noul,   double T_choice,   double T_score,
    double A_noul,   double A_choice,   double A_score,
    int n_bins = 15);

} // namespace pjev
#endif // INC_PJEV_CALIBRATION_METRICS_H_
