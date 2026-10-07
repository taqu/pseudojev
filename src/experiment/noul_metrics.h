#ifndef INC_PJEV_NOUL_METRICS_H_
#define INC_PJEV_NOUL_METRICS_H_
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace pjev
{

// One noul sample for confidence/calibration/margin metric computation.
// Populated from ItemResult after inference; independent of LlamaBackend.
struct NoulSample
{
    std::string id;
    bool has_ground_truth  = false;
    bool ground_truth      = false;   // true = semantic True
    bool correct           = false;
    double p_true_calib    = 0.0;     // calibrated P(True)
    // logit_true - logit_false before temperature scaling.
    // Derived from log(raw_probs[true] / raw_probs[false]).
    double semantic_margin = 0.0;
    // E1 ensemble fields (populated when NoulEnsembleMode::BINARY_ORDER is active)
    bool   has_ensemble    = false;
    bool   ord1_true       = false;   // ordering-1 predicted True (m1 > 0)
    bool   ord2_true       = false;   // ordering-2 predicted True (m2 > 0)
    double m1              = 0.0;
    double m2              = 0.0;
};

struct NoulMetrics
{
    // NLL, Brier, ECE — calibrated probabilities, labeled samples only.
    // -1.0 when no labeled samples.
    double nll        = -1.0;
    double brier      = -1.0;
    double ece        = -1.0;
    int    ece_n_bins = 15;

    // Signed semantic margin statistics — labeled samples only.
    // JSON null (NaN sentinel) when no labeled samples.
    double mean_signed_margin   = std::numeric_limits<double>::quiet_NaN();
    double median_signed_margin = std::numeric_limits<double>::quiet_NaN();
    double p10_signed_margin    = std::numeric_limits<double>::quiet_NaN();
    double min_signed_margin    = std::numeric_limits<double>::quiet_NaN();

    // Mean max(P(True),P(False)) over correct predictions.
    // JSON null when there are no correct predictions.
    double mean_confidence_correct = std::numeric_limits<double>::quiet_NaN();

    // E1: order disagreement across samples with ensemble data.
    // order_disagreement_rate = -1.0 when no ensemble samples are present.
    int    order_disagreement_count = 0;
    double order_disagreement_rate  = -1.0;

    nlohmann::json to_json() const;
};

// Compute NoulMetrics from a set of NoulSamples.
// ece_n_bins: number of equal-width confidence bins (default 15).
// Percentile interpolation: linear (numpy default / C++ standard method).
// NLL epsilon: 1e-12 (probabilities clamped before log).
NoulMetrics compute_noul_metrics(const std::vector<NoulSample>& samples,
                                  int ece_n_bins = 15);

// Per-sample margin gain between two matched runs.
struct MarginGainMetrics
{
    int    n_matched        = 0;    // samples matched by ID
    double mean_gain        = -1.0;
    double median_gain      = -1.0;
    int    improved_count   = 0;
    int    degraded_count   = 0;
    int    unchanged_count  = 0;

    nlohmann::json to_json() const;
};

// Match labeled NoulSamples from two runs by ID and compute signed margin gain.
// tolerance: |gain| < tolerance is classified as unchanged.
MarginGainMetrics compute_margin_gain(const std::vector<NoulSample>& e1_samples,
                                       const std::vector<NoulSample>& baseline_samples,
                                       double tolerance = 1e-9);

} // namespace pjev
#endif // INC_PJEV_NOUL_METRICS_H_
