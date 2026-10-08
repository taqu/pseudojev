#ifndef INC_PJEV_NOUL_METRICS_H_
#define INC_PJEV_NOUL_METRICS_H_
// Confidence, calibration and decision-margin metrics for binary (noul) decisions.
//
// Pipeline:  inference result -> semantic normalization (NoulSample) -> aggregation
//
// Nothing here depends on the inference backend: NoulSample can be rebuilt from a
// stored result JSON ("noul_items") so metrics can be recomputed without llama.cpp.
// See doc/metrics.md for metric definitions.
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace pjev
{
// Probabilities are clamped to [NOUL_PROB_EPS, 1 - NOUL_PROB_EPS] before log().
constexpr double NOUL_PROB_EPS = 1e-12;
// |gain| below this is classified as "unchanged" in baseline-vs-target comparison.
constexpr double MARGIN_GAIN_TOL = 1e-9;
constexpr int32_t DEFAULT_ECE_BINS = 15;

// Undefined metric values are NaN internally and serialized as JSON null.
// (-1.0 cannot be used as a sentinel because margins may legitimately be negative.)
constexpr double METRIC_UNDEFINED = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// Semantic normalization helpers
// ---------------------------------------------------------------------------

// logit(true) - logit(false), located by candidate key ("true"/"false"), never by position.
// Returns NaN if either key is missing or sizes mismatch.
double semantic_margin(const std::vector<float>& logits, const std::vector<std::string>& keys);

// Ground-truth-oriented margin: +margin when ground truth is True, -margin when False.
// Positive => correct side of the decision boundary.
double signed_margin(double semantic_margin, bool ground_truth);

// -log(P(ground truth)), with P clamped by NOUL_PROB_EPS.
double binary_nll(double p_true, bool ground_truth);

// (P(true) - y)^2 with y = 1 for True, 0 for False.
double binary_brier(double p_true, bool ground_truth);

// Percentile with linear interpolation between closest ranks
// (Hyndman & Fan type 7, the numpy/Excel PERCENTILE.INC default):
//   h = (n - 1) * q;  x[floor(h)] + (h - floor(h)) * (x[floor(h)+1] - x[floor(h)])
// q in [0, 1]. Returns NaN for an empty input. Input need not be sorted.
double percentile_linear(std::vector<double> values, double q);

// ---------------------------------------------------------------------------
// Per-example record
// ---------------------------------------------------------------------------

struct NoulSample
{
    std::string source_id;
    bool labeled = false;
    bool ground_truth = false; // valid only when labeled
    bool prediction = false;   // semantic prediction (true = "true")

    double p_true = METRIC_UNDEFINED;     // probability the run reported (calibrated when enabled)
    double p_true_raw = METRIC_UNDEFINED; // same decision at T = 1

    // Pre-temperature semantic margins. "corrected" is what the decision used
    // (post prior-correction; equals raw when prior correction is off).
    // For E1 both are ensemble margins: (m1 + m2) / 2.
    double raw_semantic_margin = METRIC_UNDEFINED;
    double corrected_semantic_margin = METRIC_UNDEFINED;

    // E1 binary order ensemble members (corrected, pre-temperature).
    bool has_orders = false;
    double order1_semantic_margin = METRIC_UNDEFINED;
    double order2_semantic_margin = METRIC_UNDEFINED;

    bool correct() const { return labeled && prediction == ground_truth; }
    double confidence() const; // probability of predicted class, from p_true
    double signed_margin() const; // from corrected_semantic_margin; NaN if unlabeled
    bool order1_prediction() const { return order1_semantic_margin > 0.0; }
    bool order2_prediction() const { return order2_semantic_margin > 0.0; }
    bool order_disagreement() const { return has_orders && order1_prediction() != order2_prediction(); }

    nlohmann::json to_json() const;
    static NoulSample from_json(const nlohmann::json& j);
};

// ---------------------------------------------------------------------------
// Aggregates
// ---------------------------------------------------------------------------

// Probability-quality metrics over labeled samples for one probability set.
struct BinaryProbMetrics
{
    int32_t n = 0; // labeled samples used
    double nll = METRIC_UNDEFINED;
    double brier = METRIC_UNDEFINED;
    double ece = METRIC_UNDEFINED;
    double mean_confidence_correct = METRIC_UNDEFINED;
    int32_t ece_bins = DEFAULT_ECE_BINS;

    nlohmann::json to_json() const;
};

// p_true[i] / ground_truth[i] / prediction[i] must have equal sizes.
// ECE uses n_bins equal-width bins [b/n, (b+1)/n); confidence 1.0 goes to the last bin.
BinaryProbMetrics compute_binary_prob_metrics(const std::vector<double>& p_true,
                                              const std::vector<bool>& ground_truth,
                                              const std::vector<bool>& prediction,
                                              int32_t n_bins = DEFAULT_ECE_BINS);

struct MarginSummary
{
    int32_t n = 0;
    double mean = METRIC_UNDEFINED;
    double median = METRIC_UNDEFINED;
    double p10 = METRIC_UNDEFINED;
    double min = METRIC_UNDEFINED;

    nlohmann::json to_json() const;
};

MarginSummary summarize_margins(const std::vector<double>& values);

struct NoulMetrics
{
    int32_t n = 0;       // samples with a valid decision
    int32_t labeled = 0; // samples with ground truth
    double accuracy = METRIC_UNDEFINED; // recomputed from samples (matches TypeMetrics::accuracy)
    BinaryProbMetrics calibrated; // from p_true (what the run reported)
    BinaryProbMetrics raw;        // from p_true_raw (T = 1)
    MarginSummary signed_margin;     // corrected margin, ground-truth aligned
    MarginSummary raw_signed_margin; // pre prior-correction margin, ground-truth aligned
    // E1 only
    bool has_orders = false;
    int32_t order_n = 0; // samples with both order members (labels not required)
    int32_t order_disagreement_count = 0;

    double order_disagreement_rate() const
    {
        return order_n > 0 ? (double)order_disagreement_count / order_n : METRIC_UNDEFINED;
    }
    // Merges the new fields into an existing per-primitive JSON object
    // (never overwrites n / labeled / accuracy / eval_ms).
    void merge_into(nlohmann::json& j) const;
};

NoulMetrics compute_noul_metrics(const std::vector<NoulSample>& samples,
                                 int32_t n_bins = DEFAULT_ECE_BINS);

// Baseline vs target, matched by source_id over labeled samples present in both.
struct MarginGain
{
    int32_t n_matched = 0;
    double mean_gain = METRIC_UNDEFINED;
    double median_gain = METRIC_UNDEFINED;
    int32_t improved = 0;
    int32_t degraded = 0;
    int32_t unchanged = 0;
    int32_t correct_to_wrong = 0; // baseline correct -> target wrong
    int32_t wrong_to_correct = 0; // baseline wrong -> target correct

    nlohmann::json to_json() const;
};

// One labeled sample reduced to what a baseline-vs-target comparison needs (any primitive).
struct MatchedOutcome
{
    std::string source_id;
    double signed_margin = METRIC_UNDEFINED;
    bool correct = false;
};

MarginGain compute_margin_gain(const std::vector<MatchedOutcome>& baseline,
                               const std::vector<MatchedOutcome>& target,
                               double tol = MARGIN_GAIN_TOL);
MarginGain compute_margin_gain(const std::vector<NoulSample>& baseline,
                               const std::vector<NoulSample>& target,
                               double tol = MARGIN_GAIN_TOL);

// Human-readable side-by-side table. eval_ms < 0 is printed as "-".
std::string format_noul_comparison(const std::string& title,
                                   const std::string& baseline_label, const NoulMetrics& baseline, int64_t baseline_eval_ms,
                                   const std::string& target_label, const NoulMetrics& target, int64_t target_eval_ms,
                                   const MarginGain& gain);
// Margin-gain and correctness-flip lines shared by the noul and choice reports.
std::string format_margin_gain(const std::string& baseline_label, const std::string& target_label,
                               const MarginGain& gain);
} // namespace pjev
#endif // INC_PJEV_NOUL_METRICS_H_
