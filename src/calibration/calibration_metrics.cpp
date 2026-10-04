#include "calibration_metrics.h"
#include <algorithm>
#include <cmath>

namespace pjev
{

static double safe_log(double p)
{
    static const double EPS = 1e-15;
    return std::log(std::max(p, EPS));
}

PrimitiveCalibrationMetrics compute_primitive_metrics(
    const std::vector<CalibrationSample>& samples,
    double temperature,
    int n_bins)
{
    PrimitiveCalibrationMetrics m;
    m.n_bins = n_bins;
    if (samples.empty() || temperature <= 0.0) return m;

    bool is_score = (samples[0].type == "score");

    double sum_nll   = 0.0;
    double sum_brier = 0.0;
    double sum_mae   = 0.0;
    int    n_correct = 0;
    int    n_valid   = 0;

    std::vector<int>    bin_count(n_bins, 0);
    std::vector<double> bin_conf_sum(n_bins, 0.0);
    std::vector<int>    bin_correct(n_bins, 0);

    for (const auto& s : samples) {
        if (s.logits.empty() ||
            s.correct_index < 0 ||
            s.correct_index >= (int)s.logits.size()) continue;

        std::vector<double> probs;
        if (!temperature_softmax(s.logits, temperature, probs).empty()) continue;

        int n_cands = (int)probs.size();
        n_valid++;

        // NLL
        sum_nll -= safe_log(probs[s.correct_index]);

        // Brier: mean over candidates of (p_i - y_i)^2
        double brier_item = 0.0;
        for (int i = 0; i < n_cands; i++) {
            double y = (i == s.correct_index) ? 1.0 : 0.0;
            double d = probs[i] - y;
            brier_item += d * d;
        }
        sum_brier += brier_item;

        // Accuracy
        int pred = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
        if (pred == s.correct_index) n_correct++;

        // Score MAE: expected value vs ground truth level
        if (is_score) {
            double exp_score = 0.0;
            for (int i = 0; i < n_cands; i++) exp_score += (double)i * probs[i];
            sum_mae += std::fabs(exp_score - s.expected_score);
        }

        // ECE binning using max-class confidence
        double conf = probs[pred];
        int bin = (int)(conf * n_bins);
        if (bin >= n_bins) bin = n_bins - 1;
        if (bin < 0) bin = 0;
        bin_count[bin]++;
        bin_conf_sum[bin] += conf;
        if (pred == s.correct_index) bin_correct[bin]++;
    }

    if (n_valid == 0) return m;
    m.n        = n_valid;
    m.nll      = sum_nll   / n_valid;
    m.brier    = sum_brier / n_valid;
    m.accuracy = (double)n_correct / n_valid;
    if (is_score) m.mae = sum_mae / n_valid;

    // ECE and reliability bins
    double ece_sum = 0.0;
    for (int b = 0; b < n_bins; b++) {
        if (bin_count[b] == 0) continue;
        double bin_acc  = (double)bin_correct[b] / bin_count[b];
        double bin_conf = bin_conf_sum[b] / bin_count[b];
        ece_sum += (double)bin_count[b] * std::fabs(bin_acc - bin_conf);

        ReliabilityBin rb;
        rb.confidence_min  = (double)b / n_bins;
        rb.confidence_max  = (double)(b + 1) / n_bins;
        rb.count           = bin_count[b];
        rb.mean_confidence = bin_conf;
        rb.accuracy        = bin_acc;
        m.reliability_bins.push_back(rb);
    }
    m.ece = ece_sum / n_valid;

    return m;
}

nlohmann::json PrimitiveCalibrationMetrics::to_json() const
{
    nlohmann::json j = {
        {"n",        n},
        {"accuracy", accuracy},
        {"nll",      nll},
        {"brier",    brier},
        {"ece",      ece},
        {"n_bins",   n_bins}
    };
    if (mae >= 0.0) j["mae"] = mae;
    nlohmann::json bins = nlohmann::json::array();
    for (const auto& b : reliability_bins) {
        bins.push_back({
            {"confidence_min",  b.confidence_min},
            {"confidence_max",  b.confidence_max},
            {"count",           b.count},
            {"mean_confidence", b.mean_confidence},
            {"accuracy",        b.accuracy}
        });
    }
    j["reliability_bins"] = bins;
    return j;
}

CalibrationReport build_calibration_report(
    const std::vector<CalibrationSample>& noul_samples,
    const std::vector<CalibrationSample>& choice_samples,
    const std::vector<CalibrationSample>& score_samples,
    double T_noul, double T_choice, double T_score,
    int n_bins)
{
    CalibrationReport r;
    r.noul_raw          = compute_primitive_metrics(noul_samples,   1.0,     n_bins);
    r.noul_calibrated   = compute_primitive_metrics(noul_samples,   T_noul,  n_bins);
    r.choice_raw        = compute_primitive_metrics(choice_samples, 1.0,     n_bins);
    r.choice_calibrated = compute_primitive_metrics(choice_samples, T_choice, n_bins);
    r.score_raw         = compute_primitive_metrics(score_samples,  1.0,     n_bins);
    r.score_calibrated  = compute_primitive_metrics(score_samples,  T_score, n_bins);
    return r;
}

static nlohmann::json primitive_compare_json(
    const PrimitiveCalibrationMetrics& raw,
    const PrimitiveCalibrationMetrics& cal)
{
    auto delta = [](double a, double b) -> double {
        return (a >= 0.0 && b >= 0.0) ? b - a : -1.0;
    };
    return {
        {"n",          raw.n},
        {"raw",        raw.to_json()},
        {"calibrated", cal.to_json()},
        {"delta", {
            {"nll",      delta(raw.nll,      cal.nll)},
            {"brier",    delta(raw.brier,    cal.brier)},
            {"ece",      delta(raw.ece,      cal.ece)},
            {"accuracy", delta(raw.accuracy, cal.accuracy)},
            {"mae",      delta(raw.mae,      cal.mae)}
        }}
    };
}

nlohmann::json CalibrationReport::to_json() const
{
    return {
        {"noul",   primitive_compare_json(noul_raw,   noul_calibrated)},
        {"choice", primitive_compare_json(choice_raw, choice_calibrated)},
        {"score",  primitive_compare_json(score_raw,  score_calibrated)}
    };
}

} // namespace pjev
