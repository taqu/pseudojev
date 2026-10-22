#include "calibration_metrics.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace pjev
{

double sample_nll(const std::vector<float>& logits, int correct_index, double T)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (logits.empty() || correct_index < 0 || correct_index >= (int)logits.size() || !(T > 0.0)) return nan;
    size_t top = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!std::isfinite((double)logits[i] / T)) return nan;
        if (logits[i] > logits[top]) top = i;
    }
    // log-sum-exp as max + log1p(sum of the other terms): exact down to the smallest doubles, so the
    // NLL of a confident correct answer stays strictly positive and decreasing in T.
    const double max_val = (double)logits[top] / T;
    double rest = 0.0;
    for (size_t i = 0; i < logits.size(); ++i)
        if (i != top) rest += std::exp((double)logits[i] / T - max_val);
    const double log_norm = std::log1p(rest);  // log-sum-exp - max
    const double nll = (size_t)correct_index == top ? log_norm
                                                    : log_norm + max_val - (double)logits[(size_t)correct_index] / T;
    return std::isfinite(nll) ? std::max(nll, 0.0) : nan;
}

PrimitiveCalibrationMetrics compute_primitive_metrics(
    const std::vector<CalibrationSample>& samples,
    double temperature,
    int n_bins)
{
    PrimitiveCalibrationMetrics m;
    m.n_bins = n_bins;
    if (samples.empty() || temperature <= 0.0 || n_bins <= 0) return m;

    const bool is_score = (samples[0].type == "score");

    double sum_nll   = 0.0;
    double sum_brier = 0.0;
    double sum_mae   = 0.0;
    double sum_conf  = 0.0, sum_conf_ok = 0.0, sum_conf_bad = 0.0;
    int    n_correct = 0;
    int    n_valid   = 0;

    std::vector<int>    bin_count(n_bins, 0);
    std::vector<double> bin_conf_sum(n_bins, 0.0);
    std::vector<int>    bin_correct(n_bins, 0);

    for (const auto& s : samples) {
        const double nll = sample_nll(s.logits, s.correct_index, temperature);
        if (!std::isfinite(nll)) continue;
        std::vector<double> probs;
        if (!temperature_softmax(s.logits, temperature, probs).empty()) continue;

        const int n_cands = (int)probs.size();
        n_valid++;
        sum_nll += nll;

        double brier_item = 0.0;
        for (int i = 0; i < n_cands; i++) {
            const double d = probs[i] - (i == s.correct_index ? 1.0 : 0.0);
            brier_item += d * d;
        }
        sum_brier += brier_item;

        const int  pred = (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
        const bool ok   = pred == s.correct_index;
        const double conf = probs[pred];
        if (ok) { n_correct++; sum_conf_ok += conf; } else { sum_conf_bad += conf; }
        sum_conf += conf;

        if (is_score) {
            double exp_score = 0.0;
            for (int i = 0; i < n_cands; i++) exp_score += (double)i * probs[i];
            sum_mae += std::fabs(exp_score - s.expected_score);
        }

        int bin = (int)(conf * n_bins);
        bin = std::clamp(bin, 0, n_bins - 1);
        bin_count[bin]++;
        bin_conf_sum[bin] += conf;
        if (ok) bin_correct[bin]++;
    }

    if (n_valid == 0) return m;
    m.n        = n_valid;
    m.nll      = sum_nll   / n_valid;
    m.brier    = sum_brier / n_valid;
    m.accuracy = (double)n_correct / n_valid;
    m.mean_max_prob = sum_conf / n_valid;
    if (n_correct > 0)           m.mean_conf_correct   = sum_conf_ok / n_correct;
    if (n_valid - n_correct > 0) m.mean_conf_incorrect = sum_conf_bad / (n_valid - n_correct);
    if (is_score) m.mae = sum_mae / n_valid;

    double ece_sum = 0.0;
    for (int b = 0; b < n_bins; b++) {
        if (bin_count[b] == 0) continue;
        const double bin_acc  = (double)bin_correct[b] / bin_count[b];
        const double bin_conf = bin_conf_sum[b] / bin_count[b];
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
        {"n",                   n},
        {"accuracy",            accuracy},
        {"nll",                 nll},
        {"brier",               brier},
        {"ece",                 ece},
        {"mean_max_prob",       mean_max_prob},
        {"mean_conf_correct",   mean_conf_correct},
        {"mean_conf_incorrect", mean_conf_incorrect},
        {"n_bins",              n_bins}
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

std::vector<CalibrationSample> apply_prior_correction(const std::vector<CalibrationSample>& samples, double alpha)
{
    std::vector<CalibrationSample> out;
    out.reserve(samples.size());
    for (const auto& s : samples) {
        CalibrationSample cs = s;
        if (!s.raw_logits.empty() && s.raw_logits.size() == s.prior_logits.size()) {
            cs.logits.resize(s.raw_logits.size());
            for (size_t i = 0; i < s.raw_logits.size(); i++)
                cs.logits[i] = s.raw_logits[i] - (float)(alpha * (double)s.prior_logits[i]);
        }
        out.push_back(std::move(cs));
    }
    return out;
}

} // namespace pjev
