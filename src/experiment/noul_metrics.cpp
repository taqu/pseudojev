#include "noul_metrics.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace pjev
{

// Probabilities are clamped to [eps, 1-eps] before log.
static constexpr double kNllEps = 1e-12;

static double safe_log(double p)
{
    return std::log(std::max(p, kNllEps));
}

// Linear interpolation percentile (numpy default / C++ "linear" method).
// Caller must pass a sorted vector.
static double percentile_sorted(const std::vector<double>& sv, double p)
{
    if(sv.empty())
        return std::numeric_limits<double>::quiet_NaN();
    if(sv.size() == 1)
        return sv[0];
    double pos = p * (double)(sv.size() - 1);
    int lo = (int)pos;
    int hi = lo + 1;
    if(hi >= (int)sv.size())
        return sv.back();
    double frac = pos - lo;
    return sv[lo] + frac * (sv[hi] - sv[lo]);
}

NoulMetrics compute_noul_metrics(const std::vector<NoulSample>& samples, int ece_n_bins)
{
    NoulMetrics m;
    m.ece_n_bins = ece_n_bins;

    double sum_nll   = 0.0;
    double sum_brier = 0.0;
    int    n_labeled = 0;

    std::vector<double> signed_margins;

    std::vector<int>    bin_count(ece_n_bins, 0);
    std::vector<double> bin_conf_sum(ece_n_bins, 0.0);
    std::vector<int>    bin_correct_count(ece_n_bins, 0);

    double sum_conf_correct = 0.0;
    int    n_correct        = 0;

    int n_ensemble = 0;
    int n_disagree = 0;

    for(const auto& s: samples) {
        // E1 disagreement uses all ensemble samples (labeled or not)
        if(s.has_ensemble) {
            n_ensemble++;
            if(s.ord1_true != s.ord2_true)
                n_disagree++;
        }

        if(!s.has_ground_truth)
            continue;
        n_labeled++;

        double p_true = std::max(kNllEps, std::min(1.0 - kNllEps, s.p_true_calib));

        // NLL: -log(P(correct class))
        sum_nll -= s.ground_truth ? safe_log(p_true) : safe_log(1.0 - p_true);

        // Brier: binary (p_true - y)^2 where y = 1 or 0
        double y = s.ground_truth ? 1.0 : 0.0;
        double d = p_true - y;
        sum_brier += d * d;

        // Signed semantic margin
        double signed_m = s.ground_truth ? s.semantic_margin : -s.semantic_margin;
        signed_margins.push_back(signed_m);

        // ECE: confidence = max(P(True), P(False))
        double conf = std::max(p_true, 1.0 - p_true);
        int bin = (int)(conf * ece_n_bins);
        if(bin >= ece_n_bins)
            bin = ece_n_bins - 1;
        if(bin < 0)
            bin = 0;
        bin_count[bin]++;
        bin_conf_sum[bin] += conf;
        if(s.correct)
            bin_correct_count[bin]++;

        // Confidence on correct predictions
        if(s.correct) {
            sum_conf_correct += conf;
            n_correct++;
        }
    }

    if(n_labeled == 0)
        return m;

    m.nll   = sum_nll   / n_labeled;
    m.brier = sum_brier / n_labeled;

    // ECE: sum over bins of (bin_fraction * |mean_conf - mean_acc|)
    double ece_sum = 0.0;
    for(int b = 0; b < ece_n_bins; b++) {
        if(bin_count[b] == 0)
            continue;
        double bin_acc  = (double)bin_correct_count[b] / bin_count[b];
        double bin_conf = bin_conf_sum[b] / bin_count[b];
        ece_sum += (double)bin_count[b] * std::fabs(bin_acc - bin_conf);
    }
    m.ece = ece_sum / n_labeled;

    // Margin statistics (sorted for percentiles)
    if(!signed_margins.empty()) {
        std::sort(signed_margins.begin(), signed_margins.end());
        double sum = 0.0;
        for(double v: signed_margins)
            sum += v;
        m.mean_signed_margin   = sum / (double)signed_margins.size();
        m.median_signed_margin = percentile_sorted(signed_margins, 0.5);
        m.p10_signed_margin    = percentile_sorted(signed_margins, 0.1);
        m.min_signed_margin    = signed_margins.front();
    }

    // Mean confidence on correct predictions
    if(n_correct > 0)
        m.mean_confidence_correct = sum_conf_correct / n_correct;

    // E1 order disagreement
    if(n_ensemble > 0) {
        m.order_disagreement_count = n_disagree;
        m.order_disagreement_rate  = (double)n_disagree / n_ensemble;
    }

    return m;
}

nlohmann::json NoulMetrics::to_json() const
{
    auto nan_or_null = [](double v) -> nlohmann::json {
        return std::isnan(v) ? nlohmann::json(nullptr) : nlohmann::json(v);
    };
    return nlohmann::json{
        {"nll",                      nll},
        {"brier",                    brier},
        {"ece",                      ece},
        {"ece_n_bins",               ece_n_bins},
        {"mean_signed_margin",       nan_or_null(mean_signed_margin)},
        {"median_signed_margin",     nan_or_null(median_signed_margin)},
        {"p10_signed_margin",        nan_or_null(p10_signed_margin)},
        {"min_signed_margin",        nan_or_null(min_signed_margin)},
        {"mean_confidence_correct",  nan_or_null(mean_confidence_correct)},
        {"order_disagreement_count", order_disagreement_count},
        {"order_disagreement_rate",  order_disagreement_rate},
    };
}

MarginGainMetrics compute_margin_gain(const std::vector<NoulSample>& e1_samples,
                                       const std::vector<NoulSample>& baseline_samples,
                                       double tolerance)
{
    MarginGainMetrics mg;

    // Build ID → signed_margin map for baseline (labeled samples only)
    std::map<std::string, double> baseline_map;
    for(const auto& s: baseline_samples) {
        if(!s.has_ground_truth || s.id.empty())
            continue;
        double signed_m = s.ground_truth ? s.semantic_margin : -s.semantic_margin;
        baseline_map[s.id] = signed_m;
    }

    std::vector<double> gains;
    for(const auto& s: e1_samples) {
        if(!s.has_ground_truth || s.id.empty())
            continue;
        auto it = baseline_map.find(s.id);
        if(it == baseline_map.end())
            continue;
        double e1_signed = s.ground_truth ? s.semantic_margin : -s.semantic_margin;
        double gain = e1_signed - it->second;
        gains.push_back(gain);
        mg.n_matched++;
        if(std::fabs(gain) < tolerance)
            mg.unchanged_count++;
        else if(gain > 0.0)
            mg.improved_count++;
        else
            mg.degraded_count++;
    }

    if(gains.empty())
        return mg;

    double sum = 0.0;
    for(double g: gains)
        sum += g;
    mg.mean_gain = sum / (double)gains.size();

    std::sort(gains.begin(), gains.end());
    int n = (int)gains.size();
    if(n % 2 == 1)
        mg.median_gain = gains[n / 2];
    else
        mg.median_gain = (gains[n / 2 - 1] + gains[n / 2]) / 2.0;

    return mg;
}

nlohmann::json MarginGainMetrics::to_json() const
{
    return nlohmann::json{
        {"n_matched",      n_matched},
        {"mean_gain",      mean_gain},
        {"median_gain",    median_gain},
        {"improved_count", improved_count},
        {"degraded_count", degraded_count},
        {"unchanged_count",unchanged_count},
    };
}

} // namespace pjev
