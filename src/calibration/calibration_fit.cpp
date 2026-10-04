#include "calibration_fit.h"
#include <algorithm>
#include <cmath>

namespace pjev
{

static double nll_at_T(const std::vector<CalibrationSample>& samples, double T)
{
    if (samples.empty() || T <= 0.0) return 1e30;
    static const double EPS = 1e-15;
    double sum   = 0.0;
    int    count = 0;
    for (const auto& s : samples) {
        if (s.logits.empty() ||
            s.correct_index < 0 ||
            s.correct_index >= (int)s.logits.size()) continue;
        std::vector<double> probs;
        if (!temperature_softmax(s.logits, T, probs).empty()) continue;
        sum -= std::log(std::max(probs[s.correct_index], EPS));
        count++;
    }
    return (count > 0) ? sum / count : 1e30;
}

FitResult fit_temperature(const std::vector<CalibrationSample>& samples,
                           double T_min, double T_max, int max_iter, double tol)
{
    FitResult result;
    result.n_samples = (int)samples.size();
    if (samples.empty()) return result;

    // Golden-section search on log(T) — convex objective for well-behaved logit distributions.
    const double gr = (std::sqrt(5.0) + 1.0) / 2.0;
    double log_a = std::log(T_min);
    double log_b = std::log(T_max);

    double log_c = log_b - (log_b - log_a) / gr;
    double log_d = log_a + (log_b - log_a) / gr;
    double fc    = nll_at_T(samples, std::exp(log_c));
    double fd    = nll_at_T(samples, std::exp(log_d));

    for (int i = 0; i < max_iter; i++) {
        if (std::fabs(log_b - log_a) < tol) {
            result.converged = true;
            break;
        }
        if (fc < fd) {
            log_b = log_d;
            log_d = log_c; fd = fc;
            log_c = log_b - (log_b - log_a) / gr;
            fc    = nll_at_T(samples, std::exp(log_c));
        } else {
            log_a = log_c;
            log_c = log_d; fc = fd;
            log_d = log_a + (log_b - log_a) / gr;
            fd    = nll_at_T(samples, std::exp(log_d));
        }
    }

    result.temperature = std::exp((log_a + log_b) / 2.0);
    result.nll         = nll_at_T(samples, result.temperature);
    return result;
}

} // namespace pjev
