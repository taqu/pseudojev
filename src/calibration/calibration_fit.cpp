#include "calibration_fit.h"
#include <cmath>
#include <limits>

namespace pjev
{

double mean_nll(const std::vector<CalibrationSample>& samples, double T)
{
    const double inf = std::numeric_limits<double>::infinity();
    if (samples.empty() || !(T > 0.0)) return inf;
    double sum = 0.0;
    for (const auto& s : samples) {
        const double nll = sample_nll(s.logits, s.correct_index, T);
        if (!std::isfinite(nll)) return inf;
        sum += nll;
    }
    return sum / (double)samples.size();
}

FitResult fit_temperature(const std::vector<CalibrationSample>& samples,
                          double T_min, double T_max, int max_iter, double tol)
{
    FitResult result;
    result.n_samples = (int)samples.size();
    if (samples.empty() || !(T_min > 0.0) || !(T_max > T_min)) return result;

    // Golden-section search on log(T).
    const double gr = (std::sqrt(5.0) + 1.0) / 2.0;
    const double lo = std::log(T_min);
    const double hi = std::log(T_max);
    double log_a = lo;
    double log_b = hi;

    double log_c = log_b - (log_b - log_a) / gr;
    double log_d = log_a + (log_b - log_a) / gr;
    double fc    = mean_nll(samples, std::exp(log_c));
    double fd    = mean_nll(samples, std::exp(log_d));

    for (int i = 0; i < max_iter; i++) {
        if (std::fabs(log_b - log_a) < tol) {
            result.converged = true;
            break;
        }
        if (fc < fd) {
            log_b = log_d;
            log_d = log_c; fd = fc;
            log_c = log_b - (log_b - log_a) / gr;
            fc    = mean_nll(samples, std::exp(log_c));
        } else {
            log_a = log_c;
            log_c = log_d; fc = fd;
            log_d = log_a + (log_b - log_a) / gr;
            fd    = mean_nll(samples, std::exp(log_d));
        }
    }

    const double log_t = (log_a + log_b) / 2.0;
    result.temperature = std::exp(log_t);
    result.nll         = mean_nll(samples, result.temperature);
    // The search interval never leaves [lo, hi]; ending within a small margin of an end means the
    // objective still decreases towards that bound.
    const double margin = 1e-4 * (hi - lo);
    result.at_lower_bound = log_t - lo < margin;
    result.at_upper_bound = hi - log_t < margin;
    if (!std::isfinite(result.nll) || !std::isfinite(result.temperature) || result.temperature <= 0.0)
        result.converged = false;
    return result;
}

AlphaTResult fit_alpha_temperature(const std::vector<CalibrationSample>& samples,
                                   double alpha_min, double alpha_max, double alpha_step)
{
    AlphaTResult best;
    best.n_samples = (int)samples.size();
    if (samples.empty() || !(alpha_step > 0.0) || alpha_max < alpha_min) return best;

    // Grid points alpha_min + i * step, each rounded to the nearest 1e-9 so that a decimal grid gives
    // the decimal values (1.7, not an accumulated or multiplied 1.7000000000000002).
    const int steps = (int)std::floor((alpha_max - alpha_min) / alpha_step + 1e-9);
    for (int i = 0; i <= steps; ++i) {
        const double alpha = std::min(alpha_max, std::round((alpha_min + i * alpha_step) * 1e9) / 1e9);
        const FitResult fr = fit_temperature(apply_prior_correction(samples, alpha));
        if (!fr.converged) continue;
        if (!best.converged || fr.nll < best.nll) {
            best.alpha       = alpha;
            best.temperature = fr.temperature;
            best.nll         = fr.nll;
            best.converged   = true;
            best.temperature_at_lower_bound = fr.at_lower_bound;
            best.temperature_at_upper_bound = fr.at_upper_bound;
        }
    }
    if (best.converged) {
        best.alpha_at_lower_bound = best.alpha <= alpha_min;
        best.alpha_at_upper_bound = best.alpha >= alpha_max;
    }
    return best;
}

} // namespace pjev
