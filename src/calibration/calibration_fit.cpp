#include "calibration_fit.h"
#include "calibration_metrics.h"
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

AlphaTResult fit_alpha_temperature(
    const std::vector<CalibrationSample>& tuning_samples,
    const std::vector<CalibrationSample>& val_samples,
    double alpha_min, double alpha_max, double alpha_step)
{
    AlphaTResult best;
    best.n_tuning = (int)tuning_samples.size();
    best.n_val    = (int)val_samples.size();

    if (tuning_samples.empty()) return best;

    bool use_val = !val_samples.empty();

    for (double alpha = alpha_min; alpha <= alpha_max + alpha_step * 0.5; alpha += alpha_step) {
        // Apply alpha correction to raw_logits
        auto apply = [&](const std::vector<CalibrationSample>& src) {
            std::vector<CalibrationSample> out;
            out.reserve(src.size());
            for (const auto& s : src) {
                CalibrationSample cs = s;
                if (!s.raw_logits.empty() && s.raw_logits.size() == s.prior_logits.size()) {
                    cs.logits.resize(s.raw_logits.size());
                    for (size_t i = 0; i < s.raw_logits.size(); i++)
                        cs.logits[i] = s.raw_logits[i] - (float)(alpha * (double)s.prior_logits[i]);
                }
                out.push_back(std::move(cs));
            }
            return out;
        };

        auto corrected_tuning = apply(tuning_samples);
        FitResult fr = fit_temperature(corrected_tuning);
        if (!fr.converged) continue;

        double score = fr.nll;  // fallback: use tuning NLL
        if (use_val) {
            auto corrected_val = apply(val_samples);
            auto m = compute_primitive_metrics(corrected_val, fr.temperature);
            score = m.nll;
        }

        bool better = !best.converged || score < (use_val ? best.val_nll : best.tuning_nll);
        if (better) {
            best.alpha      = alpha;
            best.temperature = fr.temperature;
            best.tuning_nll  = fr.nll;
            best.val_nll     = use_val ? score : 1e30;
            best.converged   = true;
        }
    }
    return best;
}

} // namespace pjev
