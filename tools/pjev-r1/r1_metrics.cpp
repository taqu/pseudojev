#include "r1_metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace pjev
{
namespace r1
{

Transition classify(bool s1_correct, bool r1_correct)
{
    if (s1_correct) return r1_correct ? Transition::BothCorrect : Transition::Harm;
    return r1_correct ? Transition::Rescue : Transition::BothIncorrect;
}

const char* to_string(Transition t)
{
    switch (t) {
    case Transition::BothCorrect:   return "both_correct";
    case Transition::Rescue:        return "rescue";
    case Transition::Harm:          return "harm";
    case Transition::BothIncorrect: return "both_incorrect";
    }
    return "unknown";
}

void TransitionCounts::add(Transition t)
{
    switch (t) {
    case Transition::BothCorrect:   both_correct++;   break;
    case Transition::Rescue:        rescue++;         break;
    case Transition::Harm:          harm++;           break;
    case Transition::BothIncorrect: both_incorrect++; break;
    }
}

double TransitionCounts::s1_accuracy() const
{
    return n() > 0 ? static_cast<double>(s1_correct()) / n() : 0.0;
}

double TransitionCounts::r1_accuracy() const
{
    return n() > 0 ? static_cast<double>(r1_correct()) / n() : 0.0;
}

double TransitionCounts::accuracy_diff() const
{
    return n() > 0 ? static_cast<double>(net()) / n() : 0.0;
}

// log C(n, k)
static double log_choose(int n, int k)
{
    return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

PairedTest paired_test(const TransitionCounts& c)
{
    PairedTest t;
    const int n = c.n();
    if (n == 0) return t;
    const double b  = c.rescue;
    const double cc = c.harm;
    t.diff = (b - cc) / n;
    // Var(d) = [(b + c) - (b - c)^2 / n] / n^2
    double var = ((b + cc) - (b - cc) * (b - cc) / n) / (static_cast<double>(n) * n);
    double se  = std::sqrt(std::max(0.0, var));
    t.ci_low  = t.diff - 1.96 * se;
    t.ci_high = t.diff + 1.96 * se;

    const int d = c.rescue + c.harm;
    if (d > 0) {
        double x = std::fabs(b - cc) - 1.0;
        t.mcnemar_chi2 = (x > 0 ? x * x : 0.0) / d;
        // chi-square with 1 df: p = erfc(sqrt(x / 2))
        t.mcnemar_p = std::erfc(std::sqrt(t.mcnemar_chi2 / 2.0));

        // Exact: 2 * P(X <= min(b, c)), X ~ Bin(d, 0.5)
        int k = std::min(c.rescue, c.harm);
        double p = 0.0;
        for (int i = 0; i <= k; ++i) p += std::exp(log_choose(d, i) - d * std::log(2.0));
        t.exact_p = std::min(1.0, 2.0 * p);
    }
    return t;
}

double winner_margin(const std::vector<float>& logits)
{
    if (logits.size() < 2) return 0.0;
    std::vector<float> v = logits;
    std::partial_sort(v.begin(), v.begin() + 2, v.end(), std::greater<float>());
    return static_cast<double>(v[0]) - static_cast<double>(v[1]);
}

double signed_margin(const std::vector<float>& logits, int ground_truth)
{
    if (ground_truth < 0 || ground_truth >= static_cast<int>(logits.size()) || logits.size() < 2) {
        return 0.0;
    }
    double best_wrong = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < logits.size(); ++i) {
        if (static_cast<int>(i) != ground_truth) best_wrong = std::max(best_wrong, (double)logits[i]);
    }
    return static_cast<double>(logits[static_cast<size_t>(ground_truth)]) - best_wrong;
}

double hamming_similarity(double hamming, int dim)
{
    return dim > 0 ? 1.0 - hamming / dim : 0.0;
}

double estimated_cosine(double hamming, int dim)
{
    static const double kPi = 3.14159265358979323846;
    return dim > 0 ? std::cos(kPi * hamming / dim) : 0.0;
}

double percentile(std::vector<double> v, double p)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double pos = (p / 100.0) * (v.size() - 1);
    size_t lo  = static_cast<size_t>(std::floor(pos));
    size_t hi  = std::min(lo + 1, v.size() - 1);
    double f   = pos - lo;
    return v[lo] * (1.0 - f) + v[hi] * f;
}

double mean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / v.size();
}

} // namespace r1
} // namespace pjev
