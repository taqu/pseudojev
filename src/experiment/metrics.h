#ifndef INC_PJEV_METRICS_H_
#define INC_PJEV_METRICS_H_
#include <cmath>
#include <string>
#include <vector>

namespace pjev
{
struct TypeMetrics
{
    int32_t n = 0;
    int32_t labeled = 0;
    int32_t correct = 0;
    double abs_err = 0.0;              // score: sum |predicted_level - actual_level|
    std::vector<int> predicted_levels; // for QWK
    std::vector<int> actual_levels;

    double accuracy() const
    {
        return (labeled > 0) ? (double)correct / labeled : -1.0;
    }
    double mae() const
    {
        return (labeled > 0) ? abs_err / labeled : -1.0;
    }
    double qwk(int n_levels) const; // quadratic weighted kappa
};

inline double TypeMetrics::qwk(int n_levels) const
{
    if(labeled < 2 || n_levels < 2)
        return -1.0;
    // Build confusion matrix
    std::vector<std::vector<int32_t>> conf(n_levels, std::vector<int32_t>(n_levels, 0));
    for(int32_t i = 0; i < (int32_t)predicted_levels.size(); i++) {
        int32_t p = predicted_levels[i], a = actual_levels[i];
        if(p >= 0 && p < n_levels && a >= 0 && a < n_levels)
            conf[a][p]++;
    }
    // Weights: w[i][j] = (i-j)^2 / (n-1)^2
    double O = 0.0, E = 0.0;
    std::vector<int> row_sum(n_levels, 0), col_sum(n_levels, 0);
    for(int32_t i = 0; i < n_levels; i++) {
        for(int32_t j = 0; j < n_levels; j++) {
            row_sum[i] += conf[i][j];
            col_sum[j] += conf[i][j];
        }
    }
    for(int32_t i = 0; i < n_levels; i++) {
        for(int32_t j = 0; j < n_levels; j++) {
            double w = (double)(i - j) * (i - j) / ((n_levels - 1) * (n_levels - 1));
            O += w * conf[i][j];
            E += w * (double)row_sum[i] * col_sum[j] / labeled;
        }
    }
    if(E < 1e-12)
        return 1.0;
    return 1.0 - O / E;
}

struct RunMetrics
{
    int32_t n_total = 0;
    int32_t n_errors = 0;
    TypeMetrics choice;
    TypeMetrics noul;
    TypeMetrics score;
    int32_t score_n_levels = 0; // max n_levels seen for score items
};

// For option-order stability measurement
struct StabilityMetrics
{
    int32_t n_questions = 0;          // unique questions tested
    int32_t n_stable = 0;             // same semantic answer across all orderings
    int32_t n_permutations_per_q = 0; // how many orderings per question
    double stability_rate() const
    {
        return (n_questions > 0) ? (double)n_stable / n_questions : -1.0;
    }
};
} // namespace pjev
#endif // INC_PJEV_METRICS_H_
