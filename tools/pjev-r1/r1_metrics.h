#pragma once

#include <string>
#include <vector>

namespace pjev
{
namespace r1
{

// Paired S1 -> R1 outcome for one query.
enum class Transition { BothCorrect, Rescue, Harm, BothIncorrect };

// Rescue: S1 incorrect, R1 correct. Harm: S1 correct, R1 incorrect.
Transition  classify(bool s1_correct, bool r1_correct);
const char* to_string(Transition t);

struct TransitionCounts {
    int both_correct   = 0;
    int rescue         = 0;
    int harm           = 0;
    int both_incorrect = 0;

    void   add(Transition t);
    int    n() const { return both_correct + rescue + harm + both_incorrect; }
    int    s1_correct() const { return both_correct + harm; }
    int    r1_correct() const { return both_correct + rescue; }
    int    net() const { return rescue - harm; }
    double s1_accuracy() const;
    double r1_accuracy() const;
    // R1 - S1 accuracy; equals net() / n() by construction.
    double accuracy_diff() const;
};

// Paired uncertainty for the accuracy difference.
struct PairedTest {
    double diff         = 0.0;  // R1 - S1 accuracy
    double ci_low       = 0.0;  // 95% Wald interval for a paired difference of proportions
    double ci_high      = 0.0;
    double mcnemar_chi2 = 0.0;  // continuity-corrected
    double mcnemar_p    = 1.0;
    double exact_p      = 1.0;  // two-sided exact binomial on discordant pairs
};
PairedTest paired_test(const TransitionCounts& c);

// Candidate-logit margins (S1 conventions).
// Winner margin: top-1 minus top-2 logit.
double winner_margin(const std::vector<float>& logits);
// Signed margin: ground-truth logit minus the best incorrect logit.
double signed_margin(const std::vector<float>& logits, int ground_truth);

// Binary embeddings are compared by Hamming distance; under sign quantization the angle between
// the original vectors is approximately pi * h / dim, so cos(pi * h / dim) estimates cosine.
double hamming_similarity(double hamming, int dim);
double estimated_cosine(double hamming, int dim);

// Linear-interpolated percentile (p in [0, 100]); 0 for an empty input.
double percentile(std::vector<double> v, double p);
double mean(const std::vector<double>& v);

} // namespace r1
} // namespace pjev
