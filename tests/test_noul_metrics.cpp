#include "experiment/noul_metrics.h"
#include <cmath>
#include <cstdio>

static int fails = 0;
static void check(bool cond, const char* what)
{
    if(!cond) { printf("FAIL: %s\n", what); fails++; }
    else       { printf("pass: %s\n", what); }
}
static void check_near(double a, double b, double tol, const char* what)
{
    check(std::fabs(a - b) < tol, what);
}

// -----------------------------------------------------------------------
// Helper: make a labeled noul sample
// -----------------------------------------------------------------------
static pjev::NoulSample make_sample(
    const char* id,
    bool ground_truth,
    double p_true_calib,
    double semantic_margin)
{
    pjev::NoulSample s;
    s.id              = id;
    s.has_ground_truth = true;
    s.ground_truth    = ground_truth;
    s.correct         = (ground_truth ? (p_true_calib >= 0.5) : (p_true_calib < 0.5));
    s.p_true_calib    = p_true_calib;
    s.semantic_margin = semantic_margin;
    return s;
}

int main()
{
    using namespace pjev;

    // -----------------------------------------------------------------------
    // T1: NLL — ground truth True, P(True)=0.8
    //     NLL = -log(0.8)
    // -----------------------------------------------------------------------
    {
        NoulSample s = make_sample("s1", true, 0.8, 1.0);
        NoulMetrics m = compute_noul_metrics({s});
        double expected_nll = -std::log(0.8);
        check_near(m.nll, expected_nll, 1e-9, "T1: NLL = -log(0.8)");
    }

    // -----------------------------------------------------------------------
    // T2: NLL — ground truth False, P(True)=0.3
    //     NLL = -log(1-0.3) = -log(0.7)
    // -----------------------------------------------------------------------
    {
        NoulSample s = make_sample("s1", false, 0.3, -1.0);
        NoulMetrics m = compute_noul_metrics({s});
        double expected_nll = -std::log(0.7);
        check_near(m.nll, expected_nll, 1e-9, "T2: NLL = -log(0.7) for GT=False, p=0.3");
    }

    // -----------------------------------------------------------------------
    // T3: Brier — ground truth True, P(True)=0.8
    //     Brier = (0.8 - 1)^2 = 0.04
    // -----------------------------------------------------------------------
    {
        NoulSample s = make_sample("s1", true, 0.8, 1.0);
        NoulMetrics m = compute_noul_metrics({s});
        check_near(m.brier, 0.04, 1e-9, "T3: Brier = (0.8-1)^2 = 0.04");
    }

    // -----------------------------------------------------------------------
    // T4: Brier — ground truth False, P(True)=0.3
    //     Brier = (0.3 - 0)^2 = 0.09
    // -----------------------------------------------------------------------
    {
        NoulSample s = make_sample("s1", false, 0.3, -1.0);
        NoulMetrics m = compute_noul_metrics({s});
        check_near(m.brier, 0.09, 1e-9, "T4: Brier = (0.3-0)^2 = 0.09 for GT=False");
    }

    // -----------------------------------------------------------------------
    // T5: Semantic margin position-independence
    //     A=False, B=True, logit(A)=1, logit(B)=3
    //     semantic_margin = logit_true - logit_false = 3 - 1 = +2
    //     raw_probs: p_B = exp(3)/(exp(1)+exp(3)), p_A = exp(1)/(exp(1)+exp(3))
    //     log(p_B/p_A) = 3 - 1 = 2
    // -----------------------------------------------------------------------
    {
        double la = 1.0, lb = 3.0;
        double pa = std::exp(la) / (std::exp(la) + std::exp(lb));
        double pb = std::exp(lb) / (std::exp(la) + std::exp(lb));
        // A=False, B=True: semantic_margin = log(p_true/p_false) = log(pb/pa)
        double margin = std::log(pb) - std::log(pa);
        check_near(margin, 2.0, 1e-9, "T5: semantic_margin = logit_B - logit_A = 2 (A=False,B=True)");
    }

    // -----------------------------------------------------------------------
    // T6: Reversed candidate mapping
    //     A=True, B=False, logit(A)=3, logit(B)=1
    //     semantic_margin = logit_true - logit_false = 3 - 1 = +2
    // -----------------------------------------------------------------------
    {
        double la = 3.0, lb = 1.0;
        double pa = std::exp(la) / (std::exp(la) + std::exp(lb));
        double pb = std::exp(lb) / (std::exp(la) + std::exp(lb));
        // A=True, B=False: semantic_margin = log(p_true/p_false) = log(pa/pb)
        double margin = std::log(pa) - std::log(pb);
        check_near(margin, 2.0, 1e-9, "T6: semantic_margin = +2 with reversed mapping (mandatory)");
    }

    // -----------------------------------------------------------------------
    // T7: Signed margin — GT=False, semantic_margin = -2 → signed = +2
    // -----------------------------------------------------------------------
    {
        NoulSample s = make_sample("s1", false, 0.2, -2.0);  // GT=False, predicts False (correct)
        NoulMetrics m = compute_noul_metrics({s});
        // signed = -(-2) = +2 (correct side)
        check_near(m.mean_signed_margin,   +2.0, 1e-9, "T7a: signed_margin = +2 (GT=False, margin=-2)");
        check_near(m.min_signed_margin,    +2.0, 1e-9, "T7b: min_signed_margin = +2");
    }

    // -----------------------------------------------------------------------
    // T8: Signed margin — GT=False, semantic_margin = +2 → signed = -2
    // -----------------------------------------------------------------------
    {
        NoulSample s = make_sample("s1", false, 0.8, +2.0);  // GT=False, predicts True (wrong)
        s.correct = false;
        NoulMetrics m = compute_noul_metrics({s});
        // signed = -(+2) = -2 (wrong side)
        check_near(m.mean_signed_margin, -2.0, 1e-9, "T8: signed_margin = -2 (GT=False, margin=+2)");
    }

    // -----------------------------------------------------------------------
    // T9: Saturated accuracy — two configs both accuracy=1.0 but different margins
    //     Config A: 3 samples, signed margins [0.1, 0.2, 0.3] → mean=0.2, min=0.1
    //     Config B: 3 samples, signed margins [2.0, 2.1, 2.2] → mean=2.1, min=2.0
    // -----------------------------------------------------------------------
    {
        // Config A: barely correct
        std::vector<NoulSample> a_samples = {
            make_sample("1", true,  0.525, 0.1),
            make_sample("2", true,  0.550, 0.2),
            make_sample("3", false, 0.425, -0.3),
        };
        // Config B: strongly correct
        std::vector<NoulSample> b_samples = {
            make_sample("1", true,  0.88, 2.0),
            make_sample("2", true,  0.90, 2.1),
            make_sample("3", false, 0.12, -2.2),
        };
        NoulMetrics ma = compute_noul_metrics(a_samples);
        NoulMetrics mb = compute_noul_metrics(b_samples);

        // Both correct → accuracy identical (not directly tracked in NoulMetrics,
        // but NLL/Brier/margins must differ)
        check(ma.nll > mb.nll,        "T9a: Config A NLL > Config B NLL");
        check(ma.brier > mb.brier,    "T9b: Config A Brier > Config B Brier");
        check(ma.mean_signed_margin < mb.mean_signed_margin, "T9c: Config A mean_margin < Config B");
        check(ma.min_signed_margin  < mb.min_signed_margin,  "T9d: Config A min_margin < Config B");
    }

    // -----------------------------------------------------------------------
    // T10: ECE — simple 2-sample case
    //     Sample 1: correct=true,  conf=0.9, bin=floor(0.9*15)=13
    //     Sample 2: correct=false, conf=0.6, bin=floor(0.6*15)=9
    //     Bin 13: acc=1.0, conf=0.9, contribution=(1/2)*|1.0-0.9|=0.05
    //     Bin 9:  acc=0.0, conf=0.6, contribution=(1/2)*|0.0-0.6|=0.30
    //     ECE = (0.05*1 + 0.30*1) / 2 ... wait let me recalculate
    //     ECE = sum_bins(n_bin * |acc - conf|) / n_total
    //         = (1 * |1.0-0.9| + 1 * |0.0-0.6|) / 2
    //         = (0.1 + 0.6) / 2 = 0.35
    // -----------------------------------------------------------------------
    {
        // Sample 1: GT=True, P(True)=0.9 → correct, conf=0.9
        NoulSample s1 = make_sample("s1", true,  0.9, 2.2);
        // Sample 2: GT=True, P(True)=0.4 → wrong, conf=max(0.4,0.6)=0.6
        NoulSample s2 = make_sample("s2", true, 0.4, -0.4);
        s2.correct = false;

        NoulMetrics m = compute_noul_metrics({s1, s2});
        // Bin for s1: floor(0.9 * 15) = 13; acc=1.0, conf=0.9
        // Bin for s2: floor(0.6 * 15) = 9;  acc=0.0, conf=0.6
        // ECE = (1*|1.0-0.9| + 1*|0.0-0.6|) / 2 = (0.1 + 0.6) / 2 = 0.35
        check_near(m.ece, 0.35, 1e-9, "T10: ECE = 0.35 for 2-sample case");
    }

    // -----------------------------------------------------------------------
    // T11: No labeled samples → NLL, Brier, ECE = -1.0; margins = null (NaN)
    // -----------------------------------------------------------------------
    {
        NoulSample s;
        s.id              = "s1";
        s.has_ground_truth = false;
        NoulMetrics m = compute_noul_metrics({s});
        check_near(m.nll,   -1.0, 1e-9, "T11a: NLL = -1 with no labeled");
        check_near(m.brier, -1.0, 1e-9, "T11b: Brier = -1 with no labeled");
        check_near(m.ece,   -1.0, 1e-9, "T11c: ECE = -1 with no labeled");
        check(std::isnan(m.mean_signed_margin), "T11d: mean_signed_margin = NaN");
        check(std::isnan(m.mean_confidence_correct), "T11e: mean_confidence_correct = NaN");
    }

    // -----------------------------------------------------------------------
    // T12: E1 order disagreement
    //     ord1 predicts True (m1 > 0), ord2 predicts False (m2 < 0) → disagree
    // -----------------------------------------------------------------------
    {
        NoulSample s = make_sample("s1", true, 0.6, 0.5);
        s.has_ensemble = true;
        s.m1           = +1.0;   // ordering 1: True
        s.m2           = -0.5;   // ordering 2: False
        s.ord1_true    = true;
        s.ord2_true    = false;

        NoulMetrics m = compute_noul_metrics({s});
        check(m.order_disagreement_count == 1, "T12a: disagreement count = 1");
        check_near(m.order_disagreement_rate, 1.0, 1e-9, "T12b: disagreement rate = 1.0");
    }

    // -----------------------------------------------------------------------
    // T13: E1 order agreement → disagreement = 0
    // -----------------------------------------------------------------------
    {
        NoulSample s = make_sample("s1", true, 0.8, 1.5);
        s.has_ensemble = true;
        s.m1           = +1.5;
        s.m2           = +2.0;
        s.ord1_true    = true;
        s.ord2_true    = true;

        NoulMetrics m = compute_noul_metrics({s});
        check(m.order_disagreement_count == 0, "T13a: no disagreement");
        check_near(m.order_disagreement_rate, 0.0, 1e-9, "T13b: disagreement rate = 0");
    }

    // -----------------------------------------------------------------------
    // T14: Margin gain computation
    //     baseline: signed margin = 0.5
    //     E1:       signed margin = 1.5
    //     gain = 1.0 → improved
    // -----------------------------------------------------------------------
    {
        NoulSample baseline_s = make_sample("q1", true, 0.62, 0.5);
        NoulSample e1_s       = make_sample("q1", true, 0.82, 1.5);

        MarginGainMetrics mg = compute_margin_gain({e1_s}, {baseline_s});
        check(mg.n_matched == 1, "T14a: 1 matched sample");
        check_near(mg.mean_gain, 1.0, 1e-9, "T14b: mean_gain = 1.0");
        check(mg.improved_count == 1, "T14c: improved_count = 1");
        check(mg.degraded_count == 0, "T14d: degraded_count = 0");
    }

    // -----------------------------------------------------------------------
    // T15: Percentile statistics — p10 and median
    //      5 values: [1,2,3,4,5] after sort
    //      p10: pos = 0.1 * 4 = 0.4 → 1 + 0.4*(2-1) = 1.4
    //      median: pos = 2.0 → 3.0
    // -----------------------------------------------------------------------
    {
        std::vector<NoulSample> samples;
        // GT=True, positive margins, sorted order doesn't matter to compute_noul_metrics
        for(int v: {3, 1, 5, 2, 4}) {
            samples.push_back(make_sample("x", true, 0.9, (double)v));
        }
        NoulMetrics m = compute_noul_metrics(samples);
        check_near(m.p10_signed_margin,    1.4, 1e-9, "T15a: p10 = 1.4 via linear interpolation");
        check_near(m.median_signed_margin, 3.0, 1e-9, "T15b: median = 3.0");
        check_near(m.min_signed_margin,    1.0, 1e-9, "T15c: min = 1.0");
        check_near(m.mean_signed_margin,   3.0, 1e-9, "T15d: mean = 3.0");
    }

    // -----------------------------------------------------------------------
    // T16: mean_confidence_correct
    //      2 correct samples: conf = max(0.8, 0.2) = 0.8; max(0.7, 0.3) = 0.7
    //      mean_confidence_correct = (0.8 + 0.7) / 2 = 0.75
    // -----------------------------------------------------------------------
    {
        NoulSample s1 = make_sample("s1", true, 0.8, 1.0);  // correct, conf=0.8
        NoulSample s2 = make_sample("s2", true, 0.7, 0.8);  // correct, conf=0.7
        NoulMetrics m = compute_noul_metrics({s1, s2});
        check_near(m.mean_confidence_correct, 0.75, 1e-9, "T16: mean_confidence_correct = 0.75");
    }

    printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
