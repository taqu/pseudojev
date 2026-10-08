#include "experiment/noul_metrics.h"
#include "experiment/runner.h"
#include "mock_backend.h"
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

static int fails = 0;
static void check(bool cond, const char* what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
    else       { printf("pass: %s\n", what); }
}
static void check_near(double a, double b, double tol, const char* what) {
    check(std::fabs(a - b) < tol, what);
}

// Returns (logit_A, logit_B) per eval call; A -> token 65, B -> token 66.
class SequentialMockBackend : public MockBackend {
public:
    void set_calls(std::vector<std::pair<float, float>> c) { calls_ = std::move(c); idx_ = 0; }
    const float* eval_tokens(const std::vector<int>&) override {
        logits_.assign(600, 0.0f);
        if (idx_ < (int)calls_.size()) {
            logits_[65] = calls_[idx_].first;
            logits_[66] = calls_[idx_].second;
            idx_++;
        }
        return logits_.data();
    }
private:
    std::vector<std::pair<float, float>> calls_;
    int idx_ = 0;
    std::vector<float> logits_;
};

// A correct, labeled sample whose ground truth is True with the given margin.
static pjev::NoulSample labeled_true(const std::string& id, double margin) {
    pjev::NoulSample s;
    s.source_id = id;
    s.labeled = true;
    s.ground_truth = true;
    s.prediction = margin > 0.0;
    s.corrected_semantic_margin = margin;
    s.raw_semantic_margin = margin;
    s.p_true = 1.0 / (1.0 + std::exp(-margin));
    s.p_true_raw = s.p_true;
    return s;
}

int main() {
    using namespace pjev;

    // NLL
    check_near(binary_nll(0.8, true), -std::log(0.8), 1e-12, "NLL: gt=True, P(True)=0.8 -> -log(0.8)");
    check_near(binary_nll(0.8, false), -std::log(0.2), 1e-12, "NLL: gt=False, P(True)=0.8 -> -log(0.2)");
    check_near(binary_nll(1.0, false), -std::log(NOUL_PROB_EPS), 1e-9, "NLL: P(gt)=0 clamped to eps, finite");

    // Brier
    check_near(binary_brier(0.8, true), 0.04, 1e-12, "Brier: gt=True, P(True)=0.8 -> 0.04");
    check_near(binary_brier(0.8, false), 0.64, 1e-12, "Brier: gt=False, P(True)=0.8 -> 0.64");

    // Semantic margin: A=False, B=True, logit(A)=1, logit(B)=3 -> +2
    check_near(semantic_margin({1.0f, 3.0f}, {"false", "true"}), 2.0, 1e-9,
               "semantic margin: A=False B=True, (1,3) -> +2");
    // Reversed candidate mapping: A=True, B=False, logit(A)=3, logit(B)=1 -> +2
    check_near(semantic_margin({3.0f, 1.0f}, {"true", "false"}), 2.0, 1e-9,
               "semantic margin: A=True B=False, (3,1) -> +2 (reversed mapping)");
    check(std::isnan(semantic_margin({1.0f, 2.0f}, {"a", "b"})), "semantic margin: missing keys -> NaN");

    // Signed margin
    check_near(signed_margin(-2.0, false), 2.0, 1e-12, "signed margin: gt=False, m=-2 -> +2");
    check_near(signed_margin(2.0, false), -2.0, 1e-12, "signed margin: gt=False, m=+2 -> -2");
    check_near(signed_margin(2.0, true), 2.0, 1e-12, "signed margin: gt=True, m=+2 -> +2");

    // Percentile (linear interpolation, type 7)
    {
        std::vector<double> v = {10, 1, 9, 2, 8, 3, 7, 4, 6, 5};
        check_near(percentile_linear(v, 0.5), 5.5, 1e-12, "percentile: median of 1..10 = 5.5");
        check_near(percentile_linear(v, 0.1), 1.9, 1e-12, "percentile: p10 of 1..10 = 1.9");
        check_near(percentile_linear(v, 0.0), 1.0, 1e-12, "percentile: p0 = min");
        check_near(percentile_linear({4.0}, 0.1), 4.0, 1e-12, "percentile: single value");
        check(std::isnan(percentile_linear({}, 0.5)), "percentile: empty -> NaN");
    }

    // ECE: hand-computed, 10 bins
    //   p_true  pred  gt   conf  bin  correct
    //   0.95    T     T    0.95  9    1
    //   0.92    T     F    0.92  9    0
    //   0.25    F     F    0.75  7    1
    //   0.62    T     T    0.62  6    1
    // bin9: 2/4 * |0.935 - 0.5| = 0.2175; bin7: 1/4 * 0.25; bin6: 1/4 * 0.38  => 0.375
    {
        std::vector<double> p = {0.95, 0.92, 0.25, 0.62};
        std::vector<bool> gt = {true, false, false, true};
        std::vector<bool> pred = {true, true, false, true};
        auto m10 = compute_binary_prob_metrics(p, gt, pred, 10);
        check_near(m10.ece, 0.375, 1e-12, "ECE: 10 bins = 0.375");
        check(m10.ece_bins == 10, "ECE: bin count recorded");
        // 15 bins: confidences fall in bins 14, 13, 11, 9 (all separate)
        // => (0.05 + 0.92 + 0.25 + 0.38) / 4 = 0.4
        auto m15 = compute_binary_prob_metrics(p, gt, pred, 15);
        check_near(m15.ece, 0.4, 1e-12, "ECE: 15 bins = 0.4");
        // mean confidence over correct predictions: (0.95 + 0.75 + 0.62) / 3
        check_near(m15.mean_confidence_correct, (0.95 + 0.75 + 0.62) / 3.0, 1e-12, "mean confidence correct");
        check(m15.n == 4, "prob metrics: n = 4");
    }
    {
        // confidence exactly 1.0 lands in the last bin; perfectly calibrated -> ECE 0
        auto m = compute_binary_prob_metrics({1.0, 0.0}, {true, false}, {true, false}, 15);
        check_near(m.ece, 0.0, 1e-12, "ECE: confidence 1.0 in last bin, ECE = 0");
    }
    {
        // no correct predictions -> mean_confidence_correct undefined (NaN / JSON null)
        auto m = compute_binary_prob_metrics({0.9}, {false}, {true}, 15);
        check(std::isnan(m.mean_confidence_correct), "mean confidence correct: none correct -> NaN");
        check(m.to_json()["mean_confidence_correct"].is_null(), "mean confidence correct: serialized as null");
    }

    // Saturated accuracy: both sets 100% correct, B has larger margins everywhere
    {
        std::vector<NoulSample> a = {labeled_true("x1", 0.1), labeled_true("x2", 0.5), labeled_true("x3", 2.0)};
        std::vector<NoulSample> b = {labeled_true("x1", 1.5), labeled_true("x2", 2.0), labeled_true("x3", 3.0)};
        auto ma = compute_noul_metrics(a);
        auto mb = compute_noul_metrics(b);
        check(ma.accuracy == 1.0 && mb.accuracy == 1.0, "saturated: accuracy identical (1.0)");
        check(mb.calibrated.nll < ma.calibrated.nll, "saturated: NLL differs (B lower)");
        check(mb.calibrated.brier < ma.calibrated.brier, "saturated: Brier differs (B lower)");
        check(mb.signed_margin.mean > ma.signed_margin.mean, "saturated: mean signed margin differs (B higher)");
        check(mb.signed_margin.min > ma.signed_margin.min, "saturated: min signed margin differs (B higher)");
        check_near(ma.signed_margin.min, 0.1, 1e-12, "saturated: min margin of A = 0.1");

        auto g = compute_margin_gain(a, b);
        check(g.n_matched == 3 && g.improved == 3 && g.degraded == 0, "margin gain: 3 improved");
        check_near(g.mean_gain, (1.4 + 1.5 + 1.0) / 3.0, 1e-12, "margin gain: mean");
        check_near(g.median_gain, 1.4, 1e-12, "margin gain: median");
    }

    // Margin gain: degraded / unchanged / unmatched / unlabeled
    {
        std::vector<NoulSample> base = {labeled_true("a", 2.0), labeled_true("b", 1.0), labeled_true("c", 1.0)};
        std::vector<NoulSample> tgt = {labeled_true("a", 0.5), labeled_true("b", 1.0), labeled_true("z", 3.0)};
        NoulSample u = labeled_true("c", 5.0);
        u.labeled = false;
        tgt.push_back(u);
        auto g = compute_margin_gain(base, tgt);
        check(g.n_matched == 2, "margin gain: only labeled ids present in both are matched");
        check(g.degraded == 1 && g.unchanged == 1 && g.improved == 0, "margin gain: 1 degraded, 1 unchanged");
    }

    // Unlabeled samples excluded from ground-truth metrics
    {
        NoulSample u = labeled_true("u", -3.0);
        u.labeled = false;
        std::vector<NoulSample> v = {labeled_true("l", 1.0), u};
        auto m = compute_noul_metrics(v);
        check(m.n == 2 && m.labeled == 1, "unlabeled: n = 2, labeled = 1");
        check(m.calibrated.n == 1 && m.signed_margin.n == 1, "unlabeled: excluded from NLL and margin denominators");
        check(m.accuracy == 1.0, "unlabeled: accuracy over labeled only");
    }

    // E1 disagreement: order1 predicts True, order2 predicts False
    {
        NoulSample s = labeled_true("d", 0.25);
        s.has_orders = true;
        s.order1_semantic_margin = 1.0;
        s.order2_semantic_margin = -0.5;
        check(s.order1_prediction() && !s.order2_prediction(), "E1: order1 = True, order2 = False");
        check(s.order_disagreement(), "E1: disagreement = true");
        NoulSample t = labeled_true("e", 1.5);
        t.has_orders = true;
        t.order1_semantic_margin = 1.0;
        t.order2_semantic_margin = 2.0;
        check(!t.order_disagreement(), "E1: agreeing orders -> no disagreement");
        auto m = compute_noul_metrics({s, t});
        check(m.order_n == 2 && m.order_disagreement_count == 1, "E1: disagreement count 1 / 2");
        check_near(m.order_disagreement_rate(), 0.5, 1e-12, "E1: disagreement rate 0.5");
    }

    // JSON round-trip (metrics can be recomputed from stored results)
    {
        NoulSample s = labeled_true("rt", 0.7);
        s.ground_truth = false;
        s.prediction = true;
        s.has_orders = true;
        s.order1_semantic_margin = 0.4;
        s.order2_semantic_margin = 1.0;
        NoulSample r = NoulSample::from_json(s.to_json());
        check(r.source_id == "rt" && r.labeled && !r.ground_truth && r.prediction, "JSON: labels round-trip");
        check_near(r.signed_margin(), -0.7, 1e-12, "JSON: signed margin round-trip");
        check(r.has_orders && std::fabs(r.order2_semantic_margin - 1.0) < 1e-12, "JSON: order margins round-trip");
        NoulSample un = labeled_true("un", 1.0);
        un.labeled = false;
        check(!NoulSample::from_json(un.to_json()).labeled, "JSON: unlabeled stays unlabeled");
    }

    // End-to-end via run_experiment: semantic remapping is position independent
    {
        DatasetRow row;
        row.id = "e2e";
        row.input.type = Type::Noul;
        row.input.state = "s";
        row.input.question = "q";
        row.input.options = {{"false", ""}, {"true", ""}};
        row.expected = "true";
        std::vector<DatasetRow> rows = {row};

        // Baseline: A=False(1), B=True(3) -> margin +2
        SequentialMockBackend mock;
        mock.set_calls({{1.0f, 3.0f}});
        ExperimentConfig base_cfg;
        base_cfg.scheme = Scheme::LETTERS;
        RunResult base = run_experiment(mock, rows, base_cfg);
        auto bs = base.noul_samples();
        check(bs.size() == 1, "e2e baseline: one noul sample");
        if (bs.size() == 1) {
            check_near(bs[0].corrected_semantic_margin, 2.0, 1e-5, "e2e baseline: semantic margin +2");
            check_near(bs[0].signed_margin(), 2.0, 1e-5, "e2e baseline: signed margin +2");
            check(!bs[0].has_orders, "e2e baseline: no order diagnostics");
        }
        check(base.items[0].corrected_logits.empty(), "e2e baseline: ItemResult.corrected_logits still gated by config");

        // Reversed option order alone: A=True(3), B=False(1) -> still +2
        mock.set_calls({{3.0f, 1.0f}});
        ExperimentConfig rev_cfg = base_cfg;
        rev_cfg.option_order = OptionOrder::REVERSED;
        RunResult rev = run_experiment(mock, rows, rev_cfg);
        auto rs = rev.noul_samples();
        check(rs.size() == 1 && std::fabs(rs[0].corrected_semantic_margin - 2.0) < 1e-5,
              "e2e reversed order: semantic margin +2");

        // E1: order1 (A=F,B=T) = (1,3) -> m1=+2; order2 (A=T,B=F) = (4,1) -> m2=+3; ensemble 2.5
        mock.set_calls({{1.0f, 3.0f}, {4.0f, 1.0f}});
        ExperimentConfig e1_cfg = base_cfg;
        e1_cfg.ensemble.noul_mode = NoulEnsembleMode::BINARY_ORDER;
        RunResult e1 = run_experiment(mock, rows, e1_cfg);
        auto es = e1.noul_samples();
        check(es.size() == 1 && es[0].has_orders, "e2e E1: order diagnostics present");
        if (es.size() == 1) {
            check_near(es[0].order1_semantic_margin, 2.0, 1e-5, "e2e E1: order1 margin +2");
            check_near(es[0].order2_semantic_margin, 3.0, 1e-5, "e2e E1: order2 margin +3");
            check_near(es[0].corrected_semantic_margin, 2.5, 1e-5, "e2e E1: ensemble margin 2.5");
            check_near(es[0].raw_semantic_margin, 2.5, 1e-5, "e2e E1: raw ensemble margin 2.5 (no prior)");
            check(!es[0].order_disagreement(), "e2e E1: no disagreement");
        }
        auto g = compute_margin_gain(bs, es);
        check(g.n_matched == 1 && g.improved == 1, "e2e: E1 margin gain matched and improved");
        check_near(g.mean_gain, 0.5, 1e-5, "e2e: E1 margin gain = 0.5");

        // JSON output: existing fields untouched, new fields added
        auto j = e1.to_json();
        const auto& nj = j["metrics"]["noul"];
        check(nj["accuracy"].get<double>() == 1.0 && nj["labeled"] == 1 && nj["n"] == 1,
              "e2e JSON: existing accuracy / labeled / n");
        check(nj.contains("eval_ms"), "e2e JSON: eval_ms kept");
        check(nj.contains("nll") && nj.contains("brier") && nj.contains("ece") && nj.contains("ece_bins"),
              "e2e JSON: nll / brier / ece / ece_bins");
        check(nj.contains("raw") && nj.contains("calibrated"), "e2e JSON: raw and calibrated blocks");
        check(std::fabs(nj["min_signed_margin"].get<double>() - 2.5) < 1e-5, "e2e JSON: min_signed_margin");
        check(nj["order_disagreement_count"] == 0, "e2e JSON: order_disagreement_count");
        check(j["noul_items"].size() == 1 && j["noul_items"][0]["source_id"] == "e2e", "e2e JSON: noul_items");
        check(j["noul_ensemble"] == "binary-order", "e2e JSON: noul_ensemble mode");
    }

    printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails > 0 ? 1 : 0;
}
