#include "calibration/calibration.h"
#include "decision/decision_engine.h"
#include "decision/prompt_strategy.h"
#include "experiment/runner.h"
#include "experiment/score_metrics.h"
#include "mock_backend.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
static void check(bool cond, const char* what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
    else       { printf("pass: %s\n", what); }
}
static void check_near(double a, double b, double tol, const char* what) {
    check(std::fabs(a - b) < tol, what);
}

static const std::string F1_FILLER = ". . . . . . . . . . . . . . . .";

// Returns fixed raw logits: A=raw[0], B=raw[1], ..., regardless of prompt.
class FixedLogitMock : public MockBackend {
public:
    std::vector<float> raw;
    int calls = 0;
    const float* eval_tokens(const std::vector<int>&) override {
        calls++;
        buf_.assign(600, 0.0f);
        for (size_t i = 0; i < raw.size(); i++) buf_[65 + i] = raw[i];
        return buf_.data();
    }
private: std::vector<float> buf_;
};

static pjev::DatasetRow score_row(const std::string& id, int expected, int n_opts = 4) {
    pjev::DatasetRow row;
    row.id = id;
    row.input.type = pjev::Type::Score;
    row.input.state = "The response was mostly accurate.";
    row.input.question = "Rate the accuracy.";
    const char* d[] = {"poor", "fair", "good", "excellent", "outstanding"};
    for (int i = 0; i < n_opts; i++) row.input.options.push_back({std::to_string(i), d[i]});
    row.expected = expected;
    return row;
}

// Apply alpha offline to raw logits + prior -> probs (helper for test #8)
static std::vector<double> apply_alpha(const std::vector<double>& raw_logits,
                                       const std::vector<float>& prior,
                                       double alpha)
{
    int n = (int)raw_logits.size();
    std::vector<double> corr(n);
    for (int i = 0; i < n; i++) corr[i] = raw_logits[i] - alpha * (double)prior[i];
    double m = *std::max_element(corr.begin(), corr.end());
    double z = 0.0;
    for (double c : corr) z += std::exp(c - m);
    std::vector<double> p(n);
    for (int i = 0; i < n; i++) p[i] = std::exp(corr[i] - m) / z;
    return p;
}

int main() {
    using namespace pjev;

    const PromptConfig F0_CFG{Layout::AUTO, Scheme::LETTERS};
    const PromptConfig F1_CFG{Layout::AUTO, Scheme::LETTERS, F1_FILLER};

    // 1. Alpha = 0 produces same raw probs as F1 (no correction)
    {
        FixedLogitMock mock;
        mock.raw = {4.0f, 3.0f, 2.0f, 1.0f};

        // F1: no correction
        ExperimentConfig cfg_f1;
        cfg_f1.scheme = Scheme::LETTERS;
        cfg_f1.filler_text = F1_FILLER;
        auto s1 = run_experiment(mock, {score_row("a1", 1)}, cfg_f1).score_samples();

        // F2 alpha=0: should be identical to F1
        ExperimentConfig cfg_f2_a0;
        cfg_f2_a0.scheme = Scheme::LETTERS;
        cfg_f2_a0.filler_text = F1_FILLER;
        cfg_f2_a0.prior_correction = true;
        cfg_f2_a0.calibration.score_prior_alpha = 0.0;
        auto s2 = run_experiment(mock, {score_row("a1", 1)}, cfg_f2_a0).score_samples();

        check(s1.size() == 1 && s2.size() == 1, "alpha0: one sample each");
        if (s1.size() == 1 && s2.size() == 1) {
            bool probs_eq = s1[0].probs.size() == s2[0].probs.size();
            for (size_t i = 0; i < s1[0].probs.size() && probs_eq; i++)
                probs_eq = probs_eq && std::fabs(s1[0].probs[i] - s2[0].probs[i]) < 1e-6;
            check(probs_eq, "alpha0: F2 alpha=0 probs == F1 probs");
            check(s1[0].prediction == s2[0].prediction, "alpha0: same prediction");
        }
    }

    // 2. Alpha = 1 formula: raw=[4,3,2,1], prior=[2,1,0,-1] → corrected=[2,2,2,2] → uniform
    {
        FixedLogitMock mock;
        mock.raw = {4.0f, 3.0f, 2.0f, 1.0f};

        CalibrationConfig calib;
        calib.score_prior_alpha = 1.0;
        DecisionEngine eng(mock, F1_CFG, calib);

        DecisionInput inp;
        inp.type = Type::Score;
        inp.state = "s";
        inp.question = "q";
        for (int i = 0; i < 4; i++) inp.options.push_back({std::to_string(i), "opt"});
        inp.prior_correction = true;
        inp.prior_logits = {2.0f, 1.0f, 0.0f, -1.0f}; // manually set F1-compatible prior
        inp.collect_corrected_logits = true;

        auto out = eng.decide(inp);
        check(out.ok, "alpha1: decision ok");
        check(out.probs.size() == 4, "alpha1: 4 probs");
        for (auto p : out.probs) check_near(p, 0.25, 1e-5, "alpha1: uniform after full correction");

        // raw_logits preserved
        check(out.raw_logits.size() == 4, "alpha1: raw_logits populated");
        if (out.raw_logits.size() == 4) {
            check_near(out.raw_logits[0], 4.0, 1e-5, "alpha1: raw A = 4.0");
            check_near(out.raw_logits[1], 3.0, 1e-5, "alpha1: raw B = 3.0");
        }
        // corrected_logits preserved
        check(out.corrected_logits.size() == 4, "alpha1: corrected_logits populated");
        if (out.corrected_logits.size() == 4) {
            check_near(out.corrected_logits[0], 2.0, 1e-5, "alpha1: corrected A = 2.0");
            check_near(out.corrected_logits[3], 2.0, 1e-5, "alpha1: corrected D = 2.0");
        }
    }

    // 3. Intermediate alpha = 0.5: raw=[4,3,2,1], prior=[2,1,0,-1] → corrected=[3,2.5,2,1.5]
    {
        FixedLogitMock mock;
        mock.raw = {4.0f, 3.0f, 2.0f, 1.0f};

        CalibrationConfig calib;
        calib.score_prior_alpha = 0.5;
        DecisionEngine eng(mock, F1_CFG, calib);

        DecisionInput inp;
        inp.type = Type::Score;
        inp.state = "s";
        inp.question = "q";
        for (int i = 0; i < 4; i++) inp.options.push_back({std::to_string(i), "opt"});
        inp.prior_correction = true;
        inp.prior_logits = {2.0f, 1.0f, 0.0f, -1.0f};
        inp.collect_corrected_logits = true;

        auto out = eng.decide(inp);
        check(out.ok, "alpha05: ok");
        check(out.corrected_logits.size() == 4, "alpha05: corrected_logits populated");
        if (out.corrected_logits.size() == 4) {
            check_near(out.corrected_logits[0], 3.0, 1e-5, "alpha05: corrected A = 3.0");
            check_near(out.corrected_logits[1], 2.5, 1e-5, "alpha05: corrected B = 2.5");
            check_near(out.corrected_logits[2], 2.0, 1e-5, "alpha05: corrected C = 2.0");
            check_near(out.corrected_logits[3], 1.5, 1e-5, "alpha05: corrected D = 1.5");
        }
        // argmax should be A (corrected 3.0 > 2.5 > 2.0 > 1.5)
        check(out.selected == 0, "alpha05: argmax at A");
    }

    // 4. No extra inference per item: prior correction is post-logit only
    {
        FixedLogitMock mock;
        mock.raw = {1.0f, 2.0f, 3.0f, 4.0f};

        ExperimentConfig cfg;
        cfg.scheme = Scheme::LETTERS;
        cfg.filler_text = F1_FILLER;
        cfg.prior_correction = true;
        cfg.calibration.score_prior_alpha = 1.0;

        mock.calls = 0;
        int n_items = 3;
        std::vector<DatasetRow> rows;
        for (int i = 0; i < n_items; i++) rows.push_back(score_row("r" + std::to_string(i), i % 4));
        run_experiment(mock, rows, cfg);

        // 1 blank-logit call + n_items inference calls
        check(mock.calls == n_items + 1, "no_extra_inference: exactly 1 blank + n_item calls");
    }

    // 5. Prior compatibility: F0 blank prompt differs from F1 blank prompt (text level)
    {
        // Verify that the user content with filler ends differently than without.
        PromptStrategy st_f0(F0_CFG), st_f1(F1_CFG);
        std::vector<Candidate> c(4);
        for (int i = 0; i < 4; i++) { c[i].key = std::to_string(i); c[i].description = "opt"; }
        st_f0.assign_labels(Type::Score, c);
        auto segs_f0 = st_f0.build_prompt_segments(Type::Score, "", "", c);
        auto segs_f1 = st_f1.build_prompt_segments(Type::Score, "", "", c);
        // seg[1] is user content; F1 has filler appended
        std::string content_f0, content_f1;
        for (const auto& s : segs_f0) content_f0 += s.text;
        for (const auto& s : segs_f1) content_f1 += s.text;
        check(content_f0 != content_f1, "prior_compat: F0 and F1 blank prompts differ");
        check(content_f1.find(F1_FILLER) != std::string::npos, "prior_compat: F1 blank prompt contains filler");
        check(content_f0.find(F1_FILLER) == std::string::npos, "prior_compat: F0 blank prompt has no filler");
    }

    // 6. Candidate semantics unchanged: A->0, B->1, C->2, D->3
    {
        MockBackend mock;
        CalibrationConfig calib;
        calib.score_prior_alpha = 1.0;
        DecisionEngine eng(mock, F1_CFG, calib);
        auto labels = eng.candidate_labels(Type::Score, 4);
        check(labels == std::vector<std::string>({"A", "B", "C", "D"}),
              "semantics: F2 level 0..3 -> A..D");
    }

    // 7. F1 reproducibility: F2 with alpha=0 and record_score_candidates=true still
    //    gives same prediction as plain F1 run.
    {
        FixedLogitMock mock;
        mock.raw = {0.5f, 2.0f, 3.0f, 1.0f}; // C wins

        ExperimentConfig cfg_f1;
        cfg_f1.scheme = Scheme::LETTERS;
        cfg_f1.filler_text = F1_FILLER;
        auto s1 = run_experiment(mock, {score_row("b1", 2)}, cfg_f1).score_samples();

        ExperimentConfig cfg_f2_a0;
        cfg_f2_a0.scheme = Scheme::LETTERS;
        cfg_f2_a0.filler_text = F1_FILLER;
        cfg_f2_a0.prior_correction = true;
        cfg_f2_a0.calibration.score_prior_alpha = 0.0;
        cfg_f2_a0.record_score_candidates = true;
        auto s2 = run_experiment(mock, {score_row("b1", 2)}, cfg_f2_a0).score_samples();

        check(s1.size() == 1 && s2.size() == 1, "f1_repro: one sample each");
        if (s1.size() == 1 && s2.size() == 1) {
            check(s1[0].prediction == s2[0].prediction, "f1_repro: same prediction at alpha=0");
            check(s1[0].selected_label() == s2[0].selected_label(), "f1_repro: same label at alpha=0");
        }
    }

    // 8. Offline sweep equivalence: applying alpha offline to stored logits matches runtime
    {
        FixedLogitMock mock;
        mock.raw = {1.0f, 3.0f, 2.0f, 0.5f}; // B wins raw

        // Run F2 at alpha=0.75 through the engine (runtime path)
        CalibrationConfig calib;
        calib.score_prior_alpha = 0.75;
        DecisionEngine eng(mock, F1_CFG, calib);

        DecisionInput inp;
        inp.type = Type::Score;
        inp.state = "s";
        inp.question = "q";
        for (int i = 0; i < 4; i++) inp.options.push_back({std::to_string(i), "opt"});
        inp.prior_correction = true;
        inp.prior_logits = {1.0f, 2.0f, 0.5f, -0.5f};
        inp.collect_corrected_logits = true;

        auto out = eng.decide(inp);
        check(out.ok && out.raw_logits.size() == 4, "sweep_equiv: runtime ok");

        // Offline: apply same alpha to stored raw_logits
        std::vector<double> raw_d(out.raw_logits.begin(), out.raw_logits.end());
        auto p_offline = apply_alpha(raw_d, inp.prior_logits, 0.75);

        check(p_offline.size() == out.probs.size(), "sweep_equiv: same size");
        bool equiv = true;
        for (size_t i = 0; i < p_offline.size(); i++)
            equiv = equiv && std::fabs(p_offline[i] - out.probs[i]) < 1e-6;
        check(equiv, "sweep_equiv: offline alpha=0.75 matches runtime alpha=0.75");
    }

    printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails > 0 ? 1 : 0;
}
