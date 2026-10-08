#include "decision/decision_engine.h"
#include "experiment/runner.h"
#include "experiment/score_metrics.h"
#include "mock_backend.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
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

// Digit candidates "0".."9" are tokens 202..211 in MockBackend; this mock sets their logits.
class DigitMockBackend : public MockBackend {
public:
    std::vector<float> digit_logits;
    int calls = 0;
    const float* eval_tokens(const std::vector<int>&) override {
        calls++;
        buf_.assign(600, 0.0f);
        for (size_t i = 0; i < digit_logits.size(); i++) buf_[202 + i] = digit_logits[i];
        return buf_.data();
    }
private:
    std::vector<float> buf_;
};

// Same as MockBackend except "3" tokenizes to two tokens.
class BadDigitBackend : public MockBackend {
public:
    std::vector<int> tokenize(const std::string& text, bool a, bool b) const override {
        if (text == "3") return {1051, 1052};
        return MockBackend::tokenize(text, a, b);
    }
};

static pjev::ScoreSample sample(int gt, int pred, std::vector<double> logits) {
    pjev::ScoreSample s;
    s.source_id = "s" + std::to_string(gt) + std::to_string(pred);
    for (int i = 0; i < (int)logits.size(); i++) {
        s.levels.push_back(i);
        s.labels.push_back(std::to_string(i));
    }
    s.ground_truth = gt;
    s.prediction = pred;
    s.logits = logits;
    s.raw_logits = logits;
    double m = -1e300, z = 0.0;
    for (double l : logits) m = std::max(m, l);
    for (double l : logits) z += std::exp(l - m);
    for (double l : logits) s.probs.push_back(std::exp(l - m) / z);
    s.raw_probs = s.probs;
    return s;
}

static std::vector<double> log_of(std::vector<double> p) {
    for (double& x : p) x = std::log(x);
    return p;
}

int main() {
    using namespace pjev;

    // 1. Digit mapping (S0: natural scheme, score)
    {
        MockBackend mock;
        DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::NATURAL});
        auto labels = eng.candidate_labels(Type::Score, 4);
        check(labels == std::vector<std::string>({"0", "1", "2", "3"}), "digits: level i -> \"i\" for 4 levels");
        bool ids = true;
        for (int i = 0; i < 4; i++) ids = ids && eng.candidate_token_id(labels[i]) == 202 + i;
        check(ids, "digits: token ids recorded (202..205 in mock)");
        auto letters = DecisionEngine(mock, PromptConfig{Layout::AUTO, Scheme::LETTERS}).candidate_labels(Type::Score, 4);
        check(letters == std::vector<std::string>({"A", "B", "C", "D"}), "digits: letters scheme is a different configuration");
    }

    // 2. Single-token validation fails clearly and names the offending candidate
    {
        BadDigitBackend bad;
        std::string msg;
        try {
            DecisionEngine eng(bad, PromptConfig{Layout::AUTO, Scheme::NATURAL});
        } catch (const std::runtime_error& e) {
            msg = e.what();
        }
        check(!msg.empty(), "validation: multi-token digit throws");
        check(msg.find("\"3\"") != std::string::npos && msg.find("2 tokens") != std::string::npos,
              "validation: error names candidate \"3\" (2 tokens)");
        MockBackend good;
        bool ok = true;
        try { DecisionEngine eng(good, PromptConfig{}); } catch (...) { ok = false; }
        check(ok, "validation: all single-token digits accepted");
    }

    // 3. Expected score
    check_near(expected_score({0.1, 0.2, 0.3, 0.4}, {0, 1, 2, 3}), 2.0, 1e-12, "expected score: [0.1,0.2,0.3,0.4] -> 2.0");
    check(std::isnan(expected_score({0.5, 0.5}, {0, 1, 2})), "expected score: size mismatch -> NaN");

    // 4. Discrete MAE
    check(sample(3, 1, {0, 0, 0, 0}).abs_error() == 2, "MAE: gt 3, pred 1 -> 2");

    // 5. Expected-score error: probs [0, 0.25, 0.25, 0.5] -> E = 2.25
    {
        ScoreSample s = sample(3, 3, {0, 0, 0, 0});
        s.probs = {0.0, 0.25, 0.25, 0.5};
        check_near(s.expected_score(), 2.25, 1e-12, "expected-score error: E = 2.25");
        check_near(s.expected_abs_error(), 0.75, 1e-12, "expected-score error: |2.25 - 3| = 0.75");
    }

    // 6. Signed margin: gt level 2, logits 0.5, 1.0, 3.0, 2.0 -> 1.0
    check_near(sample(2, 2, {0.5, 1.0, 3.0, 2.0}).signed_margin(), 1.0, 1e-12, "signed margin: 3.0 - 2.0 = 1.0");

    // 7. Error-distance histogram and large-error rate
    {
        auto h = error_distance_histogram({0, 1, 3, 2, 0}, {0, 2, 0, 2, 1});
        check(h == std::vector<int32_t>({2, 2, 0, 1}), "histogram: distances 0,1,3,0,1 -> [2,2,0,1]");
        std::vector<ScoreSample> v = {sample(0, 0, {1, 0, 0, 0}), sample(2, 1, {0, 1, 0, 0}),
                                      sample(0, 3, {0, 0, 0, 1}), sample(2, 2, {0, 0, 1, 0}),
                                      sample(1, 0, {1, 0, 0, 0})};
        auto m = compute_score_metrics(v);
        check(m.error_histogram == std::vector<int32_t>({2, 2, 0, 1}), "histogram: compute_score_metrics");
        check_near(m.large_error_rate, 0.2, 1e-12, "large-error rate (>= 2): 1/5");
        check_near(m.mae, (0 + 1 + 3 + 0 + 1) / 5.0, 1e-12, "MAE: mean absolute error 1.0");
        check_near(m.accuracy, 0.4, 1e-12, "accuracy 2/5");
    }

    // 8. NLL / Brier / ECE through the shared implementation
    //    A: p = [0.1,0.2,0.3,0.4], gt 3 (correct, conf 0.4);  B: p = [0.7,0.1,0.1,0.1], gt 1 (wrong, conf 0.7)
    {
        std::vector<ScoreSample> v = {sample(3, 3, log_of({0.1, 0.2, 0.3, 0.4})),
                                      sample(1, 0, log_of({0.7, 0.1, 0.1, 0.1}))};
        auto m = compute_score_metrics(v);
        check_near(m.raw.nll, (-std::log(0.4) - std::log(0.1)) / 2.0, 1e-6, "NLL = mean(-log 0.4, -log 0.1)");
        check_near(m.raw.brier, (0.5 + 1.32) / 2.0, 1e-6, "Brier (summed) = mean(0.50, 1.32) = 0.91");
        check_near(m.raw.ece, (0.6 + 0.7) / 2.0, 1e-6, "ECE (15 bins) = (|0.4-1| + |0.7-0|) / 2 = 0.65");
        check_near(m.raw_expected_mae, (std::fabs(2.0 - 3) + std::fabs(0.6 - 1)) / 2.0, 1e-6, "expected-score MAE = (1.0 + 0.4) / 2");
        check_near(m.raw_mean_expected_argmax_distance, (1.0 + 0.6) / 2.0, 1e-6, "expected-vs-argmax distance = (|2-3| + |0.6-0|) / 2");
        check_near(m.signed_margin.min, std::log(0.1) - std::log(0.7), 1e-6, "min signed margin = log 0.1 - log 0.7");
    }

    // 9. QWK reuses TypeMetrics::qwk
    {
        std::vector<ScoreSample> v = {sample(0, 0, {1, 0, 0, 0}), sample(1, 1, {0, 1, 0, 0}),
                                      sample(2, 2, {0, 0, 1, 0}), sample(3, 3, {0, 0, 0, 1})};
        check_near(compute_score_metrics(v).qwk, 1.0, 1e-9, "QWK: perfect agreement = 1.0");
    }

    // 10. Levels are semantic (from option keys), not positions
    {
        ScoreSample s = sample(3, 3, {0, 0, 0, 0});
        s.levels = {3, 2, 1, 0};
        s.probs = {0.4, 0.3, 0.2, 0.1};
        s.logits = {4, 3, 2, 1};
        check(s.gt_position() == 0, "semantic: level 3 found at position 0");
        check_near(s.expected_score(), 3 * 0.4 + 2 * 0.3 + 1 * 0.2, 1e-12, "semantic: expected score uses levels");
        check_near(s.signed_margin(), 1.0, 1e-12, "semantic: margin uses ground-truth position");
    }

    // 11. JSON round-trip
    {
        ScoreSample s = sample(2, 1, {0.5, 2.0, 1.0, 0.0});
        s.token_ids = {202, 203, 204, 205};
        auto r = ScoreSample::from_json(s.to_json());
        check(r.levels == s.levels && r.labels == s.labels && r.token_ids == s.token_ids &&
              r.ground_truth == 2 && r.prediction == 1 && std::fabs(r.signed_margin() - s.signed_margin()) < 1e-12,
              "JSON: score sample round-trip");
    }

    // 12. Production path end to end (PromptStrategy -> backend -> restricted softmax -> expected score)
    {
        DatasetRow row;
        row.id = "sc1";
        row.input.type = Type::Score;
        row.input.state = "s";
        row.input.question = "q";
        for (int i = 0; i < 4; i++) row.input.options.push_back({std::to_string(i), "level " + std::to_string(i)});
        row.expected = 2;
        DigitMockBackend mock;
        mock.digit_logits = {0.5f, 1.0f, 3.0f, 2.0f};
        ExperimentConfig cfg;
        cfg.record_score_candidates = true;
        RunResult r = run_experiment(mock, {row}, cfg);
        auto ss = r.score_samples();
        check(ss.size() == 1, "e2e: one score sample");
        if (ss.size() == 1) {
            const auto& s = ss[0];
            check(s.prediction == 2 && s.correct(), "e2e: argmax level 2 (correct)");
            check_near(s.signed_margin(), 1.0, 1e-6, "e2e: signed margin 1.0");
            check(s.labels == std::vector<std::string>({"0", "1", "2", "3"}) &&
                  s.token_ids == std::vector<int32_t>({202, 203, 204, 205}), "e2e: digit labels and token ids stored");
            check(s.raw_logits.size() == 4 && std::fabs(s.raw_logits[2] - 3.0) < 1e-6, "e2e: raw digit logits stored");
            check_near(s.raw_expected_score(), s.expected_score(), 1e-12, "e2e: T = 1 -> raw and reported expected score agree");
            check(s.evaluations == 1 && mock.calls == 2, "e2e: 1 evaluation per item (+1 content-free prior)");
        }
        auto j = r.to_json();
        const auto& sj = j["metrics"]["score"];
        check(sj["accuracy"].get<double>() == 1.0 && sj.contains("mae") && sj.contains("qwk") && sj.contains("eval_ms"),
              "e2e JSON: existing accuracy / mae / qwk / eval_ms kept");
        check(sj.contains("expected_score_mae") && sj.contains("nll") && sj.contains("brier") && sj.contains("ece") &&
              sj.contains("min_signed_margin") && sj.contains("error_distance_histogram") &&
              sj.contains("large_error_rate") && sj.contains("mean_expected_argmax_distance") &&
              sj["evaluations_per_item"].get<double>() == 1.0,
              "e2e JSON: S0 metric set present");
        check(j["score_items"].size() == 1, "e2e JSON: score_items");
        const auto& c = j["score_candidates"];
        check(c.size() == 1 && c[0]["n_levels"] == 4 && c[0]["token_ids"] == std::vector<int>({202, 203, 204, 205}) &&
              c[0]["all_single_token"] == true, "e2e JSON: candidate diagnostics (labels, ids, single-token)");
        check(c.size() == 1 && c[0]["answer_anchor"] == PromptStrategy::ANSWER_ANCHOR, "e2e JSON: answer anchor recorded");
        check(c.size() == 1 && c[0]["prior_logits"].is_array() && c[0]["prior_logits"].size() == 4,
              "e2e JSON: content-free prior logits recorded");
        check(c.size() == 1 && c[0]["forms"][0]["separate_token_after_anchor"].is_boolean(),
              "e2e JSON: in-context token check recorded");
    }

    printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails > 0 ? 1 : 0;
}
