#include "decision/decision_engine.h"
#include "decision/prompt_strategy.h"
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

// Letter candidates "A".."J" are tokens 65..74 in MockBackend; this mock sets their logits.
class LetterMockBackend : public MockBackend {
public:
    std::vector<float> letter_logits;
    int calls = 0;
    const float* eval_tokens(const std::vector<int>&) override {
        calls++;
        buf_.assign(600, 0.0f);
        for (size_t i = 0; i < letter_logits.size(); i++) buf_[65 + i] = letter_logits[i];
        return buf_.data();
    }
private:
    std::vector<float> buf_;
};

// "B" shares token id 65 with "A".
class DuplicateLetterBackend : public MockBackend {
public:
    std::vector<int> tokenize(const std::string& text, bool a, bool b) const override {
        if (text == "B") return {65};
        return MockBackend::tokenize(text, a, b);
    }
};

// "C" is two tokens.
class MultiTokenLetterBackend : public MockBackend {
public:
    std::vector<int> tokenize(const std::string& text, bool a, bool b) const override {
        if (text == "C") return {1067, 1068};
        return MockBackend::tokenize(text, a, b);
    }
};

// Digits are multi-token: irrelevant for the letters scheme (no silent digit dependency).
class NoDigitBackend : public MockBackend {
public:
    std::vector<int> tokenize(const std::string& text, bool a, bool b) const override {
        if (text.size() == 1 && text[0] >= '0' && text[0] <= '9') return {1100, 1101};
        return MockBackend::tokenize(text, a, b);
    }
};

static pjev::DatasetRow score_row(const std::string& id, int expected) {
    pjev::DatasetRow row;
    row.id = id;
    row.input.type = pjev::Type::Score;
    row.input.state = "The response was mostly accurate.";
    row.input.question = "Rate the accuracy.";
    const char* d[] = {"poor", "fair", "good", "excellent"};
    for (int i = 0; i < 4; i++) row.input.options.push_back({std::to_string(i), d[i]});
    row.expected = expected;
    return row;
}

static pjev::ScoreSample sample(const std::vector<std::string>& labels, int gt, std::vector<double> logits) {
    pjev::ScoreSample s;
    s.source_id = "x";
    s.labels = labels;
    for (int i = 0; i < (int)logits.size(); i++) s.levels.push_back(i);
    s.ground_truth = gt;
    s.logits = logits;
    s.raw_logits = logits;
    double m = -1e300, z = 0.0;
    for (double l : logits) m = std::max(m, l);
    for (double l : logits) z += std::exp(l - m);
    for (double l : logits) s.probs.push_back(std::exp(l - m) / z);
    s.raw_probs = s.probs;
    int best = 0;
    for (int i = 1; i < (int)logits.size(); i++) if (logits[i] > logits[best]) best = i;
    s.prediction = s.levels[best];
    return s;
}

int main() {
    using namespace pjev;
    const PromptConfig S0{Layout::AUTO, Scheme::NATURAL};
    const PromptConfig S1{Layout::AUTO, Scheme::LETTERS};

    // 1. Letter mapping
    {
        MockBackend mock;
        DecisionEngine eng(mock, S1);
        check(eng.candidate_labels(Type::Score, 4) == std::vector<std::string>({"A", "B", "C", "D"}),
              "mapping: level 0..3 -> A..D");
        check(eng.candidate_labels(Type::Score, 5).back() == "E", "mapping: level 4 -> E");
        bool ids = true;
        for (int i = 0; i < 4; i++) ids = ids && eng.candidate_token_id(std::string(1, (char)('A' + i))) == 65 + i;
        check(ids, "mapping: token ids recorded (65..68 in mock)");
    }

    // 2. Semantic expected score: letters do not enter the computation
    {
        auto a = sample({"A", "B", "C", "D"}, 3, {0, 0, 0, 0});
        a.probs = {0.1, 0.2, 0.3, 0.4};
        check_near(a.expected_score(), 2.0, 1e-12, "expected score: A..D with [0.1,0.2,0.3,0.4] -> 2.0");
        auto z = a;
        z.labels = {"Z", "Y", "X", "W"};
        check_near(z.expected_score(), 2.0, 1e-12, "expected score: independent of label text");
    }

    // 3. Candidate-token validation: unique single tokens, clear failures, no digit fallback
    {
        std::string dup, multi;
        try { DuplicateLetterBackend b; DecisionEngine e(b, S1); } catch (const std::runtime_error& ex) { dup = ex.what(); }
        try { MultiTokenLetterBackend b; DecisionEngine e(b, S1); } catch (const std::runtime_error& ex) { multi = ex.what(); }
        check(dup.find("\"B\"") != std::string::npos && dup.find("duplicate") != std::string::npos,
              "validation: duplicate token id for \"B\" rejected and named");
        check(multi.find("\"C\"") != std::string::npos && multi.find("2 tokens") != std::string::npos,
              "validation: multi-token \"C\" rejected and named");
        bool ok = true;
        try { NoDigitBackend b; DecisionEngine e(b, S1); } catch (...) { ok = false; }
        check(ok, "validation: letters scheme does not require digit tokens");
        bool s0_fails = false;
        try { NoDigitBackend b; DecisionEngine e(b, S0); } catch (...) { s0_fails = true; }
        check(s0_fails, "validation: digits scheme still requires digit tokens");
    }

    // 4. Semantic prediction through the production path
    {
        LetterMockBackend mock;
        mock.letter_logits = {0.1f, 0.2f, 2.0f, 0.5f};
        ExperimentConfig cfg;
        cfg.scheme = Scheme::LETTERS;
        RunResult r = run_experiment(mock, {score_row("p1", 2)}, cfg);
        auto s = r.score_samples();
        check(s.size() == 1 && s[0].selected_label() == "C" && s[0].prediction == 2,
              "prediction: logits (0.1,0.2,2.0,0.5) -> label C, level 2");
        check(s.size() == 1 && s[0].labels == std::vector<std::string>({"A", "B", "C", "D"}) &&
              s[0].levels == std::vector<int32_t>({0, 1, 2, 3}) &&
              s[0].token_ids == std::vector<int32_t>({65, 66, 67, 68}),
              "prediction: explicit label -> level -> token id metadata stored");
        auto j = r.to_json();
        check(j["score_items"][0]["selected_label"] == "C", "prediction: selected_label in JSON");
    }

    // 5. Signed margin: gt level 2, A..D = 0.5, 1.0, 3.0, 2.0 -> 1.0
    {
        LetterMockBackend mock;
        mock.letter_logits = {0.5f, 1.0f, 3.0f, 2.0f};
        ExperimentConfig cfg;
        cfg.scheme = Scheme::LETTERS;
        auto s = run_experiment(mock, {score_row("m1", 2)}, cfg).score_samples();
        check(s.size() == 1, "margin: one sample");
        if (s.size() == 1) check_near(s[0].signed_margin(), 1.0, 1e-6, "margin: C(3.0) - D(2.0) = 1.0");
    }

    // 6. No rotation: one mapping, one evaluation, even with E1 / E2 switched on
    {
        LetterMockBackend mock;
        mock.letter_logits = {1, 2, 3, 4};
        ExperimentConfig cfg;
        cfg.scheme = Scheme::LETTERS;
        cfg.ensemble.noul_mode = NoulEnsembleMode::BINARY_ORDER;
        cfg.ensemble.choice_mode = ChoiceEnsembleMode::CYCLIC_ROTATION;
        auto r = run_experiment(mock, {score_row("r1", 3)}, cfg);
        check(mock.calls == 1, "no rotation: exactly one inference evaluation");
        auto s = r.score_samples();
        check(s.size() == 1 && s[0].evaluations == 1 && s[0].labels.size() == 4, "no rotation: one candidate mapping");
    }

    // 7. Metric equivalence: identical semantic distributions, digit vs letter labels
    {
        std::vector<ScoreSample> s0 = {sample({"0", "1", "2", "3"}, 2, {0.5, 1.0, 3.0, 2.0}),
                                       sample({"0", "1", "2", "3"}, 0, {0.2, 1.5, 0.1, -1.0}),
                                       sample({"0", "1", "2", "3"}, 3, {0.0, 0.3, 0.9, 1.2})};
        for (size_t i = 0; i < s0.size(); i++) s0[i].source_id = "eq" + std::to_string(i);
        std::vector<ScoreSample> s1 = s0;
        for (auto& s : s1) s.labels = {"A", "B", "C", "D"};
        nlohmann::json j0, j1;
        compute_score_metrics(s0).merge_into(j0);
        compute_score_metrics(s1).merge_into(j1);
        check(j0["selection_by_label"] != j1["selection_by_label"], "equivalence: only label-keyed selection counts differ");
        j0.erase("selection_by_label");
        j1.erase("selection_by_label");
        check(j0 == j1, "equivalence: every semantic metric identical for digits and letters");
        auto g = compute_margin_gain(s0, s1);
        check(g.unchanged == 3 && g.correct_to_wrong == 0 && g.wrong_to_correct == 0, "equivalence: zero margin gain");
    }

    // 8. Prompt isolation: S1 prompt == S0 prompt with only the labels replaced
    {
        auto user_text = [](const PromptConfig& cfg) {
            PromptStrategy st(cfg);
            std::vector<Candidate> c(4);
            const char* d[] = {"poor", "fair", "good", "excellent"};
            for (int i = 0; i < 4; i++) { c[i].key = std::to_string(i); c[i].description = d[i]; }
            st.assign_labels(Type::Score, c);
            auto segs = st.build_prompt_segments(Type::Score, "STATE", "QUESTION", c);
            std::string all;
            for (const auto& s : segs) all += s.text;
            return all;
        };
        std::string p0 = user_text(S0), p1 = user_text(S1);
        std::string expect = p0;
        auto replace = [&](const std::string& from, const std::string& to) {
            auto pos = expect.find(from);
            if (pos != std::string::npos) expect.replace(pos, from.size(), to);
        };
        for (int i = 0; i < 4; i++)
            replace("\n" + std::to_string(i) + ": ", "\n" + std::string(1, (char)('A' + i)) + ": ");
        replace("one of 0, 1, 2, 3 and", "one of A, B, C, D and");
        check(p1 == expect, "prompt: S1 differs from S0 only in candidate labels");
        check(p1.find("ordered from lowest to highest") != std::string::npos, "prompt: ordinal wording kept");
        check(p1.size() >= std::string(PromptStrategy::ANSWER_ANCHOR).size() &&
              p1.compare(p1.size() - std::string(PromptStrategy::ANSWER_ANCHOR).size(), std::string::npos,
                         PromptStrategy::ANSWER_ANCHOR) == 0,
              "prompt: same answer anchor");
    }

    // 9. Candidate diagnostics for S1
    {
        LetterMockBackend mock;
        mock.letter_logits = {1, 2, 3, 4};
        ExperimentConfig cfg;
        cfg.scheme = Scheme::LETTERS;
        cfg.record_score_candidates = true;
        auto j = run_experiment(mock, {score_row("d1", 1)}, cfg).to_json();
        const auto& c = j["score_candidates"];
        check(c.size() == 1 && c[0]["labels"] == std::vector<std::string>({"A", "B", "C", "D"}) &&
              c[0]["token_ids"] == std::vector<int>({65, 66, 67, 68}),
              "diagnostics: letter labels and token ids");
        check(c.size() == 1 && c[0]["all_single_token"] == true && c[0]["all_unique_token_ids"] == true,
              "diagnostics: single-token and unique-id flags");
        check(c.size() == 1 && c[0]["label_to_level"][2]["label"] == "C" && c[0]["label_to_level"][2]["level"] == 2,
              "diagnostics: explicit label -> level mapping");
        check(c.size() == 1 && c[0]["prior_logits"].is_array() && c[0]["prior_logits"].size() == 4,
              "diagnostics: content-free prior logits for letters");
        const auto& sj = j["metrics"]["score"];
        check(sj["selection_by_label"]["D"] == 1 && sj["ground_truth_by_level"]["1"] == 1 &&
              sj["prediction_by_level"]["3"] == 1, "diagnostics: selection and ground-truth frequencies");
    }

    printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails > 0 ? 1 : 0;
}
