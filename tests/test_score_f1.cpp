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

// Frozen F1 filler — must match SCORE_F1_FILLER in runner.cpp.
static const std::string F1_FILLER = ". . . . . . . . . . . . . . . .";

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

// Build the full prompt text for the given config.
static std::string prompt_text(const pjev::PromptConfig& cfg) {
    pjev::PromptStrategy st(cfg);
    std::vector<pjev::Candidate> c(4);
    const char* d[] = {"poor", "fair", "good", "excellent"};
    for (int i = 0; i < 4; i++) { c[i].key = std::to_string(i); c[i].description = d[i]; }
    st.assign_labels(pjev::Type::Score, c);
    auto segs = st.build_prompt_segments(pjev::Type::Score, "STATE", "QUESTION", c);
    std::string all;
    for (const auto& s : segs) all += s.text;
    return all;
}

int main() {
    using namespace pjev;
    const PromptConfig F0{Layout::AUTO, Scheme::LETTERS};
    const PromptConfig F1{Layout::AUTO, Scheme::LETTERS, F1_FILLER};

    // 1. Filler insertion position: options block -> filler -> ANSWER_ANCHOR
    {
        std::string p = prompt_text(F1);
        std::string anchor = PromptStrategy::ANSWER_ANCHOR;

        // Filler must appear somewhere in the prompt
        auto filler_pos = p.find(F1_FILLER);
        check(filler_pos != std::string::npos, "position: filler found in prompt");

        // ANSWER_ANCHOR must come after filler
        auto anchor_pos = p.rfind(anchor);
        check(anchor_pos != std::string::npos && anchor_pos > filler_pos,
              "position: ANSWER_ANCHOR follows filler");

        // Filler must appear exactly once
        auto filler_pos2 = p.find(F1_FILLER, filler_pos + 1);
        check(filler_pos2 == std::string::npos, "position: filler appears exactly once");

        // Options header must precede filler
        auto opts_pos = p.find("Possible answers");
        check(opts_pos != std::string::npos && opts_pos < filler_pos,
              "position: options header precedes filler");
    }

    // 2. F0 unchanged: with filler_text="" the prompt is identical to S1
    {
        std::string p0 = prompt_text(F0);
        std::string pf = prompt_text(F1);
        check(p0 != pf, "f0_unchanged: F1 and F0 prompts differ");
        check(p0.find(F1_FILLER) == std::string::npos, "f0_unchanged: filler absent in F0 prompt");

        // F0 prompt is a prefix of F1 prompt up to where filler was inserted (the filler is
        // inserted before ANSWER_ANCHOR, so F0 == F1 with the filler+newline removed).
        std::string anchor = PromptStrategy::ANSWER_ANCHOR;
        std::string filler_block = "\n" + F1_FILLER;
        auto pos = pf.find(filler_block);
        check(pos != std::string::npos, "f0_unchanged: filler block found in F1 text");
        if (pos != std::string::npos) {
            std::string pf_stripped = pf.substr(0, pos) + pf.substr(pos + filler_block.size());
            check(pf_stripped == p0, "f0_unchanged: removing filler from F1 yields exact F0");
        }
    }

    // 3. Filler content: prompt text ends with <filler>\n<ANSWER_ANCHOR> sequence
    {
        std::string p = prompt_text(F1);
        std::string anchor = PromptStrategy::ANSWER_ANCHOR;
        std::string expected_tail = "\n" + F1_FILLER + anchor;
        bool ends_with_tail = p.size() >= expected_tail.size() &&
            p.compare(p.size() - expected_tail.size(), std::string::npos, expected_tail) == 0;
        check(ends_with_tail, "content: prompt ends with \\nFILLER + ANSWER_ANCHOR");
    }

    // 4. Candidate mapping unchanged: A->0, B->1, C->2, D->3
    {
        MockBackend mock;
        DecisionEngine eng(mock, F1);
        auto labels = eng.candidate_labels(Type::Score, 4);
        check(labels == std::vector<std::string>({"A", "B", "C", "D"}),
              "mapping: F1 level 0..3 -> A..D");
    }

    // 5. Candidate token IDs unchanged: same as F0
    {
        MockBackend mock;
        DecisionEngine eng_f0(mock, F0);
        DecisionEngine eng_f1(mock, F1);
        bool same = true;
        for (int i = 0; i < 4; i++) {
            std::string label(1, (char)('A' + i));
            same = same && eng_f0.candidate_token_id(label) == eng_f1.candidate_token_id(label);
        }
        check(same, "token_ids: F1 uses same candidate token IDs as F0");
    }

    // 6. One evaluation per item: filler does not add a second inference pass
    {
        LetterMockBackend mock;
        mock.letter_logits = {0.1f, 0.2f, 2.0f, 0.5f};
        ExperimentConfig cfg;
        cfg.scheme = Scheme::LETTERS;
        cfg.filler_text = F1_FILLER;
        mock.calls = 0;
        run_experiment(mock, {score_row("e1", 2)}, cfg);
        check(mock.calls == 1, "evaluations: exactly one inference call per item");
        auto s = run_experiment(mock, {score_row("e2", 2), score_row("e3", 1)}, cfg).score_samples();
        for (const auto& sa : s)
            check(sa.evaluations == 1, "evaluations: sample.evaluations == 1");
    }

    // 7. Semantic prediction identical to S1 when same logits: filler does not change interpretation
    {
        LetterMockBackend mock;
        mock.letter_logits = {0.1f, 0.2f, 2.0f, 0.5f};

        ExperimentConfig cfg_s1, cfg_f1;
        cfg_s1.scheme = Scheme::LETTERS;
        cfg_f1.scheme = Scheme::LETTERS;
        cfg_f1.filler_text = F1_FILLER;

        auto row = score_row("s", 2);
        auto s1 = run_experiment(mock, {row}, cfg_s1).score_samples();
        auto f1 = run_experiment(mock, {row}, cfg_f1).score_samples();

        check(s1.size() == 1 && f1.size() == 1, "prediction: one sample each");
        if (s1.size() == 1 && f1.size() == 1) {
            check(s1[0].prediction == f1[0].prediction, "prediction: same argmax (same mock logits)");
            check(s1[0].labels == f1[0].labels, "prediction: same label list");
        }
    }

    // 8. Token count: F1 prompt tokens > F0 prompt tokens by exactly len("\n" + F1_FILLER) chars
    //    (MockBackend tokenizes each char as one token for multi-char strings)
    {
        LetterMockBackend mock;
        mock.letter_logits = {1.0f, 2.0f, 3.0f, 4.0f};

        DecisionInput inp;
        inp.type = Type::Score;
        inp.state = "The response was mostly accurate.";
        inp.question = "Rate the accuracy.";
        const char* d[] = {"poor", "fair", "good", "excellent"};
        for (int i = 0; i < 4; i++) inp.options.push_back({std::to_string(i), d[i]});

        DecisionEngine eng_f0(mock, F0);
        DecisionEngine eng_f1(mock, F1);
        auto out0 = eng_f0.decide(inp);
        auto out1 = eng_f1.decide(inp);

        int expected_delta = (int)(1 + F1_FILLER.size()); // "\n" + filler chars, 1 token each in mock
        int actual_delta = out1.prompt_token_count - out0.prompt_token_count;
        check(actual_delta == expected_delta, "token_count: prompt tokens increased by 1+len(filler)");
    }

    // 9. Metric code: F1 results go through identical metric computation as F0 (ScoreSample fields match)
    {
        LetterMockBackend mock;
        mock.letter_logits = {0.5f, 1.0f, 3.0f, 2.0f};
        ExperimentConfig cfg_f1;
        cfg_f1.scheme = Scheme::LETTERS;
        cfg_f1.filler_text = F1_FILLER;
        auto s = run_experiment(mock, {score_row("m1", 2)}, cfg_f1).score_samples();
        check(s.size() == 1 && s[0].labeled() && s[0].levels.size() == 4 &&
              s[0].labels.size() == 4 && s[0].logits.size() == 4,
              "metrics: ScoreSample fields populated for F1");
        check(!s[0].rotations.empty() == false, "metrics: no rotation diagnostics for F1 (single pass)");
        auto m = compute_score_metrics(s);
        check(m.n == 1 && m.labeled == 1, "metrics: compute_score_metrics runs on F1 samples");
    }

    printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails > 0 ? 1 : 0;
}
