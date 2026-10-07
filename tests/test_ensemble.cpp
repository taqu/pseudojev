#include "decision/decision_engine.h"
#include "mock_backend.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

static int fails = 0;
static void check(bool cond, const char* what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
    else        { printf("pass: %s\n", what); }
}

// Returns a sigmoid value
static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Mock backend that returns a specific sequence of (logit_A, logit_B) per eval call.
// logit_A -> token 65, logit_B -> token 66.
class SequentialMockBackend : public pjev::ILlamaBackend {
public:
    void set_calls(std::vector<std::pair<float,float>> c) { calls_ = std::move(c); idx_ = 0; }

    std::vector<int> tokenize(const std::string& text, bool, bool) const override {
        int id = MockBackend::single_token_id(text);
        if (id >= 0) return {id};
        std::vector<int> toks;
        for (unsigned char c : text) toks.push_back(1000 + (int)c);
        return toks;
    }
    int vocab_size() const override { return 600; }
    std::string token_to_piece(int token_id) const override {
        if (token_id >= 65 && token_id <= 74) return std::string(1, (char)token_id);
        if (token_id == 200) return "No";
        if (token_id == 201) return "Yes";
        for (int i = 0; i < 10; i++) if (token_id == 202+i) return std::to_string(i);
        return "?";
    }
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
    std::vector<std::pair<float,float>> calls_;
    int idx_ = 0;
    mutable std::vector<float> logits_;
};

int main() {
    using namespace pjev;

    // Test 1: Semantic remapping
    // Run 1: A=1.0, B=3.0; A=False, B=True => m1 = 3.0 - 1.0 = 2.0
    // Run 2: A=4.0, B=1.0; A=True, B=False => m2 = 4.0 - 1.0 = 3.0
    // m_ensemble = 2.5; p_true = sigmoid(2.5)
    {
        SequentialMockBackend mock;
        mock.set_calls({{1.0f, 3.0f}, {4.0f, 1.0f}});
        EnsembleConfig ens; ens.noul_mode = NoulEnsembleMode::BINARY_ORDER;
        DecisionEngine eng(mock, PromptConfig{}, {}, ens);

        DecisionInput inp;
        inp.type = Type::Noul;
        inp.state   = "test";
        inp.question = "q";
        inp.options = {{"false",""}, {"true",""}};
        auto out = eng.decide(inp);

        check(out.ok,   "T1: ok");
        check(out.ensemble_diag.has_value(), "T1: diag present");
        if (out.ensemble_diag) {
            check(std::fabs(out.ensemble_diag->m1 - 2.0) < 1e-5, "T1: m1=2.0");
            check(std::fabs(out.ensemble_diag->m2 - 3.0) < 1e-5, "T1: m2=3.0");
            check(std::fabs(out.ensemble_diag->m_ensemble - 2.5) < 1e-5, "T1: m_ensemble=2.5");
        }
        double expected_p = sigmoid(2.5);
        check(std::fabs(out.p_true - expected_p) < 1e-6, "T1: p_true=sigmoid(2.5)");
        check(out.selected == 1, "T1: selected=true (index 1)");
        check(out.keys[out.selected] == "true", "T1: key=true");
    }

    // Test 2: Position-bias cancellation
    // Model always prefers A (3.0) over B (2.0) in both orderings.
    // Run 1: A=False, B=True => m1 = 2.0 - 3.0 = -1.0
    // Run 2: A=True,  B=False => m2 = 3.0 - 2.0 = +1.0
    // m_ensemble = 0.0; p_true = sigmoid(0) = 0.5
    {
        SequentialMockBackend mock;
        mock.set_calls({{3.0f, 2.0f}, {3.0f, 2.0f}});
        EnsembleConfig ens; ens.noul_mode = NoulEnsembleMode::BINARY_ORDER;
        DecisionEngine eng(mock, PromptConfig{}, {}, ens);

        DecisionInput inp;
        inp.type = Type::Noul;
        inp.state   = "test";
        inp.question = "q";
        inp.options = {{"false",""}, {"true",""}};
        auto out = eng.decide(inp);

        check(out.ok, "T2: ok");
        if (out.ensemble_diag) {
            check(std::fabs(out.ensemble_diag->m1 - (-1.0)) < 1e-5, "T2: m1=-1.0");
            check(std::fabs(out.ensemble_diag->m2 -   1.0 ) < 1e-5, "T2: m2=+1.0");
            check(std::fabs(out.ensemble_diag->m_ensemble)   < 1e-5, "T2: m_ensemble=0");
        }
        check(std::fabs(out.p_true - 0.5) < 1e-6, "T2: p_true=0.5 (bias cancelled)");
    }

    // Test 3: Semantic evidence survives permutation
    // Run 1: A=False=1.0, B=True=3.0  => m1 = 3.0 - 1.0 = 2.0
    // Run 2: A=True=3.5,  B=False=1.5 => m2 = 3.5 - 1.5 = 2.0
    // m_ensemble = 2.0; p_true = sigmoid(2.0) > 0.88
    {
        SequentialMockBackend mock;
        mock.set_calls({{1.0f, 3.0f}, {3.5f, 1.5f}});
        EnsembleConfig ens; ens.noul_mode = NoulEnsembleMode::BINARY_ORDER;
        DecisionEngine eng(mock, PromptConfig{}, {}, ens);

        DecisionInput inp;
        inp.type = Type::Noul;
        inp.state   = "test";
        inp.question = "q";
        inp.options = {{"false",""}, {"true",""}};
        auto out = eng.decide(inp);

        check(out.ok, "T3: ok");
        check(out.p_true > 0.88, "T3: strong True preference survives permutation");
        check(out.selected == 1, "T3: selected=true");
    }

    // Test 4: Prior correction applied per ordering in candidate space
    // prior_logits = [1.0, 1.0] (one entry per candidate, alpha=1 default)
    // Run 1: raw A=2.0, B=4.0 -> corrected A=1.0, B=3.0 -> m1=2.0 (A=False)
    // Run 2: raw A=5.0, B=2.0 -> corrected A=4.0, B=1.0 -> m2=3.0 (A=True)
    // m_ensemble=2.5; same as test 1 but via prior correction
    {
        SequentialMockBackend mock;
        mock.set_calls({{2.0f, 4.0f}, {5.0f, 2.0f}});
        EnsembleConfig ens; ens.noul_mode = NoulEnsembleMode::BINARY_ORDER;
        DecisionEngine eng(mock, PromptConfig{}, {}, ens);

        DecisionInput inp;
        inp.type = Type::Noul;
        inp.state   = "test";
        inp.question = "q";
        inp.options = {{"false",""}, {"true",""}};
        inp.prior_correction = true;
        inp.prior_logits = {1.0f, 1.0f};
        auto out = eng.decide(inp);

        check(out.ok, "T4: ok");
        if (out.ensemble_diag) {
            check(std::fabs(out.ensemble_diag->m1 - 2.0) < 1e-4, "T4: m1=2.0 after prior correction");
            check(std::fabs(out.ensemble_diag->m2 - 3.0) < 1e-4, "T4: m2=3.0 after prior correction");
            check(std::fabs(out.ensemble_diag->m_ensemble - 2.5) < 1e-4, "T4: m_ensemble=2.5");
        }
        double expected_p = sigmoid(2.5);
        check(std::fabs(out.p_true - expected_p) < 1e-5, "T4: p_true matches sigmoid(2.5)");
    }

    // Test 5: Temperature scaling
    // m_ensemble=2.0; T=1 => sigmoid(2.0); T=2 => sigmoid(1.0)
    {
        // T=1 (default)
        SequentialMockBackend mock1;
        mock1.set_calls({{1.0f, 3.0f}, {3.0f, 1.0f}});
        EnsembleConfig ens; ens.noul_mode = NoulEnsembleMode::BINARY_ORDER;
        DecisionEngine eng1(mock1, PromptConfig{}, {}, ens);
        DecisionInput inp;
        inp.type = Type::Noul; inp.state = "s"; inp.question = "q";
        inp.options = {{"false",""}, {"true",""}};
        auto out1 = eng1.decide(inp);
        check(out1.ok, "T5: T=1 ok");
        check(std::fabs(out1.p_true - sigmoid(2.0)) < 1e-6, "T5: T=1 matches sigmoid(2.0)");

        // T=2
        SequentialMockBackend mock2;
        mock2.set_calls({{1.0f, 3.0f}, {3.0f, 1.0f}});
        CalibrationConfig calib;
        calib.noul_temperature = 2.0;
        calib.enabled = true;
        DecisionEngine eng2(mock2, PromptConfig{}, calib, ens);
        auto out2 = eng2.decide(inp);
        check(out2.ok, "T5: T=2 ok");
        check(std::fabs(out2.p_true - sigmoid(1.0)) < 1e-6, "T5: T=2 matches sigmoid(m/2)");
    }

    // Test 6: NONE mode uses single call (no ensemble)
    {
        int eval_count = 0;
        // Use a regular MockBackend and count calls via the sequential mock
        SequentialMockBackend mock;
        mock.set_calls({{2.0f, 1.0f}, {2.0f, 1.0f}});
        // NONE mode — should call eval_tokens exactly once
        DecisionEngine eng(mock, PromptConfig{}, {}, {});  // default EnsembleConfig = NONE
        DecisionInput inp;
        inp.type = Type::Noul; inp.state = "s"; inp.question = "q";
        inp.options = {{"false",""}, {"true",""}};
        auto out = eng.decide(inp);
        check(out.ok, "T6: NONE mode ok");
        check(!out.ensemble_diag.has_value(), "T6: no diag in NONE mode");
        // With NONE mode, only one eval call consumed → idx_ should be 1, not 2
        // We verify indirectly: p_true from single call A=2.0, B=1.0 (A=False, B=True)
        // m = 1.0-2.0 = -1.0; p_true = softmax[1] ≈ sigmoid(-1) ≈ 0.269
        check(out.p_true < 0.3, "T6: NONE mode p_true from single call");
    }

    printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails > 0 ? 1 : 0;
}
