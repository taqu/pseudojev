#include "decision/decision_engine.h"
#include "decision/prompt_strategy.h"
#include "mock_backend.h"
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

// Extended mock backend that records calls and supports deterministic logits
class RecordingMockBackend : public pjev::ILlamaBackend {
public:
    static int single_token_id(const std::string& text) {
        return MockBackend::single_token_id(text);
    }

    std::vector<int> tokenize(const std::string& text, bool, bool) const override {
        int id = single_token_id(text);
        if (id >= 0) return {id};
        std::vector<int> toks;
        for (unsigned char c : text) toks.push_back(1000 + c);
        return toks;
    }

    int vocab_size() const override { return 600; }

    std::string token_to_piece(int token_id) const override {
        if (token_id >= 65 && token_id <= 74) return std::string(1, (char)token_id);
        if (token_id == 200) return "No";
        if (token_id == 201) return "Yes";
        for (int i = 0; i < 10; i++) {
            if (token_id == 202 + i) return std::to_string(i);
        }
        return "?";
    }

    void set_winner(int token_id) { winner_id_ = token_id; }

    const float* eval_tokens(const std::vector<int>& tokens) override {
        last_eval_tokens_ = tokens;
        eval_calls_++;
        logits_.assign(600, 0.0f);
        if (winner_id_ >= 0 && winner_id_ < 600) {
            logits_[winner_id_] = 100.0f;
        }
        return logits_.data();
    }

    const float* eval_tokens_reuse_prefix(const std::vector<int>& tokens, int32_t prefix_len) override {
        reuse_calls_++;
        last_reuse_tokens_ = tokens;
        last_prefix_len_   = prefix_len;
        // Falls back to eval_tokens (ignores prefix_len, safe for tests)
        return eval_tokens(tokens);
    }

    int eval_calls()  const { return eval_calls_; }
    int reuse_calls() const { return reuse_calls_; }
    const std::vector<int>& last_eval_tokens()  const { return last_eval_tokens_; }
    const std::vector<int>& last_reuse_tokens() const { return last_reuse_tokens_; }
    int32_t last_prefix_len() const { return last_prefix_len_; }

private:
    int   winner_id_     = 65;
    int   eval_calls_    = 0;
    int   reuse_calls_   = 0;
    int32_t last_prefix_len_ = 0;
    std::vector<int> last_eval_tokens_;
    std::vector<int> last_reuse_tokens_;
    mutable std::vector<float> logits_;
};

// ---------------------------------------------------------------------------

static pjev::DecisionInput make_noul_input(const std::string& state, const std::string& question) {
    pjev::DecisionInput inp;
    inp.type = pjev::Type::Noul;
    inp.state    = state;
    inp.question = question;
    inp.options  = {{"false", ""}, {"true", ""}};
    return inp;
}

static pjev::DecisionInput make_choice_input(const std::string& state, const std::string& question) {
    pjev::DecisionInput inp;
    inp.type = pjev::Type::Choice;
    inp.state    = state;
    inp.question = question;
    inp.options  = {{"opt1", "Option 1"}, {"opt2", "Option 2"}};
    return inp;
}

static pjev::DecisionInput make_score_input(const std::string& state, const std::string& question) {
    pjev::DecisionInput inp;
    inp.type = pjev::Type::Score;
    inp.state    = state;
    inp.question = question;
    inp.options  = {{"0", "low"}, {"1", "medium"}, {"2", "high"}};
    return inp;
}

// ---------------------------------------------------------------------------
// test_prefix_info: build_prefix_info returns valid=true for noul/choice and valid=false for score
// ---------------------------------------------------------------------------

static void test_prefix_info() {
    pjev::PromptStrategy strategy;
    const std::string state = "Some state text.";

    auto noul_info = strategy.build_prefix_info(pjev::Type::Noul, state);
    assert(noul_info.valid);
    assert(!noul_info.seg0_text.empty());
    assert(!noul_info.partial_content.empty());
    assert(noul_info.partial_content.find(state) != std::string::npos);

    auto choice_info = strategy.build_prefix_info(pjev::Type::Choice, state);
    assert(choice_info.valid);

    auto score_info = strategy.build_prefix_info(pjev::Type::Score, state);
    assert(!score_info.valid);

    // STATE_LAST layout disables prefix reuse
    pjev::PromptConfig cfg_sl;
    cfg_sl.layout = pjev::Layout::STATE_LAST;
    pjev::PromptStrategy strategy_sl(cfg_sl);
    auto noul_sl = strategy_sl.build_prefix_info(pjev::Type::Noul, state);
    assert(!noul_sl.valid);

    // QUESTION_FIRST layout disables prefix reuse
    pjev::PromptConfig cfg_qf;
    cfg_qf.layout = pjev::Layout::QUESTION_FIRST;
    pjev::PromptStrategy strategy_qf(cfg_qf);
    auto noul_qf = strategy_qf.build_prefix_info(pjev::Type::Noul, state);
    assert(!noul_qf.valid);

    printf("test_prefix_info: PASS\n");
}

// ---------------------------------------------------------------------------
// test_batch_equivalence: decide_batch results equal sequential decide() results
// ---------------------------------------------------------------------------

static void test_batch_equivalence() {
    MockBackend backend;
    pjev::DecisionEngine engine(backend);

    const std::string state = "Shared state for all questions.";
    std::vector<pjev::DecisionInput> inputs = {
        make_noul_input(state, "Question 1?"),
        make_noul_input(state, "Question 2?"),
        make_noul_input(state, "Question 3?"),
    };

    // Sequential results
    std::vector<pjev::DecisionOutput> seq_results;
    for (const auto& inp : inputs) seq_results.push_back(engine.decide(inp));

    // Batch results
    pjev::BatchConfig bcfg;
    auto batch_results = engine.decide_batch(inputs, bcfg);

    assert(batch_results.size() == 3);
    for (size_t i = 0; i < 3; i++) {
        assert(batch_results[i].ok);
        assert(seq_results[i].ok);
        assert(batch_results[i].probs.size() == seq_results[i].probs.size());
        for (size_t j = 0; j < batch_results[i].probs.size(); j++) {
            assert(std::abs(batch_results[i].probs[j] - seq_results[i].probs[j]) < 1e-9);
        }
        assert(batch_results[i].selected == seq_results[i].selected);
        assert(batch_results[i].p_true == seq_results[i].p_true);
    }

    printf("test_batch_equivalence: PASS\n");
}

// ---------------------------------------------------------------------------
// test_batch_isolation: Q2 call uses eval_tokens_reuse_prefix with correct prefix
// ---------------------------------------------------------------------------

static void test_batch_isolation() {
    RecordingMockBackend backend;
    pjev::DecisionEngine engine(backend);

    const std::string state = "Shared state text.";
    std::vector<pjev::DecisionInput> inputs = {
        make_noul_input(state, "Q1?"),
        make_noul_input(state, "Q2?"),
    };

    pjev::BatchConfig bcfg;
    bcfg.use_kv_reuse = true;
    auto results = engine.decide_batch(inputs, bcfg);

    assert(results.size() == 2);
    assert(results[0].ok);
    assert(results[1].ok);

    // Second question should have used eval_tokens_reuse_prefix
    assert(backend.reuse_calls() >= 1);

    // The reuse call should pass a prefix_len > 0
    assert(backend.last_prefix_len() > 0);

    // The full token sequence passed to the second call should start with
    // the same tokens as the first call up to prefix_len
    const auto& q1_toks = backend.last_eval_tokens();
    const auto& q2_toks = backend.last_reuse_tokens();
    int32_t pfx = backend.last_prefix_len();
    assert((int32_t)q1_toks.size() >= pfx);
    assert((int32_t)q2_toks.size() >= pfx);
    for (int32_t k = 0; k < pfx; k++) {
        assert(q1_toks[k] == q2_toks[k]);
    }

    printf("test_batch_isolation: PASS\n");
}

// ---------------------------------------------------------------------------
// test_batch_fallback_mixed: batch with one score question falls back to sequential
// ---------------------------------------------------------------------------

static void test_batch_fallback_mixed() {
    MockBackend backend;
    pjev::DecisionEngine engine(backend);

    const std::string state = "Shared state.";
    std::vector<pjev::DecisionInput> inputs = {
        make_noul_input(state, "Q1?"),
        make_score_input(state, "Q2 score?"),
    };

    pjev::BatchConfig bcfg;
    bcfg.use_kv_reuse = true;
    auto results = engine.decide_batch(inputs, bcfg);

    assert(results.size() == 2);
    assert(results[0].ok);
    assert(results[1].ok);

    // Verify score result is reasonable
    assert(results[1].probs.size() == 3);

    printf("test_batch_fallback_mixed: PASS\n");
}

// ---------------------------------------------------------------------------
// test_batch_single: batch with 1 question gives same result as decide()
// ---------------------------------------------------------------------------

static void test_batch_single() {
    MockBackend backend;
    pjev::DecisionEngine engine(backend);

    pjev::DecisionInput inp = make_noul_input("Some state.", "Single question?");

    pjev::DecisionOutput single_out = engine.decide(inp);
    auto batch_out = engine.decide_batch({inp});

    assert(batch_out.size() == 1);
    assert(batch_out[0].ok == single_out.ok);
    assert(batch_out[0].probs.size() == single_out.probs.size());
    for (size_t i = 0; i < single_out.probs.size(); i++) {
        assert(std::abs(batch_out[0].probs[i] - single_out.probs[i]) < 1e-9);
    }

    printf("test_batch_single: PASS\n");
}

// ---------------------------------------------------------------------------
// test_batch_kv_reuse_disabled: BatchConfig{use_kv_reuse=false} gives same results as sequential
// ---------------------------------------------------------------------------

static void test_batch_kv_reuse_disabled() {
    MockBackend backend;
    pjev::DecisionEngine engine(backend);

    const std::string state = "State for disabled reuse test.";
    std::vector<pjev::DecisionInput> inputs = {
        make_noul_input(state, "Q1?"),
        make_noul_input(state, "Q2?"),
        make_choice_input(state, "Q3?"),
    };

    // Sequential
    std::vector<pjev::DecisionOutput> seq;
    for (const auto& inp : inputs) seq.push_back(engine.decide(inp));

    // Batch with reuse disabled
    pjev::BatchConfig bcfg;
    bcfg.use_kv_reuse = false;
    auto batch = engine.decide_batch(inputs, bcfg);

    assert(batch.size() == 3);
    for (size_t i = 0; i < 3; i++) {
        assert(batch[i].ok);
        assert(batch[i].selected == seq[i].selected);
        for (size_t j = 0; j < batch[i].probs.size(); j++) {
            assert(std::abs(batch[i].probs[j] - seq[i].probs[j]) < 1e-9);
        }
    }

    printf("test_batch_kv_reuse_disabled: PASS\n");
}

// ---------------------------------------------------------------------------

int main() {
    test_prefix_info();
    test_batch_equivalence();
    test_batch_isolation();
    test_batch_fallback_mixed();
    test_batch_single();
    test_batch_kv_reuse_disabled();
    printf("All test_perf_opt tests passed.\n");
    return 0;
}
