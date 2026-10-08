#include "decision/decision_engine.h"
#include "experiment/runner.h"
#include "experiment/score_metrics.h"
#include "mock_backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
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

// Backend whose logits are supplied per-call via callback(call_index).
// Also validates prefix-reuse calls.
class RotationScoreMock : public MockBackend {
public:
    std::function<std::vector<float>(int)> logits_for;
    int calls = 0;
    int reuse_calls = 0;
    bool reuse_valid = true;
    std::vector<int> last_tokens;

    const float* eval_tokens(const std::vector<int>& tokens) override {
        last_tokens = tokens;
        return fill();
    }
    const float* eval_tokens_reuse_prefix(const std::vector<int>& tokens, int32_t prefix_len) override {
        reuse_calls++;
        if (prefix_len <= 0 || prefix_len >= (int32_t)tokens.size() || prefix_len > (int32_t)last_tokens.size())
            reuse_valid = false;
        else
            for (int32_t i = 0; i < prefix_len; i++)
                if (tokens[i] != last_tokens[i]) reuse_valid = false;
        last_tokens = tokens;
        return fill();
    }
private:
    const float* fill() {
        buf_.assign(600, 0.0f);
        if (logits_for) {
            auto l = logits_for(calls);
            for (size_t i = 0; i < l.size(); i++) buf_[65 + i] = l[i];
        }
        calls++;
        return buf_.data();
    }
    std::vector<float> buf_;
};

static pjev::DatasetRow score_row(const std::string& id, int expected) {
    pjev::DatasetRow row;
    row.id = id;
    row.input.type = pjev::Type::Score;
    row.input.state = "The assistant response was helpful.";
    row.input.question = "Rate the quality.";
    const char* d[] = {"poor", "fair", "good", "excellent"};
    for (int i = 0; i < 4; i++) row.input.options.push_back({std::to_string(i), d[i]});
    row.expected = expected;
    return row;
}

static std::vector<double> softmax(const std::vector<double>& v) {
    double m = *std::max_element(v.begin(), v.end());
    std::vector<double> p;
    double z = 0.0;
    for (double x : v) { p.push_back(std::exp(x - m)); z += p.back(); }
    for (double& x : p) x /= z;
    return p;
}

int main() {
    using namespace pjev;
    const PromptConfig S2cfg{Layout::AUTO, Scheme::LETTERS};

    // 1. Rotation mapping for score (S2): level i -> label (i+r)%n
    {
        // For rotation r=0: level i -> label i  (A B C D)
        // For rotation r=1: level i -> label (i+1)%4  (B C D A)
        // For rotation r=2: level i -> label (i+2)%4  (C D A B)
        // For rotation r=3: level i -> label (i+3)%4  (D A B C)
        const std::vector<std::vector<int32_t>> expected_l2l = {
            {0,1,2,3},  // r=0
            {1,2,3,0},  // r=1
            {2,3,0,1},  // r=2
            {3,0,1,2},  // r=3
        };
        RotationScoreMock mock;
        mock.logits_for = [](int) { return std::vector<float>{1,2,3,4}; };
        EnsembleConfig ens;
        ens.score_mode = ScoreEnsembleMode::LABEL_ROTATION;
        ExperimentConfig cfg;
        cfg.scheme = Scheme::LETTERS;
        cfg.ensemble = ens;
        auto r = run_experiment(mock, {score_row("r1", 3)}, cfg);
        auto s = r.score_samples();
        check(s.size() == 1 && !s[0].rotations.empty(), "mapping: S2 produces rotation diagnostics");
        if (s.size() == 1 && s[0].rotations.size() == 4) {
            bool ok = true;
            for (int rot = 0; rot < 4; rot++)
                if (s[0].rotations[rot].level_to_label != expected_l2l[rot]) ok = false;
            check(ok, "mapping: rot0..rot3 level_to_label = 0123/1230/2301/3012");
            // Every level appears exactly once at every label across all rotations
            std::vector<std::vector<int>> seen(4, std::vector<int>(4, 0));
            for (int rot = 0; rot < 4; rot++)
                for (int lv = 0; lv < 4; lv++)
                    seen[lv][s[0].rotations[rot].level_to_label[lv]]++;
            bool once = true;
            for (auto& row : seen) for (int c : row) if (c != 1) once = false;
            check(once, "mapping: every level appears exactly once at every label");
        }
        check(mock.calls == 4, "mapping: exactly 4 evaluations for 4 levels");
        check(s.size() == 1 && s[0].evaluations == 4, "mapping: evaluations == n_levels");
    }

    // 2. Semantic order preservation: S2 never changes the description order in the prompt.
    {
        MockBackend mock;
        PromptStrategy st(S2cfg);
        const char* desc[] = {"poor", "fair", "good", "excellent"};
        for (int r = 0; r < 4; r++) {
            std::vector<Candidate> cands;
            for (int i = 0; i < 4; i++) {
                Candidate c;
                c.key = std::to_string(i);
                c.description = desc[i];
                c.internal = std::string(1, (char)('A' + (i + r) % 4));
                cands.push_back(c);
            }
            auto segs = st.build_prompt_segments(Type::Score, "S", "Q", cands);
            std::string full;
            for (auto& s : segs) full += s.text;
            // Verify descriptions appear in semantic order within the prompt
            size_t p0 = full.find("poor"), p1 = full.find("fair");
            size_t p2 = full.find("good"), p3 = full.find("excellent");
            bool ordered = (p0 != std::string::npos && p1 != std::string::npos &&
                            p2 != std::string::npos && p3 != std::string::npos &&
                            p0 < p1 && p1 < p2 && p2 < p3);
            std::string msg = "semantic order preserved in rotation " + std::to_string(r);
            check(ordered, msg.c_str());
        }
    }

    // 3. Label-bias cancellation: if model always returns logit A=5, B=C=D=1,
    //    after full rotation averaging each semantic level should get equal mean logits.
    {
        RotationScoreMock mock;
        // Letter positions: A=65, B=66, C=67, D=68 (0-indexed offset from 65)
        // For each call, A token gets 5, others get 1 (in ASCII-offset space)
        mock.logits_for = [](int) { return std::vector<float>{5, 1, 1, 1}; }; // A=5, B=1, C=1, D=1

        EnsembleConfig ens;
        ens.score_mode = ScoreEnsembleMode::LABEL_ROTATION;
        ExperimentConfig cfg;
        cfg.scheme = Scheme::LETTERS;
        cfg.ensemble = ens;
        auto r = run_experiment(mock, {score_row("bias", 0)}, cfg);
        auto s = r.score_samples();
        check(s.size() == 1, "bias cancel: one sample");
        if (s.size() == 1) {
            // Mean semantic logits should all be equal: (5+1+1+1)/4 = 2.0
            const auto& lg = s[0].logits; // mean semantic logits
            bool equal = true;
            for (size_t i = 1; i < lg.size(); i++)
                if (std::fabs(lg[i] - lg[0]) > 1e-5) equal = false;
            check(equal, "bias cancel: uniform letter preference -> equal mean semantic logits");
            // Final probabilities should be approximately equal
            bool eq_probs = true;
            for (size_t i = 1; i < s[0].probs.size(); i++)
                if (std::fabs(s[0].probs[i] - s[0].probs[0]) > 1e-5) eq_probs = false;
            check(eq_probs, "bias cancel: uniform letter preference -> equal final probabilities");
        }
    }

    // 4. Semantic ordinal evidence survives rotation: level 2 should win when its
    //    letter always gets the highest logit regardless of which letter that is.
    {
        RotationScoreMock mock;
        // For each rotation r, level 2 uses letter (2+r)%4.
        // We give that letter logit 5; others get 1, 2, 1 in their natural order.
        // callback(call_index): call_index == r
        mock.logits_for = [](int call) -> std::vector<float> {
            // letter_logits[letter_idx] for this call where call == rotation r
            // level 2 -> letter (2+call)%4 should get logit 5
            std::vector<float> l = {1, 1, 1, 1};
            l[(2 + call) % 4] = 5.0f;
            return l;
        };
        EnsembleConfig ens;
        ens.score_mode = ScoreEnsembleMode::LABEL_ROTATION;
        ExperimentConfig cfg;
        cfg.scheme = Scheme::LETTERS;
        cfg.ensemble = ens;
        auto r = run_experiment(mock, {score_row("ord", 2)}, cfg);
        auto s = r.score_samples();
        check(s.size() == 1 && s[0].prediction == 2, "ordinal evidence: level 2 wins when its letter always has highest logit");
        if (s.size() == 1) {
            // Each rotation's prediction should also be level 2
            bool all2 = true;
            for (const auto& rot : s[0].rotations) if (rot.prediction != 2) all2 = false;
            check(all2, "ordinal evidence: per-rotation prediction also level 2");
            check(!s[0].rotation_disagreement(), "ordinal evidence: no rotation disagreement");
        }
    }

    // 5. Expected score is computed from semantic probabilities, not from letter positions.
    {
        RotationScoreMock mock;
        // Uniform logits -> all probs equal -> expected score = (0+1+2+3)/4 = 1.5
        mock.logits_for = [](int) { return std::vector<float>{0, 0, 0, 0}; };
        EnsembleConfig ens;
        ens.score_mode = ScoreEnsembleMode::LABEL_ROTATION;
        ExperimentConfig cfg;
        cfg.scheme = Scheme::LETTERS;
        cfg.ensemble = ens;
        auto s = run_experiment(mock, {score_row("es", 1)}, cfg).score_samples();
        check(s.size() == 1, "expected score: one sample");
        if (s.size() == 1)
            check_near(s[0].expected_score(), 1.5, 1e-6, "expected score: uniform logits -> 1.5");
    }

    // 6. Prior correction happens in letter space before semantic aggregation.
    //    For rotation r, level i uses letter (i+r)%n; the prior for letter (i+r)%n is subtracted.
    //    Use a flat-logit mock (all letter logits equal to 0) so raw logits are identical
    //    and the difference in corrected logits comes purely from the prior.
    {
        // Custom mock: always returns 0 for all letter tokens
        class FlatMock : public MockBackend {
        public:
            const float* eval_tokens(const std::vector<int>&) override {
                buf_.assign(600, 0.0f);
                return buf_.data();
            }
            const float* eval_tokens_reuse_prefix(const std::vector<int>& t, int32_t) override {
                return eval_tokens(t);
            }
        private:
            std::vector<float> buf_;
        } mock;

        EnsembleConfig ens;
        ens.score_mode = ScoreEnsembleMode::LABEL_ROTATION;
        CalibrationConfig calib;
        calib.score_prior_alpha = 1.0;
        DecisionEngine eng(mock, S2cfg, calib, ens);

        DecisionInput inp;
        inp.type = Type::Score;
        inp.state = "test";
        inp.question = "q";
        for (int i = 0; i < 4; i++) inp.options.push_back({std::to_string(i), std::string(1, 'a'+i)});
        inp.prior_correction = true;
        inp.prior_logits = {1.0f, 0.5f, 0.5f, 0.5f}; // prior: A=1.0, B=C=D=0.5
        inp.collect_corrected_logits = true;
        auto out = eng.decide(inp);
        check(out.ok, "prior correction: decide succeeds with prior");
        check(out.score_ensemble_diag && !out.score_ensemble_diag->rotations.empty(),
              "prior correction: ensemble diag populated");
        if (out.ok && out.score_ensemble_diag && out.score_ensemble_diag->rotations.size() >= 2) {
            // Rotation 0: level 0 -> letter A (prior=1.0) → corrected = 0 - 1.0 = -1.0
            // Rotation 1: level 0 -> letter B (prior=0.5) → corrected = 0 - 0.5 = -0.5
            const auto& r0 = out.score_ensemble_diag->rotations[0];
            const auto& r1 = out.score_ensemble_diag->rotations[1];
            double c0 = r0.corrected_logits[0];  // should be -1.0
            double c1 = r1.corrected_logits[0];  // should be -0.5
            check(std::fabs(c0 - (-1.0)) < 1e-5, "prior correction: r0 level-0 corrected = -1.0");
            check(std::fabs(c1 - (-0.5)) < 1e-5, "prior correction: r1 level-0 corrected = -0.5");
            // The difference confirms prior is applied in letter space
            check(c0 < c1, "prior correction: A prior(1.0) > B prior(0.5) -> r0 level-0 more penalized");
        }
    }

    // 7. No semantic permutation: S2 never reorders descriptions, even with E1/E2 also on.
    {
        MockBackend mock;
        PromptStrategy st(S2cfg);
        for (int r = 0; r < 4; r++) {
            std::vector<Candidate> cands;
            const char* desc[] = {"level0", "level1", "level2", "level3"};
            for (int i = 0; i < 4; i++) {
                Candidate c; c.key = std::to_string(i); c.description = desc[i];
                c.internal = std::string(1, (char)('A' + (i + r) % 4));
                cands.push_back(c);
            }
            auto segs = st.build_prompt_segments(Type::Score, "s", "q", cands);
            std::string full; for (auto& s : segs) full += s.text;
            auto pos = [&](const std::string& t) { return full.find(t); };
            bool no_permute = pos("level0") < pos("level1") && pos("level1") < pos("level2") && pos("level2") < pos("level3");
            std::string msg = "no permutation: descriptions in semantic order for rotation " + std::to_string(r);
            check(no_permute, msg.c_str());
        }
    }

    // 8. KV prefix reuse: predict matches sequential path, and reuse_calls > 0.
    {
        RotationScoreMock mock;
        mock.logits_for = [](int call) -> std::vector<float> {
            // level 1 always wins: for call r, level 1 -> letter (1+r)%4 gets logit 4
            std::vector<float> l = {1, 1, 1, 1};
            l[(1 + call) % 4] = 4.0f;
            return l;
        };

        // Sequential reference path
        EnsembleConfig ens_seq;
        ens_seq.score_mode = ScoreEnsembleMode::LABEL_ROTATION;
        ExperimentConfig cfg_seq;
        cfg_seq.scheme = Scheme::LETTERS;
        cfg_seq.ensemble = ens_seq;
        auto seq = run_experiment(mock, {score_row("kv", 1)}, cfg_seq).score_samples();

        // Optimized path with prefix reuse
        RotationScoreMock mock2;
        mock2.logits_for = mock.logits_for;
        EnsembleConfig ens_kv;
        ens_kv.score_mode = ScoreEnsembleMode::LABEL_ROTATION;
        ens_kv.score_prefix_reuse = true;
        ExperimentConfig cfg_kv;
        cfg_kv.scheme = Scheme::LETTERS;
        cfg_kv.ensemble = ens_kv;
        auto kv_samples = run_experiment(mock2, {score_row("kv", 1)}, cfg_kv).score_samples();

        check(seq.size() == 1 && kv_samples.size() == 1, "kv reuse: both paths produce a sample");
        if (seq.size() == 1 && kv_samples.size() == 1) {
            check(seq[0].prediction == kv_samples[0].prediction,
                  "kv reuse: prediction matches sequential path");
            check_near(seq[0].expected_score(), kv_samples[0].expected_score(), 1e-5,
                       "kv reuse: expected score matches sequential path");
            bool probs_match = true;
            for (size_t i = 0; i < seq[0].probs.size() && i < kv_samples[0].probs.size(); i++)
                if (std::fabs(seq[0].probs[i] - kv_samples[0].probs[i]) > 1e-5) probs_match = false;
            check(probs_match, "kv reuse: probabilities match sequential path");
        }
        check(mock2.reuse_calls > 0, "kv reuse: prefix reuse was invoked at least once");
        check(mock2.reuse_valid, "kv reuse: reused prefix tokens match previous sequence");
    }

    // 9. Rotation diagnostics in JSON output.
    {
        RotationScoreMock mock;
        mock.logits_for = [](int call) -> std::vector<float> {
            std::vector<float> l = {1, 1, 1, 1};
            l[(2 + call) % 4] = 4.0f;  // level 2 always wins
            return l;
        };
        EnsembleConfig ens;
        ens.score_mode = ScoreEnsembleMode::LABEL_ROTATION;
        ExperimentConfig cfg;
        cfg.scheme = Scheme::LETTERS;
        cfg.ensemble = ens;
        auto rr = run_experiment(mock, {score_row("json", 2)}, cfg);
        auto j = rr.to_json();
        const auto& si = j["score_items"];
        check(si.size() == 1, "json: one score item");
        if (si.size() == 1) {
            check(si[0].contains("rotations") && si[0]["rotations"].size() == 4, "json: 4 rotation entries");
            check(si[0]["evaluations"] == 4, "json: evaluations == 4");
            if (si[0].contains("rotations") && si[0]["rotations"].size() == 4) {
                check(si[0]["rotations"][1]["rotation"] == 1, "json: rotation index stored");
                check(si[0]["rotations"][1]["level_to_label"].is_array(), "json: level_to_label is array");
            }
        }
        // S2 metrics in the metrics block
        const auto& sm = j["metrics"]["score"];
        check(sm.contains("rotation_disagreement_rate"), "json metrics: rotation_disagreement_rate present");
        check(sm.contains("mean_distinct_winners"), "json metrics: mean_distinct_winners present");
    }

    printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails > 0 ? 1 : 0;
}
