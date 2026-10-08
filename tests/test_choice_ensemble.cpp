#include "decision/decision_engine.h"
#include "decision/rotation.h"
#include "experiment/choice_metrics.h"
#include "experiment/runner.h"
#include "mock_backend.h"
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

static int fails = 0;
static void check(bool cond, const char* what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
    else       { printf("pass: %s\n", what); }
}
static void check_near(double a, double b, double tol, const char* what) {
    check(std::fabs(a - b) < tol, what);
}

// Candidate logits for A..D (tokens 65..68) chosen per call by a callback(call_index).
// Records prefix-reuse calls and verifies the reused prefix matches the previous sequence.
class RotationMockBackend : public MockBackend {
public:
    std::function<std::vector<float>(int)> logits_for;
    int calls = 0;
    int reuse_calls = 0;
    bool reuse_prefix_valid = true;
    std::vector<int> last_tokens;

    const float* eval_tokens(const std::vector<int>& tokens) override {
        last_tokens = tokens;
        return fill();
    }
    const float* eval_tokens_reuse_prefix(const std::vector<int>& tokens, int32_t prefix_len) override {
        reuse_calls++;
        if (prefix_len <= 0 || prefix_len >= (int32_t)tokens.size() || prefix_len > (int32_t)last_tokens.size())
            reuse_prefix_valid = false;
        else
            for (int32_t i = 0; i < prefix_len; i++)
                if (tokens[i] != last_tokens[i]) reuse_prefix_valid = false;
        last_tokens = tokens;
        return fill();
    }
private:
    const float* fill() {
        buf_.assign(600, 0.0f);
        std::vector<float> l = logits_for ? logits_for(calls) : std::vector<float>{};
        for (size_t i = 0; i < l.size(); i++) buf_[65 + i] = l[i];
        calls++;
        return buf_.data();
    }
    std::vector<float> buf_;
};

static pjev::DecisionInput choice_input(int n) {
    pjev::DecisionInput inp;
    inp.type = pjev::Type::Choice;
    inp.state = "server log shows a crash";
    inp.question = "Which action should be taken?";
    const char* descs[] = {"Restart the server", "Ignore the error", "Delete the database", "Shut down the network", "Call support"};
    for (int i = 0; i < n; i++) inp.options.push_back({std::string(1, (char)('a' + i)), descs[i]});
    return inp;
}

static std::vector<double> softmax(const std::vector<double>& v, double T = 1.0) {
    double m = -1e300;
    for (double x : v) m = std::max(m, x / T);
    std::vector<double> p;
    double s = 0.0;
    for (double x : v) { p.push_back(std::exp(x / T - m)); s += p.back(); }
    for (double& x : p) x /= s;
    return p;
}

int main() {
    using namespace pjev;

    // 1. Rotation mapping and inverse for N = 4
    {
        const std::vector<std::vector<int32_t>> expected = {{0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2}};
        bool ok = true, inv_ok = true;
        std::vector<std::vector<int>> seen(4, std::vector<int>(4, 0)); // [semantic][position]
        for (int r = 0; r < 4; r++) {
            RotationMapping m = make_cyclic_rotation(4, r);
            if (m.cand_to_sem != expected[r]) ok = false;
            for (int j = 0; j < 4; j++) {
                if (m.sem_to_cand[m.cand_to_sem[j]] != j) inv_ok = false;
                seen[m.cand_to_sem[j]][j]++;
            }
        }
        check(ok, "rotation: rot0..rot3 = 0123 / 1230 / 2301 / 3012");
        check(inv_ok, "rotation: sem_to_cand is the inverse of cand_to_sem");
        bool once = true;
        for (auto& row : seen) for (int c : row) if (c != 1) once = false;
        check(once, "rotation: every semantic option appears exactly once at every position");
    }

    // 2. Semantic remapping: A->2, B->3, C->0, D->1 with A..D = 1,2,3,4 => sem = 3,4,1,2
    {
        RotationMapping m = make_cyclic_rotation(4, 2);
        check(m.cand_to_sem == std::vector<int32_t>({2, 3, 0, 1}), "remap: rotation 2 binds A->2 B->3 C->0 D->1");
        auto sem = to_semantic_order(std::vector<float>{1, 2, 3, 4}, m);
        check(sem == std::vector<float>({3, 4, 1, 2}), "remap: semantic logits = 3, 4, 1, 2");
        auto back = to_candidate_order(sem, m);
        check(back == std::vector<float>({1, 2, 3, 4}), "remap: candidate order round-trips");
    }

    EnsembleConfig e2;
    e2.choice_mode = ChoiceEnsembleMode::CYCLIC_ROTATION;

    // 3. Position-bias cancellation: every rotation prefers A regardless of meaning
    {
        RotationMockBackend mock;
        mock.logits_for = [](int) { return std::vector<float>{5, 1, 1, 1}; };
        DecisionEngine eng(mock, PromptConfig{}, {}, e2);
        auto out = eng.decide(choice_input(4));
        check(out.ok, "bias: ok");
        check(mock.calls == 4, "bias: N = 4 evaluations");
        bool equal = out.choice_ensemble_diag.has_value();
        if (equal)
            for (double s : out.choice_ensemble_diag->mean_semantic_logits) equal = equal && std::fabs(s - 2.0) < 1e-6;
        check(equal, "bias: all semantic scores equal (mean = 2)");
        bool uniform = out.probs.size() == 4;
        for (double p : out.probs) uniform = uniform && std::fabs(p - 0.25) < 1e-9;
        check(uniform, "bias: probabilities uniform (0.25)");
        if (out.choice_ensemble_diag) {
            ChoiceSample s;
            s.keys = out.keys;
            for (const auto& rd : out.choice_ensemble_diag->rotations) {
                ChoiceRotation r;
                r.prediction = rd.semantic_prediction;
                r.semantic_logits.assign(rd.semantic_logits.begin(), rd.semantic_logits.end());
                s.rotations.push_back(r);
            }
            check(s.distinct_winners() == 4 && s.rotation_disagreement(), "bias: each rotation picks a different semantic winner");
            check_near(s.mean_semantic_logit_variance(), 3.0, 1e-9, "bias: per-option variance = 3 (values 5,1,1,1)");
        }
    }

    // 4. Semantic evidence survives rotation: semantic option 2 strongest wherever it is placed
    {
        RotationMockBackend mock;
        mock.logits_for = [](int call) {
            RotationMapping m = make_cyclic_rotation(4, call);
            std::vector<float> l(4, 1.0f);
            l[m.sem_to_cand[2]] = 4.0f;
            return l;
        };
        DecisionEngine eng(mock, PromptConfig{}, {}, e2);
        auto out = eng.decide(choice_input(4));
        check(out.ok && out.selected == 2 && out.keys[out.selected] == "c", "evidence: ensemble selects semantic option 2");
        check(out.probs[2] > 0.85, "evidence: strong probability for option 2");
        bool all2 = out.choice_ensemble_diag.has_value();
        if (all2) for (const auto& r : out.choice_ensemble_diag->rotations) all2 = all2 && r.semantic_prediction == 2;
        check(all2, "evidence: every rotation's semantic winner is option 2");
        check_near(out.choice_ensemble_diag->mean_semantic_logits[2], 4.0, 1e-6, "evidence: mean semantic logit 4");
    }

    // 5. Prior correction is applied in candidate space before remapping
    //    raw: A gets +3 position bias, semantic option 2 gets +1 evidence; prior = {3,0,0,0}
    {
        auto raw_for = [](int call) {
            RotationMapping m = make_cyclic_rotation(4, call);
            std::vector<float> l(4, 0.0f);
            l[0] += 3.0f;
            l[m.sem_to_cand[2]] += 1.0f;
            return l;
        };
        RotationMockBackend mock;
        mock.logits_for = raw_for;
        DecisionEngine eng(mock, PromptConfig{}, {}, e2); // choice_prior_alpha defaults to 1
        DecisionInput inp = choice_input(4);
        inp.prior_correction = true;
        inp.prior_logits = {3.0f, 0.0f, 0.0f, 0.0f};
        auto out = eng.decide(inp);
        bool corrected_ok = true, sem_ok = true, all2 = true;
        for (const auto& rd : out.choice_ensemble_diag->rotations) {
            auto raw = raw_for(rd.rotation);
            for (int j = 0; j < 4; j++) {
                if (std::fabs(rd.raw_logits[j] - raw[j]) > 1e-6) corrected_ok = false;
                if (std::fabs(rd.corrected_logits[j] - (raw[j] - inp.prior_logits[j])) > 1e-6) corrected_ok = false;
                if (std::fabs(rd.semantic_logits[rd.cand_to_sem[j]] - rd.corrected_logits[j]) > 1e-6) sem_ok = false;
            }
            if (rd.semantic_prediction != 2) all2 = false;
        }
        check(corrected_ok, "prior: corrected = raw - alpha * prior, in candidate order");
        check(sem_ok, "prior: semantic logits are the remapped corrected logits");
        check(all2, "prior: with the A prior removed every rotation picks option 2");
        check(out.selected == 2, "prior: ensemble selects option 2");

        RotationMockBackend mock2;
        mock2.logits_for = raw_for;
        DecisionEngine eng2(mock2, PromptConfig{}, {}, e2);
        auto out2 = eng2.decide(choice_input(4));
        int non2 = 0;
        for (const auto& rd : out2.choice_ensemble_diag->rotations) if (rd.semantic_prediction != 2) non2++;
        check(non2 == 3, "prior: without correction 3 of 4 rotations follow the A bias");
        check(out2.selected == 2, "prior: rotation averaging alone still recovers option 2");
    }

    // 6. Temperature applied after aggregation; T = 1 reproduces the unscaled softmax
    {
        auto logits_for = [](int call) {
            RotationMapping m = make_cyclic_rotation(3, call);
            std::vector<float> sem = {0.5f, 2.0f, 1.0f};
            std::vector<float> cand(3);
            for (int j = 0; j < 3; j++) cand[j] = sem[m.cand_to_sem[j]] + (j == 0 ? 0.7f : 0.0f);
            return cand;
        };
        for (double T : {1.0, 2.5}) {
            RotationMockBackend mock;
            mock.logits_for = logits_for;
            CalibrationConfig cal;
            cal.enabled = true;
            cal.choice_temperature = T;
            DecisionEngine eng(mock, PromptConfig{}, cal, e2);
            auto out = eng.decide(choice_input(3));
            const auto& mean = out.choice_ensemble_diag->mean_semantic_logits;
            auto expect = softmax(mean, T);
            auto expect_raw = softmax(mean, 1.0);
            bool ok = out.probs.size() == 3, raw_ok = true;
            for (int i = 0; i < 3; i++) {
                ok = ok && std::fabs(out.probs[i] - expect[i]) < 1e-9;
                raw_ok = raw_ok && std::fabs(out.raw_probs[i] - expect_raw[i]) < 1e-9;
            }
            check_near(mean[0], 0.5 + 0.7 / 3.0, 1e-6, T == 1.0 ? "temperature T=1: A bias spread as 0.7/3" : "temperature T=2.5: A bias spread as 0.7/3");
            check(ok, T == 1.0 ? "temperature T=1: probs = softmax(mean semantic logits)" : "temperature T=2.5: probs = softmax(mean / T)");
            check(raw_ok, T == 1.0 ? "temperature T=1: raw_probs = softmax(mean)" : "temperature T=2.5: raw_probs unscaled");
        }
    }

    // 7. Sequential reference vs prefix-reuse path
    {
        auto logits_for = [](int call) {
            return std::vector<float>{0.3f * call, 1.0f - 0.2f * call, 0.7f, 0.1f * call * call};
        };
        RotationMockBackend ref_mock, opt_mock;
        ref_mock.logits_for = logits_for;
        opt_mock.logits_for = logits_for;
        EnsembleConfig opt = e2;
        opt.choice_prefix_reuse = true;
        DecisionEngine ref_eng(ref_mock, PromptConfig{}, {}, e2);
        DecisionEngine opt_eng(opt_mock, PromptConfig{}, {}, opt);
        auto r = ref_eng.decide(choice_input(4));
        auto o = opt_eng.decide(choice_input(4));
        check(ref_mock.reuse_calls == 0, "equivalence: reference path never reuses KV");
        check(opt_mock.reuse_calls == 3, "equivalence: optimized path reuses KV for rotations 1..3");
        check(opt_mock.reuse_prefix_valid, "equivalence: reused prefix always matches the previous sequence");
        bool same = r.ok && o.ok && r.selected == o.selected && r.probs.size() == o.probs.size();
        for (size_t i = 0; same && i < r.probs.size(); i++) {
            same = same && std::fabs(r.probs[i] - o.probs[i]) < 1e-12;
            same = same && std::fabs(r.choice_ensemble_diag->mean_semantic_logits[i] - o.choice_ensemble_diag->mean_semantic_logits[i]) < 1e-12;
        }
        check(same, "equivalence: semantic logits, probabilities and selection match");
        bool reused = o.choice_ensemble_diag->rotations[0].prefix_reused == 0;
        for (int k = 1; k < 4; k++) reused = reused && o.choice_ensemble_diag->rotations[k].prefix_reused > 0;
        check(reused, "equivalence: prefix_reused recorded (0 for rotation 0, > 0 after)");
    }

    // 8. Baseline unchanged when E2 is off; E1 noul unaffected by E2 config
    {
        RotationMockBackend mock;
        mock.logits_for = [](int) { return std::vector<float>{1, 3, 2, 0}; };
        DecisionEngine eng(mock, PromptConfig{}, {}, {});
        auto out = eng.decide(choice_input(4));
        check(mock.calls == 1 && out.selected == 1 && !out.choice_ensemble_diag, "baseline: single evaluation, no rotation diag");

        RotationMockBackend nm;
        nm.logits_for = [](int) { return std::vector<float>{1, 3}; };
        DecisionEngine neng(nm, PromptConfig{}, {}, e2);
        DecisionInput ni;
        ni.type = Type::Noul;
        ni.state = "s";
        ni.question = "q";
        ni.options = {{"false", ""}, {"true", ""}};
        auto nout = neng.decide(ni);
        check(nm.calls == 1 && nout.ok && !nout.ensemble_diag, "baseline: E2 config does not change noul");

        RotationMockBackend bm;
        bm.logits_for = [](int) { return std::vector<float>{5, 1, 1, 1}; };
        DecisionEngine beng(bm, PromptConfig{}, {}, e2);
        auto outs = beng.decide_batch({choice_input(4), choice_input(4)});
        check(bm.calls == 8 && outs.size() == 2 && outs[1].choice_ensemble_diag.has_value(),
              "batch: decide_batch routes E2 through the rotation ensemble");
    }

    // 9. Metric helpers
    {
        check_near(signed_winner_margin({1.0, 3.0, 2.5}, 1), 0.5, 1e-12, "winner margin: correct side = 3 - 2.5");
        check_near(signed_winner_margin({1.0, 3.0, 2.5}, 0), -2.0, 1e-12, "winner margin: wrong side = 1 - 3");
        check(std::isnan(signed_winner_margin({1.0, 2.0}, -1)), "winner margin: unlabeled -> NaN");

        ChoiceSample a;
        a.source_id = "q1"; a.keys = {"x", "y", "z"}; a.ground_truth = 1; a.prediction = 1;
        a.logits = {0.0, 2.0, 1.0}; a.raw_logits = a.logits; a.probs = softmax(a.logits);
        ChoiceSample b = a;
        b.source_id = "q2"; b.ground_truth = 2; b.prediction = 1;
        b.logits = {0.0, 1.0, 0.5}; b.raw_logits = b.logits;
        auto m = compute_choice_metrics({a, b});
        check_near(m.accuracy, 0.5, 1e-12, "metrics: accuracy 1/2");
        check_near(m.signed_margin.min, -0.5, 1e-12, "metrics: min winner margin -0.5");
        check_near(m.signed_margin.mean, 0.25, 1e-12, "metrics: mean winner margin 0.25");
        // Same definitions as compute_primitive_metrics (summed multi-class Brier)
        std::vector<CalibrationSample> cs(2);
        cs[0].type = cs[1].type = "choice";
        cs[0].logits = {0.0f, 2.0f, 1.0f}; cs[0].correct_index = 1;
        cs[1].logits = {0.0f, 1.0f, 0.5f}; cs[1].correct_index = 2;
        auto ref = compute_primitive_metrics(cs, 1.0, 15);
        check_near(m.raw.nll, ref.nll, 1e-12, "metrics: NLL matches compute_primitive_metrics");
        check_near(m.raw.brier, ref.brier, 1e-12, "metrics: Brier matches compute_primitive_metrics");
        check_near(m.raw.ece, ref.ece, 1e-12, "metrics: ECE matches compute_primitive_metrics");
        auto p = softmax(a.logits);
        double brier_a = p[0] * p[0] + (p[1] - 1) * (p[1] - 1) + p[2] * p[2];
        auto single = compute_choice_metrics({a});
        check_near(single.raw.brier, brier_a, 1e-6, "metrics: Brier = sum over options of (p - y)^2");

        // rotations: disagreement and distinct winners
        ChoiceSample r = a;
        for (int k = 0; k < 3; k++) {
            ChoiceRotation cr;
            cr.rotation = k;
            cr.prediction = (k == 2) ? 0 : 1;
            cr.semantic_logits = {k == 2 ? 3.0 : 0.0, 2.0, 1.0};
            r.rotations.push_back(cr);
        }
        ChoiceSample stable = b;
        for (int k = 0; k < 3; k++) {
            ChoiceRotation cr;
            cr.prediction = 1;
            cr.semantic_logits = {0.0, 1.0, 0.5};
            stable.rotations.push_back(cr);
        }
        check(r.distinct_winners() == 2 && r.rotation_disagreement(), "rotations: 2 distinct winners -> disagreement");
        check_near(r.winner_agreement(), 2.0 / 3.0, 1e-12, "rotations: winner agreement 2/3");
        check(r.winner_counts() == std::vector<int32_t>({1, 2, 0}), "rotations: semantic winner frequency");
        check_near(r.max_semantic_logit_variance(), 2.0, 1e-12, "rotations: variance of {0,0,3} = 2");
        check_near(stable.mean_semantic_logit_variance(), 0.0, 1e-12, "rotations: stable sample variance 0");
        auto mr = compute_choice_metrics({r, stable});
        check(mr.samples_with_any_rotation_disagreement == 1, "rotations: 1 sample with disagreement");
        check_near(mr.rotation_disagreement_rate(), 0.5, 1e-12, "rotations: disagreement rate 0.5");
        check_near(mr.mean_distinct_winners, 1.5, 1e-12, "rotations: mean distinct winners 1.5");
        check_near(mr.mean_rotations, 3.0, 1e-12, "rotations: 3 evaluations per item");

        // margin gain + correctness flips
        ChoiceSample a2 = a; a2.prediction = 2; a2.logits = {0.0, 1.0, 2.0};
        ChoiceSample b2 = b; b2.prediction = 2; b2.logits = {0.0, 0.5, 1.5};
        auto g = compute_margin_gain(std::vector<ChoiceSample>{a, b}, std::vector<ChoiceSample>{a2, b2});
        check(g.n_matched == 2 && g.improved == 1 && g.degraded == 1, "gain: 1 improved, 1 degraded");
        check(g.correct_to_wrong == 1 && g.wrong_to_correct == 1, "gain: baseline correct->wrong 1, wrong->correct 1");

        auto rt = ChoiceSample::from_json(r.to_json());
        check(rt.source_id == "q1" && rt.ground_truth == 1 && rt.rotations.size() == 3 &&
              rt.rotations[2].prediction == 0 && std::fabs(rt.signed_margin() - r.signed_margin()) < 1e-12,
              "JSON: choice sample round-trip");
    }

    // 10. End to end through run_experiment
    {
        DatasetRow row;
        row.id = "c1";
        row.input = choice_input(4);
        row.expected = "c";
        std::vector<DatasetRow> rows = {row};
        auto logits_for = [](int call) {
            RotationMapping m = make_cyclic_rotation(4, call % 4);
            std::vector<float> l(4, 1.0f);
            l[0] += 2.0f;               // position bias
            l[m.sem_to_cand[2]] += 1.5f; // semantic evidence for "c"
            return l;
        };
        RotationMockBackend bm;
        bm.logits_for = logits_for;
        ExperimentConfig base_cfg;
        RunResult base = run_experiment(bm, rows, base_cfg);
        RotationMockBackend em;
        em.logits_for = logits_for;
        ExperimentConfig e2_cfg;
        e2_cfg.ensemble.choice_mode = ChoiceEnsembleMode::CYCLIC_ROTATION;
        RunResult e2r = run_experiment(em, rows, e2_cfg);

        auto bs = base.choice_samples();
        auto es = e2r.choice_samples();
        check(bs.size() == 1 && !bs[0].correct(), "e2e: baseline follows the A bias (wrong)");
        check(es.size() == 1 && es[0].correct() && es[0].rotations.size() == 4, "e2e: E2 correct with 4 rotations");
        check_near(bs[0].signed_margin(), -0.5, 1e-5, "e2e: baseline winner margin -0.5");
        check_near(es[0].signed_margin(), 1.5, 1e-5, "e2e: E2 winner margin 1.5");
        auto g = compute_margin_gain(bs, es);
        check(g.wrong_to_correct == 1 && g.correct_to_wrong == 0, "e2e: baseline wrong -> E2 correct");

        auto j = e2r.to_json();
        const auto& cj = j["metrics"]["choice"];
        check(cj["accuracy"].get<double>() == 1.0 && cj["n"] == 1 && cj.contains("eval_ms"), "e2e JSON: existing fields kept");
        check(cj.contains("nll") && cj.contains("brier") && cj.contains("ece"), "e2e JSON: nll / brier / ece");
        check(cj.contains("min_signed_winner_margin") && cj.contains("rotation_disagreement_rate") &&
              cj.contains("mean_semantic_logit_variance") && cj["rotations_per_item"].get<double>() == 4.0,
              "e2e JSON: winner margin and rotation diagnostics");
        check(j["choice_ensemble"] == "cyclic-rotation" && j["choice_items"].size() == 1, "e2e JSON: mode and choice_items");
        check(!base.to_json()["metrics"]["choice"].contains("rotation_disagreement_rate"), "e2e JSON: baseline has no rotation fields");
    }

    printf("\n%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails > 0 ? 1 : 0;
}
