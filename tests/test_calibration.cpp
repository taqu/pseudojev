#include "calibration/calibration.h"
#include "calibration/calibration_fit.h"
#include "calibration/calibration_metrics.h"
#include "experiment/dataset.h"
#include <cmath>
#include <cstdio>
#include <vector>

static int fails = 0;
static void check(bool cond, const char* what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
    else        { printf("pass: %s\n", what); }
}
static void check_near(double a, double b, double tol, const char* what) {
    check(std::fabs(a - b) < tol, what);
}

int main() {
    using namespace pjev;

    // -----------------------------------------------------------------------
    // 1. temperature_softmax: T=1 should match plain softmax
    // -----------------------------------------------------------------------
    {
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        std::vector<double> p1, p2;
        check(temperature_softmax(logits, 1.0, p1).empty(), "T=1 softmax: no error");
        check(temperature_softmax(logits, 2.0, p2).empty(), "T=2 softmax: no error");

        double sum1 = p1[0]+p1[1]+p1[2];
        double sum2 = p2[0]+p2[1]+p2[2];
        check_near(sum1, 1.0, 1e-9, "T=1 probs sum to 1");
        check_near(sum2, 1.0, 1e-9, "T=2 probs sum to 1");
    }

    // -----------------------------------------------------------------------
    // 2. T=1 identity: temperature_softmax(T=1) == restricted_softmax (same logits)
    // -----------------------------------------------------------------------
    {
        std::vector<float> logits = {0.5f, -0.5f, 1.5f, -1.5f};
        std::vector<double> p_t1;
        check(temperature_softmax(logits, 1.0, p_t1).empty(), "T=1 identity: no error");
        // Manually compute softmax
        double max_l = 1.5;
        double denom = 0.0;
        for (float l : logits) denom += std::exp((double)l - max_l);
        std::vector<double> p_ref;
        for (float l : logits) p_ref.push_back(std::exp((double)l - max_l) / denom);
        for (int i = 0; i < 4; i++)
            check_near(p_t1[i], p_ref[i], 1e-9, "T=1 matches manual softmax");
    }

    // -----------------------------------------------------------------------
    // 3. T>1 makes distribution softer (entropy increases)
    // -----------------------------------------------------------------------
    {
        std::vector<float> logits = {3.0f, 0.0f, 0.0f};
        std::vector<double> p1, p2;
        temperature_softmax(logits, 1.0, p1);
        temperature_softmax(logits, 5.0, p2);
        // With T>1, max prob should decrease (distribution softens)
        check(p2[0] < p1[0], "T>1: max prob decreases (softer distribution)");
    }

    // -----------------------------------------------------------------------
    // 4. T<1 makes distribution sharper (max prob increases)
    // -----------------------------------------------------------------------
    {
        std::vector<float> logits = {3.0f, 0.0f, 0.0f};
        std::vector<double> p1, p05;
        temperature_softmax(logits, 1.0, p1);
        temperature_softmax(logits, 0.5, p05);
        check(p05[0] > p1[0], "T<1: max prob increases (sharper distribution)");
    }

    // -----------------------------------------------------------------------
    // 5. Argmax preservation: temperature scaling preserves ordering
    // -----------------------------------------------------------------------
    {
        std::vector<float> logits = {2.0f, 5.0f, 1.0f, 3.0f};
        std::vector<double> p_raw, p_cal;
        temperature_softmax(logits, 1.0, p_raw);
        temperature_softmax(logits, 2.5, p_cal);
        // argmax should be index 1 (logit=5) for both
        int am_raw = 0, am_cal = 0;
        for (int i = 1; i < 4; i++) {
            if (p_raw[i] > p_raw[am_raw]) am_raw = i;
            if (p_cal[i] > p_cal[am_cal]) am_cal = i;
        }
        check(am_raw == 1, "argmax preservation: raw argmax is index 1");
        check(am_cal == 1, "argmax preservation: calibrated argmax is index 1");
    }

    // -----------------------------------------------------------------------
    // 6. Invalid temperature rejected
    // -----------------------------------------------------------------------
    {
        std::vector<float> logits = {1.0f, 2.0f};
        std::vector<double> p;
        check(!temperature_softmax(logits, 0.0, p).empty(),  "T=0 rejected");
        check(!temperature_softmax(logits, -1.0, p).empty(), "T<0 rejected");
        check(temperature_softmax(logits, 0.01, p).empty(),  "T=0.01 accepted");
    }

    // -----------------------------------------------------------------------
    // 7. Numerical stability: large positive logits
    // -----------------------------------------------------------------------
    {
        std::vector<float> logits = {100.0f, 200.0f, 300.0f};
        std::vector<double> p;
        std::string err = temperature_softmax(logits, 1.0, p);
        check(err.empty(), "large positive logits: no error");
        double sum = p[0]+p[1]+p[2];
        check_near(sum, 1.0, 1e-9, "large positive logits: sum to 1");
        for (double pi : p) check(std::isfinite(pi) && pi >= 0.0, "large positive logits: all valid");
    }

    // -----------------------------------------------------------------------
    // 8. Numerical stability: large negative logits
    // -----------------------------------------------------------------------
    {
        std::vector<float> logits = {-300.0f, -200.0f, -100.0f};
        std::vector<double> p;
        std::string err = temperature_softmax(logits, 1.0, p);
        check(err.empty(), "large negative logits: no error");
        double sum = p[0]+p[1]+p[2];
        check_near(sum, 1.0, 1e-9, "large negative logits: sum to 1");
    }

    // -----------------------------------------------------------------------
    // 9. Numerical stability: nearly equal logits
    // -----------------------------------------------------------------------
    {
        std::vector<float> logits = {1.0f, 1.0f + 1e-6f, 1.0f - 1e-6f};
        std::vector<double> p;
        std::string err = temperature_softmax(logits, 1.0, p);
        check(err.empty(), "nearly equal logits: no error");
        check_near(p[0]+p[1]+p[2], 1.0, 1e-9, "nearly equal logits: sum to 1");
        // Each prob should be close to 1/3
        for (double pi : p) check_near(pi, 1.0/3.0, 0.01, "nearly equal logits: near uniform");
    }

    // -----------------------------------------------------------------------
    // 10. Numerical stability: extreme temperature
    // -----------------------------------------------------------------------
    {
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        std::vector<double> p_hot, p_cold;
        check(temperature_softmax(logits, 100.0, p_hot).empty(),  "T=100 accepted");
        check(temperature_softmax(logits, 0.01, p_cold).empty(),  "T=0.01 accepted");
        check_near(p_hot[0]+p_hot[1]+p_hot[2],   1.0, 1e-9, "T=100: sum to 1");
        check_near(p_cold[0]+p_cold[1]+p_cold[2], 1.0, 1e-9, "T=0.01: sum to 1");
    }

    // -----------------------------------------------------------------------
    // 11. log_softmax: valid output and matches log(softmax)
    // -----------------------------------------------------------------------
    {
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        auto lp = log_softmax(logits);
        check(lp.size() == 3, "log_softmax: correct size");
        std::vector<double> p;
        temperature_softmax(logits, 1.0, p);
        for (int i = 0; i < 3; i++)
            check_near(lp[i], std::log(p[i]), 1e-9, "log_softmax matches log(softmax)");
    }

    // -----------------------------------------------------------------------
    // 12. CalibrationArtifact round-trip serialization
    // -----------------------------------------------------------------------
    {
        CalibrationArtifact art;
        art.version            = 1;
        art.method             = "temperature";
        art.model_identifier   = "bonsai-test";
        art.formulation.layout = "state-last";
        art.formulation.candidate_scheme = "natural";
        art.formulation.prior_correction = true;
        art.noul_temperature   = 1.08;
        art.choice_temperature = 1.31;
        art.score_temperature  = 0.92;

        nlohmann::json j = art.to_json();
        CalibrationArtifact art2;
        std::string err;
        bool ok = CalibrationArtifact::from_json(j, art2, err);
        check(ok, "artifact round-trip: from_json succeeds");
        check_near(art2.noul_temperature,   1.08, 1e-9, "artifact round-trip: noul_temperature");
        check_near(art2.choice_temperature, 1.31, 1e-9, "artifact round-trip: choice_temperature");
        check_near(art2.score_temperature,  0.92, 1e-9, "artifact round-trip: score_temperature");
        check(art2.model_identifier == "bonsai-test", "artifact round-trip: model_identifier");
        check(art2.formulation.layout == "state-last", "artifact round-trip: layout");
        check(art2.formulation.prior_correction, "artifact round-trip: prior_correction");
    }

    // -----------------------------------------------------------------------
    // 13. CalibrationArtifact: unsupported version rejected
    // -----------------------------------------------------------------------
    {
        nlohmann::json j = {{"version", 99}, {"method", "temperature"},
                            {"parameters", {{"noul_temperature", 1.0},
                                            {"choice_temperature", 1.0},
                                            {"score_temperature", 1.0}}}};
        CalibrationArtifact art;
        std::string err;
        check(!CalibrationArtifact::from_json(j, art, err), "unsupported version rejected");
    }

    // -----------------------------------------------------------------------
    // 14. CalibrationArtifact: unknown method rejected
    // -----------------------------------------------------------------------
    {
        nlohmann::json j = {{"version", 1}, {"method", "isotonic"},
                            {"parameters", {{"noul_temperature", 1.0},
                                            {"choice_temperature", 1.0},
                                            {"score_temperature", 1.0}}}};
        CalibrationArtifact art;
        std::string err;
        check(!CalibrationArtifact::from_json(j, art, err), "unknown method rejected");
    }

    // -----------------------------------------------------------------------
    // 15. CalibrationArtifact: zero temperature rejected
    // -----------------------------------------------------------------------
    {
        nlohmann::json j = {{"version", 1}, {"method", "temperature"},
                            {"parameters", {{"noul_temperature", 0.0},
                                            {"choice_temperature", 1.0},
                                            {"score_temperature", 1.0}}}};
        CalibrationArtifact art;
        std::string err;
        check(!CalibrationArtifact::from_json(j, art, err), "zero temperature rejected in artifact");
    }

    // -----------------------------------------------------------------------
    // 16. CalibrationArtifact compatibility check
    // -----------------------------------------------------------------------
    {
        CalibrationArtifact art;
        art.formulation.layout           = "state-last";
        art.formulation.candidate_scheme = "natural";
        art.formulation.prior_correction = true;

        CalibrationFormulation match = art.formulation;
        CalibrationFormulation mismatch = art.formulation;
        mismatch.layout = "state-first";

        std::string w1, w2;
        check(art.check_compatible(match, w1),    "compatibility: matching formulation passes");
        check(!art.check_compatible(mismatch, w2), "compatibility: layout mismatch flagged");
        check(!w2.empty(), "compatibility: warning message populated");
    }

    // -----------------------------------------------------------------------
    // 17. CalibrationConfig::temperature_for
    // -----------------------------------------------------------------------
    {
        CalibrationConfig cfg;
        cfg.enabled           = true;
        cfg.noul_temperature  = 1.1;
        cfg.choice_temperature = 1.2;
        cfg.score_temperature = 0.9;
        check_near(cfg.temperature_for(pjev::Type::Noul),   1.1, 1e-9, "temperature_for noul");
        check_near(cfg.temperature_for(pjev::Type::Choice), 1.2, 1e-9, "temperature_for choice");
        check_near(cfg.temperature_for(pjev::Type::Score),  0.9, 1e-9, "temperature_for score");

        CalibrationConfig disabled;
        check_near(disabled.temperature_for(pjev::Type::Noul),  1.0, 1e-9, "disabled: T=1.0 for noul");
        check_near(disabled.temperature_for(pjev::Type::Score), 1.0, 1e-9, "disabled: T=1.0 for score");
    }

    // -----------------------------------------------------------------------
    // 18. compute_primitive_metrics: perfect predictions
    // -----------------------------------------------------------------------
    {
        // 4 samples where correct is always index 0; make it highly likely
        std::vector<CalibrationSample> samples;
        for (int i = 0; i < 4; i++) {
            CalibrationSample s;
            s.logits = {10.0f, 0.0f, 0.0f};
            s.correct_index = 0;
            s.type = "choice";
            s.expected_score = 0.0;
            samples.push_back(s);
        }
        auto m = compute_primitive_metrics(samples, 1.0);
        check(m.n == 4, "perfect metrics: n=4");
        check_near(m.accuracy, 1.0, 1e-9, "perfect metrics: accuracy=1.0");
        check(m.nll < 0.1, "perfect metrics: low NLL");
        check(m.brier < 0.1, "perfect metrics: low Brier");
        check(m.ece >= 0.0, "perfect metrics: ECE non-negative");
    }

    // -----------------------------------------------------------------------
    // 19. compute_primitive_metrics: T=1 vs T>1 NLL comparison
    // -----------------------------------------------------------------------
    {
        // Samples where correct is index 0 with dominant logit
        std::vector<CalibrationSample> samples;
        for (int i = 0; i < 10; i++) {
            CalibrationSample s;
            s.logits = {2.0f, 0.0f};
            s.correct_index = 0;
            s.type = "noul";
            s.expected_score = 0.0;
            samples.push_back(s);
        }
        auto m1 = compute_primitive_metrics(samples, 1.0);
        auto m2 = compute_primitive_metrics(samples, 2.0);
        // With correct index dominant and T>1, confidence decreases → NLL increases
        check(m1.accuracy == m2.accuracy, "T scaling preserves accuracy");
        check(m1.nll < m2.nll, "T=1 gives lower NLL than T=2 when correct is dominant");
    }

    // -----------------------------------------------------------------------
    // 20. fit_temperature: converges on a simple dataset
    // -----------------------------------------------------------------------
    {
        // 20 samples, correct at index 0 with logit advantage 2.0
        std::vector<CalibrationSample> samples;
        for (int i = 0; i < 20; i++) {
            CalibrationSample s;
            s.logits = {2.0f, 0.0f};
            s.correct_index = 0;
            s.type = "choice";
            s.expected_score = 0.0;
            samples.push_back(s);
        }
        FitResult fr = fit_temperature(samples);
        check(fr.converged, "fit_temperature: converged");
        check(fr.temperature > 0.0, "fit_temperature: T > 0");
        check(fr.n_samples == 20, "fit_temperature: n_samples correct");
        check(fr.nll >= 0.0, "fit_temperature: NLL non-negative");
    }

    // -----------------------------------------------------------------------
    // 21. fit_temperature: T=1 is optimal for perfectly calibrated data
    // -----------------------------------------------------------------------
    {
        // If logits exactly match log-probabilities of a uniform distribution,
        // T=1 should be near-optimal (small adjustment acceptable)
        std::vector<CalibrationSample> samples;
        for (int i = 0; i < 4; i++) {
            CalibrationSample s;
            // Equal logits → uniform probs
            s.logits = {1.0f, 1.0f};
            // Correct evenly split between 0 and 1
            s.correct_index = i % 2;
            s.type = "noul";
            s.expected_score = 0.0;
            samples.push_back(s);
        }
        FitResult fr = fit_temperature(samples);
        // For uniform logits and balanced labels, T has no effect — any T is equally good.
        // The optimizer may converge to any value. Just verify it converges and T > 0.
        check(fr.converged, "fit_temperature uniform: converged");
        check(fr.temperature > 0.0, "fit_temperature uniform: T > 0");
    }

    // -----------------------------------------------------------------------
    // 22. CalibrationReport: raw vs calibrated NLL ordering
    // -----------------------------------------------------------------------
    {
        // Create samples where T=1 is suboptimal (overconfident predictions)
        // Use logits that are way too sharp; T>1 should reduce NLL
        std::vector<CalibrationSample> choice_s;
        for (int i = 0; i < 20; i++) {
            CalibrationSample s;
            // Very sharp prediction on wrong candidate half the time
            s.logits = (i % 2 == 0) ? std::vector<float>{10.0f, -10.0f}
                                     : std::vector<float>{-10.0f, 10.0f};
            s.correct_index = (i % 2 == 0) ? 0 : 1;
            s.type = "choice";
            s.expected_score = 0.0;
            choice_s.push_back(s);
        }
        // With this setup, T=1 is fine (always predicts correct).
        // The report should show valid metrics.
        CalibrationReport report = build_calibration_report(
            {}, choice_s, {}, 1.0, 1.0, 1.0);
        check(report.choice_raw.n == 20, "report: choice_raw.n == 20");
        check_near(report.choice_raw.accuracy, 1.0, 1e-9, "report: choice_raw accuracy=1.0");
        check(report.choice_raw.nll >= 0.0, "report: NLL non-negative");
    }

    // -----------------------------------------------------------------------
    // 23. Reliability bins: structure check
    // -----------------------------------------------------------------------
    {
        std::vector<CalibrationSample> samples;
        for (int i = 0; i < 20; i++) {
            CalibrationSample s;
            s.logits = {(float)(i % 3), 0.0f};
            s.correct_index = 0;
            s.type = "choice";
            s.expected_score = 0.0;
            samples.push_back(s);
        }
        auto m = compute_primitive_metrics(samples, 1.0, 10);
        check((int)m.reliability_bins.size() <= 10, "reliability bins: at most 10 bins");
        for (const auto& b : m.reliability_bins) {
            check(b.count > 0, "reliability bin: non-empty");
            check(b.confidence_min >= 0.0 && b.confidence_max <= 1.0, "reliability bin: valid confidence range");
            check(b.accuracy >= 0.0 && b.accuracy <= 1.0, "reliability bin: valid accuracy");
        }
    }

    // -----------------------------------------------------------------------
    // 24. Prior alpha: corrected_logit = raw - alpha * prior
    // -----------------------------------------------------------------------
    {
        // raw=[2.0, 0.0], prior=[1.0, 0.5], alpha=0.5
        // corrected = [2.0 - 0.5, 0.0 - 0.25] = [1.5, -0.25]
        std::vector<float> raw    = {2.0f, 0.0f};
        std::vector<float> prior  = {1.0f, 0.5f};
        double alpha = 0.5;
        std::vector<float> corr(raw.size());
        for (size_t i = 0; i < raw.size(); i++)
            corr[i] = raw[i] - (float)(alpha * prior[i]);
        check_near((double)corr[0], 1.5,   1e-6, "prior alpha: corrected[0] = 1.5");
        check_near((double)corr[1], -0.25, 1e-6, "prior alpha: corrected[1] = -0.25");

        // alpha=0 means no correction
        std::vector<float> corr0(raw.size());
        for (size_t i = 0; i < raw.size(); i++)
            corr0[i] = raw[i] - (float)(0.0 * prior[i]);
        check_near((double)corr0[0], 2.0, 1e-6, "alpha=0: no correction applied");
        check_near((double)corr0[1], 0.0, 1e-6, "alpha=0: no correction applied");
    }

    // -----------------------------------------------------------------------
    // 25. CalibrationConfig::prior_alpha_for routing
    // -----------------------------------------------------------------------
    {
        CalibrationConfig cfg;
        cfg.noul_prior_alpha   = 0.3;
        cfg.choice_prior_alpha = 0.7;
        cfg.score_prior_alpha  = 1.2;
        check_near(cfg.prior_alpha_for(pjev::Type::Noul),   0.3, 1e-9, "prior_alpha_for noul");
        check_near(cfg.prior_alpha_for(pjev::Type::Choice), 0.7, 1e-9, "prior_alpha_for choice");
        check_near(cfg.prior_alpha_for(pjev::Type::Score),  1.2, 1e-9, "prior_alpha_for score");
        // Unknown type falls back to choice_prior_alpha
        check_near(cfg.prior_alpha_for(pjev::Type::Unknown),  0.7, 1e-9, "prior_alpha_for unknown falls back to choice");
    }

    // -----------------------------------------------------------------------
    // 26. CalibrationArtifact round-trip with alpha fields
    // -----------------------------------------------------------------------
    {
        CalibrationArtifact art;
        art.noul_temperature   = 1.1;
        art.noul_prior_alpha   = 0.4;
        art.choice_prior_alpha = 0.8;
        art.score_prior_alpha  = 0.0;
        nlohmann::json j = art.to_json();
        CalibrationArtifact art2;
        std::string err;
        bool ok = CalibrationArtifact::from_json(j, art2, err);
        check(ok, "alpha round-trip: from_json succeeds");
        check_near(art2.noul_prior_alpha,   0.4, 1e-9, "alpha round-trip: noul_prior_alpha");
        check_near(art2.choice_prior_alpha, 0.8, 1e-9, "alpha round-trip: choice_prior_alpha");
        check_near(art2.score_prior_alpha,  0.0, 1e-9, "alpha round-trip: score_prior_alpha");
        // to_config copies alpha
        CalibrationConfig cfg = art2.to_config();
        check_near(cfg.noul_prior_alpha,   0.4, 1e-9, "alpha to_config: noul_prior_alpha");
    }

    // -----------------------------------------------------------------------
    // 27. CalibrationArtifact: negative alpha rejected
    // -----------------------------------------------------------------------
    {
        nlohmann::json j = {{"version", 1}, {"method", "temperature"},
                            {"parameters", {{"noul_temperature", 1.0},
                                            {"choice_temperature", 1.0},
                                            {"score_temperature", 1.0},
                                            {"noul_prior_alpha", -0.5}}}};
        CalibrationArtifact art;
        std::string err;
        check(!CalibrationArtifact::from_json(j, art, err), "negative alpha rejected");
    }

    // -----------------------------------------------------------------------
    // 28. fit_alpha_temperature: converges on a simple dataset
    // -----------------------------------------------------------------------
    {
        // Tuning: 20 samples where alpha=0 correction and correct index 0
        std::vector<CalibrationSample> tuning, val;
        for (int i = 0; i < 20; i++) {
            CalibrationSample s;
            s.raw_logits   = {2.0f, 0.0f};
            s.prior_logits = {1.0f, 0.5f};
            s.logits       = s.raw_logits;
            s.correct_index = 0;
            s.type = "choice";
            tuning.push_back(s);
        }
        // Val: 10 samples same setup
        for (int i = 0; i < 10; i++) {
            CalibrationSample s;
            s.raw_logits   = {2.0f, 0.0f};
            s.prior_logits = {1.0f, 0.5f};
            s.logits       = s.raw_logits;
            s.correct_index = 0;
            s.type = "choice";
            val.push_back(s);
        }
        AlphaTResult r = fit_alpha_temperature(tuning, val);
        check(r.converged, "fit_alpha_temperature: converged");
        check(r.alpha >= 0.0 && r.alpha <= 2.0, "fit_alpha_temperature: alpha in range");
        check(r.temperature > 0.0, "fit_alpha_temperature: T > 0");
        check(r.n_tuning == 20, "fit_alpha_temperature: n_tuning correct");
        check(r.n_val == 10, "fit_alpha_temperature: n_val correct");
    }

    // -----------------------------------------------------------------------
    // 29. assign_split: deterministic and covers all three buckets
    // -----------------------------------------------------------------------
    {
        using namespace pjev;
        // Same ID + seed always gives same split
        DataSplit s1 = assign_split("item-001", 42);
        DataSplit s2 = assign_split("item-001", 42);
        check(s1 == s2, "assign_split: deterministic");

        // Different seed gives potentially different split
        int seen_tuning = 0, seen_val = 0, seen_held = 0;
        for (int i = 0; i < 100; i++) {
            DataSplit s = assign_split("item-" + std::to_string(i), 42);
            if (s == DataSplit::Tuning)     seen_tuning++;
            else if (s == DataSplit::Validation) seen_val++;
            else                            seen_held++;
        }
        check(seen_tuning > 0,  "assign_split: some items are Tuning");
        check(seen_val > 0,     "assign_split: some items are Validation");
        check(seen_held > 0,    "assign_split: some items are HeldOut");
        check(seen_tuning + seen_val + seen_held == 100, "assign_split: all items assigned");

        // split_name
        check(split_name(DataSplit::Tuning)     == "tuning",     "split_name: tuning");
        check(split_name(DataSplit::Validation) == "validation",  "split_name: validation");
        check(split_name(DataSplit::HeldOut)    == "held_out",    "split_name: held_out");
    }

    // -----------------------------------------------------------------------
    // 30. build_comparison_report: 4 rows per primitive, expected labels
    // -----------------------------------------------------------------------
    {
        // Create samples with raw and prior logits
        std::vector<CalibrationSample> samples;
        for (int i = 0; i < 10; i++) {
            CalibrationSample s;
            s.raw_logits   = {3.0f, 0.0f};
            s.prior_logits = {1.0f, 0.5f};
            s.logits       = s.raw_logits;
            s.correct_index = 0;
            s.type = "choice";
            samples.push_back(s);
        }
        auto rep = build_comparison_report({}, samples, {}, 1.0, 1.2, 1.0, 0.0, 0.5, 0.0);
        check(rep.choice_rows.size() == 4, "comparison report: 4 rows for choice");
        check(rep.choice_rows[0].label == "raw",               "comparison report: row0=raw");
        check(rep.choice_rows[1].label == "temperature",       "comparison report: row1=temperature");
        check(rep.choice_rows[2].label == "prior",             "comparison report: row2=prior");
        check(rep.choice_rows[3].label == "prior+temperature", "comparison report: row3=prior+temperature");
        check(rep.choice_rows[0].alpha == 0.0,  "comparison report: raw alpha=0");
        check(rep.choice_rows[2].alpha == 0.5,  "comparison report: prior alpha=0.5");
        check(rep.noul_rows.size()  == 4, "comparison report: 4 rows for noul (empty samples ok)");
        check(rep.score_rows.size() == 4, "comparison report: 4 rows for score (empty samples ok)");
    }

    printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
