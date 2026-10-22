// Unit tests for production calibration (no model): loading (default and overridden paths, strict
// artifact validation, exact parameter values, formulation and model compatibility, the missing-file
// policy) and generation (NLL / Brier / ECE, deterministic temperature and prior-alpha fitting, boundary
// and degenerate cases, group partitioning, query construction, atomic output).

#include "calibration/calibration.h"
#include "calibration/calibration_fit.h"
#include "calibration/calibration_generator.h"
#include "calibration/calibration_metrics.h"
#include "distribution/release_config.h"
#include "routing/task_policy.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <string>

using namespace pjev;

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); g_failures++; } \
    else         { printf("pass: %s\n", msg); } \
} while (0)

// The supplied models/calibration.json, verbatim.
static const char* kSupplied = R"({
  "formulation": {
    "candidate_scheme": "natural",
    "layout": "auto",
    "prior_correction": true
  },
  "method": "temperature",
  "model": "",
  "parameters": {
    "choice_prior_alpha": 0.9999999999999999,
    "choice_temperature": 2.046567219106836,
    "noul_prior_alpha": 0.0,
    "noul_temperature": 99.99999952490862,
    "score_prior_alpha": 1.0999999999999999,
    "score_temperature": 11.960309669703673
  },
  "version": 1
})";

// The same parameters fitted for the production formulation.
static std::string letters_artifact(bool prior_correction = true)
{
    std::string s = kSupplied;
    s.replace(s.find("\"natural\""), 9, "\"letters\"");
    if (!prior_correction) s.replace(s.find("\"prior_correction\": true"), 24, "\"prior_correction\": false");
    return s;
}

static std::string write_file(const std::string& name, const std::string& content)
{
    std::ofstream(name, std::ios::binary) << content;
    return name;
}

static CalibrationFormulation production()
{
    CalibrationFormulation f;
    f.layout           = routing::kFormulationLayout;
    f.candidate_scheme = routing::kFormulationCandidateScheme;
    return f;
}

static void test_default_path()
{
    namespace fs = std::filesystem;
    RuntimePaths p = ReleaseConfig::resolve_runtime_paths(RuntimePaths{}, "");
    CHECK(p.calibration == "models/calibration.json", "path: default is models/calibration.json");

    const fs::path dir = fs::absolute("test_calib_exe_dir");
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::ofstream((dir / "calibration.json").string()) << "{}";
    p = ReleaseConfig::resolve_runtime_paths(RuntimePaths{}, dir.string());
    CHECK(p.calibration == (dir / "calibration.json").string(), "path: calibration.json next to the executable");

    std::ofstream((dir / "pjev.json").string()) << R"({"calibration": "custom-calib.json"})";
    p = ReleaseConfig::resolve_runtime_paths(RuntimePaths{}, dir.string());
    CHECK(p.calibration == (dir / "custom-calib.json").string(), "path: pjev.json overrides the default");

    RuntimePaths flags;
    flags.calibration = "/explicit/calib.json";
    p = ReleaseConfig::resolve_runtime_paths(flags, dir.string());
    CHECK(p.calibration == "/explicit/calib.json", "path: --calibration overrides pjev.json and the default");
    fs::remove_all(dir);
}

static void test_supplied_file()
{
    // The repository's default file, when present, loads as written and is reported incompatible.
    const std::string path = "models/calibration.json";
    if (!std::ifstream(path).good()) {
        printf("skip: %s not present\n", path.c_str());
        return;
    }
    CalibrationLoad c;
    std::string err;
    CHECK(load_calibration(path, false, production(), c, err), "supplied: parses (not a configuration error)");
    CHECK(c.state == CalibrationState::Incompatible, "supplied: natural-scheme artifact is incompatible with letters");
    CHECK(c.message.find("candidate_scheme mismatch (artifact=natural current=letters)") != std::string::npos,
          "supplied: message names the candidate scheme mismatch");
    CHECK(!c.config().enabled && !c.prior_correction(), "supplied: not applied (uncalibrated engine, no prior)");
}

static void test_exact_values()
{
    const std::string path = write_file("test_calib_letters.json", letters_artifact());
    CalibrationLoad c;
    std::string err;
    CHECK(load_calibration(path, true, production(), c, err) && c.state == CalibrationState::Enabled,
          "load: letters artifact enabled");
    const CalibrationConfig cfg = c.config();
    // Values exactly as stored (the literals parse to the same doubles).
    CHECK(cfg.enabled && cfg.method == "temperature", "load: temperature calibration enabled");
    CHECK(cfg.choice_temperature == 2.046567219106836 && cfg.choice_prior_alpha == 0.9999999999999999,
          "values: choice parameters exact");
    CHECK(cfg.noul_temperature == 99.99999952490862 && cfg.noul_prior_alpha == 0.0, "values: noul parameters exact");
    CHECK(cfg.score_temperature == 11.960309669703673 && cfg.score_prior_alpha == 1.0999999999999999,
          "values: score parameters exact (not rounded to 1.1)");
    CHECK(cfg.temperature_for(Type::Choice) == cfg.choice_temperature &&
          cfg.temperature_for(Type::Noul) == cfg.noul_temperature &&
          cfg.temperature_for(Type::Score) == cfg.score_temperature, "values: temperature per task kind");
    CHECK(cfg.prior_alpha_for(Type::Choice) == cfg.choice_prior_alpha &&
          cfg.prior_alpha_for(Type::Noul) == cfg.noul_prior_alpha &&
          cfg.prior_alpha_for(Type::Score) == cfg.score_prior_alpha, "values: prior alpha per task kind");
    CHECK(c.prior_correction(), "load: prior correction from the artifact");
    std::remove(path.c_str());

    const std::string no_prior = write_file("test_calib_noprior.json", letters_artifact(false));
    CHECK(load_calibration(no_prior, true, production(), c, err) && c.state == CalibrationState::Enabled &&
          !c.prior_correction(), "load: artifact without prior correction");
    std::remove(no_prior.c_str());
}

static void test_missing()
{
    CalibrationLoad c;
    std::string err;
    CHECK(load_calibration("does-not-exist/calibration.json", false, production(), c, err) &&
          c.state == CalibrationState::Missing, "missing default: not an error");
    CHECK(c.message.find("not found") != std::string::npos && c.message.find("uncalibrated") != std::string::npos,
          "missing default: warning says decisions are uncalibrated");
    CHECK(!c.config().enabled && !c.prior_correction(), "missing default: uncalibrated");
    CHECK(!load_calibration("does-not-exist/calibration.json", true, production(), c, err) &&
          err.find("not found") != std::string::npos, "missing explicit path: configuration error");
    CHECK(load_calibration("", true, production(), c, err) && c.state == CalibrationState::Disabled &&
          !c.config().enabled, "empty path: disabled");
}

static void test_invalid()
{
    struct Case { const char* what; std::string json; const char* expect; };
    auto with = [](const std::string& from, const std::string& to) {
        std::string s = letters_artifact();
        const size_t at = s.find(from);
        if (at == std::string::npos) return std::string("<pattern not found>");
        s.replace(at, from.size(), to);
        return s;
    };
    const Case cases[] = {
        {"malformed JSON", "{\"version\": 1,", "parse error"},
        {"unsupported version", with("\"version\": 1", "\"version\": 2"), "unsupported artifact version"},
        {"unsupported method", with("\"method\": \"temperature\"", "\"method\": \"platt\""), "unsupported calibration method"},
        {"missing formulation field", with("\"layout\": \"auto\",", ""), "formulation.layout"},
        {"non-bool prior_correction", with("\"prior_correction\": true", "\"prior_correction\": 1"), "prior_correction"},
        {"missing parameter", with("\"noul_prior_alpha\": 0.0,", ""), "noul_prior_alpha"},
        {"non-numeric parameter", with("\"score_temperature\": 11.960309669703673", "\"score_temperature\": \"12\""),
         "score_temperature"},
        // JSON has no inf / nan literal; an overflowing number is rejected by the parser.
        {"non-finite parameter", with("\"choice_temperature\": 2.046567219106836", "\"choice_temperature\": 1e999"),
         "number overflow"},
        {"zero temperature", with("\"noul_temperature\": 99.99999952490862", "\"noul_temperature\": 0"), "temperature"},
        {"negative temperature", with("\"noul_temperature\": 99.99999952490862", "\"noul_temperature\": -1"), "temperature"},
        {"negative prior alpha", with("\"choice_prior_alpha\": 0.9999999999999999", "\"choice_prior_alpha\": -0.5"),
         "prior_alpha"},
        {"missing parameters", "{\"version\":1,\"method\":\"temperature\",\"formulation\":{\"layout\":\"auto\","
                               "\"candidate_scheme\":\"letters\",\"prior_correction\":false}}", "parameters"},
    };
    for (const auto& c : cases) {
        const std::string path = write_file("test_calib_invalid.json", c.json);
        CalibrationLoad out;
        std::string err;
        const bool ok = load_calibration(path, false, production(), out, err);
        const std::string msg = std::string("invalid: ") + c.what + " -> configuration error";
        CHECK(!ok && err.find(c.expect) != std::string::npos && err.find(path) != std::string::npos, msg.c_str());
        std::remove(path.c_str());
    }
}

static void test_incompatible()
{
    auto load_text = [](const std::string& text, CalibrationLoad& c) {
        const std::string path = write_file("test_calib_incompat.json", text);
        std::string err;
        const bool ok = load_calibration(path, true, production(), c, err);
        std::remove(path.c_str());
        return ok;
    };
    CalibrationLoad c;
    CHECK(load_text(kSupplied, c) && c.state == CalibrationState::Incompatible &&
          c.message.find("candidate_scheme") != std::string::npos, "incompatible: natural vs letters detected");
    CHECK(!c.config().enabled, "incompatible: parameters not applied");
    std::string layout = letters_artifact();
    layout.replace(layout.find("\"layout\": \"auto\""), 16, "\"layout\": \"state_first\"");
    CHECK(load_text(layout, c) && c.state == CalibrationState::Incompatible &&
          c.message.find("layout mismatch") != std::string::npos, "incompatible: layout mismatch detected");
    CHECK(std::string(to_string(CalibrationState::Incompatible)) == "incompatible" &&
          std::string(to_string(CalibrationState::Enabled)) == "enabled" &&
          std::string(to_string(CalibrationState::Missing)) == "missing" &&
          std::string(to_string(CalibrationState::Disabled)) == "disabled", "state names");
}

static const std::string kSha  = std::string(64, 'a');
static const std::string kSha2 = std::string(64, 'b');

static void test_model_identity()
{
    auto load = [](const std::string& model) {
        nlohmann::json j = nlohmann::json::parse(letters_artifact());
        j["model"] = model;
        const std::string path = write_file("test_calib_model.json", j.dump());
        CalibrationLoad c;
        std::string err;
        load_calibration(path, true, production(), c, err);
        std::remove(path.c_str());
        return c;
    };
    CalibrationLoad c = load(kSha);
    CHECK(check_calibration_model(c, kSha) && c.state == CalibrationState::Enabled, "model: matching SHA-256 verified");
    c = load(std::string(64, 'A'));
    CHECK(check_calibration_model(c, kSha) && c.state == CalibrationState::Enabled, "model: SHA-256 compared case-insensitively");
    c = load(kSha2);
    CHECK(!check_calibration_model(c, kSha) && c.state == CalibrationState::Incompatible &&
          c.message.find("another inference model") != std::string::npos && !c.config().enabled,
          "model: another model's calibration is incompatible (not applied)");
    c = load("bonsai2_2b");
    CHECK(!check_calibration_model(c, kSha) && c.state == CalibrationState::Incompatible &&
          c.message.find("cannot be verified") != std::string::npos, "model: unverifiable identity is incompatible");
    c = load("");
    CHECK(!check_calibration_model(c, kSha) && c.state == CalibrationState::Enabled,
          "model: no recorded identity (version-1 file) is not checked");
}

// ---------------------------------------------------------------------------
// Metrics and fitting
// ---------------------------------------------------------------------------

static CalibrationSample sample(std::vector<float> logits, int correct, std::vector<float> prior = {})
{
    CalibrationSample s;
    s.raw_logits    = logits;
    s.logits        = logits;
    s.prior_logits  = prior.empty() ? std::vector<float>(logits.size(), 0.0f) : prior;
    s.correct_index = correct;
    s.type          = "choice";
    return s;
}

static bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

static void test_metrics()
{
    // NLL of softmax([0, ln 3]) = [0.25, 0.75].
    CHECK(near(sample_nll({0.0f, (float)std::log(3.0)}, 1, 1.0), -std::log(0.75), 1e-7), "nll: -log p(correct)");
    CHECK(near(sample_nll({0.0f, (float)std::log(3.0)}, 0, 1.0), -std::log(0.25), 1e-7), "nll: wrong answer");
    // Log-sum-exp keeps extreme logits finite (a naive exp overflows; a probability clamp would cap it).
    const double big = sample_nll({1e4f, -1e4f}, 1, 1.0);
    CHECK(std::isfinite(big) && near(big, 2e4, 1e-3), "nll: stable for extreme logits (log-sum-exp)");
    CHECK(near(sample_nll({1e4f, -1e4f}, 0, 1.0), 0.0), "nll: confident correct answer");
    CHECK(near(sample_nll({2.0f, 0.0f}, 1, 2.0), std::log(1.0 + std::exp(1.0)), 1e-9), "nll: temperature divides the logits");
    CHECK(std::isnan(sample_nll({1.0f, NAN}, 0, 1.0)) && std::isnan(sample_nll({1.0f, 2.0f}, 2, 1.0)) &&
          std::isnan(sample_nll({1.0f, 2.0f}, 0, 0.0)), "nll: invalid samples are NaN");

    // Two samples: p = [0.25, 0.75] (correct 1) and [0.25, 0.75] (correct 0).
    std::vector<CalibrationSample> s = {sample({0.0f, (float)std::log(3.0)}, 1), sample({0.0f, (float)std::log(3.0)}, 0)};
    PrimitiveCalibrationMetrics m = compute_primitive_metrics(s, 1.0, 10);
    CHECK(m.n == 2 && near(m.accuracy, 0.5), "metrics: accuracy from the exact candidate index");
    CHECK(near(m.nll, (-std::log(0.75) - std::log(0.25)) / 2, 1e-7), "metrics: mean NLL");
    // Brier: (0.25^2 + 0.25^2) and (0.75^2 + 0.75^2), averaged.
    CHECK(near(m.brier, (0.125 + 1.125) / 2, 1e-7), "metrics: multi-class Brier score");
    // Both confidences 0.75 in one bin with accuracy 0.5: ECE = |0.5 - 0.75|.
    CHECK(near(m.ece, 0.25, 1e-7) && m.reliability_bins.size() == 1 && m.reliability_bins[0].count == 2,
          "metrics: ECE and reliability bins");
    CHECK(near(m.mean_max_prob, 0.75, 1e-7) && near(m.mean_conf_correct, 0.75, 1e-7) &&
          near(m.mean_conf_incorrect, 0.75, 1e-7), "metrics: mean confidence (all / correct / incorrect)");
    m = compute_primitive_metrics({sample({0.0f, 5.0f}, 1)}, 1.0);
    CHECK(m.mean_conf_incorrect == -1.0 && m.mean_conf_correct > 0.99, "metrics: no incorrect predictions -> -1");

    // Prior correction in the decision engine's float arithmetic.
    const auto c = apply_prior_correction({sample({1.0f, 2.0f}, 0, {0.5f, -1.0f})}, 0.9999999999999999);
    CHECK(c[0].logits[0] == 1.0f - (float)(0.9999999999999999 * 0.5) && c[0].logits[1] == 2.0f - (float)(0.9999999999999999 * -1.0),
          "metrics: prior correction = raw - (float)(alpha * prior)");
}

// Samples drawn from softmax(z / T0) with a known temperature (fixed seed).
static std::vector<CalibrationSample> synthetic(double T0, int n, unsigned seed, double prior_bias = 0.0)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 2.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    std::vector<CalibrationSample> out;
    const std::vector<float> prior = {1.0f, 0.0f, -1.0f};
    for (int i = 0; i < n; ++i) {
        std::vector<float> z(3);
        for (auto& v : z) v = (float)nd(rng);
        // True logits z; the model reports z + prior_bias * prior (a candidate-position bias).
        std::vector<double> p(3);
        double sum = 0.0;
        for (int k = 0; k < 3; ++k) sum += (p[k] = std::exp(z[k] / T0));
        double u = ud(rng) * sum;
        int y = 0;
        while (y < 2 && u > p[y]) u -= p[y++];
        std::vector<float> reported(3);
        for (int k = 0; k < 3; ++k) reported[k] = z[k] + (float)(prior_bias * prior[k]);
        out.push_back(sample(reported, y, prior));
    }
    return out;
}

static void test_fit()
{
    const auto data = synthetic(2.5, 4000, 7);
    const FitResult a = fit_temperature(data);
    const FitResult b = fit_temperature(data);
    CHECK(a.converged && a.temperature == b.temperature && a.nll == b.nll, "fit: deterministic temperature");
    CHECK(std::isfinite(a.temperature) && a.temperature > 0.0, "fit: finite positive temperature");
    CHECK(std::fabs(a.temperature - 2.5) < 0.25, "fit: recovers the generating temperature");
    CHECK(a.nll <= mean_nll(data, 1.0) && a.nll <= mean_nll(data, a.temperature * 1.1) &&
          a.nll <= mean_nll(data, a.temperature / 1.1), "fit: temperature minimizes the NLL");
    CHECK(!a.at_lower_bound && !a.at_upper_bound, "fit: interior solution not flagged");

    // Perfectly separable data: NLL keeps falling as T -> 0, so the fit stops at the lower bound.
    std::vector<CalibrationSample> sep = {sample({5.0f, 0.0f}, 0), sample({0.0f, 5.0f}, 1)};
    const FitResult lo = fit_temperature(sep);
    CHECK(lo.converged && lo.at_lower_bound && near(lo.temperature, kFitTemperatureMin, 1e-3),
          "fit: separable data -> temperature at the lower bound (flagged)");
    // Uninformative logits whose argmax is always wrong: T -> infinity, stops at the upper bound.
    std::vector<CalibrationSample> anti = {sample({5.0f, 0.0f}, 1), sample({0.0f, 5.0f}, 0)};
    const FitResult hi = fit_temperature(anti);
    CHECK(hi.converged && hi.at_upper_bound && near(hi.temperature, kFitTemperatureMax, 0.1),
          "fit: anti-correlated data -> temperature at the upper bound (flagged)");

    // Joint fit: a candidate-position bias of +1 x prior is removed by alpha ~ 1.
    const auto biased = synthetic(1.5, 4000, 11, 1.0);
    const AlphaTResult j1 = fit_alpha_temperature(biased);
    const AlphaTResult j2 = fit_alpha_temperature(biased);
    CHECK(j1.converged && j1.alpha == j2.alpha && j1.temperature == j2.temperature && j1.nll == j2.nll,
          "fit: deterministic joint fit");
    CHECK(std::fabs(j1.alpha - 1.0) < 0.15 && std::fabs(j1.temperature - 1.5) < 0.2,
          "fit: recovers prior alpha and temperature");
    CHECK(j1.alpha >= kFitAlphaMin && j1.alpha <= kFitAlphaMax, "fit: prior alpha within [0, 2]");
    CHECK(j1.alpha == 0.9 || j1.alpha == 1.0 || j1.alpha == 1.1, "fit: prior alpha is a decimal grid point");
    bool decimal = true;
    for (int b = 0; b <= 20; ++b) {
        const AlphaTResult r = fit_alpha_temperature(synthetic(1.0, 400, 3, b / 10.0), kFitAlphaMin, kFitAlphaMax, kFitAlphaStep);
        decimal &= r.alpha == std::stod(std::to_string(r.alpha));  // e.g. exactly 1.7, not 1.7000000000000002
    }
    CHECK(decimal, "fit: every fitted prior alpha is the double nearest its decimal grid value");
    CHECK(j1.nll <= mean_nll(apply_prior_correction(biased, 0.0), 1.0), "fit: joint fit does not increase NLL");

    // A bias against the prior cannot be corrected by alpha >= 0: alpha = 0, reported as a boundary.
    const AlphaTResult j0 = fit_alpha_temperature(synthetic(1.0, 3000, 5, -1.0));
    CHECK(j0.converged && j0.alpha_at_lower_bound && j0.alpha == 0.0, "fit: alpha 0 reported at the lower bound");
    // A bias beyond the grid stops at alpha_max (flagged).
    const AlphaTResult j3 = fit_alpha_temperature(synthetic(1.0, 3000, 9, 4.0));
    CHECK(j3.converged && j3.alpha_at_upper_bound && j3.alpha == kFitAlphaMax, "fit: alpha at the upper bound flagged");

    CHECK(!fit_temperature({}).converged && !fit_alpha_temperature({}).converged, "fit: no samples -> not converged");
    CHECK(!std::isfinite(mean_nll({sample({1.0f, NAN}, 0)}, 1.0)), "fit: non-finite logits never score");
}

// ---------------------------------------------------------------------------
// Partition and queries
// ---------------------------------------------------------------------------

using namespace pjev::calibration;
using retrieval::ExemplarRow;
using retrieval::LoadResult;

static std::vector<ExemplarRow> synthetic_rows(int groups)
{
    std::vector<ExemplarRow> rows;
    int64_t id = 1000;
    for (int g = 0; g < groups; ++g) {
        for (const char* k : {"choice", "choice", "noul", "score"}) {
            ExemplarRow r;
            r.id = id++;
            r.group_id = 50000 + g;
            r.kind = k;
            r.one_hot = true;
            r.number_of_levels = std::string(k) == "score" ? 5 : 0;
            rows.push_back(r);
        }
    }
    return rows;
}

static void test_partition()
{
    const std::vector<ExemplarRow> rows = synthetic_rows(400);
    std::set<int64_t> not_training, invalid;
    for (const auto& r : rows) {
        if (r.id % 7 == 0) not_training.insert(r.id);
        else if (r.id % 11 == 0) invalid.insert(r.id);
    }
    auto training = [&](int64_t id) { return not_training.count(id) == 0; };
    std::set<int64_t> validated;
    RowValidator validate = [&](const ExemplarRow& r, std::string& reason) {
        validated.insert(r.id);
        if (invalid.count(r.id)) { reason = "ambiguous_answer"; return LoadResult::Invalid; }
        return LoadResult::Ok;
    };
    GeneratorOptions o;
    o.max_samples = 120;
    o.validation_ratio = 0.25;
    Partition a, b;
    std::string err;
    CHECK(partition_rows(rows, training, validate, o, a, err), "partition: succeeds");
    CHECK(partition_rows(rows, training, validate, o, b, err), "partition: repeat");
    bool same = a.queries.size() == b.queries.size() && a.held_out == b.held_out;
    for (size_t i = 0; same && i < a.queries.size(); ++i)
        same = a.queries[i].id == b.queries[i].id && a.queries[i].split == b.queries[i].split;
    CHECK(same, "partition: deterministic for the same seed");

    std::map<int64_t, int64_t> group_of;
    std::map<int64_t, std::set<int64_t>> rows_of_group;
    for (const auto& r : rows) { group_of[r.id] = r.group_id; rows_of_group[r.group_id].insert(r.id); }
    std::map<int64_t, std::set<Split>> splits;
    std::map<std::pair<int64_t, Type>, int> per_group_kind;
    bool training_only = true, valid_only = true, held = true;
    for (const auto& q : a.queries) {
        splits[q.group_id].insert(q.split);
        per_group_kind[{q.group_id, q.type}]++;
        training_only &= not_training.count(q.id) == 0;
        valid_only &= invalid.count(q.id) == 0;
        for (int64_t id : rows_of_group[q.group_id])
            if (!not_training.count(id)) held &= a.held_out.count(id) == 1;
    }
    bool one_split = true, one_per_kind = true;
    for (const auto& kv : splits) one_split &= kv.second.size() == 1;
    for (const auto& kv : per_group_kind) one_per_kind &= kv.second == 1;
    CHECK(one_split, "partition: a group belongs wholly to fitting or validation");
    CHECK(one_per_kind, "partition: one query per task kind per group");
    CHECK(held, "partition: every training row of a query group is held out of retrieval");
    bool held_only_selected = true;
    for (int64_t id : a.held_out) held_only_selected &= splits.count(group_of[id]) == 1 && !not_training.count(id);
    CHECK(held_only_selected, "partition: only the selected groups' training rows are held out");
    CHECK(training_only, "partition: validation / test rows (not in the allowlist) are never queries");
    CHECK(valid_only, "partition: invalid rows are never queries");
    CHECK(a.rows_not_training == not_training.size(), "partition: rows outside the training split counted");

    for (Type t : {Type::Choice, Type::Noul, Type::Score}) {
        const KindCounts& k = a.kinds.at(t);
        CHECK(k.selected_fit + k.selected_validation == 120, (std::string("partition: ") + to_string(t) + " quota filled").c_str());
        CHECK(k.selected_validation > 10 && k.selected_validation < 60,
              (std::string("partition: ") + to_string(t) + " validation share near the ratio").c_str());
    }
    size_t rejected = 0;
    for (const auto& kv : a.kinds) rejected += kv.second.rejected.count("ambiguous_answer") ? kv.second.rejected.at("ambiguous_answer") : 0;
    size_t examined_invalid = 0;
    for (int64_t id : validated) examined_invalid += invalid.count(id);
    CHECK(rejected == examined_invalid && rejected > 0, "partition: rejected rows recorded with their reason");

    GeneratorOptions o2 = o;
    o2.seed = 2;
    Partition c;
    partition_rows(rows, training, validate, o2, c, err);
    CHECK(c.held_out != a.held_out, "partition: another seed selects other groups");

    // Insufficient data: never weakened, reported per kind.
    CHECK(check_sufficient(a, false, err), "partition: enough queries for the minimums");
    GeneratorOptions small = o;
    small.max_samples = 40;
    Partition d;
    partition_rows(rows, training, validate, small, d, err);
    CHECK(!check_sufficient(d, false, err) && err.find("choice") != std::string::npos &&
          err.find("at least 50 fitting") != std::string::npos, "partition: too few queries -> insufficient data");

    // A database failure stops partitioning.
    RowValidator broken = [](const ExemplarRow&, std::string& reason) { reason = "disk I/O error"; return LoadResult::Failed; };
    CHECK(!partition_rows(rows, training, broken, o, d, err) && err == "disk I/O error", "partition: database failure propagates");

    std::string why;
    GeneratorOptions bad;
    bad.validation_ratio = 1.0;
    CHECK(!bad.validate(why), "options: validation ratio must be < 1");
    bad = GeneratorOptions{};
    bad.max_samples = 0;
    CHECK(!bad.validate(why), "options: max samples must be >= 1");
}

static retrieval::ExemplarPayload payload(const char* kind, std::vector<std::string> options, int answer)
{
    retrieval::ExemplarPayload p;
    p.id = 1;
    p.kind = kind;
    p.question = "Which one?";
    p.state = nlohmann::ordered_json{{"x", 1}};
    p.options = options;
    p.answer_index = answer;
    p.number_of_levels = std::string(kind) == "score" ? (int)options.size() : 0;
    return p;
}

static void test_make_query()
{
    QueryRef q;
    q.type = Type::Choice;
    q.domain = "d";
    q.family = "f";
    DecisionInput in;
    int correct = -1;
    std::string reason;
    CHECK(make_query(payload("choice", {"red", "green", "blue"}, 2), q, in, correct, reason) && correct == 2 &&
          in.options.size() == 3 && in.options[2].first == "blue" && in.options[2].second == "blue" &&
          in.state == "{\"x\":1}" && in.domain == "d" && in.family == "f", "query: choice options as keys, answer index");
    CHECK(!make_query(payload("choice", {"a", "a", "b"}, 0), q, in, correct, reason) && reason == "ambiguous_answer",
          "query: repeated choice options -> ambiguous answer rejected");
    CHECK(!make_query(payload("choice", {"a", "b"}, -1), q, in, correct, reason) && reason == "no_one_hot_answer",
          "query: no one-hot answer rejected");
    auto empty_q = payload("choice", {"a", "b"}, 0);
    empty_q.question.clear();
    CHECK(!make_query(empty_q, q, in, correct, reason) && reason == "missing_question", "query: missing question rejected");
    CHECK(!make_query(payload("choice", {"a"}, 0), q, in, correct, reason) && reason == "invalid_option_count",
          "query: a single option rejected");
    CHECK(!make_query(payload("noul", {"no", "yes"}, 0), q, in, correct, reason) && reason == "kind_mismatch",
          "query: kind must match the row");

    q.type = Type::Score;
    CHECK(make_query(payload("score", {"low", "mid", "high"}, 1), q, in, correct, reason) && correct == 1 &&
          in.options[0].first == "0" && in.options[2].second == "high", "query: score levels keyed 0..N-1, lowest first");
    auto levels = payload("score", {"low", "mid", "high"}, 1);
    levels.number_of_levels = 5;
    CHECK(!make_query(levels, q, in, correct, reason) && reason == "level_count_mismatch", "query: level count mismatch rejected");

    q.type = Type::Noul;
    CHECK(make_query(payload("noul", {"yes", "no"}, 0), q, in, correct, reason) && correct == 1 &&
          in.options[0].first == "false" && in.options[1].first == "true", "query: noul (false, true), 'yes' -> true");
    CHECK(make_query(payload("noul", {"no", "yes"}, 0), q, in, correct, reason) && correct == 0, "query: noul 'no' -> false");
    CHECK(!make_query(payload("noul", {"no", "maybe"}, 0), q, in, correct, reason) && reason == "invalid_noul_options",
          "query: noul options must be no / yes");
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

static CalibrationArtifact fitted()
{
    CalibrationArtifact a;
    a.model_identifier   = kSha;
    a.formulation        = calibration::production_formulation();
    a.choice_temperature = 1.7;
    a.choice_prior_alpha = 0.3;
    a.noul_temperature   = 2.2;
    a.noul_prior_alpha   = 0.0;
    a.score_temperature  = 3.1;
    a.score_prior_alpha  = 1.2;
    return a;
}

static std::string read_text(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static bool no_temp_files(const std::string& dir)
{
    for (const auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".tmp") return false;
    return true;
}

static void test_output()
{
    namespace fs = std::filesystem;
    const std::string dir = "test_calib_out";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string dest = dir + "/calibration.json";
    const std::string old = letters_artifact();
    write_file(dest, old);
    std::string err;

    const CalibrationArtifact a = fitted();
    CHECK(write_calibration_file(dest, a.to_json().dump(2), kSha, &a, err), "output: valid calibration written");
    CalibrationLoad c;
    CHECK(load_calibration(dest, true, production(), c, err) && c.state == CalibrationState::Enabled &&
          check_calibration_model(c, kSha) && c.artifact.score_prior_alpha == 1.2, "output: replaced file loads for the model");
    CHECK(no_temp_files(dir), "output: no temporary file left");

    const std::string before = read_text(dest);
    CHECK(!write_calibration_file(dest, "{\"version\": 1, \"method\": \"temperature\"", kSha, nullptr, err) &&
          read_text(dest) == before && no_temp_files(dir), "output: malformed output never replaces the existing file");
    nlohmann::json bad = a.to_json();
    bad["parameters"]["noul_temperature"] = -1.0;
    CHECK(!write_calibration_file(dest, bad.dump(), kSha, nullptr, err) && read_text(dest) == before,
          "output: invalid parameters never replace the existing file");
    CHECK(!write_calibration_file(dest, a.to_json().dump(), kSha2, &a, err) && read_text(dest) == before &&
          err.find("another inference model") != std::string::npos, "output: calibration for another model rejected");
    nlohmann::json natural = a.to_json();
    natural["formulation"]["candidate_scheme"] = "natural";
    CHECK(!write_calibration_file(dest, natural.dump(), kSha, nullptr, err) && read_text(dest) == before,
          "output: another formulation rejected");
    CalibrationArtifact other = a;
    other.choice_temperature = 1.75;
    CHECK(!write_calibration_file(dest, other.to_json().dump(), kSha, &a, err) && read_text(dest) == before,
          "output: content that differs from the fitted parameters rejected");
    CHECK(no_temp_files(dir), "output: failed writes leave no temporary file");

    CHECK(fs::path(report_path_for("models/calibration.json", true)) == fs::path("models/calibration.report.json") &&
          fs::path(report_path_for("models/calibration.json", false)) == fs::path("models/calibration.failed-report.json"),
          "output: report file names");
    fs::remove_all(dir);
}

int main()
{
    test_default_path();
    test_supplied_file();
    test_exact_values();
    test_missing();
    test_invalid();
    test_incompatible();
    test_model_identity();
    test_metrics();
    test_fit();
    test_partition();
    test_make_query();
    test_output();

    if (g_failures == 0) {
        printf("\nPASS (0 failure(s))\n");
        return 0;
    }
    fprintf(stderr, "\nFAIL (%d failure(s))\n", g_failures);
    return 1;
}
