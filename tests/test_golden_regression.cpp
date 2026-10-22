// Golden regression (integration): the release decision path must reproduce the frozen R4 policy's
// recorded predictions on tests/data/golden_r4.jsonl. Needs the runtime resources; exits with 77
// (reported as skipped) when they are absent.
//
// Compared per case: strategy, exemplars (demonstrations in prompt order / kNN neighbours), prompt
// token count, candidate logits (exact, as in the R4 reference checks), kNN vote shares, and the
// prediction. Every case is also sent through the HTTP API handler.
//
// Two validation targets (R5):
//   uncalibrated compatibility - the run above, with calibration disabled, reproduces R4 exactly;
//   calibrated behaviour       - the default calibration file loads automatically, and with a compatible
//                                (letters) artifact every LLM output is the existing calibration
//                                transformation of the uncalibrated logits while kNN-5 is unchanged.
//                                Calibrated accuracy is not measured here (and is not R4's).
//   model identity (R6)        - that artifact, recorded for this model's SHA-256, is enabled; recorded
//                                for another model it is incompatible and decisions stay uncalibrated.
//
// Environment (optional): PJEV_GOLDEN_DATA, PJEV_MODEL, PJEV_EMBEDDING_MODEL, PJEV_EXEMPLAR_DB,
// PJEV_EXEMPLAR_ALLOWLIST override the default paths (relative to the repository root).

#include "calibration/calibration.h"
#include "decision/decision_service.h"
#include "retrieval/exemplar_allowlist.h"
#include "routing/router_runtime.h"
#include "routing/task_policy.h"
#include "server/jev_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace pjev;
using json = nlohmann::ordered_json;

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", std::string(msg).c_str()); g_failures++; } \
} while (0)

static std::string env_or(const char* name, const char* fallback)
{
    const char* v = std::getenv(name);
    return v && *v ? v : fallback;
}

static DecisionInput to_input(const json& in)
{
    DecisionInput d;
    d.type     = to_type(in["type"].get<std::string>());
    d.state    = in["state"].get<std::string>();
    d.question = in["question"].get<std::string>();
    d.domain   = in.value("domain", "");
    d.family   = in.value("family", "");
    for (const auto& o : in["options"]) d.options.push_back({o[0].get<std::string>(), o[1].get<std::string>()});
    return d;
}

static std::string api_request(const json& in)
{
    json q;
    q["type"] = in["type"];
    q["instructions"] = in["question"];
    q["domain"] = in["domain"];
    q["family"] = in["family"];
    const std::string type = in["type"].get<std::string>();
    if (type == "noul") {
        q["criteria"] = json{{"false", in["options"][0][1]}, {"true", in["options"][1][1]}};
    } else if (type == "choice") {
        q["criteria"] = json::object();
        for (const auto& o : in["options"]) q["criteria"][o[0].get<std::string>()] = o[1];
    } else {
        q["criteria"] = json::array();
        for (const auto& o : in["options"]) q["criteria"].push_back(o[1]);
    }
    return json{{"state", in["state"]}, {"questions", {{"q", q}}}}.dump();
}

static std::vector<std::string> hex_ids(const std::vector<int64_t>& ids)
{
    std::vector<std::string> out;
    for (int64_t id : ids) out.push_back(retrieval::format_exemplar_id(id));
    return out;
}

int main(int argc, char** argv)
{
    bool record_mode = false;
    std::string record_path;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--record" && i + 1 < argc) {
            record_mode = true;
            record_path = argv[++i];
        }
    }
    const std::string data_path = env_or("PJEV_GOLDEN_DATA", "tests/data/golden_r7.jsonl");
    RuntimeOptions opts;
    opts.paths.model              = env_or("PJEV_MODEL", "models/bonsai2_2b.gguf");
    opts.paths.embedding_model    = env_or("PJEV_EMBEDDING_MODEL", "models/nomic-embed-text-v1.5.gguf");
    opts.paths.exemplar_db        = env_or("PJEV_EXEMPLAR_DB", "models/examplar.db");
    opts.paths.exemplar_allowlist = env_or("PJEV_EXEMPLAR_ALLOWLIST", "models/examplar-allowlist.txt");
    opts.n_ctx = 4096;  // the R4 reference configuration
    opts.paths.calibration.clear();  // uncalibrated: the R4 reference

    for (const std::string* p : std::initializer_list<const std::string*>{&data_path, &opts.paths.model, &opts.paths.embedding_model, &opts.paths.exemplar_db,
                                 &opts.paths.exemplar_allowlist}) {
        if (!std::filesystem::is_regular_file(*p) || is_lfs_pointer(*p)) {
            printf("SKIP: %s not found or a Git LFS pointer (golden regression needs the runtime resources)\n", p->c_str());
            return 77;
        }
    }

    std::vector<json> cases;
    {
        std::ifstream f(data_path, std::ios::binary);
        std::string line;
        while (std::getline(f, line))
            if (!line.empty()) cases.push_back(json::parse(line));
    }
    CHECK(cases.size() == 24, "golden data has 24 cases");

    // Resource loading
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<RouterRuntime> rt = RouterRuntime::load(opts, err);
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!rt) {
        fprintf(stderr, "FAIL: runtime load: %s\n", err.c_str());
        return 1;
    }
    printf("runtime loaded in %.0f ms: model %s (sha256 %.16s...), embedding fingerprint %s, %lld exemplar rows, %zu eligible\n",
           load_ms, rt->model_identity().name.c_str(), rt->model_identity().sha256.c_str(),
           rt->embedding_fingerprint().c_str(), (long long)rt->exemplar_count(), rt->eligible_exemplars());
    CHECK(rt->embedding_fingerprint() == "3e24342164b3d949", "embedding fingerprint matches the R4 reference");
    CHECK(rt->model_identity().sha256 == "36061ad406df8a4903a6db005be350c50fabfaf717f9ef0acf4b76adcd9929fa",
          "inference model matches the R4 reference");
    CHECK(rt->eligible_exemplars() == 111596, "eligible exemplar count matches R4 (111596)");
    CHECK(rt->calibration().state == CalibrationState::Disabled, "uncalibrated run: calibration disabled");

    DecisionService& service = rt->service();
    JevApiHandler api(service, "golden");
    std::map<std::string, std::vector<double>> ms_by_kind;
    int agree = 0, logits_exact = 0, logit_cases = 0;
    double max_logit_diff = 0.0;
    std::vector<DecisionOutput> uncalibrated;
    std::vector<json> updated_cases;

    for (const auto& c : cases) {
        const std::string name = c["case"].get<std::string>();
        const json& exp = c["expected"];
        const DecisionInput in = to_input(c["input"]);

        const auto ts = std::chrono::steady_clock::now();
        const DecisionOutput out = service.decide(in);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
        ms_by_kind[to_string(in.type)].push_back(ms);
        uncalibrated.push_back(out);

        if (!out.ok) {
            fprintf(stderr, "FAIL: %s: decision failed: %s %s\n", name.c_str(), out.error_code.c_str(), out.error.c_str());
            g_failures++;
            continue;
        }
        const auto& r = *out.routing;
        const std::string want_strategy = exp["strategy"].get<std::string>();
        bool ok = true;
        auto expect = [&](bool cond, const std::string& what) {
            if (!cond) { fprintf(stderr, "FAIL: %s: %s\n", name.c_str(), what.c_str()); g_failures++; ok = false; }
        };
        expect(r.strategy == want_strategy, "strategy " + r.strategy + " != " + want_strategy);
        expect(!r.fallback, "unexpected fallback (" + r.strategy + ": " + r.fallback_reason + ")");

        if (exp.contains("logits")) {
            expect(hex_ids(r.demonstration_ids) == exp["demonstration_ids"].get<std::vector<std::string>>(),
                   "demonstration exemplars differ");
            expect(out.prompt_token_count == exp["prompt_tokens"].get<int>(),
                   "prompt tokens " + std::to_string(out.prompt_token_count) + " != " + std::to_string(exp["prompt_tokens"].get<int>()));
            const auto want = exp["logits"].get<std::vector<float>>();
            logit_cases++;
            if (out.raw_logits.size() != want.size()) {
                expect(false, "logit count differs");
            } else {
                double diff = 0.0;
                for (size_t i = 0; i < want.size(); ++i) diff = std::max(diff, (double)std::fabs(out.raw_logits[i] - want[i]));
                max_logit_diff = std::max(max_logit_diff, diff);
                if (diff == 0.0) logits_exact++;
                expect(diff == 0.0, "candidate logits differ (max abs diff " + std::to_string(diff) + ")");
            }
        } else {
            expect(hex_ids(r.neighbour_ids) == exp["neighbour_ids"].get<std::vector<std::string>>(), "kNN neighbours differ");
            expect(r.neighbour_labels == exp["neighbour_labels"].get<std::vector<int32_t>>(), "kNN neighbour labels differ");
            expect(r.neighbour_hamming == exp["neighbour_hamming"].get<std::vector<double>>(), "kNN Hamming distances differ");
            const auto shares = exp["vote_shares"].get<std::vector<double>>();
            bool same = shares.size() == out.probs.size();
            for (size_t i = 0; same && i < shares.size(); ++i) same = std::fabs(shares[i] - out.probs[i]) <= 1e-12;
            expect(same, "vote shares differ");
        }
        const int want_pred = exp["prediction_index"].get<int>();
        expect(out.selected == want_pred, "prediction " + std::to_string(out.selected) + " != " + std::to_string(want_pred));
        if (out.selected == want_pred) agree++;

        // Public API path
        int32_t status = 0;
        std::string body;
        api.handle(api_request(c["input"]), status, body);
        const json resp = json::parse(body);
        const json& a = resp["answers"]["q"];
        const std::string want_key = exp["prediction_key"].get<std::string>();
        if (status != 200) {
            expect(false, "API status " + std::to_string(status) + ": " + body);
        } else if (in.type == Type::Choice) {
            expect(a["choice"].get<std::string>() == want_key, "API choice differs");
        } else if (in.type == Type::Noul) {
            expect((a["noul"].get<double>() > 0.5) == (want_key == "true"), "API noul differs");
        } else {
            std::string best;
            double bp = -1.0;
            for (const auto& [k, v] : a["probabilities"].items()) if (v.get<double>() > bp) { bp = v.get<double>(); best = k; }
            expect(best == want_key, "API score argmax differs");
        }
        printf("%s %-9s %-14s pred %d (expected %d) %7.1f ms (retrieval %.1f ms) %s\n", ok ? "ok  " : "FAIL", name.c_str(),
               r.strategy.c_str(), out.selected, want_pred, ms, out.retrieval_us / 1000.0, c["tags"].dump().c_str());
        if (record_mode && out.ok) {
            json upd = c;
            json& e = upd["expected"];
            e["strategy"] = r.strategy;
            e["prediction_index"] = out.selected;
            e["prediction_key"] = c["input"]["options"][out.selected][0];
            if (exp.contains("logits")) {
                e["demonstration_ids"] = hex_ids(r.demonstration_ids);
                e["prompt_tokens"] = out.prompt_token_count;
                e["logits"] = out.raw_logits;
            } else {
                e["neighbour_ids"] = hex_ids(r.neighbour_ids);
                e["neighbour_labels"] = r.neighbour_labels;
                e["neighbour_hamming"] = r.neighbour_hamming;
                e["vote_shares"] = out.probs;
            }
            updated_cases.push_back(std::move(upd));
        }
    }
    if (record_mode) {
        std::ofstream rf(record_path);
        for (const auto& u : updated_cases) rf << u.dump() << "\n";
        printf("recorded %zu cases to %s\n", updated_cases.size(), record_path.c_str());
    }

    // Determinism: repeat one case per kind.
    for (const char* kind : {"choice", "noul", "score"}) {
        for (const auto& c : cases) {
            if (c["input"]["type"] != kind) continue;
            const DecisionInput in = to_input(c["input"]);
            const DecisionOutput a = service.decide(in), b = service.decide(in);
            CHECK(a.ok && b.ok && a.selected == b.selected && a.raw_logits == b.raw_logits && a.probs == b.probs &&
                  a.routing->neighbour_ids == b.routing->neighbour_ids, std::string("deterministic repeat: ") + kind);
            break;
        }
    }

    printf("\nprediction agreement: %d/%zu; logits exact: %d/%d (max abs diff %.3g)\n", agree, cases.size(),
           logits_exact, logit_cases, max_logit_diff);
    const std::string model_sha256 = rt->model_identity().sha256;
    rt.reset();

    // Calibrated behaviour. Every case against its uncalibrated output: same routing, exemplars and raw
    // logits; LLM probabilities = temperature softmax of (raw - alpha * prior), with one prior per kind
    // and candidate count; kNN-5 unchanged. Uncalibrated states (missing / incompatible) must equal R4.
    auto run_calibrated = [&](const RuntimeOptions& o, const char* label) -> CalibrationState {
        std::string e;
        std::unique_ptr<RouterRuntime> crt = RouterRuntime::load(o, e);
        if (!crt) {
            fprintf(stderr, "FAIL: %s: runtime load: %s\n", label, e.c_str());
            g_failures++;
            return CalibrationState::Disabled;
        }
        const CalibrationLoad& cal = crt->calibration();
        const CalibrationConfig cfg = cal.config();
        printf("\n%s: calibration %s (%s)%s%s\n", label, to_string(cal.state), o.paths.calibration.c_str(),
               cal.message.empty() ? "" : ": ", cal.message.c_str());
        std::map<std::pair<int, size_t>, std::vector<float>> priors;  // raw - corrected, per kind / count
        int changed = 0;
        for (size_t i = 0; i < cases.size(); ++i) {
            const std::string name = cases[i]["case"].get<std::string>();
            const DecisionInput in = to_input(cases[i]["input"]);
            const DecisionOutput& u = uncalibrated[i];
            const DecisionOutput out = crt->service().decide(in);
            auto expect = [&](bool cond, const std::string& what) {
                if (!cond) { fprintf(stderr, "FAIL: %s: %s: %s\n", label, name.c_str(), what.c_str()); g_failures++; }
            };
            if (!out.ok || !u.ok) { expect(false, "decision failed: " + out.error); continue; }
            expect(out.routing->strategy == u.routing->strategy && out.routing->demonstration_ids == u.routing->demonstration_ids &&
                   out.routing->neighbour_ids == u.routing->neighbour_ids, "routing changed");
            if (out.selected != u.selected) changed++;
            if (cal.state != CalibrationState::Enabled || out.routing->strategy == routing::kKnn5) {
                expect(out.selected == u.selected && out.probs == u.probs && out.raw_logits == u.raw_logits,
                       "output differs from the uncalibrated decision");
                continue;
            }
            expect(out.raw_logits == u.raw_logits, "raw candidate logits changed");
            std::vector<float> delta(out.raw_logits.size());
            for (size_t k = 0; k < delta.size(); ++k) delta[k] = out.raw_logits[k] - out.corrected_logits[k];
            const auto key = std::make_pair(static_cast<int>(in.type), in.options.size());
            auto [it, inserted] = priors.emplace(key, delta);
            // Tolerance: delta = raw - corrected is computed per-decision; floating-point
            // non-associativity can produce tiny variation (~1e-5) when raw values differ.
            bool prior_ok = it->second.size() == delta.size();
            for (size_t k = 0; prior_ok && k < delta.size(); ++k)
                prior_ok = std::fabs(it->second[k] - delta[k]) < 1e-4f;
            expect(prior_ok, "prior correction differs between decisions of the same kind and size");
            if (!cal.prior_correction()) expect(out.corrected_logits == out.raw_logits, "prior correction without prior");
            std::vector<double> probs;
            temperature_softmax(out.corrected_logits, cfg.temperature_for(in.type), probs);
            expect(out.probs == probs, "probabilities are not the temperature softmax of the corrected logits");
            const int best = static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
            expect(out.selected == best, "prediction is not the calibrated argmax");
        }
        printf("%s: %d/%zu predictions differ from the uncalibrated run (not an accuracy measurement)\n", label,
               changed, cases.size());
        return cal.state;
    };

    // The default calibration path is loaded without being configured.
    RuntimeOptions dflt = opts;
    dflt.paths.calibration = ReleaseConfig::resolve_runtime_paths(RuntimePaths{}, "").calibration;
    CHECK(dflt.paths.calibration == "models/calibration.json", "default calibration path");
    const CalibrationState dstate = run_calibrated(dflt, "default calibration");
    CHECK(dstate == CalibrationState::Enabled, "bundled calibration is enabled and model-verified");

    // A compatible artifact (the supplied parameters, letters formulation) is applied.
    CalibrationArtifact art;
    art.formulation        = {"auto", "letters", true};
    art.model_identifier   = model_sha256;  // as `pjev calibrate` records it
    art.choice_temperature = 2.046567219106836;
    art.choice_prior_alpha = 0.9999999999999999;
    art.noul_temperature   = 99.99999952490862;
    art.noul_prior_alpha   = 0.0;
    art.score_temperature  = 11.960309669703673;
    art.score_prior_alpha  = 1.0999999999999999;
    const std::string letters_path = "test_golden_calibration_letters.json";
    std::ofstream(letters_path) << art.to_json().dump(2);
    RuntimeOptions copts = opts;
    copts.paths.calibration  = letters_path;
    copts.calibration_required = true;
    CHECK(run_calibrated(copts, "letters calibration") == CalibrationState::Enabled,
          "compatible calibration for this model is enabled");

    // The same parameters recorded for another model are not applied (R6 model identity check).
    art.model_identifier = std::string(64, '0');
    std::ofstream(letters_path, std::ios::trunc) << art.to_json().dump(2);
    CHECK(run_calibrated(copts, "calibration for another model") == CalibrationState::Incompatible,
          "calibration fitted for another model is incompatible (decisions uncalibrated)");
    std::remove(letters_path.c_str());
    for (const auto& [kind, v] : ms_by_kind) {
        double s = 0;
        for (double x : v) s += x;
        printf("latency %-6s n=%zu mean %.1f ms (first decision of the process includes warm-up)\n", kind.c_str(), v.size(),
               s / v.size());
    }
    if (g_failures == 0) {
        printf("PASS (0 failure(s))\n");
        return 0;
    }
    fprintf(stderr, "FAIL (%d failure(s))\n", g_failures);
    return 1;
}
