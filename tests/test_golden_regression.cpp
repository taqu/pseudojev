// Golden regression (integration): the release decision path must reproduce the frozen R4 policy's
// recorded predictions on tests/data/golden_r4.jsonl. Needs the runtime resources; exits with 77
// (reported as skipped) when they are absent.
//
// Compared per case: strategy, exemplars (demonstrations in prompt order / kNN neighbours), prompt
// token count, candidate logits (exact, as in the R4 reference checks), kNN vote shares, and the
// prediction. Every case is also sent through the HTTP API handler.
//
// Environment (optional): PJEV_GOLDEN_DATA, PJEV_MODEL, PJEV_EMBEDDING_MODEL, PJEV_EXEMPLAR_DB,
// PJEV_EXEMPLAR_ALLOWLIST override the default paths (relative to the repository root).

#include "decision/decision_service.h"
#include "retrieval/exemplar_allowlist.h"
#include "routing/router_runtime.h"
#include "routing/task_policy.h"
#include "server/jev_api.h"

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

int main()
{
    const std::string data_path = env_or("PJEV_GOLDEN_DATA", "tests/data/golden_r4.jsonl");
    RuntimeOptions opts;
    opts.paths.model              = env_or("PJEV_MODEL", "models/bonsai2_2b.gguf");
    opts.paths.embedding_model    = env_or("PJEV_EMBEDDING_MODEL", "models/nomic-embed-text-v1.5.gguf");
    opts.paths.exemplar_db        = env_or("PJEV_EXEMPLAR_DB", "models/examplar.db");
    opts.paths.exemplar_allowlist = env_or("PJEV_EXEMPLAR_ALLOWLIST", "models/examplar-allowlist.txt");
    opts.n_ctx = 4096;  // the R4 reference configuration

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

    DecisionService& service = rt->service();
    JevApiHandler api(service, "golden");
    std::map<std::string, std::vector<double>> ms_by_kind;
    int agree = 0, logits_exact = 0, logit_cases = 0;
    double max_logit_diff = 0.0;

    for (const auto& c : cases) {
        const std::string name = c["case"].get<std::string>();
        const json& exp = c["expected"];
        const DecisionInput in = to_input(c["input"]);

        const auto ts = std::chrono::steady_clock::now();
        const DecisionOutput out = service.decide(in);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
        ms_by_kind[to_string(in.type)].push_back(ms);

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
        expect(!r.fallback, "unexpected 1-shot fallback");

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
