// Unit tests for Phase 5: model identity, candidate compatibility, value profile,
// performance stats, Phase5Decision, common-valid-subset.
// No real model files are used; mock backends and synthetic RunResults are constructed directly.

#include "model/model_identity.h"
#include "model/value_profile.h"
#include "calibration/calibration_metrics.h"
#include "multilingual/multilingual.h"
#include "experiment/runner.h"
#include "experiment/config.h"
#include "inference/backend.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>
#include <map>

// ---------------------------------------------------------------------------
// Test framework
// ---------------------------------------------------------------------------
static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_failures++; } \
    else         { printf("pass: %s\n", msg); } \
} while(0)

// ---------------------------------------------------------------------------
// Mock backend: each candidate character tokenises to its ASCII code (single token)
// except multi-char strings -> multi-token
// ---------------------------------------------------------------------------
struct MockBackend : public pjev::ILlamaBackend {
    std::vector<int> tokenize(const std::string& text,
                              bool, bool) const override {
        // 1-char strings → single token (ascii code)
        // multi-char strings → one token per character
        std::vector<int> r;
        for (char c : text) r.push_back((int)(unsigned char)c);
        return r;
    }
    int vocab_size() const override { return 256; }
    std::string token_to_piece(int id) const override { return std::string(1, (char)id); }
    const float* eval_tokens(const std::vector<int>&) override {
        static float logits[256] = {};
        return logits;
    }
    // metadata defaults are inherited (return "" / 0)
};

// ---------------------------------------------------------------------------
// Helpers to build synthetic RunResult
// ---------------------------------------------------------------------------
static pjev::RunResult make_run(const std::vector<std::tuple<std::string,std::string,bool,std::string,int64_t>>& items)
{
    // items: (id, type_str, correct, difficulty, eval_ms)
    pjev::RunResult rr;
    rr.config.name = "synthetic";
    rr.dataset_path = "fake";
    for (const auto& tup : items) {
        pjev::ItemResult ir;
        ir.id         = std::get<0>(tup);
        ir.type       = pjev::to_type(std::get<1>(tup));
        ir.correct    = std::get<2>(tup);
        ir.difficulty = std::get<3>(tup);
        ir.eval_ms    = std::get<4>(tup);
        ir.ok         = true;
        if (std::get<2>(tup)) {
            rr.metrics.choice.correct += (ir.type == pjev::Type::Choice);
            rr.metrics.noul.correct   += (ir.type == pjev::Type::Noul);
            rr.metrics.score.correct  += (ir.type == pjev::Type::Score);
        }
        if (ir.type == pjev::Type::Choice) { rr.metrics.choice.n++; rr.metrics.choice.labeled++; }
        if (ir.type == pjev::Type::Noul)   { rr.metrics.noul.n++;   rr.metrics.noul.labeled++; }
        if (ir.type == pjev::Type::Score)  { rr.metrics.score.n++;  rr.metrics.score.labeled++; }
        rr.metrics.n_total++;
        rr.items.push_back(ir);
    }
    return rr;
}

// ---------------------------------------------------------------------------
// SHA-256 tests
// ---------------------------------------------------------------------------
static void test_sha256()
{
    using pjev::sha256_data;

    CHECK(sha256_data("").size() == 64,
          "sha256 empty: 64 hex chars");
    CHECK(sha256_data("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "sha256 empty: known value");
    CHECK(sha256_data("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "sha256 'abc': known value");
    // Same input → same output
    CHECK(sha256_data("hello") == sha256_data("hello"),
          "sha256 deterministic");
    // Different inputs → different output
    CHECK(sha256_data("a") != sha256_data("b"),
          "sha256 distinct inputs produce distinct hashes");
}

// ---------------------------------------------------------------------------
// CandidateCompatibility tests
// ---------------------------------------------------------------------------
static void test_candidate_compat()
{
    MockBackend mb;

    // Single-char candidates → single token (length 1 string → 1 token)
    auto cc1 = pjev::probe_candidates(mb, {"A", "B", "C", "D"});
    CHECK(cc1.all_single_token,   "ABCD: all single-token");
    CHECK(cc1.records.size() == 4, "ABCD: 4 records");
    CHECK(cc1.records[0].single_token, "A: single_token=true");
    CHECK(cc1.records[0].token_count == 1, "A: token_count=1");
    CHECK(cc1.records[0].token_id == (int)'A', "A: token_id == ascii('A')");
    CHECK(cc1.issues.empty(), "ABCD: no issues");

    // Multi-char string → multi-token
    auto cc2 = pjev::probe_candidates(mb, {"Yes", "No"});
    CHECK(!cc2.all_single_token, "Yes/No: not all single-token");
    CHECK(!cc2.issues.empty(),   "Yes/No: issues reported");
    CHECK(!cc2.records[0].single_token, "Yes: single_token=false");
    CHECK(cc2.records[0].token_count == 3, "Yes: token_count=3");

    // to_json round-trip
    auto j = cc1.to_json();
    CHECK(j.contains("all_single_token"), "compat json: all_single_token field");
    CHECK(j.contains("records"),          "compat json: records field");
    CHECK(j["all_single_token"].get<bool>(), "compat json: all_single_token=true");
}

// ---------------------------------------------------------------------------
// ModelIdentity tests
// ---------------------------------------------------------------------------
static void test_model_identity()
{
    // from_path without a backend: only file-level fields
    pjev::ModelIdentity id = pjev::ModelIdentity::from_path("/nonexistent/model.gguf");
    CHECK(id.gguf_path == "/nonexistent/model.gguf", "identity: gguf_path set");
    CHECK(id.name == "model",                         "identity: name from filename stem");
    CHECK(id.size_bytes == 0,                         "identity: size_bytes=0 for missing file");
    CHECK(id.sha256.empty(),                          "identity: sha256 empty for missing file");

    // to_json includes all required fields
    auto j = id.to_json();
    CHECK(j.contains("name"),         "identity json: name");
    CHECK(j.contains("sha256"),       "identity json: sha256");
    CHECK(j.contains("size_bytes"),   "identity json: size_bytes");
    CHECK(j.contains("quantization"), "identity json: quantization");
    CHECK(j.contains("architecture"), "identity json: architecture");
    CHECK(j.contains("model_desc"),   "identity json: model_desc");
    CHECK(j.contains("n_params"),     "identity json: n_params");
}

// ---------------------------------------------------------------------------
// compute_quality_stats tests
// ---------------------------------------------------------------------------
static void test_quality_stats()
{
    // 3 choice items: 2 correct, 1 wrong; 2 noul items: 1 correct
    auto rr = make_run({
        {"q1", "choice", true,  "easy",     100},
        {"q2", "choice", true,  "hard",     150},
        {"q3", "choice", false, "easy",     120},
        {"n1", "noul",   true,  "original", 80},
        {"n2", "noul",   false, "original", 90}
    });

    auto qs = pjev::compute_quality_stats(rr);

    CHECK(qs.choice.n == 3, "quality: choice.n=3");
    CHECK(std::abs(qs.choice.accuracy - 2.0/3.0) < 1e-9, "quality: choice accuracy=2/3");
    CHECK(qs.noul.n == 2,   "quality: noul.n=2");
    CHECK(std::abs(qs.noul.accuracy - 0.5) < 1e-9, "quality: noul accuracy=0.5");
    CHECK(qs.score.n == 0,  "quality: score.n=0 (no score items)");

    // Difficulty breakdown
    CHECK(qs.choice.n_by_difficulty.count("easy") &&
          qs.choice.n_by_difficulty.at("easy") == 2,  "quality: choice easy n=2");
    CHECK(qs.choice.accuracy_by_difficulty.count("easy") &&
          std::abs(qs.choice.accuracy_by_difficulty.at("easy") - 0.5) < 1e-9,
          "quality: choice easy accuracy=0.5");
    CHECK(qs.choice.accuracy_by_difficulty.count("hard") &&
          std::abs(qs.choice.accuracy_by_difficulty.at("hard") - 1.0) < 1e-9,
          "quality: choice hard accuracy=1.0");

    // to_json
    auto j = qs.to_json();
    CHECK(j.contains("choice"), "quality json: choice");
    CHECK(j.contains("noul"),   "quality json: noul");
    CHECK(j.contains("score"),  "quality json: score");
    CHECK(j["choice"].contains("accuracy_by_difficulty"),
          "quality json: accuracy_by_difficulty present");
}

// ---------------------------------------------------------------------------
// compute_perf_stats tests
// ---------------------------------------------------------------------------
static void test_perf_stats()
{
    // 10 evenly spaced samples: 100, 200, ..., 1000
    std::vector<int64_t> warm_ms;
    for (int i = 1; i <= 10; i++) warm_ms.push_back(i * 100LL);

    auto ps = pjev::compute_perf_stats(440, 987654321, warm_ms, "TestOS", "TestCPU", 8);

    CHECK(ps.model_load_ms == 440,          "perf: load_ms=440");
    CHECK(ps.rss_post_load_bytes == 987654321, "perf: rss set");
    CHECK(ps.n_warm_samples == 10,          "perf: n_warm_samples=10");
    CHECK(ps.os == "TestOS",                "perf: os");
    CHECK(ps.n_threads == 8,                "perf: n_threads=8");

    // p50 of [100,200,...,1000] = interpolate at index 4.5 → 550
    CHECK(std::abs(ps.warm_p50_ms - 550.0) < 1.0, "perf: p50=550");
    // p90 at index 8.1 → 810 + 0.1*100 = 910 (approx check)
    CHECK(ps.warm_p90_ms > 800.0 && ps.warm_p90_ms < 1000.0, "perf: p90 in range");
    // p95 at index 8.55 → roughly 955
    CHECK(ps.warm_p95_ms >= ps.warm_p90_ms, "perf: p95 >= p90");

    // Edge: empty samples
    auto ps2 = pjev::compute_perf_stats(100, -1, {}, "Linux", "", 4);
    CHECK(ps2.warm_p50_ms < 0.0, "perf: empty samples → p50=-1");
    CHECK(ps2.n_warm_samples == 0, "perf: empty samples n=0");

    // to_json
    auto j = ps.to_json();
    CHECK(j.contains("model_load_ms"),  "perf json: model_load_ms");
    CHECK(j.contains("warm_p50_ms"),    "perf json: warm_p50_ms");
    CHECK(j.contains("warm_p95_ms"),    "perf json: warm_p95_ms");
    CHECK(j.contains("rss_post_load_bytes"), "perf json: rss_post_load_bytes");
}

// ---------------------------------------------------------------------------
// Phase5Decision tests
// ---------------------------------------------------------------------------
static void test_phase5_decision()
{
    pjev::Phase5Decision d;
    d.phase                  = 5;
    d.timestamp              = "2026-10-04T00:00:00Z";
    d.baseline_model         = "Bonsai-1.7B-Q4_K_M";
    d.evaluated_models       = {"Bonsai-1.7B-Q4_K_M"};
    d.selected_default_model = "Bonsai-1.7B-Q4_K_M";
    d.evidence_artifacts     = {"artifacts/bonsai_profile.json"};
    d.notes                  = {"Bonsai satisfies quality and latency targets."};

    // to_json
    auto j = d.to_json();
    CHECK(j["phase"].get<int>() == 5,                     "decision json: phase=5");
    CHECK(j["baseline_model"].get<std::string>() == "Bonsai-1.7B-Q4_K_M",
          "decision json: baseline_model");
    CHECK(j["evaluated_models"].size() == 1,              "decision json: evaluated_models size=1");
    CHECK(j["notes"].size() == 1,                         "decision json: notes size=1");
    CHECK(j.contains("selected_default_model"),           "decision json: selected_default_model field");

    // from_json round-trip
    pjev::Phase5Decision d2;
    std::string err;
    bool ok = pjev::Phase5Decision::from_json(j, d2, err);
    CHECK(ok,                                  "decision from_json: ok");
    CHECK(d2.phase == 5,                       "decision from_json: phase=5");
    CHECK(d2.baseline_model == d.baseline_model, "decision from_json: baseline_model");
    CHECK(d2.selected_default_model == d.selected_default_model,
          "decision from_json: selected_default_model");
    CHECK(d2.evaluated_models.size() == 1,     "decision from_json: evaluated_models size");
    CHECK(d2.evidence_artifacts.size() == 1,   "decision from_json: evidence_artifacts size");
    CHECK(d2.notes.size() == 1,                "decision from_json: notes size");

    // Bad JSON: wrong phase
    auto bad = j;
    bad["phase"] = 3;
    pjev::Phase5Decision d3;
    std::string err2;
    CHECK(!pjev::Phase5Decision::from_json(bad, d3, err2), "decision from_json: reject wrong phase");

    // Bad JSON: missing phase
    nlohmann::json bad2 = {{"baseline_model", "x"}};
    pjev::Phase5Decision d4;
    std::string err3;
    CHECK(!pjev::Phase5Decision::from_json(bad2, d4, err3), "decision from_json: reject missing phase");
}

// ---------------------------------------------------------------------------
// CommonValidSubset tests
// ---------------------------------------------------------------------------
static void test_common_valid_subset()
{
    // Model A: items q1,q2,q3 correct=[true,true,false]; q4 incompatible
    // Model B: items q1,q2,q3 correct=[true,false,false]; q4 present but incompatible in model A
    auto rr_a = make_run({
        {"q1","choice",true,"",50},
        {"q2","choice",true,"",60},
        {"q3","choice",false,"",70},
    });
    auto rr_b = make_run({
        {"q1","choice",true,"",50},
        {"q2","choice",false,"",60},
        {"q3","choice",false,"",70},
        {"q4","choice",true,"",80},  // q4 incompatible in model A
    });

    std::map<std::string, pjev::RunResult> results;
    results["A"] = rr_a;
    results["B"] = rr_b;

    // q4 is incompatible for model A
    std::map<std::string, std::vector<std::string>> incompat;
    incompat["A"] = {"q4"};

    auto cvs = pjev::build_common_valid_subset(results, incompat);

    CHECK(cvs.total_items == 4, "cvs: total_items=4");
    CHECK(cvs.valid_items == 3, "cvs: valid_items=3 (q4 excluded)");
    CHECK(cvs.full_accuracy.count("A"),    "cvs: model A in full_accuracy");
    CHECK(cvs.subset_accuracy.count("A"),  "cvs: model A in subset_accuracy");
    CHECK(cvs.n_compat_issues.count("A"),  "cvs: model A in n_compat_issues");
    CHECK(cvs.n_compat_issues.at("A") == 1, "cvs: model A has 1 compat issue");
    CHECK(cvs.n_compat_issues.count("B") && cvs.n_compat_issues.at("B") == 0,
          "cvs: model B has 0 compat issues");

    // Model A: full set = q1,q2,q3 (q4 excluded as incompatible) → 2/3
    CHECK(std::abs(cvs.full_accuracy.at("A") - 2.0/3.0) < 1e-9,
          "cvs: model A full accuracy=2/3");
    // Model B: full set includes q4 → q1,q2,q3,q4 → correct: q1, q4 → 2/4
    CHECK(std::abs(cvs.full_accuracy.at("B") - 0.5) < 1e-9,
          "cvs: model B full accuracy=0.5");

    // to_json
    auto j = cvs.to_json();
    CHECK(j.contains("total_items"),     "cvs json: total_items");
    CHECK(j.contains("valid_items"),     "cvs json: valid_items");
    CHECK(j.contains("full_accuracy"),   "cvs json: full_accuracy");
    CHECK(j.contains("subset_accuracy"), "cvs json: subset_accuracy");
    CHECK(j.contains("n_compat_issues"), "cvs json: n_compat_issues");
}

// ---------------------------------------------------------------------------
// ValueProfile serialization test
// ---------------------------------------------------------------------------
static void test_value_profile_json()
{
    pjev::ValueProfile vp;
    vp.version           = 1;
    vp.timestamp         = "2026-10-04T00:00:00Z";
    vp.pjev_commit       = "abc1234";
    vp.llamacpp_revision = "deadbeef";
    vp.model.name        = "Bonsai-1.7B-Q4_K_M";
    vp.model.sha256      = "aabbccdd";
    vp.model.size_bytes  = 1234567890;
    vp.performance.model_load_ms = 440;
    vp.performance.rss_post_load_bytes = 987654321;

    auto j = vp.to_json();
    CHECK(j["version"].get<int>() == 1,               "vp json: version=1");
    CHECK(j["pjev_commit"].get<std::string>() == "abc1234", "vp json: pjev_commit");
    CHECK(j.contains("model"),              "vp json: model field");
    CHECK(j.contains("quality_en"),         "vp json: quality_en field");
    CHECK(j.contains("quality_ja"),         "vp json: quality_ja field");
    CHECK(j.contains("calibration_en"),     "vp json: calibration_en field");
    CHECK(j.contains("calibration_ja"),     "vp json: calibration_ja field");
    CHECK(j.contains("multilingual"),       "vp json: multilingual field");
    CHECK(j.contains("performance"),        "vp json: performance field");
    CHECK(j.contains("candidate_compat"),   "vp json: candidate_compat field");
    CHECK(j.contains("frozen_config"),      "vp json: frozen_config field");
    CHECK(j["frozen_config"].contains("layout"), "vp json: frozen_config.layout");
    CHECK(j["model"]["name"].get<std::string>() == "Bonsai-1.7B-Q4_K_M", "vp json: model.name");
    CHECK(j["performance"]["model_load_ms"].get<int64_t>() == 440, "vp json: load_ms=440");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    test_sha256();
    test_candidate_compat();
    test_model_identity();
    test_quality_stats();
    test_perf_stats();
    test_phase5_decision();
    test_common_valid_subset();
    test_value_profile_json();

    if (g_failures == 0) {
        printf("\nPASS (0 failure(s))\n");
        return 0;
    }
    fprintf(stderr, "\nFAIL (%d failure(s))\n", g_failures);
    return 1;
}
