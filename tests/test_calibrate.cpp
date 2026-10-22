// Integration tests for `pjev calibrate` generation (no model): a small fixture exemplar database (built
// with pjev-filldb's writer) read through the production ExemplarStore, TaskRouter and DecisionEngine,
// with a mock inference backend whose letter logits favour the answer marked in the query text and a
// mock embedder that places every row of a group at the same point. Checks the generated calibration end
// to end: partitioning and leakage prevention (from the evaluated prompts), inference paths and
// fallback recording, deterministic output, the production loader and model identity, read-only
// database access, atomic replacement, insufficient / degenerate data, validation rejection, and the
// generated parameters applied by the router (kNN-5 unchanged, zero-shot fallback calibrated).

#include "exemplar_database.h"
#include "exemplar_record.h"

#include "calibration/calibration.h"
#include "calibration/calibration_generator.h"
#include "lz4/xxhash.h"
#include "mock_backend.h"
#include "model/model_identity.h"
#include "retrieval/exemplar_allowlist.h"
#include "retrieval/exemplar_store.h"
#include "retrieval/id_hash.h"
#include "routing/task_policy.h"
#include "routing/task_router.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace pjev;
using namespace pjev::calibration;
using pjev::filldb::ExemplarDatabase;
using pjev::filldb::ExemplarRecord;
namespace fs = std::filesystem;

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); g_failures++; } \
    else         { printf("pass: %s\n", msg); } \
} while (0)

static const int         kDim   = 16;
static const std::string kModel = std::string(64, 'c');

// "grp<N>:" tags every row's question with its fixture group.
static int group_at(const std::string& text, size_t at)
{
    return std::atoi(text.c_str() + at + 3);
}

static std::vector<float> group_vec(int g)
{
    const uint64_t h = XXH64(&g, sizeof g, 7);
    std::vector<float> v(kDim);
    for (int i = 0; i < kDim; ++i) v[static_cast<size_t>(i)] = ((h >> i) & 1) ? 1.0f : -1.0f;
    return v;
}

// Every text of a group embeds to the group's point: without held-out groups, a query's siblings would
// be its nearest neighbours.
struct GroupEmbedder : routing::ITextEmbedder {
    bool encode(const std::string& text, std::vector<float>& out) override {
        const size_t at = text.find("grp");
        out = group_vec(at == std::string::npos ? -1 : group_at(text, at));
        return true;
    }
    int dimension() const override { return kDim; }
};

// Letter logits: seeded noise plus a boost for the query's marked answer ("[ans=K]", the last one in the
// prompt), scaled up (an overconfident model: what temperature scaling corrects), plus a position bias
// towards A (what prior correction removes; a blank prompt shows only this bias). Groups in `inverted`
// get the boost on another letter (a model that is wrong in a way the fitting data does not show).
class MarkedBackend : public MockBackend {
public:
    std::vector<std::string> prompts;
    std::set<int> inverted;
    float boost = 1.5f;
    float scale = 8.0f;

    const float* eval_tokens(const std::vector<int>& tokens) override {
        std::string text;
        for (int t : tokens) text += t >= 1000 ? std::string(1, (char)(t - 1000)) : token_to_piece(t);
        prompts.push_back(text);
        logits_.assign(600, 0.0f);
        const uint64_t h = XXH64(text.data(), text.size(), 3);
        const size_t at = text.rfind("[ans=");
        if (at != std::string::npos) {
            for (int i = 0; i < 20; ++i) logits_[65 + i] = static_cast<float>((h >> (3 * i)) & 7) * 0.4f;
            int k = text[at + 5] - '0';
            const size_t g = text.rfind("grp", at);
            if (g != std::string::npos && inverted.count(group_at(text, g))) k = (k + 1) % 2;
            logits_[65 + k] += boost;
        }
        for (int i = 0; i < 20; ++i) logits_[65 + i] *= scale;
        logits_[65] += 6.0f;
        return logits_.data();
    }

private:
    std::vector<float> logits_;
};

struct Fixture {
    std::string db, allow;
    std::map<int64_t, int> group_of;  // exemplar ID -> fixture group
    std::map<int64_t, int> answer_of; // exemplar ID -> correct candidate in decision order
};

static void remove_db(const std::string& p)
{
    for (const char* s : {"", ".tmp", "-wal", "-shm"}) std::remove((p + s).c_str());
}

// 600 groups: two choice rows each; noul in groups 0-199 only (all of them become calibration queries,
// so noul is decided by the zero-shot fallback); 5-level score in groups 200-599; a few rows with
// ambiguous answers (rejected) and rows outside the training split (never used).
static Fixture build_fixture(const std::string& name, bool degenerate = false)
{
    Fixture f;
    f.db    = name + ".db";
    f.allow = name + "-allow.txt";
    remove_db(f.db);
    ExemplarDatabase db(f.db);
    db.open(false);
    db.create_schema(kDim, "test-embedding.gguf", "0123456789abcdef");
    db.begin_transaction();
    std::vector<int64_t> allowed;
    auto add = [&](const std::string& id, int g, const std::string& kind, std::vector<std::string> options, int answer,
                   bool train) {
        ExemplarRecord rec;
        rec.id       = retrieval::hash_exemplar_id(id);
        rec.group_id = retrieval::hash_exemplar_id("group-" + std::to_string(g));
        rec.kind     = kind;
        rec.source   = "unit";
        rec.split    = train ? "train" : "validation";
        // The answer in the router's decision order (noul: false, true).
        const int k = kind == "noul" ? (options[static_cast<size_t>(answer)] == "yes" ? 1 : 0) : answer;
        rec.question = "Question grp" + std::to_string(g) + ": " + id + " [ans=" + std::to_string(k) + "]";
        rec.options  = options;
        rec.target.assign(options.size(), 0.0f);
        rec.target[static_cast<size_t>(answer)] = 1.0f;
        rec.answer_index = answer;
        rec.answer_text  = options[static_cast<size_t>(answer)];
        rec.state = nlohmann::ordered_json{{"group", g}};
        rec.number_of_levels = kind == "score" ? static_cast<int>(options.size()) : 0;
        rec.language = "en";
        rec.domain   = "unit";
        rec.embedding_text = "Task: " + kind + " grp" + std::to_string(g) + ":";
        int64_t rowid = 0;
        db.insert_group(rec);
        db.insert_example(rec, rowid);
        db.insert_vector(rowid, group_vec(g), kind, rec.number_of_levels);
        f.group_of[rec.id] = g;
        f.answer_of[rec.id] = k;
        if (train) allowed.push_back(rec.id);
    };
    const std::vector<std::string> colours = {"red", "green", "blue"};
    for (int g = 0; g < 600; ++g) {
        add("c" + std::to_string(g) + "a", g, "choice", colours, degenerate ? 0 : g % 3, true);
        add("c" + std::to_string(g) + "b", g, "choice", colours, degenerate ? 0 : (g + 1) % 3, true);
        if (g < 200) {
            add("n" + std::to_string(g), g, "noul", g % 2 ? std::vector<std::string>{"no", "yes"} : std::vector<std::string>{"yes", "no"},
                g % 3 == 0 ? 0 : 1, true);
        } else {
            add("s" + std::to_string(g), g, "score", {"l0", "l1", "l2", "l3", "l4"}, g % 5, true);
        }
        if (g % 30 == 0) add("dup" + std::to_string(g), g, "choice", {"x", "x", "y"}, 0, true);
        if (g % 10 == 0) add("heldback" + std::to_string(g), g, "choice", colours, 1, false);
    }
    db.commit_transaction();
    std::string err;
    retrieval::write_allowlist(f.allow, allowed, {"calibrate fixture"}, err);
    return f;
}

static const EnsembleConfig kSingleOrder{NoulEnsembleMode::NONE, ChoiceEnsembleMode::NONE, ScoreEnsembleMode::NONE};

// The production decision path on the fixture, uncalibrated (as `pjev calibrate` loads it).
struct Pipeline {
    retrieval::ExemplarStore store;
    MarkedBackend            backend;
    GroupEmbedder            embedder;
    std::unique_ptr<DecisionEngine>      engine;
    std::unique_ptr<routing::TaskRouter> router;

    explicit Pipeline(const Fixture& f, const CalibrationLoad& cal = {}) {
        std::string err;
        if (!store.open(f.db, f.allow, err)) fprintf(stderr, "store: %s\n", err.c_str());
        engine = std::make_unique<DecisionEngine>(backend, PromptConfig{Layout::AUTO, Scheme::LETTERS}, cal.config(), kSingleOrder);
        router = std::make_unique<routing::TaskRouter>(*engine, embedder, store, cal.prior_correction());
    }
    GenerateOutcome run(const std::string& output, GeneratorOptions o = {}) {
        GenerateRequest req;
        req.output_path  = output;
        req.model_sha256 = kModel;
        req.options      = o;
        req.provenance   = {{"fixture", true}};
        return generate_calibration(store, *router, *engine, req);
    }
};

static std::string read_text(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static bool no_temp_files(const std::string& dir)
{
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".tmp" || e.path().extension() == ".probe") return false;
    return true;
}

static GeneratorOptions options()
{
    GeneratorOptions o;
    o.max_samples      = 200;
    o.validation_ratio = 0.4;
    o.batch_size       = 16;
    return o;
}

// An existing artifact at the output path (the previous calibration).
static const char* kPrevious = R"({"version":1,"method":"temperature","model":"","formulation":{"layout":"auto",
"candidate_scheme":"letters","prior_correction":true},"parameters":{"choice_temperature":2.0,"choice_prior_alpha":0.5,
"noul_temperature":2.0,"noul_prior_alpha":0.5,"score_temperature":2.0,"score_prior_alpha":0.5}})";

static void test_generate(const Fixture& f)
{
    const std::string dir = "test_calibrate_out/a";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string out_path = dir + "/calibration.json";
    std::ofstream(out_path, std::ios::binary) << kPrevious;
    const std::string db_sha = sha256_file(f.db);

    Pipeline p(f);
    const GenerateOutcome r = p.run(out_path, options());
    CHECK(r.ok, ("generate: succeeds (" + r.error + ")").c_str());
    if (!r.ok) return;
    CHECK(sha256_file(f.db) == db_sha, "generate: database file unchanged (read-only access)");
    CHECK(p.store.held_out_size() == 0, "generate: the store's held-out set is cleared afterwards");
    CHECK(read_text(out_path) != kPrevious && no_temp_files(dir), "generate: previous calibration replaced atomically");

    // Production loader and model identity.
    CalibrationLoad c;
    std::string err;
    CHECK(load_calibration(out_path, true, production_formulation(), c, err) && c.state == CalibrationState::Enabled,
          "generate: output loads through the production loader (letters, auto, prior correction)");
    CHECK(c.artifact.version == 1 && c.artifact.method == "temperature" && c.artifact.formulation.candidate_scheme == "letters" &&
          c.artifact.formulation.layout == "auto" && c.artifact.formulation.prior_correction, "generate: production formulation recorded");
    CHECK(c.artifact.model_identifier == kModel && check_calibration_model(c, kModel), "generate: model identity recorded and verified");
    CalibrationLoad other = c;
    CHECK(!check_calibration_model(other, std::string(64, 'd')) && other.state == CalibrationState::Incompatible,
          "generate: the calibration is not applied to another model");
    const CalibrationArtifact& a = c.artifact;
    bool in_bounds = true;
    for (double t : {a.choice_temperature, a.noul_temperature, a.score_temperature}) in_bounds &= std::isfinite(t) && t >= 0.01 && t <= 100.0;
    for (double al : {a.choice_prior_alpha, a.noul_prior_alpha, a.score_prior_alpha}) in_bounds &= al >= 0.0 && al <= 2.0;
    CHECK(in_bounds, "generate: six task-specific parameters within bounds");
    CHECK(a.choice_prior_alpha > 0.0, "generate: prior correction fitted against the A bias");

    // Report.
    const nlohmann::json& rep = r.report;
    CHECK(r.report_path == report_path_for(out_path, true) && fs::exists(r.report_path) &&
          nlohmann::json::parse(read_text(r.report_path))["status"] == "accepted", "generate: report written next to the output");
    const auto& kinds = rep["kinds"];
    CHECK(kinds["choice"]["counts"]["candidate_rows"] == 1220 && rep["partition"]["rows_not_in_training_split"] == 60,
          "generate: candidate and non-training rows counted");
    bool quotas = true;
    for (const char* k : {"choice", "noul", "score"}) {
        const auto& cnt = kinds[k]["counts"];
        quotas &= cnt["selected_fit"].get<int>() + cnt["selected_validation"].get<int>() == 200 &&
                  cnt["accepted_fit"] == cnt["selected_fit"] && cnt["accepted_validation"] == cnt["selected_validation"];
        quotas &= kinds[k]["validation"]["uncalibrated"]["n"] == cnt["accepted_validation"];
    }
    CHECK(quotas, "generate: quotas filled, every query accepted, validation on the validation queries");
    CHECK(kinds["choice"]["counts"]["examined"].get<int>() >= 200, "generate: examined rows counted");

    // Every sample's path; leakage from the evaluated prompts.
    std::set<int> query_groups;
    std::map<std::string, std::map<std::string, int>> strategies;
    std::map<int, std::set<std::string>> group_splits;
    bool aligned = true;
    for (const auto& s : rep["samples"]) {
        int64_t id = 0;
        retrieval::parse_exemplar_id(s["id"].get<std::string>(), id);
        aligned &= s["correct"].get<int>() == f.answer_of.at(id) &&
                   s["raw_logits"].size() == (s["kind"] == "choice" ? 3u : s["kind"] == "noul" ? 2u : 5u);
        query_groups.insert(f.group_of.at(id));
        group_splits[f.group_of.at(id)].insert(s["split"].get<std::string>());
        strategies[s["kind"].get<std::string>()][s["strategy"].get<std::string>() +
                                                 (s.contains("fallback_reason") ? "/" + s["fallback_reason"].get<std::string>() : "")]++;
    }
    CHECK(rep["samples"].size() == 600, "generate: one sample per accepted query");
    CHECK(aligned, "generate: ground truth aligned with the candidate index (noul 'yes' -> true), one logit per candidate");
    CHECK(strategies["choice"].size() == 1 && strategies["choice"]["nearest_2shot"] == 200,
          "generate: choice queries use nearest 2-shot");
    CHECK(strategies["noul"].size() == 1 && strategies["noul"]["zero_shot/no_eligible_exemplar"] == 200,
          "generate: noul queries without exemplars use the zero-shot fallback (recorded)");
    CHECK(strategies["score"].size() == 1 && strategies["score"]["zero_shot"] == 200,
          "generate: score queries use S1 zero-shot, not kNN-5");
    CHECK(kinds["noul"]["paths"]["zero_shot"]["role"] == "fallback" &&
          kinds["noul"]["paths"]["zero_shot"]["fallback_reason"] == "no_eligible_exemplar" &&
          kinds["choice"]["paths"]["nearest_2shot"]["role"] == "primary", "generate: per-path report");
    bool one_split = true;
    for (const auto& kv : group_splits) one_split &= kv.second.size() == 1;
    CHECK(one_split, "generate: no group in both fitting and validation");

    int few_shot = 0;
    bool own_once = true, no_held_out_demo = true;
    for (const std::string& prompt : p.backend.prompts) {
        const size_t q = prompt.rfind("[ans=");
        if (q == std::string::npos) continue;  // blank prompt (prior)
        const int query_group = group_at(prompt, prompt.rfind("grp", q));
        int own = 0, demos = 0;
        for (size_t at = prompt.find("grp"); at != std::string::npos; at = prompt.find("grp", at + 3)) {
            const int g = group_at(prompt, at);
            if (g == query_group) own++;
            else if (query_groups.count(g)) no_held_out_demo = false;
            else demos++;
        }
        own_once &= own == 1;
        if (demos > 0) few_shot++;
    }
    CHECK(few_shot == 200, "generate: demonstrations shown for the 2-shot queries");
    CHECK(own_once, "generate: no query sees itself or a sibling of its group");
    CHECK(no_held_out_demo, "generate: no demonstration from any calibration group");

    // Deterministic: the same inputs give the same file and samples.
    Pipeline p2(f);
    const std::string out2 = "test_calibrate_out/b/calibration.json";
    fs::remove_all("test_calibrate_out/b");
    const GenerateOutcome r2 = p2.run(out2, options());
    CHECK(r2.ok && read_text(out2) == read_text(out_path) && r2.report["samples"] == rep["samples"],
          "generate: deterministic for identical inputs and settings");
    GeneratorOptions o3 = options();
    o3.seed = 7;
    Pipeline p3(f);
    const GenerateOutcome r3 = p3.run("test_calibrate_out/c/calibration.json", o3);
    CHECK(r3.ok && r3.report["samples"] != rep["samples"], "generate: another seed samples other queries");

    // The generated calibration applied by the router.
    Pipeline cal(f, c);
    Pipeline raw(f);
    DecisionInput score5;
    score5.type     = Type::Score;
    score5.question = "Question grp350: rate [ans=2]";
    score5.state    = "{\"group\":350}";
    for (int i = 0; i < 5; ++i) score5.options.push_back({std::to_string(i), "l" + std::to_string(i)});
    const DecisionOutput k_cal = cal.router->decide(score5);
    const DecisionOutput k_raw = raw.router->decide(score5);
    CHECK(k_cal.ok && k_cal.routing->strategy == routing::kKnn5 && k_cal.probs == k_raw.probs &&
          k_cal.selected == k_raw.selected && cal.backend.prompts.empty(), "applied: score kNN-5 unchanged, no LLM evaluation");

    DecisionInput score4 = score5;
    score4.options.resize(4);  // no 4-level exemplars: zero-shot fallback
    const DecisionOutput z = cal.router->decide(score4);
    const std::vector<float> prior = cal.engine->compute_blank_logits(Type::Score, 4);
    std::vector<float> corrected = z.raw_logits;
    for (size_t i = 0; i < corrected.size(); ++i) corrected[i] -= (float)(a.score_prior_alpha * (double)prior[i]);
    std::vector<double> probs;
    temperature_softmax(corrected, a.score_temperature, probs);
    CHECK(z.ok && z.routing->strategy == routing::kZeroShot && z.probs == probs,
          "applied: score zero-shot fallback uses the generated score parameters");

    DecisionInput choice;
    choice.type     = Type::Choice;
    choice.question = "Question grp250: pick [ans=1]";
    choice.state    = "{\"group\":250}";
    choice.options  = {{"red", "red"}, {"green", "green"}, {"blue", "blue"}};
    const DecisionOutput ch = cal.router->decide(choice);
    const std::vector<float> cprior = cal.engine->compute_blank_logits(Type::Choice, 3);
    corrected = ch.raw_logits;
    for (size_t i = 0; i < corrected.size(); ++i) corrected[i] -= (float)(a.choice_prior_alpha * (double)cprior[i]);
    temperature_softmax(corrected, a.choice_temperature, probs);
    CHECK(ch.ok && ch.routing->strategy == routing::kNearest2Shot && ch.probs == probs,
          "applied: choice 2-shot uses the generated choice parameters");
}

static void test_failures(const Fixture& f, const Fixture& degenerate)
{
    const std::string dir = "test_calibrate_out/fail";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string out_path = dir + "/calibration.json";
    std::ofstream(out_path, std::ios::binary) << kPrevious;

    // Insufficient data is reported, never weakened; nothing is written.
    GeneratorOptions small = options();
    small.max_samples = 60;
    Pipeline p(f);
    GenerateOutcome r = p.run(out_path, small);
    CHECK(!r.ok && r.error.find("insufficient calibration data") != std::string::npos && p.backend.prompts.empty(),
          "insufficient: fails before any inference");
    CHECK(read_text(out_path) == kPrevious && !fs::exists(report_path_for(out_path, true)) && no_temp_files(dir),
          "insufficient: existing calibration untouched, nothing written");

    // Degenerate labels: every choice answer is the first option.
    Pipeline d(degenerate);
    r = d.run(out_path, options());
    CHECK(!r.ok && r.error.find("degenerate choice data") != std::string::npos && read_text(out_path) == kPrevious,
          "degenerate: single-label data rejected, existing calibration untouched");

    // Validation rejection: a model that is right on the fitting groups and wrong on the validation
    // groups (an overconfident fit) never replaces the existing file.
    Pipeline v(f);
    v.backend.boost = 6.0f;
    {
        // The validation groups, from the same deterministic partition.
        Partition part;
        std::string err;
        std::vector<retrieval::ExemplarRow> rows;
        v.store.scan(rows, err);
        partition_rows(rows, [&](int64_t id) { return v.store.is_allowed(id); },
                       [](const retrieval::ExemplarRow&, std::string&) { return retrieval::LoadResult::Ok; }, options(), part, err);
        for (const auto& q : part.queries)
            if (q.split == Split::Validation) v.backend.inverted.insert(f.group_of.at(q.id));
    }
    r = v.run(out_path, options());
    CHECK(!r.ok && r.error.find("calibration rejected") != std::string::npos && read_text(out_path) == kPrevious,
          "validation: a calibration that worsens validation NLL is rejected; existing file untouched");
    const std::string failed = report_path_for(out_path, false);
    CHECK(r.report_path == failed && fs::exists(failed) &&
          nlohmann::json::parse(read_text(failed))["status"] == "rejected" && no_temp_files(dir),
          "validation: failed report written, no accepted report");
}

int main()
{
    const Fixture f = build_fixture("test_calibrate_fixture");
    const Fixture degenerate = build_fixture("test_calibrate_degenerate", true);

    test_generate(f);
    test_failures(f, degenerate);

    fs::remove_all("test_calibrate_out");
    for (const Fixture* x : {&f, &degenerate}) {
        remove_db(x->db);
        std::remove(x->allow.c_str());
    }
    if (g_failures == 0) {
        printf("\nPASS (0 failure(s))\n");
        return 0;
    }
    fprintf(stderr, "\nFAIL (%d failure(s))\n", g_failures);
    return 1;
}
