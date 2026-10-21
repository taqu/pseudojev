// pjev-r4: R4 nearest 2-shot, embedding kNN classification and task-aware routing.
//
//   --mode run --partition tuning|evaluation
//                     For every query of the partition, in one session: S1, nearest 1-shot (R1), nearest
//                     2-shot in orders A and B, random 1-shot (seed 42) and the top-5 neighbour list used by
//                     kNN. Every component is timed. S1 / R1 are checked against the reference R1 run on
//                     every query (and random 1-shot against R3 seed 42 on the evaluation partition).
//                     The evaluation partition refuses to run without results/r4/frozen_policy.json.
//   --mode tune       Tuning partition only: choose the 2-shot order, kNN voting, routing finalists;
//                     write frozen_policy.json + .sha256.
//   --mode evaluate   Apply the frozen configuration to the evaluation run; write the report.
//   --mode policy-timing
//                     Execute the frozen finalist policies end to end (live) on the first N evaluation
//                     queries and record wall time and predictions, to check the offline cost model.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "decision/decision_engine.h"
#include "embedding/embedding.h"
#include "inference/llama_backend.h"
#include "inference/llama_instance.h"
#include "model/model_identity.h"

#include "embedding_text.h"
#include "exemplar_record.h"
#include "id_hash.h"
#include "r0_retrieval.h"
#include "r1_metrics.h"
#include "r1_output.h"
#include "r1_task.h"
#include "r2_offline.h"
#include "r3_select.h"
#include "r4_core.h"
#include "r4_eval.h"
#include "vec_autoext.h"

#include "sqlite/sqlite3.h"

#include "nlohmann/json.hpp"

using namespace pjev;
using namespace pjev::r1;
using namespace pjev::filldb;
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::string iso_timestamp()
{
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

static std::string hex_id(int64_t id)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%016" PRIx64, static_cast<uint64_t>(id));
    return buf;
}

static std::string dump(const json& j)
{
    return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

static void print_usage(const char* prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s --mode run --partition tuning|evaluation [options]\n"
        "  %s --mode tune [--dir results/r4]\n"
        "  %s --mode evaluate [--dir results/r4] [--bootstrap 2000] [--seed 20261020]\n"
        "  %s --mode policy-timing [--limit 300] [options]\n"
        "\n"
        "Run options:\n"
        "  --input PATH            Query dataset (default: datasets/validation.jsonl)\n"
        "  --db PATH               Exemplar database, read-only (default: models/examplar.db)\n"
        "  --train PATH            Training split allowlist (default: datasets/train.jsonl)\n"
        "  --embedding-model PATH  (default: models/nomic-embed-text-v1.5.gguf)\n"
        "  --model PATH            Inference model (default: models/bonsai2_2b.gguf)\n"
        "  --reference PATH        Reference R1 predictions (default: results/r1-2b/predictions.jsonl)\n"
        "  --split-assignments PATH Saved R2 partition (default: results/r2/predictions.jsonl)\n"
        "  --r3-seed42 PATH        R3 seed-42 predictions, checked on the evaluation partition\n"
        "                          (default: results/r3-random-1shot/seed-42/predictions.jsonl)\n"
        "  --dir DIR               R4 result root (default: results/r4)\n"
        "  --limit N               First N queries only (smoke test; preliminary)\n"
        "  --output DIR            Override the run output directory (default: <dir>/<partition>)\n"
        "  --ctx-size N            Inference context (default 4096; must match the reference)\n",
        prog, prog, prog, prog);
}

struct Query {
    std::string    id;
    std::string    group_id;
    ExemplarRecord rec;
    TaskMapping    task;
    int            ground_truth = -1;
    uint64_t       content = 0;
    std::vector<std::string> labels, texts;
};

// One neighbour as the runtime builds it.
struct LiveNeighbour {
    int64_t       id = 0;
    double        hamming = 0.0, similarity = 0.0;
    r4::LabelMap  map, text_map;
    double        cum_load_ms = 0.0;        // row fetch + LZ4 + parse, cumulative over inspected candidates
    double        cum_decompress_ms = 0.0;  // the LZ4 + parse part of cum_load_ms
    bool          same_group = false;
    std::string   answer_text;
    Demonstration demo;
};

struct NeighbourBuild {
    std::string status = "not_run";   // ok | payload_error | eligibility_violation
    std::string error;
    std::vector<LiveNeighbour> nb;
    int duplicates_skipped = 0, self_skipped = 0, inspected = 0;
    bool query_content_in_bank = false;   // a candidate had the query's content (leakage diagnostic)
};

// Models, database and the deployable operations of every action.
class Runtime
{
public:
    std::string db_path, train_path, emb_model_path, model_path;
    int ctx_size = 4096;

    std::unordered_set<int64_t> train_ids, r1_allow;
    r3::ExemplarPool pool;
    r3::PoolStats    pstats;
    int64_t kind_metadata_mismatch = 0;

    std::unique_ptr<r0::R0Reader>   reader;
    std::unique_ptr<Embedding>      embedding;
    std::unique_ptr<LlamaBackend>   backend;
    std::unique_ptr<DecisionEngine> engine;
    std::unique_ptr<PromptStrategy> strategy;
    LlamaInstance llama_guard;
    int emb_dim = 0;
    std::string emb_fp;
    ModelIdentity model_id;

    bool load_allowlist(std::string& err)
    {
        std::ifstream tf(train_path);
        if (!tf.is_open()) { err = "cannot open training split: " + train_path; return false; }
        std::string line;
        while (std::getline(tf, line)) {
            if (line.empty()) continue;
            try {
                auto j = json::parse(line);
                if (j.value("split", "") == "train" && j.contains("id") && j["id"].is_string())
                    train_ids.insert(hash_exemplar_id(j["id"].get<std::string>()));
            } catch (...) {}
        }
        // Eligible pool for random 1-shot: the same scan as pjev-r3.
        register_sqlite_vec_autoext();
        std::vector<r3::PoolRow> rows;
        sqlite3* db = nullptr;
        if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
            err = "cannot open database read-only: " + db_path;
            return false;
        }
        sqlite3_stmt* st = nullptr;
        const char* sql = "SELECT v.rowid, e.id, v.kind, v.number_of_levels, e.answer_index, e.kind, e.number_of_levels "
                          "FROM exemplar_vectors v JOIN examples e ON e.rowid = v.rowid";
        if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) {
            err = std::string("pool scan failed: ") + sqlite3_errmsg(db);
            sqlite3_close(db);
            return false;
        }
        while (sqlite3_step(st) == SQLITE_ROW) {
            r3::PoolRow r;
            r.id = sqlite3_column_int64(st, 1);
            const char* k = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
            r.kind = k ? k : "";
            r.number_of_levels = sqlite3_column_int(st, 3);
            r.answer_index = sqlite3_column_type(st, 4) == SQLITE_NULL ? -1 : sqlite3_column_int(st, 4);
            const char* ek = reinterpret_cast<const char*>(sqlite3_column_text(st, 5));
            if ((ek ? ek : "") != r.kind || (r.kind == "score" && sqlite3_column_int(st, 6) != r.number_of_levels))
                kind_metadata_mismatch++;
            rows.push_back(std::move(r));
        }
        sqlite3_finalize(st);
        sqlite3_close(db);
        pool = r3::ExemplarPool::build(rows, train_ids, &pstats);
        r1_allow = train_ids;
        for (const auto& r : rows) if (r.answer_index < 0) r1_allow.erase(r.id);
        return true;
    }

    bool load_models(std::string& err)
    {
        reader = std::make_unique<r0::R0Reader>(db_path);
        if (!reader->open()) { err = "cannot open database read-only: " + db_path; return false; }
        EmbeddingConfig emb_cfg;
        emb_cfg.model_path = emb_model_path;
        embedding = std::make_unique<Embedding>(emb_cfg);
        emb_dim = embedding->dimension();
        ModelIdentity emb_id = ModelIdentity::from_path(emb_model_path, nullptr);
        emb_fp = emb_id.sha256.size() >= 16 ? emb_id.sha256.substr(0, 16) : "";
        if (!reader->check_compatible(emb_dim, emb_fp, err)) return false;

        LlamaConfig llama_cfg;
        llama_cfg.model_path = model_path;
        llama_cfg.n_ctx = llama_cfg.n_batch = ctx_size;
        backend  = std::make_unique<LlamaBackend>(llama_cfg);
        model_id = ModelIdentity::from_path(model_path, backend.get());
        // Decision configuration: identical to pjev-r1.
        PromptConfig prompt_cfg;
        prompt_cfg.layout = Layout::AUTO;
        prompt_cfg.scheme = Scheme::LETTERS;
        EnsembleConfig ensemble;
        ensemble.noul_mode   = NoulEnsembleMode::NONE;
        ensemble.choice_mode = ChoiceEnsembleMode::NONE;
        ensemble.score_mode  = ScoreEnsembleMode::NONE;
        engine   = std::make_unique<DecisionEngine>(*backend, prompt_cfg, CalibrationConfig{}, ensemble);
        strategy = std::make_unique<PromptStrategy>(prompt_cfg);
        return true;
    }

    // --- Stages ---

    bool embed(const Query& q, std::vector<float>& vec, double& ms)
    {
        const auto t = Clock::now();
        const bool ok = embedding->encode(vec, std::string(r0::kQueryPrefix) +
            build_embedding_text(q.rec.kind, q.rec.domain, q.rec.family, q.rec.question, q.rec.state, q.rec.options));
        ms = ms_since(t);
        return ok;
    }

    // R1's retrieval call, unchanged.
    bool search(const Query& q, const std::vector<float>& vec, int top_k, r0::QueryResult& qr, double& ms, int& fetched_k)
    {
        const auto t = Clock::now();
        const int64_t self_id = q.rec.id;
        const bool ok = reader->search_filtered(q.rec, vec, top_k,
            [&](int64_t id) { return id != self_id && r1_allow.count(id) > 0; },
            r0::R0Reader::kMaxFetchK, qr, &fetched_k);
        ms = ms_since(t);
        return ok;
    }

    // Neighbour list (see r4::merge_neighbours): candidates = R1 nearest, then the top-K results.
    // Payloads are loaded lazily in candidate order until `need` neighbours are picked.
    NeighbourBuild neighbours(const Query& q, const std::vector<r0::RetrievedExample>& cands, size_t need)
    {
        NeighbourBuild b;
        std::unordered_set<int64_t>  ids;
        std::unordered_set<uint64_t> contents;
        double cum = 0.0, cum_dec = 0.0;
        for (const auto& c : cands) {
            if (b.nb.size() >= need) break;
            if (!ids.insert(c.example_id).second) continue;
            if (c.example_id == q.rec.id) { b.self_skipped++; continue; }
            r0::ExemplarPayload p;
            std::string err;
            const auto t = Clock::now();
            const bool loaded = reader->load_payload(c.example_id, p, err);
            cum += ms_since(t);
            if (loaded) cum_dec += p.decompress_ms;
            b.inspected++;
            LiveNeighbour n;
            if (!loaded || !build_demonstration(p, n.demo, err)) { b.status = "payload_error"; b.error = err; return b; }
            if (!r4::exemplar_eligible(p.example_id, p.kind, p.number_of_levels, q.rec.id, q.rec.kind,
                                       q.rec.number_of_levels, r1_allow)) {
                b.status = "eligibility_violation";  // unreachable through R1's filter
                return b;
            }
            const uint64_t ck = r4::content_key(p.kind, p.question, state_text(p.state), p.options);
            if (ck == q.content) { b.self_skipped++; b.query_content_in_bank = true; continue; }
            if (!contents.insert(ck).second) { b.duplicates_skipped++; continue; }
            n.id          = c.example_id;
            n.hamming     = c.distance;
            n.similarity  = estimated_cosine(c.distance, emb_dim);
            n.map         = r4::map_label(q.rec.kind, q.rec.options, q.task.option_to_candidate, p.kind, p.options, p.answer_index);
            if (q.rec.kind == "choice")
                n.text_map = r4::map_label_text_identity(q.rec.options, q.task.option_to_candidate, p.options, p.answer_index);
            n.cum_load_ms = cum;
            n.cum_decompress_ms = cum_dec;
            n.same_group  = p.group_id == q.rec.group_id;
            n.answer_text = p.options[static_cast<size_t>(p.answer_index)];
            b.nb.push_back(std::move(n));
        }
        b.status = "ok";
        return b;
    }

    ConditionResult s1(const Query& q)
    {
        ConditionResult c;
        const auto t = Clock::now();
        DecisionOutput o = engine->decide(q.task.input);
        fill_condition(o, q.labels, q.texts, q.ground_truth, ms_since(t), c);
        return c;
    }

    ConditionResult few_shot(const Query& q, const std::vector<Demonstration>& demos)
    {
        ConditionResult c;
        const auto t = Clock::now();
        DecisionOutput o = demos.size() == 1 ? engine->decide_with_demonstration(q.task.input, demos[0])
                                             : engine->decide_with_demonstrations(q.task.input, demos);
        fill_condition(o, q.labels, q.texts, q.ground_truth, ms_since(t), c);
        return c;
    }

    struct RandomRun {
        bool ok = false;
        std::string status;
        int64_t id = 0, pool_size = 0, index = -1;
        double select_ms = 0.0, load_ms = 0.0, decompress_ms = 0.0;
        ConditionResult cond;
    };
    RandomRun random_1shot(const Query& q)
    {
        RandomRun r;
        const auto ts = Clock::now();
        const auto pick = r3::select_random(pool.candidates(r3::pool_key(q.rec.kind, q.rec.number_of_levels)),
                                            q.rec.id, r4::kRandomSeed, q.id);
        r.select_ms = ms_since(ts);
        if (!pick.ok) { r.status = "no_eligible_exemplar"; return r; }
        r.id = pick.id; r.pool_size = pick.pool_size; r.index = pick.index;
        r0::ExemplarPayload p;
        Demonstration d;
        std::string err;
        const auto tl = Clock::now();
        const bool loaded = reader->load_payload(pick.id, p, err) && build_demonstration(p, d, err);
        r.load_ms = ms_since(tl);
        r.decompress_ms = p.decompress_ms;
        if (!loaded) { r.status = "payload_error"; return r; }
        r.cond = few_shot(q, {d});
        r.ok = r.cond.ok();
        r.status = r.cond.status;
        return r;
    }

    // kNN vote from live neighbours (timed by the caller).
    static r4::VoteResult vote(const NeighbourBuild& b, const r4::KnnParams& p)
    {
        std::vector<r4::VoteInput> v;
        for (const auto& n : b.nb) v.push_back({n.map.status == r4::MapStatus::Ok ? n.map.candidate : -1, n.similarity});
        return r4::vote(v, p.k, p.voting, p.tau);
    }
};

struct Common {
    std::string input_path = "datasets/validation.jsonl";
    std::string reference_path = "results/r1-2b/predictions.jsonl";
    std::string split_path = "results/r2/predictions.jsonl";
    std::string r3_path = "results/r3-random-1shot/seed-42/predictions.jsonl";
    std::string dir = "results/r4";
    std::string output_dir;
    std::string partition;
    int limit = -1;
    int bootstrap = 2000;
    uint64_t seed = 20261020;
};

static bool parse_common(int argc, char** argv, Common& o, Runtime& rt)
{
    rt.db_path = "models/examplar.db";
    rt.train_path = "datasets/train.jsonl";
    rt.emb_model_path = "models/nomic-embed-text-v1.5.gguf";
    rt.model_path = "models/bonsai2_2b.gguf";
    for (int i = 1; i < argc; ++i) {
        auto arg  = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s requires an argument\n", flag); exit(1); }
            return argv[++i];
        };
        if (arg("--mode")) next("--mode");
        else if (arg("--partition"))        o.partition       = next("--partition");
        else if (arg("--input"))            o.input_path      = next("--input");
        else if (arg("--db"))               rt.db_path        = next("--db");
        else if (arg("--train"))            rt.train_path     = next("--train");
        else if (arg("--embedding-model"))  rt.emb_model_path = next("--embedding-model");
        else if (arg("--model"))            rt.model_path     = next("--model");
        else if (arg("--reference"))        o.reference_path  = next("--reference");
        else if (arg("--split-assignments")) o.split_path     = next("--split-assignments");
        else if (arg("--r3-seed42"))        o.r3_path         = next("--r3-seed42");
        else if (arg("--dir"))              o.dir             = next("--dir");
        else if (arg("--output"))           o.output_dir      = next("--output");
        else if (arg("--limit"))            o.limit           = atoi(next("--limit"));
        else if (arg("--ctx-size"))         rt.ctx_size       = atoi(next("--ctx-size"));
        else if (arg("--bootstrap"))        o.bootstrap       = atoi(next("--bootstrap"));
        else if (arg("--seed"))             o.seed            = std::strtoull(next("--seed"), nullptr, 10);
        else { fprintf(stderr, "error: unknown argument: %s\n", argv[i]); print_usage(argv[0]); return false; }
    }
    return true;
}

// Reference predictions, recomputed partition (checked against the saved one) and the partition's queries.
struct Setup {
    json ref_config;
    std::string ref_sha;
    std::unordered_map<std::string, json> reference;
    r2::PartitionInfo pinfo;
    int saved_eval = 0, saved_tuning = 0, partition_disagreements = 0, rejected_train = 0;
    size_t partition_size = 0;
    std::vector<Query> queries;
};

static bool setup(const Common& o, Runtime& rt, r2::Partition want, Setup& s)
{
    const std::string ref_dir = std::filesystem::path(o.reference_path).parent_path().string();
    {
        std::ifstream cf(ref_dir + "/config.json");
        if (!cf.is_open()) { fprintf(stderr, "error: reference has no config.json\n"); return false; }
        s.ref_config = json::parse(cf);
    }
    if (s.ref_config.value("preliminary", true)) { fprintf(stderr, "error: reference R1 run is preliminary\n"); return false; }
    s.ref_sha = sha256_file(o.reference_path);
    {
        std::ifstream rf(o.reference_path, std::ios::binary);
        std::string line;
        while (std::getline(rf, line)) {
            if (line.empty()) continue;
            json j = json::parse(line);
            std::string id = j["query_id"].get<std::string>();
            s.reference[id] = std::move(j);
        }
    }
    std::unordered_map<std::string, std::string> group_of;
    std::set<std::string> want_ids;
    {
        std::vector<r2::Record> recs;
        r2::LoadStats ls;
        std::string err;
        if (!r2::load_predictions(o.reference_path, 768, recs, ls, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return false; }
        if (!r2::apply_partition(o.input_path, recs, s.pinfo, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return false; }
        std::unordered_map<std::string, r2::Partition> part;
        for (const auto& r : recs) {
            part[r.query_id] = r.partition;
            group_of[r.query_id] = r.group_id;
            if (r.partition == want) want_ids.insert(r.query_id);
        }
        std::ifstream sf(o.split_path, std::ios::binary);
        if (!sf.is_open()) { fprintf(stderr, "error: cannot open saved split assignments %s\n", o.split_path.c_str()); return false; }
        std::string line;
        std::set<std::string> seen;
        while (std::getline(sf, line)) {
            if (line.empty()) continue;
            json j = json::parse(line);
            const std::string id = j["query_id"].get<std::string>();
            const std::string p  = j.value("partition", "");
            seen.insert(id);
            (p == "evaluation" ? s.saved_eval : s.saved_tuning)++;
            auto it = part.find(id);
            if (it == part.end() || p != r2::to_string(it->second)) s.partition_disagreements++;
        }
        s.partition_disagreements += static_cast<int>(part.size()) - static_cast<int>(seen.size());
        printf("Partition: saved tuning=%d evaluation=%d; disagreements=%d; group overlap=%d; %s=%zu\n",
               s.saved_tuning, s.saved_eval, s.partition_disagreements, s.pinfo.group_overlap,
               r2::to_string(want), want_ids.size());
        const size_t expect = want == r2::Partition::Evaluation ? static_cast<size_t>(s.saved_eval) : static_cast<size_t>(s.saved_tuning);
        if (s.partition_disagreements != 0 || s.pinfo.group_overlap != 0 || s.pinfo.train_split_records != 0 ||
            want_ids.size() != expect) {
            fprintf(stderr, "error: the R2 partition cannot be reproduced exactly; refusing to run\n");
            return false;
        }
    }
    std::string err;
    if (!rt.load_allowlist(err)) { fprintf(stderr, "error: %s\n", err.c_str()); return false; }
    std::ifstream fin(o.input_path);
    if (!fin.is_open()) { fprintf(stderr, "error: cannot open input\n"); return false; }
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        Query q;
        try {
            json j = json::parse(line);
            if (!parse_exemplar(j, q.rec).empty()) continue;
            q.id = j["id"].get<std::string>();
        } catch (...) { continue; }
        if (!want_ids.count(q.id)) continue;
        if (q.rec.language != "en" || q.rec.answer_index < 0) continue;
        if (q.rec.split == "train" || rt.train_ids.count(q.rec.id)) { s.rejected_train++; continue; }
        if (!build_task(q.rec.kind, q.rec.question, q.rec.state, q.rec.options, q.task, err)) continue;
        q.ground_truth = q.task.option_to_candidate[static_cast<size_t>(q.rec.answer_index)];
        q.group_id = group_of[q.id];
        q.content  = r4::content_key(q.rec.kind, q.rec.question, state_text(q.rec.state), q.rec.options);
        s.queries.push_back(std::move(q));
    }
    if (s.queries.size() != want_ids.size()) {
        fprintf(stderr, "error: %zu queries rebuilt, expected %zu\n", s.queries.size(), want_ids.size());
        return false;
    }
    for (const auto& q : s.queries)
        if (!s.reference.count(q.id)) { fprintf(stderr, "error: %s missing from reference\n", q.id.c_str()); return false; }
    s.partition_size = s.queries.size();
    if (o.limit >= 0 && static_cast<size_t>(o.limit) < s.queries.size()) s.queries.resize(static_cast<size_t>(o.limit));
    return true;
}

// Identity configuration (same keys as pjev-r1) checked against the reference.
static json identity_config(const Runtime& rt, const Setup& s, const Common& o)
{
    json c;
    c["input_path"]            = o.input_path;
    c["input_sha256"]          = sha256_file(o.input_path);
    c["db_path"]               = rt.db_path;
    c["train_path"]            = rt.train_path;
    c["inference_model"]       = rt.model_path;
    c["inference_model_identity"] = rt.model_id.to_json();
    c["embedding_model"]       = rt.emb_model_path;
    c["embedding_fingerprint"] = rt.emb_fp;
    c["embedding_dim"]         = rt.emb_dim;
    c["db_info"]               = s.ref_config.value("db_info", json::object());
    c["query_prefix"]          = r0::kQueryPrefix;
    c["prompt_format_version"] = kPromptFormatVersion;
    c["prompt_format"]         = "S1 user turn for the exemplar, assistant turn = exemplar answer label, "
                                 "then the unchanged S1 query prompt";
    c["few_shot_prompt_format"] = "r1-demo-turn-v1 repeated: each demonstration is an S1 user turn followed by an "
                                  "assistant turn holding its answer label, in the stated order, then the unchanged "
                                  "S1 query prompt";
    c["scheme"]                = "letters";
    c["layout"]                = "auto";
    c["ensemble"]              = "none (single order)";
    c["calibration"]           = "none (temperature 1.0, no prior correction)";
    c["n_ctx"]                 = rt.ctx_size;
    c["top_k"]                 = 1;
    c["leakage_policy"]        = "demonstrations restricted to IDs whose split is 'train' in train_path; "
                                 "the query's own ID is rejected; query text never includes the target";
    c["max_fetch_k"]           = r0::R0Reader::kMaxFetchK;
    const json& rdi = s.ref_config["db_info"];
    const bool same_db = rdi.value("schema_version", "") == rt.reader->get_info("schema_version") &&
                         rdi.value("embedding_fingerprint", "") == rt.reader->get_info("embedding_fingerprint") &&
                         rdi.value("example_count", int64_t{-1}) == rt.reader->example_count();
    if (!same_db) c["db_info"] = json{{"schema_version", rt.reader->get_info("schema_version")}};
    return c;
}

static std::vector<float> logits_of(const json& j)
{
    std::vector<float> v;
    if (j.contains("logits")) for (const auto& x : j["logits"]) v.push_back(x.get<float>());
    return v;
}

static int run_mode(int argc, char** argv)
{
    Common o;
    Runtime rt;
    if (!parse_common(argc, argv, o, rt)) return 1;
    r2::Partition want;
    if (o.partition == "tuning") want = r2::Partition::Tuning;
    else if (o.partition == "evaluation") want = r2::Partition::Evaluation;
    else { fprintf(stderr, "error: --partition must be tuning or evaluation\n"); return 1; }
    const bool eval_stage = want == r2::Partition::Evaluation;
    if (o.output_dir.empty()) o.output_dir = o.dir + "/" + o.partition;

    // Stage separation: the evaluation partition is only run once a frozen configuration exists.
    r4::FrozenPolicy frozen;
    if (eval_stage) {
        std::string err;
        if (!r4::load_frozen(o.dir + "/frozen_policy.json", o.dir + "/frozen_policy.sha256", frozen, err)) {
            fprintf(stderr, "error: evaluation run requires the frozen configuration: %s\n", err.c_str());
            return 1;
        }
        printf("Frozen configuration %s (sha256 %s)\n", (o.dir + "/frozen_policy.json").c_str(), frozen.sha256.c_str());
    }

    const std::string run_timestamp = iso_timestamp();
    const auto t_run = Clock::now();
    Setup s;
    if (!setup(o, rt, want, s)) return 1;
    const bool preliminary = s.queries.size() < s.partition_size;
    printf("Queries: %zu of %zu%s\n", s.queries.size(), s.partition_size, preliminary ? " [preliminary]" : "");
    printf("Pool: %" PRId64 " eligible of %" PRId64 " vector rows\n", rt.pstats.eligible, rt.pstats.rows);

    // R3 seed-42 predictions (evaluation partition): the random 1-shot action must reproduce them.
    std::unordered_map<std::string, json> r3_ref;
    if (eval_stage) {
        std::ifstream rf(o.r3_path, std::ios::binary);
        std::string line;
        while (std::getline(rf, line)) {
            if (line.empty()) continue;
            json j = json::parse(line);
            std::string id = j["query_id"].get<std::string>();
            r3_ref[id] = std::move(j);
        }
        printf("R3 seed-42 reference: %zu records\n", r3_ref.size());
    }

    std::string err;
    try { if (!rt.load_models(err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; } }
    catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
    json c = identity_config(rt, s, o);
    const auto diffs = r2::config_differences(s.ref_config, c);
    if (!diffs.empty()) {
        fprintf(stderr, "error: configuration differs from the reference R1 run:");
        for (const auto& d : diffs) fprintf(stderr, " %s", d.c_str());
        fprintf(stderr, "\n");
        return 1;
    }
    printf("Configuration identical to the reference on all identity keys\n");

    for (auto& q : s.queries) {
        q.labels = candidate_labels(*rt.strategy, q.task.input.type, q.task.input.options.size());
        for (int oi : q.task.candidate_to_option) q.texts.push_back(q.rec.options[static_cast<size_t>(oi)]);
    }

    std::error_code ec;
    std::filesystem::create_directories(o.output_dir, ec);
    std::ofstream out(o.output_dir + "/queries.jsonl", std::ios::binary);
    if (!out.is_open()) { fprintf(stderr, "error: cannot write %s\n", o.output_dir.c_str()); return 1; }

    // Warm-up (untimed): embedding, search, S1, 1-shot and 2-shot on the first query.
    {
        const Query& q = s.queries.front();
        std::vector<float> v;
        double ms = 0;
        int fk = 0;
        rt.embed(q, v, ms);
        r0::QueryResult qr;
        rt.search(q, v, r4::kSearchK, qr, ms, fk);
        rt.s1(q);
        auto b = rt.neighbours(q, qr.retrieved, 2);
        if (b.nb.size() >= 2) { rt.few_shot(q, {b.nb[0].demo}); rt.few_shot(q, {b.nb[1].demo, b.nb[0].demo}); }
    }

    struct Counters {
        int s1_mismatch = 0, r1_mismatch = 0, nearest_mismatch = 0, topk_rank1_differs = 0;
        int random_checked = 0, random_mismatch = 0;
        int retrieval_fail = 0, short_lists = 0, dup_skipped = 0, self_skipped = 0, query_content_in_bank = 0;
        int same_group_neighbours = 0;
        std::map<std::string, std::map<std::string, int>> status;  // condition -> status -> count
        std::map<std::string, std::map<std::string, int>> context_limit;  // condition -> kind -> count
    } k;

    for (size_t qi = 0; qi < s.queries.size(); ++qi) {
        const Query& q = s.queries[qi];
        const json& ref = s.reference.at(q.id);
        json j;
        j["query_id"]   = q.id;
        j["query_hash"] = hex_id(q.rec.id);
        j["group_id"]   = q.group_id;
        j["partition"]  = o.partition;
        j["kind"]       = q.rec.kind;
        if (q.rec.kind == "score") j["number_of_levels"] = q.rec.number_of_levels;
        j["options_count"]      = static_cast<int>(q.task.input.options.size());
        j["option_labels"]      = q.labels;
        j["ground_truth_index"] = q.ground_truth;   // used only after every decision, for scoring

        // --- Retrieval: embedding, R1's top-1 search, top-K search, neighbour list ---
        json rj;
        std::vector<float> vec;
        double embed_ms = 0, s1_search_ms = 0, sk_ms = 0;
        int fk1 = 0, fkk = 0;
        r0::QueryResult q1, qk;
        NeighbourBuild nb;
        const bool emb_ok = rt.embed(q, vec, embed_ms);
        bool ok1 = emb_ok && rt.search(q, vec, 1, q1, s1_search_ms, fk1) && !q1.retrieved.empty();
        bool okk = emb_ok && rt.search(q, vec, r4::kSearchK, qk, sk_ms, fkk) && !qk.retrieved.empty();
        if (ok1 && okk) {
            if (qk.retrieved.front().example_id != q1.retrieved.front().example_id) k.topk_rank1_differs++;
            std::vector<r0::RetrievedExample> cands;
            cands.push_back(q1.retrieved.front());
            cands.insert(cands.end(), qk.retrieved.begin(), qk.retrieved.end());
            nb = rt.neighbours(q, cands, r4::kMaxNeighbours);
        }
        std::string rstatus = !emb_ok ? "embedding_error" : !(ok1 && okk) ? "no_eligible_exemplar" : nb.status;
        if (rstatus != "ok") k.retrieval_fail++;
        if (nb.nb.size() < static_cast<size_t>(r4::kMaxNeighbours) && rstatus == "ok") k.short_lists++;
        k.dup_skipped += nb.duplicates_skipped;
        k.self_skipped += nb.self_skipped;
        if (nb.query_content_in_bank) k.query_content_in_bank++;
        // Vote cost: the most expensive configuration (k=5, weighted), timed.
        double vote_ms = 0;
        {
            const auto t = Clock::now();
            volatile int sink = Runtime::vote(nb, {5, r4::Voting::Weighted, 0.01}).label;
            (void)sink;
            vote_ms = ms_since(t);
        }
        rj["status"]     = rstatus;
        if (!nb.error.empty()) rj["error"] = nb.error;
        rj["embed_ms"]   = embed_ms;
        rj["search1_ms"] = s1_search_ms;
        rj["search1_fetched_k"] = fk1;
        rj["searchk_ms"] = sk_ms;
        rj["searchk_fetched_k"] = fkk;
        rj["search_k"]   = r4::kSearchK;
        rj["topk_rank1_matches_r1"] = ok1 && okk && qk.retrieved.front().example_id == q1.retrieved.front().example_id;
        rj["duplicates_skipped"] = nb.duplicates_skipped;
        rj["self_skipped"] = nb.self_skipped;
        rj["query_content_in_bank"] = nb.query_content_in_bank;
        rj["payloads_inspected"] = nb.inspected;
        rj["vote_ms"]    = vote_ms;
        json arr = json::array();
        for (size_t i = 0; i < nb.nb.size(); ++i) {
            const auto& n = nb.nb[i];
            if (n.same_group) k.same_group_neighbours++;
            json x;
            x["rank"]             = static_cast<int>(i + 1);
            x["example_id"]       = hex_id(n.id);
            x["hamming_distance"] = n.hamming;
            x["estimated_cosine"] = n.similarity;
            x["map_status"]       = r4::to_string(n.map.status);
            x["label"]            = n.map.status == r4::MapStatus::Ok ? n.map.candidate : -1;
            if (q.rec.kind == "choice") {
                x["text_identity_status"] = r4::to_string(n.text_map.status);
                x["text_identity_label"]  = n.text_map.status == r4::MapStatus::Ok ? n.text_map.candidate : -1;
            }
            x["answer_label"]     = q.rec.kind == "choice" ? std::string() : (n.map.candidate >= 0 ? q.labels[static_cast<size_t>(n.map.candidate)] : "");
            x["demo_answer_index"] = n.demo.answer_index;
            x["answer_text"]      = n.answer_text;
            x["same_group"]       = n.same_group;
            x["cum_load_ms"]      = n.cum_load_ms;
            x["cum_decompress_ms"] = n.cum_decompress_ms;
            arr.push_back(std::move(x));
        }
        rj["neighbours"] = std::move(arr);
        j["retrieval"] = std::move(rj);

        // --- Decisions ---
        auto record = [&](const char* name, const ConditionResult& cr, json extra = json::object()) {
            json cj = condition_json(cr);
            for (auto& [kk, v] : extra.items()) cj[kk] = v;
            k.status[name][cr.status]++;
            if (cr.status == "context_limit") k.context_limit[name][q.rec.kind]++;
            j[name] = std::move(cj);
        };
        const ConditionResult s1c = rt.s1(q);
        {
            const bool m = s1c.ok() && s1c.logits == logits_of(ref["s1"]) && s1c.prompt_tokens == ref["s1"].value("prompt_tokens", -1);
            if (!m) k.s1_mismatch++;
            record("s1", s1c, {{"matches_reference", m}});
        }
        const bool have1 = rstatus == "ok" && !nb.nb.empty();
        const bool have2 = have1 && nb.nb.size() >= 2;
        {
            ConditionResult r1c;
            if (have1) r1c = rt.few_shot(q, {nb.nb[0].demo});
            const bool ex_m = have1 && hex_id(nb.nb[0].id) == ref["retrieval"].value("example_id", "");
            const bool m = ex_m && r1c.ok() && r1c.logits == logits_of(ref["r1"]);
            if (!ex_m) k.nearest_mismatch++;
            if (!m) k.r1_mismatch++;
            record("r1", r1c, {{"example_ids", have1 ? json::array({hex_id(nb.nb[0].id)}) : json::array()},
                               {"matches_reference", m}});
        }
        // Order A: second-nearest, nearest, query. Order B: nearest, second-nearest, query.
        {
            ConditionResult a, b;
            if (have2) {
                a = rt.few_shot(q, {nb.nb[1].demo, nb.nb[0].demo});
                b = rt.few_shot(q, {nb.nb[0].demo, nb.nb[1].demo});
            }
            record("two_shot_a", a, {{"example_ids", have2 ? json::array({hex_id(nb.nb[1].id), hex_id(nb.nb[0].id)}) : json::array()}});
            record("two_shot_b", b, {{"example_ids", have2 ? json::array({hex_id(nb.nb[0].id), hex_id(nb.nb[1].id)}) : json::array()}});
        }
        {
            const auto rr = rt.random_1shot(q);
            json extra{{"seed", r4::kRandomSeed}, {"select_ms", rr.select_ms}, {"load_ms", rr.load_ms},
                       {"decompress_ms", rr.decompress_ms},
                       {"example_ids", rr.id ? json::array({hex_id(rr.id)}) : json::array()},
                       {"pool_size", rr.pool_size}, {"pick_index", rr.index}};
            if (eval_stage) {
                auto it = r3_ref.find(q.id);
                bool m = false;
                if (it != r3_ref.end()) {
                    k.random_checked++;
                    m = it->second["exemplar"].value("example_id", "") == hex_id(rr.id) &&
                        rr.cond.ok() && rr.cond.logits == logits_of(it->second["r3"]);
                }
                if (!m) k.random_mismatch++;
                extra["matches_r3_seed42"] = m;
            }
            ConditionResult cr = rr.cond;
            if (!rr.ok && cr.status == "not_run") cr.status = rr.status;
            record("random_1shot", cr, extra);
        }
        j["completion_tokens"] = 0;
        out << dump(j) << "\n";
        out.flush();

        if ((qi + 1) % 25 == 0 || qi + 1 == s.queries.size()) {
            printf("  [%zu/%zu] elapsed %.0fs  mismatches s1=%d r1=%d nearest=%d random=%d\n", qi + 1, s.queries.size(),
                   ms_since(t_run) / 1000.0, k.s1_mismatch, k.r1_mismatch, k.nearest_mismatch, k.random_mismatch);
        }
    }

    c["experiment"]    = "R4 measurement run";
    c["partition"]     = o.partition;
    c["run_timestamp"] = run_timestamp;
    c["preliminary"]   = preliminary;
    c["frozen_policy_sha256"] = eval_stage ? json(frozen.sha256) : json(nullptr);
    json cond_status;
    for (const auto& [n, m] : k.status) cond_status[n] = m;
    json ctx;
    for (const auto& [n, m] : k.context_limit) ctx[n] = m;
    c["detail"] = {
        {"reference_predictions", o.reference_path},
        {"reference_predictions_sha256", s.ref_sha},
        {"split_assignments", o.split_path},
        {"split_assignments_sha256", sha256_file(o.split_path)},
        {"r3_seed42_predictions", eval_stage ? json(o.r3_path) : json(nullptr)},
        {"partition", {{"name", o.partition}, {"size", s.partition_size}, {"saved_evaluation", s.saved_eval},
                       {"saved_tuning", s.saved_tuning}, {"disagreements", s.partition_disagreements},
                       {"group_overlap", s.pinfo.group_overlap}, {"train_split_records", s.pinfo.train_split_records},
                       {"evaluation_families", s.pinfo.evaluation_families}, {"tuning_families", s.pinfo.tuning_families},
                       {"queries_rejected_as_training", s.rejected_train}}},
        {"evaluated_queries", s.queries.size()},
        {"pool_stats", {{"vector_rows", rt.pstats.rows}, {"eligible", rt.pstats.eligible},
                        {"outside_train", rt.pstats.not_in_train}, {"no_one_hot", rt.pstats.no_one_hot},
                        {"kind_metadata_mismatch", rt.kind_metadata_mismatch}}},
        {"search_k", r4::kSearchK},
        {"max_neighbours", r4::kMaxNeighbours},
        {"neighbour_rule", "rank 1 = R1's top-1 search result; ranks 2.. = top-K search order (distance, rowid), "
                           "skipping repeated IDs, the query itself (ID or identical content) and exemplars whose "
                           "content (kind, question, state, options) duplicates an earlier neighbour"},
        {"two_shot_orders", {{"A", "second-nearest, nearest, query"}, {"B", "nearest, second-nearest, query"}}},
        {"random_seed", r4::kRandomSeed},
        {"random_selection", "XXH64(\"r3-random-v1|42|<query_id>\") mod pool size (pjev-r3, seed 42)"},
        {"checks", {{"s1_reference_mismatch", k.s1_mismatch}, {"r1_reference_mismatch", k.r1_mismatch},
                    {"nearest_reference_mismatch", k.nearest_mismatch}, {"topk_rank1_differs_from_r1", k.topk_rank1_differs},
                    {"random_checked", k.random_checked}, {"random_r3_mismatch", k.random_mismatch},
                    {"retrieval_failures", k.retrieval_fail}, {"short_neighbour_lists", k.short_lists},
                    {"duplicates_skipped", k.dup_skipped}, {"self_skipped", k.self_skipped},
                    {"query_content_in_bank", k.query_content_in_bank},
                    {"same_group_neighbours", k.same_group_neighbours}}},
        {"condition_status", cond_status},
        {"context_limit_by_kind", ctx},
        {"latency_conditions", "models loaded once; one untimed warm-up of every stage; sequential; every decision "
                               "evaluates its full prompt from an empty KV cache; all actions of a query measured "
                               "back to back in this process"},
        {"wall_seconds", ms_since(t_run) / 1000.0}};
    {
        std::ofstream cf(o.output_dir + "/run_config.json", std::ios::binary);
        cf << c.dump(2, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
    }
    printf("\nChecks: S1 mismatches %d, R1 mismatches %d, nearest mismatches %d, top-K rank-1 differs %d, "
           "random vs R3 %d/%d mismatched, retrieval failures %d, short lists %d, duplicates skipped %d, "
           "query content found in bank %d\n", k.s1_mismatch, k.r1_mismatch, k.nearest_mismatch, k.topk_rank1_differs,
           k.random_mismatch, k.random_checked, k.retrieval_fail, k.short_lists, k.dup_skipped, k.query_content_in_bank);
    for (const auto& [n, m] : k.status) {
        printf("  %-13s", n.c_str());
        for (const auto& [st, cnt] : m) printf(" %s=%d", st.c_str(), cnt);
        printf("\n");
    }
    const bool clean = k.s1_mismatch == 0 && k.r1_mismatch == 0 && k.nearest_mismatch == 0 && k.random_mismatch == 0;
    return clean ? 0 : 2;
}

// ---------------------------------------------------------------------------
// Live end-to-end execution of the frozen finalist policies
// ---------------------------------------------------------------------------

struct LiveOutcome {
    int pred = -1;
    std::string terminal;
    int llm = 0, emb = 0;
};

static LiveOutcome live_action(Runtime& rt, const Query& q, r4::Action a, const r4::KnnConfig& knn,
                               const std::vector<float>* have_vec = nullptr)
{
    LiveOutcome lo;
    lo.terminal = r4::to_string(a);
    auto decide = [&](const ConditionResult& c) { lo.llm++; lo.pred = c.ok() ? c.prediction : -1; };
    switch (a) {
    case r4::Action::ZeroShot: decide(rt.s1(q)); return lo;
    case r4::Action::Random1: {
        const auto rr = rt.random_1shot(q);
        if (rr.status != "no_eligible_exemplar" && rr.status != "payload_error") lo.llm++;
        lo.pred = rr.ok ? rr.cond.prediction : -1;
        return lo;
    }
    default: break;
    }
    std::vector<float> vec;
    double ms = 0;
    int fk = 0;
    if (have_vec) vec = *have_vec;
    else { lo.emb++; if (!rt.embed(q, vec, ms)) return lo; }
    const int k = r4::is_knn(a) ? r4::knn_k(a) : (a == r4::Action::Nearest1 ? 1 : 2);
    const bool top_k = k > 1;
    // Deployable plan: one search (top-1 or top-K), as in the offline cost model. Offline, rank 1 is R1's
    // top-1 result; the measurement run reports whether the top-K search ever ranks another exemplar first.
    r0::QueryResult qr;
    if (!rt.search(q, vec, top_k ? r4::kSearchK : 1, qr, ms, fk) || qr.retrieved.empty()) return lo;
    const std::vector<r0::RetrievedExample>& cands = qr.retrieved;
    const NeighbourBuild b = rt.neighbours(q, cands, static_cast<size_t>(k));
    if (b.status != "ok" || b.nb.empty()) return lo;
    if (r4::is_knn(a)) {
        const auto v = Runtime::vote(b, knn.of(a));
        lo.pred = v.ok ? v.label : -1;
        return lo;
    }
    if (a == r4::Action::Nearest1 || b.nb.size() < 2) { decide(rt.few_shot(q, {b.nb[0].demo})); return lo; }
    const bool order_a = a == r4::Action::Nearest2A;
    const ConditionResult two = order_a ? rt.few_shot(q, {b.nb[1].demo, b.nb[0].demo}) : rt.few_shot(q, {b.nb[0].demo, b.nb[1].demo});
    if (two.status == "context_limit") { decide(rt.few_shot(q, {b.nb[0].demo})); lo.terminal = "nearest_1shot"; return lo; }
    decide(two);
    return lo;
}

static LiveOutcome live_rule(Runtime& rt, const Query& q, const r4::Rule& rule, const r4::KnnConfig& knn)
{
    using T = r4::Rule::Type;
    if (rule.type == T::Fixed) return live_action(rt, q, rule.action, knn);
    if (rule.type == T::MarginGate) {
        LiveOutcome lo;
        const ConditionResult s1c = rt.s1(q);
        lo.llm = 1;
        if (s1c.ok() && s1c.margin >= rule.threshold) { lo.pred = s1c.prediction; lo.terminal = "zero_shot"; return lo; }
        LiveOutcome f = live_action(rt, q, rule.fallback, knn);
        f.llm += 1;
        return f;
    }
    // Retrieval-first rules.
    LiveOutcome lo;
    std::vector<float> vec;
    double ms = 0;
    int fk = 0;
    lo.emb = 1;
    if (!rt.embed(q, vec, ms)) return lo;
    const int k = std::max(1, rule.gate_k);
    const bool two_fb = rule.fallback == r4::Action::Nearest2A || rule.fallback == r4::Action::Nearest2B;
    const size_t need = rule.type == T::SimGate ? 1 : std::max<size_t>(static_cast<size_t>(k), two_fb ? 2 : 1);
    r0::QueryResult qr;
    if (!rt.search(q, vec, need > 1 ? r4::kSearchK : 1, qr, ms, fk) || qr.retrieved.empty()) return lo;
    const std::vector<r0::RetrievedExample>& cands = qr.retrieved;
    if (rule.type == T::SimGate) {
        const double sim = estimated_cosine(qr.retrieved.front().distance, rt.emb_dim);
        if (sim >= rule.threshold) {
            const NeighbourBuild b = rt.neighbours(q, cands, 1);
            if (b.status != "ok" || b.nb.empty()) return lo;
            const ConditionResult c = rt.few_shot(q, {b.nb[0].demo});
            lo.llm = 1;
            lo.pred = c.ok() ? c.prediction : -1;
            lo.terminal = "nearest_1shot";
            return lo;
        }
        LiveOutcome f = live_action(rt, q, rule.fallback, knn);
        f.emb += 1;
        return f;
    }
    // KnnGate
    const NeighbourBuild b = rt.neighbours(q, cands, need);
    if (b.status != "ok" || b.nb.empty()) return lo;
    const auto v = Runtime::vote(b, {k, r4::Voting::Uniform, 0.0});
    if (v.ok && v.unanimous && b.nb[0].similarity >= rule.threshold) {
        lo.pred = v.label;
        lo.terminal = "knn_" + std::to_string(k);
        return lo;
    }
    lo.terminal = r4::to_string(rule.fallback);
    lo.llm = 1;
    ConditionResult c;
    if (rule.fallback == r4::Action::ZeroShot) c = rt.s1(q);
    else if (rule.fallback == r4::Action::Nearest1 || b.nb.size() < 2) c = rt.few_shot(q, {b.nb[0].demo});
    else {
        c = rule.fallback == r4::Action::Nearest2A ? rt.few_shot(q, {b.nb[1].demo, b.nb[0].demo})
                                                   : rt.few_shot(q, {b.nb[0].demo, b.nb[1].demo});
        if (c.status == "context_limit") c = rt.few_shot(q, {b.nb[0].demo});
    }
    lo.pred = c.ok() ? c.prediction : -1;
    return lo;
}

static int policy_timing_mode(int argc, char** argv)
{
    Common o;
    Runtime rt;
    if (!parse_common(argc, argv, o, rt)) return 1;
    if (o.limit < 0) o.limit = 300;
    r4::FrozenPolicy frozen;
    std::string err;
    if (!r4::load_frozen(o.dir + "/frozen_policy.json", o.dir + "/frozen_policy.sha256", frozen, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    Setup s;
    if (!setup(o, rt, r2::Partition::Evaluation, s)) return 1;
    try { if (!rt.load_models(err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; } }
    catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
    const auto diffs = r2::config_differences(s.ref_config, identity_config(rt, s, o));
    if (!diffs.empty()) { fprintf(stderr, "error: configuration differs from the reference\n"); return 1; }
    for (auto& q : s.queries) {
        q.labels = candidate_labels(*rt.strategy, q.task.input.type, q.task.input.options.size());
        for (int oi : q.task.candidate_to_option) q.texts.push_back(q.rec.options[static_cast<size_t>(oi)]);
    }
    // Policies to time: the finalists, plus the static all-action baselines that the report compares.
    std::vector<r4::Policy> policies = frozen.finalists;
    // Warm-up (untimed): every finalist once on the first query.
    for (const auto& p : policies) {
        auto it = p.per_kind.find(s.queries.front().rec.kind);
        if (it != p.per_kind.end()) live_rule(rt, s.queries.front(), it->second, frozen.knn);
    }
    std::ofstream out(o.dir + "/policy_timing.jsonl", std::ios::binary);
    const auto t_run = Clock::now();
    for (size_t qi = 0; qi < s.queries.size(); ++qi) {
        const Query& q = s.queries[qi];
        // Rotate the policy order per query so no policy always runs first.
        for (size_t pi = 0; pi < policies.size(); ++pi) {
            const auto& p = policies[(pi + qi) % policies.size()];
            auto it = p.per_kind.find(q.rec.kind);
            if (it == p.per_kind.end()) continue;
            const auto t = Clock::now();
            const LiveOutcome lo = live_rule(rt, q, it->second, frozen.knn);
            const double wall = ms_since(t);
            json j{{"query_id", q.id}, {"kind", q.rec.kind}, {"policy", p.name}, {"prediction_index", lo.pred},
                   {"terminal", lo.terminal}, {"llm_evaluations", lo.llm}, {"embedding_evaluations", lo.emb},
                   {"wall_ms", wall}};
            out << dump(j) << "\n";
        }
        out.flush();
        if ((qi + 1) % 25 == 0 || qi + 1 == s.queries.size())
            printf("  [%zu/%zu] elapsed %.0fs\n", qi + 1, s.queries.size(), ms_since(t_run) / 1000.0);
    }
    json meta{{"frozen_policy_sha256", frozen.sha256}, {"queries", s.queries.size()}, {"timestamp", iso_timestamp()},
              {"policies", [&] { json a = json::array(); for (const auto& p : policies) a.push_back(p.name); return a; }()},
              {"conditions", "live end-to-end execution of each policy's plan; models loaded once; untimed warm-up; "
                             "policy order rotated per query; first N evaluation queries"}};
    std::ofstream mf(o.dir + "/policy_timing_config.json", std::ios::binary);
    mf << meta.dump(2) << "\n";
    return 0;
}

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string mode;
    for (int i = 1; i + 1 < argc; ++i) if (strcmp(argv[i], "--mode") == 0) mode = argv[i + 1];
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) { print_usage(argv[0]); return 0; }
    try {
        if (mode == "run") return run_mode(argc, argv);
        if (mode == "policy-timing") return policy_timing_mode(argc, argv);
        if (mode == "tune" || mode == "evaluate") {
            Common o;
            Runtime rt;
            if (!parse_common(argc, argv, o, rt)) return 1;
            r4::EvalOptions eo;
            eo.dir = o.dir;
            eo.bootstrap = o.bootstrap;
            eo.seed = o.seed;
            return mode == "tune" ? r4::tune_main(eo) : r4::evaluate_main(eo);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    print_usage(argv[0]);
    return 1;
}
