// pjev-r3: R3 random 1-shot vs nearest 1-shot.
//
//   --mode run        For every held-out query of the R1/R2 evaluation partition and every predefined
//                     seed: pick one eligible training exemplar uniformly at random (same candidate
//                     population as R1's retriever) and run the unchanged R1 1-shot decision with it.
//                     S1 and R1 are reused from the reference R1 run; the first --verify-reference
//                     queries re-run S1 and nearest R1 to prove the reference is reproducible.
//   --mode aggregate  Combine the seed runs with the reference S1 / R1 predictions (r3_analysis).

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
#include <sstream>
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
#include "r3_analysis.h"
#include "r3_select.h"
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
        "  %s --mode run [options]\n"
        "  %s --mode aggregate [--output DIR] [--reference PRED.jsonl] [--split-dataset DATA.jsonl]\n"
        "                      [--bootstrap N] [--seed N]\n"
        "\n"
        "Run options:\n"
        "  --input PATH            Query dataset (default: datasets/validation.jsonl)\n"
        "  --db PATH               Exemplar database, read-only (default: models/examplar.db)\n"
        "  --train PATH            Training split allowlist (default: datasets/train.jsonl)\n"
        "  --embedding-model PATH  (default: models/nomic-embed-text-v1.5.gguf); used for the nearest\n"
        "                          reference check and similarity analysis only, never for R3 selection\n"
        "  --model PATH            Inference model (default: models/bonsai2_2b.gguf)\n"
        "  --reference PATH        Reference R1 predictions (default: results/r1-2b/predictions.jsonl)\n"
        "  --split-assignments PATH Saved R2 partition (default: results/r2/predictions.jsonl)\n"
        "  --selection MODE        random (R3, default) | length-matched (R3-L control: uniform within the\n"
        "                          --length-window exemplars closest in demonstration tokens to the nearest one)\n"
        "  --length-window N       R3-L window size (default 100)\n"
        "  --seeds LIST            Comma-separated seeds (default: the predefined ten; R3-L: the first five)\n"
        "  --verify-reference N    Re-run S1 and nearest R1 on the first N held-out queries (default 100)\n"
        "  --limit N               Evaluate only the first N held-out queries (smoke test; preliminary)\n"
        "  --ctx-size N            Inference context (default: 4096; must match the reference)\n"
        "  --output DIR            (default: results/r3-random-1shot, R3-L: results/r3-length-matched-1shot)\n",
        prog, prog);
}

// Predefined before any R3 result existed: the five required seeds, then five more.
static const std::vector<uint64_t> kDefaultSeeds = {42, 123, 456, 789, 2026, 7, 99, 314, 1000, 31337};

struct Query {
    std::string    id;
    ExemplarRecord rec;
    TaskMapping    task;
    int            ground_truth = -1;
};

// Hamming distance between a query embedding and one stored exemplar vector, computed by sqlite-vec
// with the same quantisation as the KNN query (vec_quantize_binary). Analysis only.
class HammingProbe
{
public:
    bool open(const std::string& db_path, std::string& err)
    {
        if (sqlite3_open_v2(db_path.c_str(), &db_, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
            err = "cannot open db";
            return false;
        }
        const char* sql = "SELECT vec_distance_hamming(embedding, vec_quantize_binary(?)) "
                          "FROM exemplar_vectors WHERE rowid = ?";
        if (sqlite3_prepare_v2(db_, sql, -1, &st_, nullptr) != SQLITE_OK) {
            err = sqlite3_errmsg(db_);
            return false;
        }
        return true;
    }
    ~HammingProbe()
    {
        if (st_) sqlite3_finalize(st_);
        if (db_) sqlite3_close(db_);
    }
    // Returns -1 on failure.
    double distance(const std::vector<float>& emb, int64_t rowid)
    {
        if (!st_) return -1;
        sqlite3_reset(st_);
        sqlite3_bind_blob(st_, 1, emb.data(), static_cast<int>(emb.size() * sizeof(float)), SQLITE_STATIC);
        sqlite3_bind_int64(st_, 2, rowid);
        double d = -1;
        if (sqlite3_step(st_) == SQLITE_ROW && sqlite3_column_type(st_, 0) != SQLITE_NULL) d = sqlite3_column_double(st_, 0);
        return d;
    }

private:
    sqlite3*      db_ = nullptr;
    sqlite3_stmt* st_ = nullptr;
};

static bool parse_seeds(const std::string& s, std::vector<uint64_t>& out)
{
    out.clear();
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) return false;
        char* end = nullptr;
        out.push_back(std::strtoull(tok.c_str(), &end, 10));
        if (!end || *end) return false;
    }
    std::vector<uint64_t> sorted = out;
    std::sort(sorted.begin(), sorted.end());
    return !out.empty() && std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
}

static int run_mode(int argc, char** argv)
{
    std::string input_path     = "datasets/validation.jsonl";
    std::string db_path        = "models/examplar.db";
    std::string train_path     = "datasets/train.jsonl";
    std::string emb_model_path = "models/nomic-embed-text-v1.5.gguf";
    std::string model_path     = "models/bonsai2_2b.gguf";
    std::string reference_path = "results/r1-2b/predictions.jsonl";
    std::string split_path     = "results/r2/predictions.jsonl";
    std::string output_dir;
    std::vector<uint64_t> seeds = kDefaultSeeds;
    bool seeds_default = true;
    int  verify_n = 100;
    int  limit    = -1;
    int  ctx_size = 4096;
    r3::ExampleSelection selection = r3::ExampleSelection::Random;
    int  length_window = 100;

    for (int i = 1; i < argc; ++i) {
        auto arg  = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s requires an argument\n", flag); exit(1); }
            return argv[++i];
        };
        if (arg("--mode"))                 { next("--mode"); }
        else if (arg("--input"))           input_path     = next("--input");
        else if (arg("--db"))              db_path        = next("--db");
        else if (arg("--train"))           train_path     = next("--train");
        else if (arg("--embedding-model")) emb_model_path = next("--embedding-model");
        else if (arg("--model"))           model_path     = next("--model");
        else if (arg("--reference"))       reference_path = next("--reference");
        else if (arg("--split-assignments")) split_path   = next("--split-assignments");
        else if (arg("--output"))          output_dir     = next("--output");
        else if (arg("--verify-reference")) verify_n      = atoi(next("--verify-reference"));
        else if (arg("--limit"))           limit          = atoi(next("--limit"));
        else if (arg("--ctx-size"))        ctx_size       = atoi(next("--ctx-size"));
        else if (arg("--seeds")) {
            if (!parse_seeds(next("--seeds"), seeds)) { fprintf(stderr, "error: --seeds must be distinct integers\n"); return 1; }
            seeds_default = false;
        } else if (arg("--selection")) {
            const std::string s = next("--selection");
            if (s == "random") selection = r3::ExampleSelection::Random;
            else if (s == "length-matched") selection = r3::ExampleSelection::LengthMatched;
            else { fprintf(stderr, "error: --selection must be random or length-matched\n"); return 1; }
        } else if (arg("--length-window")) {
            length_window = atoi(next("--length-window"));
        } else { fprintf(stderr, "error: unknown argument: %s\n", argv[i]); print_usage(argv[0]); return 1; }
    }
    const bool length_matched = selection == r3::ExampleSelection::LengthMatched;
    if (output_dir.empty()) output_dir = length_matched ? "results/r3-length-matched-1shot" : "results/r3-random-1shot";
    // R3-L is a secondary control: the five required seeds unless others are given.
    if (length_matched && seeds_default) seeds.assign(kDefaultSeeds.begin(), kDefaultSeeds.begin() + 5);
    if (length_matched && length_window < 1) { fprintf(stderr, "error: --length-window must be >= 1\n"); return 1; }

    const std::string run_timestamp = iso_timestamp();
    auto t_run = Clock::now();

    // --- Reference R1 run (read-only) ---
    const std::string ref_dir = std::filesystem::path(reference_path).parent_path().string();
    json ref_config;
    {
        std::ifstream cf(ref_dir + "/config.json");
        if (!cf.is_open()) { fprintf(stderr, "error: reference has no config.json: %s\n", ref_dir.c_str()); return 1; }
        ref_config = json::parse(cf);
    }
    if (ref_config.value("preliminary", true)) {
        fprintf(stderr, "error: reference R1 run is preliminary (subset); a full run is required\n");
        return 1;
    }
    const std::string ref_sha = sha256_file(reference_path);
    std::unordered_map<std::string, json> reference;
    {
        std::ifstream rf(reference_path, std::ios::binary);
        std::string line;
        while (std::getline(rf, line)) {
            if (line.empty()) continue;
            json j = json::parse(line);
            std::string id = j["query_id"].get<std::string>();
            reference[id] = std::move(j);
        }
    }
    printf("Reference: %s (%zu records, sha256 %s)\n", reference_path.c_str(), reference.size(), ref_sha.substr(0, 16).c_str());

    // --- Held-out partition: recomputed with the R2 implementation, then checked against the saved one ---
    std::set<std::string> eval_ids;
    r2::PartitionInfo pinfo;
    int saved_eval = 0, saved_tuning = 0, partition_disagreements = 0;
    {
        std::vector<r2::Record> recs;
        r2::LoadStats ls;
        std::string err;
        if (!r2::load_predictions(reference_path, 768, recs, ls, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        if (!r2::apply_partition(input_path, recs, pinfo, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
        std::unordered_map<std::string, r2::Partition> part;
        for (const auto& r : recs) {
            part[r.query_id] = r.partition;
            if (r.partition == r2::Partition::Evaluation) eval_ids.insert(r.query_id);
        }
        std::ifstream sf(split_path, std::ios::binary);
        if (!sf.is_open()) { fprintf(stderr, "error: cannot open saved split assignments %s\n", split_path.c_str()); return 1; }
        std::string line;
        std::set<std::string> seen;
        while (std::getline(sf, line)) {
            if (line.empty()) continue;
            json j = json::parse(line);
            const std::string id = j["query_id"].get<std::string>();
            const std::string p  = j.value("partition", "");
            seen.insert(id);
            (p == "evaluation" ? saved_eval : saved_tuning)++;
            auto it = part.find(id);
            if (it == part.end() || p != r2::to_string(it->second)) partition_disagreements++;
        }
        partition_disagreements += static_cast<int>(part.size()) - static_cast<int>(seen.size());
        printf("Partition: recomputed evaluation=%zu; saved tuning=%d evaluation=%d; disagreements=%d; "
               "group overlap=%d\n", eval_ids.size(), saved_tuning, saved_eval, partition_disagreements, pinfo.group_overlap);
        if (partition_disagreements != 0 || pinfo.group_overlap != 0 || pinfo.train_split_records != 0 ||
            static_cast<int>(eval_ids.size()) != saved_eval) {
            fprintf(stderr, "error: the held-out partition cannot be reproduced exactly; refusing a direct comparison\n");
            return 1;
        }
    }

    // --- Training allowlist (identical to pjev-r1) ---
    std::unordered_set<int64_t> train_ids;
    {
        std::ifstream tf(train_path);
        if (!tf.is_open()) { fprintf(stderr, "error: cannot open training split: %s\n", train_path.c_str()); return 1; }
        std::string line;
        while (std::getline(tf, line)) {
            if (line.empty()) continue;
            try {
                auto j = json::parse(line);
                if (j.value("split", "") == "train" && j.contains("id") && j["id"].is_string())
                    train_ids.insert(hash_exemplar_id(j["id"].get<std::string>()));
            } catch (...) {}
        }
    }

    // --- Eligible exemplar pool: the rows R1's KNN can return (vector index kind / level columns) ---
    register_sqlite_vec_autoext();
    std::vector<r3::PoolRow> pool_rows;
    std::unordered_map<int64_t, int64_t> id_to_rowid;
    int64_t kind_metadata_mismatch = 0;
    {
        sqlite3* db = nullptr;
        if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
            fprintf(stderr, "error: cannot open database read-only: %s\n", db_path.c_str());
            return 1;
        }
        sqlite3_stmt* st = nullptr;
        const char* sql = "SELECT v.rowid, e.id, v.kind, v.number_of_levels, e.answer_index, e.kind, e.number_of_levels "
                          "FROM exemplar_vectors v JOIN examples e ON e.rowid = v.rowid";
        if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) {
            fprintf(stderr, "error: pool scan failed: %s\n", sqlite3_errmsg(db));
            sqlite3_close(db);
            return 1;
        }
        while (sqlite3_step(st) == SQLITE_ROW) {
            r3::PoolRow r;
            const int64_t rowid = sqlite3_column_int64(st, 0);
            r.id   = sqlite3_column_int64(st, 1);
            const char* k = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
            r.kind = k ? k : "";
            r.number_of_levels = sqlite3_column_int(st, 3);
            r.answer_index = sqlite3_column_type(st, 4) == SQLITE_NULL ? -1 : sqlite3_column_int(st, 4);
            const char* ek = reinterpret_cast<const char*>(sqlite3_column_text(st, 5));
            if ((ek ? ek : "") != r.kind || (r.kind == "score" && sqlite3_column_int(st, 6) != r.number_of_levels))
                kind_metadata_mismatch++;
            id_to_rowid[r.id] = rowid;
            pool_rows.push_back(std::move(r));
        }
        sqlite3_finalize(st);
        sqlite3_close(db);
    }
    r3::PoolStats pstats;
    const r3::ExemplarPool pool = r3::ExemplarPool::build(pool_rows, train_ids, &pstats);
    printf("Pool: %" PRId64 " vector rows, %" PRId64 " eligible (%" PRId64 " outside train, %" PRId64
           " without one-hot), kind metadata mismatches %" PRId64 "\n",
           pstats.rows, pstats.eligible, pstats.not_in_train, pstats.no_one_hot, kind_metadata_mismatch);
    for (const auto& [key, ids] : pool.by_key()) printf("  pool %-8s %zu\n", key.c_str(), ids.size());

    // R1's leakage filter accepts exactly these IDs (train allowlist minus non-one-hot rows).
    std::unordered_set<int64_t> r1_allow = train_ids;
    for (const auto& r : pool_rows) if (r.answer_index < 0) r1_allow.erase(r.id);

    // --- Queries: same filters as pjev-r1, restricted to the held-out partition ---
    std::vector<Query> queries;
    int rejected_train = 0;
    {
        std::ifstream fin(input_path);
        if (!fin.is_open()) { fprintf(stderr, "error: cannot open input: %s\n", input_path.c_str()); return 1; }
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            Query q;
            json j;
            try {
                j = json::parse(line);
                if (!parse_exemplar(j, q.rec).empty()) continue;
                q.id = j["id"].get<std::string>();
            } catch (...) { continue; }
            if (!eval_ids.count(q.id)) continue;
            if (q.rec.language != "en" || q.rec.answer_index < 0) continue;
            if (q.rec.split == "train" || train_ids.count(q.rec.id)) { rejected_train++; continue; }
            std::string err;
            if (!build_task(q.rec.kind, q.rec.question, q.rec.state, q.rec.options, q.task, err)) continue;
            q.ground_truth = q.task.option_to_candidate[static_cast<size_t>(q.rec.answer_index)];
            queries.push_back(std::move(q));
        }
    }
    if (queries.size() != eval_ids.size()) {
        fprintf(stderr, "error: %zu held-out queries rebuilt, expected %zu\n", queries.size(), eval_ids.size());
        return 1;
    }
    for (const auto& q : queries) {
        if (!reference.count(q.id)) { fprintf(stderr, "error: %s missing from reference\n", q.id.c_str()); return 1; }
    }
    const size_t n_heldout = queries.size();
    const bool preliminary = limit >= 0 && static_cast<size_t>(limit) < queries.size();
    if (preliminary) queries.resize(static_cast<size_t>(limit));
    printf("Held-out queries: %zu (evaluating %zu)%s\n", n_heldout, queries.size(), preliminary ? " [preliminary]" : "");

    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);

    // --- Models ---
    LlamaInstance llama_guard;
    r0::R0Reader reader(db_path);
    if (!reader.open()) { fprintf(stderr, "error: cannot open database read-only: %s\n", db_path.c_str()); return 1; }
    EmbeddingConfig emb_cfg;
    emb_cfg.model_path = emb_model_path;
    std::unique_ptr<Embedding> embedding;
    try { embedding = std::make_unique<Embedding>(emb_cfg); }
    catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
    const int emb_dim = embedding->dimension();
    ModelIdentity emb_id = ModelIdentity::from_path(emb_model_path, nullptr);
    const std::string emb_fp = emb_id.sha256.size() >= 16 ? emb_id.sha256.substr(0, 16) : "";
    {
        std::string err;
        if (!reader.check_compatible(emb_dim, emb_fp, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    }
    HammingProbe probe;
    std::string probe_err;
    const bool probe_ok = probe.open(db_path, probe_err);
    if (!probe_ok) printf("warning: Hamming probe unavailable (%s); similarity analysis disabled\n", probe_err.c_str());

    LlamaConfig llama_cfg;
    llama_cfg.model_path = model_path;
    llama_cfg.n_ctx = llama_cfg.n_batch = ctx_size;
    std::unique_ptr<LlamaBackend> backend;
    try { backend = std::make_unique<LlamaBackend>(llama_cfg); }
    catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
    ModelIdentity model_id = ModelIdentity::from_path(model_path, backend.get());

    // Decision configuration: identical to pjev-r1.
    PromptConfig prompt_cfg;
    prompt_cfg.layout = Layout::AUTO;
    prompt_cfg.scheme = Scheme::LETTERS;
    EnsembleConfig ensemble;
    ensemble.noul_mode   = NoulEnsembleMode::NONE;
    ensemble.choice_mode = ChoiceEnsembleMode::NONE;
    ensemble.score_mode  = ScoreEnsembleMode::NONE;
    std::unique_ptr<DecisionEngine> engine;
    try { engine = std::make_unique<DecisionEngine>(*backend, prompt_cfg, CalibrationConfig{}, ensemble); }
    catch (const std::exception& e) { fprintf(stderr, "error: %s\n", e.what()); return 1; }
    PromptStrategy strategy(prompt_cfg);

    // --- Configuration and identity check against the reference ---
    json c;
    c["experiment"]            = length_matched ? "R3-L length-matched random 1-shot" : "R3 random 1-shot";
    c["run_timestamp"]         = run_timestamp;
    c["preliminary"]           = preliminary;
    c["input_path"]            = input_path;
    c["input_sha256"]          = sha256_file(input_path);
    c["db_path"]               = db_path;
    c["train_path"]            = train_path;
    c["inference_model"]       = model_path;
    c["inference_model_identity"] = model_id.to_json();
    c["embedding_model"]       = emb_model_path;
    c["embedding_fingerprint"] = emb_fp;
    c["embedding_dim"]         = emb_dim;
    c["db_info"]               = ref_config.value("db_info", json::object());  // checked below
    c["query_prefix"]          = r0::kQueryPrefix;
    c["prompt_format_version"] = kPromptFormatVersion;
    c["prompt_format"]         = "S1 user turn for the exemplar, assistant turn = exemplar answer label, "
                                 "then the unchanged S1 query prompt";
    c["scheme"]                = "letters";
    c["layout"]                = "auto";
    c["ensemble"]              = "none (single order)";
    c["calibration"]           = "none (temperature 1.0, no prior correction)";
    c["n_ctx"]                 = ctx_size;
    c["top_k"]                 = 1;
    c["leakage_policy"]        = "demonstrations restricted to IDs whose split is 'train' in train_path; "
                                 "the query's own ID is rejected; query text never includes the target";
    c["max_fetch_k"]           = r0::R0Reader::kMaxFetchK;
    {
        // db_info must describe the same database as the reference.
        const json& rdi = ref_config["db_info"];
        const bool same_db = rdi.value("schema_version", "") == reader.get_info("schema_version") &&
                             rdi.value("embedding_fingerprint", "") == reader.get_info("embedding_fingerprint") &&
                             rdi.value("example_count", int64_t{-1}) == reader.example_count();
        if (!same_db) c["db_info"] = json{{"schema_version", reader.get_info("schema_version")}};
    }
    const auto diffs = r2::config_differences(ref_config, c);
    if (!diffs.empty()) {
        fprintf(stderr, "error: configuration differs from the reference R1 run:");
        for (const auto& d : diffs) fprintf(stderr, " %s", d.c_str());
        fprintf(stderr, "\n");
        return 1;
    }
    printf("Configuration identical to the reference on all identity keys\n");

    // --- R3-L only: demonstration length of every eligible exemplar ---
    // Demonstration tokens = tokens(1-shot prompt) - tokens(S1 prompt) for one fixed query of the same pool,
    // tokenized segment by segment exactly as the decision engine does (BOS on the first trusted segment).
    auto count_tokens = [&](const std::vector<PromptSegment>& segs) {
        size_t n = 0;
        bool first = true;
        for (const auto& seg : segs) {
            n += backend->tokenize(seg.text, first && seg.trusted, seg.trusted).size();
            first = false;
        }
        return static_cast<int>(n);
    };
    auto demo_tokens = [&](const DecisionInput& in, const Demonstration& d) {
        return count_tokens(r1_prompt_segments(strategy, in, d)) - count_tokens(s1_prompt_segments(strategy, in));
    };
    std::map<std::string, const Query*> dummy_of;  // first held-out query of each pool key
    for (const auto& q : queries) dummy_of.emplace(r3::pool_key(q.rec.kind, q.rec.number_of_levels), &q);
    std::map<std::string, r3::LengthPool> length_pools;
    int64_t length_pool_failures = 0;
    if (length_matched) {
        const auto t0 = Clock::now();
        std::ofstream lf(output_dir + "/pool_demo_tokens.csv", std::ios::binary);
        lf << "pool_key,example_id,demo_tokens\n";
        for (const auto& [key, dq] : dummy_of) {
            std::vector<std::pair<int64_t, int>> id_tok;
            for (int64_t id : pool.candidates(key)) {
                r0::ExemplarPayload p;
                Demonstration d;
                std::string err;
                if (!reader.load_payload(id, p, err) || !build_demonstration(p, d, err)) { length_pool_failures++; continue; }
                const int t = demo_tokens(dq->task.input, d);
                id_tok.emplace_back(id, t);
                lf << key << "," << hex_id(id) << "," << t << "\n";
            }
            length_pools[key] = r3::LengthPool::build(id_tok);
            const auto& bl = length_pools[key].by_length;
            printf("  length pool %-8s %zu exemplars, demo tokens %d..%d\n", key.c_str(), id_tok.size(),
                   bl.empty() ? 0 : bl.front().first, bl.empty() ? 0 : bl.back().first);
        }
        printf("Demonstration lengths computed in %.0f s (%" PRId64 " exemplars could not be built)\n",
               ms_since(t0) / 1000.0, length_pool_failures);
    }
    // Target length for R3-L: the nearest exemplar's demonstration tokens in the reference run.
    auto length_target = [&](const Query& q) {
        const json& ref = reference.at(q.id);
        return ref["r1"].value("prompt_tokens", 0) - ref["s1"].value("prompt_tokens", 0);
    };

    struct Sel {
        bool    ok = false;
        int64_t id = 0, pool_size = 0, index = -1;
        int     target = 0, tokens = 0, window_lo = 0, window_hi = 0;
        int64_t window_size = 0;
    };
    auto draw = [&](const Query& q, uint64_t seed) {
        Sel s;
        const std::string key = r3::pool_key(q.rec.kind, q.rec.number_of_levels);
        if (!length_matched) {
            const auto p = r3::select_random(pool.candidates(key), q.rec.id, seed, q.id);
            s.ok = p.ok; s.id = p.id; s.pool_size = p.pool_size; s.index = p.index;
        } else {
            s.target = length_target(q);
            const auto p = r3::select_length_matched(length_pools[key], q.rec.id, s.target, length_window, seed, q.id);
            s.ok = p.ok; s.id = p.id; s.index = p.index; s.tokens = p.tokens;
            s.window_size = p.window_size; s.window_lo = p.window_lo; s.window_hi = p.window_hi;
            s.pool_size = static_cast<int64_t>(length_pools[key].by_length.size());
        }
        return s;
    };

    // --- Selections: fixed for every (seed, query) before any inference ---
    std::vector<std::vector<Sel>> picks(seeds.size(), std::vector<Sel>(queries.size()));
    int no_eligible = 0;
    {
        std::ofstream sel(output_dir + "/selections.jsonl", std::ios::binary);
        for (size_t qi = 0; qi < queries.size(); ++qi) {
            const Query& q = queries[qi];
            const std::string key = r3::pool_key(q.rec.kind, q.rec.number_of_levels);
            for (size_t si = 0; si < seeds.size(); ++si) {
                picks[si][qi] = draw(q, seeds[si]);
                const auto& p = picks[si][qi];
                if (!p.ok) no_eligible++;
                json j;
                j["query_id"]    = q.id;
                j["seed"]        = seeds[si];
                j["selection"]   = r3::to_string(selection);
                j["pool_key"]    = key;
                j["pool_size"]   = p.pool_size;
                j["status"]      = p.ok ? "ok" : "no_eligible_exemplar";
                if (p.ok) { j["pick_index"] = p.index; j["example_id"] = hex_id(p.id); }
                if (length_matched) {
                    j["length_target"] = p.target;
                    j["demo_tokens"]   = p.tokens;
                    j["window_size"]   = p.window_size;
                    j["window_tokens"] = {p.window_lo, p.window_hi};
                }
                sel << dump(j) << "\n";
            }
        }
    }
    printf("Selections written (%zu seeds x %zu queries, %d without an eligible exemplar)\n",
           seeds.size(), queries.size(), no_eligible);

    // Warm-up (untimed).
    {
        const Query& q = queries.front();
        std::vector<float> vec;
        embedding->encode(vec, std::string(r0::kQueryPrefix) +
            build_embedding_text(q.rec.kind, q.rec.domain, q.rec.family, q.rec.question, q.rec.state, q.rec.options));
        engine->decide(q.task.input);
    }

    // --- Output streams ---
    std::vector<std::ofstream> seed_out(seeds.size());
    for (size_t si = 0; si < seeds.size(); ++si) {
        const std::string dir = output_dir + "/seed-" + std::to_string(seeds[si]);
        std::filesystem::create_directories(dir, ec);
        seed_out[si].open(dir + "/predictions.jsonl", std::ios::binary);
        if (!seed_out[si].is_open()) { fprintf(stderr, "error: cannot write %s\n", dir.c_str()); return 1; }
    }
    std::ofstream verify_out(output_dir + "/reference_check.jsonl", std::ios::binary);

    int verify_done = 0, verify_s1_mismatch = 0, verify_r1_mismatch = 0, verify_example_mismatch = 0;
    int hamming_checked = 0, hamming_mismatch = 0, length_checked = 0, length_mismatch = 0;
    std::vector<int> r3_ok(seeds.size(), 0), r3_fail(seeds.size(), 0);
    double r3_seconds = 0.0;

    for (size_t qi = 0; qi < queries.size(); ++qi) {
        const Query& q = queries[qi];
        const DecisionInput& input = q.task.input;
        const json& ref = reference.at(q.id);
        const auto labels = candidate_labels(strategy, input.type, input.options.size());
        std::vector<std::string> texts;
        for (int oi : q.task.candidate_to_option) texts.push_back(q.rec.options[static_cast<size_t>(oi)]);

        // Query embedding: analysis (and the nearest check) only; R3 selection never uses it.
        std::vector<float> qvec;
        const auto te = Clock::now();
        const bool emb_ok = embedding->encode(qvec, std::string(r0::kQueryPrefix) +
            build_embedding_text(q.rec.kind, q.rec.domain, q.rec.family, q.rec.question, q.rec.state, q.rec.options));
        const double embed_ms = ms_since(te);

        // Reference check: S1 and nearest R1 re-run in this session must reproduce the reference.
        if (verify_done < verify_n) {
            json vj;
            vj["query_id"] = q.id;
            auto t0 = Clock::now();
            DecisionOutput s1_out = engine->decide(input);
            ConditionResult s1;
            fill_condition(s1_out, labels, texts, q.ground_truth, ms_since(t0), s1);
            std::vector<float> ref_s1;
            for (const auto& v : ref["s1"]["logits"]) ref_s1.push_back(v.get<float>());
            const bool s1_match = s1.ok() && s1.logits == ref_s1 && s1.prompt_tokens == ref["s1"].value("prompt_tokens", -1);
            if (!s1_match) verify_s1_mismatch++;

            r0::QueryResult qr;
            const auto ts = Clock::now();
            const int64_t self_id = q.rec.id;
            bool found = emb_ok && reader.search_filtered(q.rec, qvec, 1,
                [&](int64_t id) { return id != self_id && r1_allow.count(id) > 0; },
                r0::R0Reader::kMaxFetchK, qr) && !qr.retrieved.empty();
            const double search_ms = ms_since(ts);
            ConditionResult r1c;
            double load_ms = 0.0;
            std::string near_hex;
            if (found) {
                const auto& top = qr.retrieved.front();
                near_hex = hex_id(top.example_id);
                r0::ExemplarPayload p;
                Demonstration demo;
                std::string err;
                const auto tf = Clock::now();
                const bool loaded = reader.load_payload(top.example_id, p, err) && build_demonstration(p, demo, err);
                load_ms = ms_since(tf);
                if (loaded) {
                    const auto t1 = Clock::now();
                    DecisionOutput o = engine->decide_with_demonstration(input, demo);
                    fill_condition(o, labels, texts, q.ground_truth, ms_since(t1), r1c);
                    // Demonstration-length model used by R3-L: must equal R1's measured extra tokens, both with
                    // this query and with the pool's fixed reference query.
                    const int target = length_target(q);
                    const Query* dq = dummy_of.at(r3::pool_key(q.rec.kind, q.rec.number_of_levels));
                    length_checked++;
                    if (demo_tokens(input, demo) != target || demo_tokens(dq->task.input, demo) != target) length_mismatch++;
                }
                if (probe_ok) {
                    const double h = probe.distance(qvec, id_to_rowid[top.example_id]);
                    hamming_checked++;
                    if (h != top.distance) hamming_mismatch++;
                }
            }
            std::vector<float> ref_r1;
            for (const auto& v : ref["r1"]["logits"]) ref_r1.push_back(v.get<float>());
            const bool ex_match = near_hex == ref["retrieval"].value("example_id", "");
            const bool r1_match = r1c.ok() && r1c.logits == ref_r1;
            if (!ex_match) verify_example_mismatch++;
            if (!r1_match) verify_r1_mismatch++;
            vj["s1_match"]         = s1_match;
            vj["nearest_example_match"] = ex_match;
            vj["r1_match"]         = r1_match;
            vj["s1_ms"]            = s1.total_ms;
            vj["s1_prompt_tokens"] = s1.prompt_tokens;
            vj["embed_ms"]         = embed_ms;
            vj["search_ms"]        = search_ms;
            vj["load_ms"]          = load_ms;
            vj["r1_ms"]            = r1c.total_ms;
            vj["r1_prompt_tokens"] = r1c.prompt_tokens;
            verify_out << dump(vj) << "\n";
            verify_out.flush();
            verify_done++;
        }

        // R3: one random exemplar per seed through the unchanged R1 1-shot path.
        for (size_t si = 0; si < seeds.size(); ++si) {
            const uint64_t seed = seeds[si];
            const std::string key = r3::pool_key(q.rec.kind, q.rec.number_of_levels);
            const auto tsel = Clock::now();
            const Sel pick = draw(q, seed);
            const double select_ms = ms_since(tsel);

            json j;
            j["query_id"]   = q.id;
            j["query_hash"] = hex_id(q.rec.id);
            j["kind"]       = q.rec.kind;
            if (q.rec.kind == "score") j["number_of_levels"] = q.rec.number_of_levels;
            j["options_count"]      = static_cast<int>(input.options.size());
            j["option_labels"]      = labels;
            j["ground_truth_index"] = q.ground_truth;
            j["ground_truth_label"] = labels[static_cast<size_t>(q.ground_truth)];
            j["strategy"]           = length_matched ? "length_matched" : "random";
            j["seed"]               = seed;

            json ex;
            ConditionResult r3c;
            double fetch_ms = 0.0, decompress_ms = 0.0;
            if (!pick.ok || pick.id != picks[si][qi].id) {
                ex["status"] = pick.ok ? "selection_mismatch" : "no_eligible_exemplar";
            } else {
                r0::ExemplarPayload p;
                Demonstration demo;
                std::string err;
                const auto tf = Clock::now();
                const bool loaded = reader.load_payload(pick.id, p, err);
                const double load_ms = ms_since(tf);
                decompress_ms = p.decompress_ms;
                fetch_ms      = std::max(0.0, load_ms - p.decompress_ms);
                if (!loaded || !build_demonstration(p, demo, err)) {
                    ex["status"] = "payload_error";
                    ex["error"]  = err;
                } else if (p.example_id == q.rec.id || !r1_allow.count(p.example_id) || p.kind != q.rec.kind ||
                           (q.rec.kind == "score" && p.number_of_levels != q.rec.number_of_levels)) {
                    ex["status"] = "eligibility_violation";  // unreachable through the pool
                } else {
                    ex["status"]       = "ok";
                    ex["example_id"]   = hex_id(p.example_id);
                    ex["pool_key"]     = key;
                    ex["pool_size"]    = pick.pool_size;
                    ex["pick_index"]   = pick.index;
                    if (length_matched) {
                        ex["length_target"]   = pick.target;
                        ex["demo_tokens"]     = pick.tokens;
                        ex["window_size"]     = pick.window_size;
                        ex["window_tokens"]   = {pick.window_lo, pick.window_hi};
                        ex["same_as_nearest"] = hex_id(p.example_id) == ref["retrieval"].value("example_id", "");
                    }
                    ex["kind"]         = p.kind;
                    if (p.kind == "score") ex["number_of_levels"] = p.number_of_levels;
                    ex["domain"]       = p.domain;
                    ex["family"]       = p.family;
                    ex["same_group"]   = p.group_id == q.rec.group_id;
                    ex["same_domain"]  = !q.rec.domain.empty() && p.domain == q.rec.domain;
                    ex["same_family"]  = !q.rec.family.empty() && p.family == q.rec.family;
                    if (p.kind == "score") ex["same_option_texts"] = p.options == q.rec.options;
                    const auto dlabels = candidate_labels(strategy, demo.type, demo.options.size());
                    ex["answer_index"] = demo.answer_index;
                    ex["answer_label"] = dlabels[static_cast<size_t>(demo.answer_index)];
                    ex["answer_text"]  = demo.options[static_cast<size_t>(demo.answer_index)].second;
                    ex["options_count"] = static_cast<int>(demo.options.size());
                    if (probe_ok && emb_ok) {
                        const double h = probe.distance(qvec, id_to_rowid[p.example_id]);
                        if (h >= 0) {
                            ex["hamming_distance"] = h;
                            ex["estimated_cosine"] = estimated_cosine(h, emb_dim);
                        }
                    }
                    const auto t1 = Clock::now();
                    DecisionOutput o = engine->decide_with_demonstration(input, demo);
                    fill_condition(o, labels, texts, q.ground_truth, ms_since(t1), r3c);
                }
            }
            ex["select_ms"]     = select_ms;
            ex["fetch_ms"]      = fetch_ms;
            ex["decompress_ms"] = decompress_ms;
            j["exemplar"] = ex;
            j["r3"] = condition_json(r3c);
            // pjev decisions score candidate-label logits; nothing is generated.
            j["completion_tokens"] = 0;
            j["response"] = "candidate-logit scoring (no generated text); prediction = argmax of r3.logits";
            const double total = select_ms + fetch_ms + decompress_ms + r3c.total_ms;
            j["r3_total_latency_ms"] = total;
            r3_seconds += total / 1000.0;
            (r3c.ok() ? r3_ok[si] : r3_fail[si])++;
            seed_out[si] << dump(j) << "\n";
        }
        for (auto& f : seed_out) f.flush();

        if ((qi + 1) % 25 == 0 || qi + 1 == queries.size()) {
            int ok = 0;
            for (int v : r3_ok) ok += v;
            printf("  [%zu/%zu] r3 ok=%d  elapsed %.0fs\n", qi + 1, queries.size(), ok, ms_since(t_run) / 1000.0);
            fflush(stdout);
        }
    }

    c["experiment_detail"] = {
        {"selection", r3::to_string(selection)},
        {"selection_version", length_matched ? r3::kLengthMatchedVersion : r3::kRandomSelectionVersion},
        {"random_index", length_matched
            ? "window = the " + std::to_string(length_window) + " eligible exemplars (query removed) whose demonstration "
              "token count is closest to the nearest exemplar's (|tokens - target|, ties by lower ID); pick = "
              "XXH64(\"r3-length-v1|<seed>|<query_id>\") mod window size over the window sorted by ID"
            : std::string("XXH64(\"r3-random-v1|<seed>|<query_id>\") mod pool_size, over the pool sorted by exemplar ID "
                          "with the query itself removed")},
        {"length_window", length_matched ? json(length_window) : json(nullptr)},
        {"length_pool_build_failures", length_pool_failures},
        {"seeds", seeds},
        {"seeds_predefined", seeds_default},
        {"pool_definition", "exemplar_vectors rows (kind, and number_of_levels for score — the KNN filter "
                            "columns) whose ID is in the training allowlist and that have a one-hot answer"},
        {"pool_sizes", [&] { json p; for (const auto& [k, v] : pool.by_key()) p[k] = v.size(); return p; }()},
        {"pool_stats", {{"vector_rows", pstats.rows}, {"eligible", pstats.eligible},
                        {"outside_train", pstats.not_in_train}, {"no_one_hot", pstats.no_one_hot},
                        {"duplicate_ids", pstats.duplicate_ids}, {"kind_metadata_mismatch", kind_metadata_mismatch}}},
        {"context_policy", "same as R1: no length filter; a prompt that does not fit n_ctx is recorded as "
                           "context_limit, never truncated or replaced"},
        {"reference_predictions", reference_path},
        {"reference_predictions_sha256", ref_sha},
        {"split_assignments", split_path},
        {"split_assignments_sha256", sha256_file(split_path)},
        {"partition", {{"heldout", n_heldout}, {"saved_evaluation", saved_eval}, {"saved_tuning", saved_tuning},
                       {"disagreements", partition_disagreements}, {"group_overlap", pinfo.group_overlap},
                       {"train_split_records", pinfo.train_split_records},
                       {"evaluation_families", pinfo.evaluation_families},
                       {"queries_rejected_as_training", rejected_train}}},
        {"evaluated_queries", queries.size()},
        {"no_eligible_exemplar", no_eligible},
        {"r3_ok_per_seed", r3_ok},
        {"r3_failed_per_seed", r3_fail},
        {"reference_check", {{"queries", verify_done}, {"s1_mismatch", verify_s1_mismatch},
                             {"nearest_example_mismatch", verify_example_mismatch},
                             {"r1_logit_mismatch", verify_r1_mismatch},
                             {"hamming_probe_checked", hamming_checked},
                             {"hamming_probe_mismatch", hamming_mismatch},
                             {"demo_length_checked", length_checked},
                             {"demo_length_mismatch", length_mismatch}}},
        {"similarity_probe", probe_ok ? "vec_distance_hamming(stored bit vector, vec_quantize_binary(query))"
                                      : "unavailable: " + probe_err},
        {"latency_conditions", "models loaded once, one untimed warm-up embedding and decision; sequential; "
                               "every decision evaluates its full prompt from an empty KV cache"},
        {"wall_seconds", ms_since(t_run) / 1000.0},
        {"r3_seconds", r3_seconds}};
    {
        std::ofstream cf(output_dir + "/config.json", std::ios::binary);
        cf << c.dump(2, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
    }
    printf("\nReference check: %d queries, S1 mismatches %d, nearest example mismatches %d, R1 logit mismatches %d, "
           "Hamming probe %d/%d exact, demo length model %d/%d exact\n", verify_done, verify_s1_mismatch,
           verify_example_mismatch, verify_r1_mismatch, hamming_checked - hamming_mismatch, hamming_checked,
           length_checked - length_mismatch, length_checked);
    for (size_t si = 0; si < seeds.size(); ++si)
        printf("  seed %" PRIu64 ": ok %d failed %d\n", seeds[si], r3_ok[si], r3_fail[si]);
    const bool clean = verify_s1_mismatch == 0 && verify_r1_mismatch == 0 && verify_example_mismatch == 0 &&
                       hamming_mismatch == 0 && (!length_matched || length_mismatch == 0);
    return clean ? 0 : 2;
}

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IONBF, 0);  // long runs: progress must survive a crash
    std::string mode;
    for (int i = 1; i + 1 < argc; ++i) if (strcmp(argv[i], "--mode") == 0) mode = argv[i + 1];
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) { print_usage(argv[0]); return 0; }
    }
    try {
        if (mode == "run") return run_mode(argc, argv);
        if (mode == "aggregate") return r3::aggregate_main(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    print_usage(argv[0]);
    return 1;
}
