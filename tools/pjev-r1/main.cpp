// pjev-r1: R1 retrieved 1-shot evaluation.
// For every query: S1 zero-shot decision, then Top-1 retrieval of a training exemplar of the same
// kind (and score level count), then the same decision with that exemplar as a solved earlier turn.

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
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
#include "r1_harness.h"
#include "r1_metrics.h"
#include "r1_output.h"
#include "r1_task.h"
#include "r2_gate.h"

#include "sqlite/sqlite3.h"

#include "nlohmann/json.hpp"

using namespace pjev;
using namespace pjev::r1;
using namespace pjev::filldb;

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

static void print_usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "  --input PATH            Evaluation queries (default: datasets/validation.jsonl)\n"
        "  --db PATH               Exemplar database, opened read-only (default: models/examplar.db)\n"
        "  --train PATH            Training split; only its record IDs may be retrieved\n"
        "                          (default: datasets/train.jsonl)\n"
        "  --embedding-model PATH  Embedding model (default: models/nomic-embed-text-v1.5.gguf)\n"
        "  --model PATH            Inference model (default: models/bonsai2_2b.gguf)\n"
        "  --output DIR            Output directory (default: results/r1)\n"
        "  --corpus NAME           Corpus label for reports (default: open-jev-validation for the\n"
        "                          default input, otherwise external:<file name>)\n"
        "  --top-k N               Demonstrations per query; R1 accepts only 1 (default: 1)\n"
        "  --kind KIND             choice | noul | score | all (default: all)\n"
        "  --limit N               Evaluate at most N queries (results marked preliminary)\n"
        "  --seed N                Shuffle queries with this seed before --limit\n"
        "  --offset N              Skip the first N queries (after --seed, before --limit)\n"
        "  --ctx-size N            Inference context size (default: 4096)\n"
        "  --parity-check N        Re-run S1 after R1 for the first N queries and require\n"
        "                          identical output (default: 20)\n"
        "  --save-prompts          Write full S1/R1 prompts to prompts.jsonl\n"
        "  --verbose               Per-query progress\n"
        "  --help                  Show this message\n",
        prog);
}

struct Query {
    std::string    id;   // textual id; parse_exemplar keeps only its hash
    ExemplarRecord rec;
    TaskMapping    task;
    int            ground_truth = -1;  // candidate index
};

int pjev::r1::run_harness(int argc, char** argv, const HarnessHooks& hooks)
{
    std::string input_path      = "datasets/validation.jsonl";
    std::string db_path         = "models/examplar.db";
    std::string train_path      = "datasets/train.jsonl";
    std::string emb_model_path  = "models/nomic-embed-text-v1.5.gguf";
    std::string model_path      = "models/bonsai2_2b.gguf";
    std::string output_dir      = hooks.default_output;
    std::string corpus;
    std::string kind_filter;
    int         top_k           = 1;
    int         limit           = -1;
    int         offset          = 0;
    bool        has_seed        = false;
    uint64_t    seed            = 0;
    int         ctx_size        = 4096;
    int         parity_n        = 20;
    bool        save_prompts    = false;
    bool        verbose         = false;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* name) { return strcmp(argv[i], name) == 0; };
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires an argument\n", flag);
                exit(1);
            }
            return argv[++i];
        };
        if (arg("--help") || arg("-h"))   { print_usage(argv[0]); return 0; }
        else if (arg("--input"))           input_path     = next("--input");
        else if (arg("--db"))              db_path        = next("--db");
        else if (arg("--train"))           train_path     = next("--train");
        else if (arg("--embedding-model")) emb_model_path = next("--embedding-model");
        else if (arg("--model"))           model_path     = next("--model");
        else if (arg("--output"))          output_dir     = next("--output");
        else if (arg("--corpus"))          corpus         = next("--corpus");
        else if (arg("--top-k"))           top_k          = atoi(next("--top-k"));
        else if (arg("--kind"))            { kind_filter = next("--kind"); if (kind_filter == "all") kind_filter.clear(); }
        else if (arg("--limit"))           limit          = atoi(next("--limit"));
        else if (arg("--offset"))          offset         = atoi(next("--offset"));
        else if (arg("--seed"))            { seed = std::strtoull(next("--seed"), nullptr, 10); has_seed = true; }
        else if (arg("--ctx-size"))        ctx_size       = atoi(next("--ctx-size"));
        else if (arg("--parity-check"))    parity_n       = atoi(next("--parity-check"));
        else if (arg("--save-prompts"))    save_prompts   = true;
        else if (arg("--verbose"))         verbose        = true;
        else {
            fprintf(stderr, "error: unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }
    if (top_k != 1) {
        fprintf(stderr, "error: R1 is an always-1-shot experiment; --top-k must be 1\n");
        return 1;
    }
    if (!kind_filter.empty() && kind_filter != "choice" && kind_filter != "noul" && kind_filter != "score") {
        fprintf(stderr, "error: --kind must be choice, noul, score or all\n");
        return 1;
    }
    if (corpus.empty()) {
        std::string base = input_path.substr(input_path.find_last_of("/\\") + 1);
        corpus = (base == "validation.jsonl") ? "open-jev-validation" : "external:" + base;
    }

    const std::string run_timestamp = iso_timestamp();
    auto t_run = Clock::now();

    // --- Exemplar database (read-only) ---
    r0::R0Reader reader(db_path);
    if (!reader.open()) {
        fprintf(stderr, "error: cannot open database read-only: %s\n", db_path.c_str());
        return 1;
    }
    const int64_t db_count = reader.example_count();
    printf("Database: %s (%" PRId64 " examples)\n", db_path.c_str(), db_count);

    // --- Training allowlist ---
    // The exemplar DB was filled from every English record, including validation and test, and
    // stores no split column. Only IDs from the training split may be used as demonstrations.
    std::unordered_set<int64_t> train_ids;
    {
        std::ifstream tf(train_path);
        if (!tf.is_open()) {
            fprintf(stderr, "error: cannot open training split: %s\n", train_path.c_str());
            return 1;
        }
        std::string line;
        int bad = 0;
        while (std::getline(tf, line)) {
            if (line.empty()) continue;
            try {
                auto j = nlohmann::ordered_json::parse(line);
                if (j.value("split", "") == "train" && j.contains("id") && j["id"].is_string()) {
                    train_ids.insert(hash_exemplar_id(j["id"].get<std::string>()));
                }
            } catch (...) { bad++; }
        }
        printf("Training allowlist: %zu IDs from %s (%d unparsable lines)\n",
               train_ids.size(), train_path.c_str(), bad);
    }
    // Training rows without a one-hot answer (soft targets) cannot be shown as a solved
    // demonstration, so they are removed from the allowlist rather than retrieved and rejected.
    int64_t db_train = 0, db_non_train = 0, db_train_no_one_hot = 0;
    {
        sqlite3* db = nullptr;
        if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK) {
            sqlite3_stmt* st = nullptr;
            sqlite3_prepare_v2(db, "SELECT id, answer_index FROM examples", -1, &st, nullptr);
            while (st && sqlite3_step(st) == SQLITE_ROW) {
                const int64_t id = sqlite3_column_int64(st, 0);
                if (!train_ids.count(id)) { db_non_train++; continue; }
                db_train++;
                if (sqlite3_column_type(st, 1) == SQLITE_NULL || sqlite3_column_int(st, 1) < 0) {
                    train_ids.erase(id);
                    db_train_no_one_hot++;
                }
            }
            sqlite3_finalize(st);
        }
        sqlite3_close(db);
    }
    printf("DB records in training split: %" PRId64 " (%" PRId64 " without a one-hot answer, never retrieved), "
           "outside it (never retrieved): %" PRId64 "\n",
           db_train, db_train_no_one_hot, db_non_train);

    // --- Queries ---
    std::vector<Query> queries;
    int n_read = 0, n_parse_err = 0, n_lang = 0, n_kind = 0, n_no_gt = 0, n_task_err = 0;
    int n_train_split = 0, n_in_allowlist = 0;
    {
        std::ifstream fin(input_path);
        if (!fin.is_open()) {
            fprintf(stderr, "error: cannot open input: %s\n", input_path.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            n_read++;
            Query q;
            try {
                auto j = nlohmann::ordered_json::parse(line);
                if (!parse_exemplar(j, q.rec).empty()) { n_parse_err++; continue; }
                q.id = j["id"].get<std::string>();
            } catch (...) { n_parse_err++; continue; }
            // The exemplar bank is English-only; R0 used the same query filter.
            if (q.rec.language != "en") { n_lang++; continue; }
            if (!kind_filter.empty() && q.rec.kind != kind_filter) { n_kind++; continue; }
            if (q.rec.split == "train") { n_train_split++; continue; }
            if (train_ids.count(q.rec.id)) { n_in_allowlist++; continue; }
            if (q.rec.answer_index < 0) { n_no_gt++; continue; }
            std::string err;
            if (!build_task(q.rec.kind, q.rec.question, q.rec.state, q.rec.options, q.task, err)) {
                n_task_err++;
                if (verbose) fprintf(stderr, "  task error %s: %s\n", q.id.c_str(), err.c_str());
                continue;
            }
            q.ground_truth = q.task.option_to_candidate[static_cast<size_t>(q.rec.answer_index)];
            queries.push_back(std::move(q));
        }
    }
    const size_t n_eligible_queries = queries.size();
    if (has_seed) {
        std::mt19937_64 rng(seed);
        for (size_t i = queries.size(); i > 1; --i) {
            std::swap(queries[i - 1], queries[static_cast<size_t>(rng() % i)]);
        }
    }
    if (offset > 0) queries.erase(queries.begin(), queries.begin() + std::min(queries.size(), static_cast<size_t>(offset)));
    if (limit >= 0 && static_cast<size_t>(limit) < queries.size()) queries.resize(static_cast<size_t>(limit));
    const bool preliminary = offset > 0 || (limit >= 0 && static_cast<size_t>(limit) < n_eligible_queries) ||
                             !kind_filter.empty() || has_seed;

    printf("Input: %s  read=%d parse_err=%d non_en=%d kind_skip=%d train_split=%d in_train_ids=%d "
           "no_one_hot=%d task_err=%d -> eligible=%zu, evaluating=%zu\n",
           input_path.c_str(), n_read, n_parse_err, n_lang, n_kind, n_train_split, n_in_allowlist,
           n_no_gt, n_task_err, n_eligible_queries, queries.size());
    if (queries.empty()) {
        fprintf(stderr, "error: no queries to evaluate\n");
        return 1;
    }

    // --- Models ---
    LlamaInstance llama_guard;

    EmbeddingConfig emb_cfg;
    emb_cfg.model_path = emb_model_path;
    emb_cfg.verbose    = verbose;
    std::unique_ptr<Embedding> embedding;
    try {
        embedding = std::make_unique<Embedding>(emb_cfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    const int emb_dim = embedding->dimension();
    ModelIdentity emb_id = ModelIdentity::from_path(emb_model_path, nullptr);
    std::string emb_fp = emb_id.sha256.size() >= 16 ? emb_id.sha256.substr(0, 16) : "";
    {
        std::string err;
        if (!reader.check_compatible(emb_dim, emb_fp, err)) {
            fprintf(stderr, "error: incompatible exemplar database — %s\n", err.c_str());
            return 1;
        }
    }
    printf("Embedding model: %s (dim=%d) — compatible with DB\n", emb_model_path.c_str(), emb_dim);

    LlamaConfig llama_cfg;
    llama_cfg.model_path = model_path;
    llama_cfg.n_ctx = llama_cfg.n_batch = ctx_size;
    llama_cfg.verbose = verbose;
    std::unique_ptr<LlamaBackend> backend;
    try {
        backend = std::make_unique<LlamaBackend>(llama_cfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    ModelIdentity model_id = ModelIdentity::from_path(model_path, backend.get());

    // S1: single-order letter baseline. No ensembles, no calibration, no prior correction.
    PromptConfig prompt_cfg;
    prompt_cfg.layout = Layout::AUTO;
    prompt_cfg.scheme = Scheme::LETTERS;
    EnsembleConfig ensemble;
    ensemble.noul_mode   = NoulEnsembleMode::NONE;
    ensemble.choice_mode = ChoiceEnsembleMode::NONE;
    ensemble.score_mode  = ScoreEnsembleMode::NONE;
    std::unique_ptr<DecisionEngine> engine;
    try {
        engine = std::make_unique<DecisionEngine>(*backend, prompt_cfg, CalibrationConfig{}, ensemble);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    PromptStrategy strategy(prompt_cfg);
    printf("Inference model: %s (n_ctx=%d)\n", model_path.c_str(), ctx_size);

    std::ofstream prompts_out;
    if (save_prompts) {
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        prompts_out.open(output_dir + "/prompts.jsonl", std::ios::binary);
    }

    // Warm-up (untimed, not recorded): latencies below assume warm models.
    {
        const Query& q = queries.front();
        std::vector<float> vec;
        embedding->encode(vec, std::string(r0::kQueryPrefix) +
            build_embedding_text(q.rec.kind, q.rec.domain, q.rec.family, q.rec.question,
                                 q.rec.state, q.rec.options));
        engine->decide(q.task.input);
    }

    // --- Evaluation loop ---
    std::vector<R1Item> items;
    items.reserve(queries.size());
    int parity_checked = 0, parity_mismatch = 0;
    double s1_seconds = 0.0, r1_seconds = 0.0;
    int gate_activated = 0, gate_rejected = 0, gate_unavailable = 0, second_passes = 0;

    for (size_t qi = 0; qi < queries.size(); ++qi) {
        const Query& q = queries[qi];
        const DecisionInput& input = q.task.input;

        R1Item it;
        it.query_id         = q.id;
        it.query_hash       = q.rec.id;
        it.kind             = q.rec.kind;
        it.number_of_levels = q.rec.number_of_levels;
        it.options_count    = static_cast<int>(input.options.size());
        it.question         = q.rec.question;
        it.state            = input.state;
        it.option_labels    = candidate_labels(strategy, input.type, input.options.size());
        for (int oi : q.task.candidate_to_option) it.option_texts.push_back(q.rec.options[static_cast<size_t>(oi)]);
        it.ground_truth     = q.ground_truth;

        // S1
        auto t0 = Clock::now();
        DecisionOutput s1_out = engine->decide(input);
        double s1_ms = ms_since(t0);
        fill_condition(s1_out, it.option_labels, it.option_texts, q.ground_truth, s1_ms, it.s1);
        s1_seconds += s1_ms / 1000.0;

        // Retrieval: never sees the target.
        auto& r = it.retrieval;
        Demonstration demo;
        bool have_demo = false;
        {
            std::string text = std::string(r0::kQueryPrefix) +
                build_embedding_text(q.rec.kind, q.rec.domain, q.rec.family, q.rec.question,
                                     q.rec.state, q.rec.options);
            std::vector<float> vec;
            auto te = Clock::now();
            bool emb_ok = embedding->encode(vec, text);
            r.embed_ms = ms_since(te);
            if (!emb_ok) {
                r.status = "embedding_error";
            } else {
                r0::QueryResult qr;
                auto ts = Clock::now();
                const int64_t self_id = q.rec.id;
                bool ok = reader.search_filtered(
                    q.rec, vec, 1,
                    [&](int64_t id) { return id != self_id && train_ids.count(id) > 0; },
                    r0::R0Reader::kMaxFetchK, qr, &r.fetched_k);
                r.search_ms = ms_since(ts);
                if (!ok) {
                    r.status = "search_error";
                } else if (qr.retrieved.empty()) {
                    r.status = "no_eligible_exemplar";
                } else if (const auto& top = qr.retrieved.front();
                           hooks.gate_enabled &&
                           r2::gate_decide(true, estimated_cosine(top.distance, emb_dim),
                                           hooks.gate_threshold) != r2::GateDecision::Activated) {
                    // R2 gate rejected (or invalid similarity): keep the retrieval diagnostics, skip
                    // the exemplar load and the 1-shot pass entirely.
                    r.status             = "ok";
                    r.example_id         = top.example_id;
                    r.kind               = q.rec.kind;
                    r.number_of_levels   = q.rec.number_of_levels;
                    r.hamming_distance   = top.distance;
                    r.hamming_similarity = hamming_similarity(top.distance, emb_dim);
                    r.estimated_cosine   = estimated_cosine(top.distance, emb_dim);
                    r.estimated_cosine_distance = 1.0 - r.estimated_cosine;
                    r.same_group         = top.same_group;
                    it.r1.status         = "gate_rejected";
                } else {
                    r0::ExemplarPayload p;
                    std::string err;
                    auto tf = Clock::now();
                    bool loaded = reader.load_payload(top.example_id, p, err);
                    double load_ms = ms_since(tf);
                    r.decompress_ms = p.decompress_ms;
                    r.fetch_ms      = std::max(0.0, load_ms - p.decompress_ms);
                    if (!loaded || !build_demonstration(p, demo, err)) {
                        r.status = "payload_error";
                        r.error  = err;
                    } else if (p.example_id == self_id) {
                        // Unreachable through the filter; kept as an explicit leakage guard.
                        r.status = "payload_error";
                        r.error  = "retrieved exemplar is the query itself";
                    } else {
                        have_demo = true;
                        r.status             = "ok";
                        r.example_id         = p.example_id;
                        r.kind               = p.kind;
                        r.number_of_levels   = p.number_of_levels;
                        r.hamming_distance   = top.distance;
                        r.hamming_similarity = hamming_similarity(top.distance, emb_dim);
                        r.estimated_cosine   = estimated_cosine(top.distance, emb_dim);
                        r.estimated_cosine_distance = 1.0 - r.estimated_cosine;
                        // Schema v4 keeps only domain/family as relation metadata.
                        r.domain             = p.domain;
                        r.family             = p.family;
                        r.same_group           = top.same_group;
                        r.same_domain          = !q.rec.domain.empty() && p.domain == q.rec.domain;
                        r.same_family          = !q.rec.family.empty() && p.family == q.rec.family;
                        r.same_option_texts    = (p.options == q.rec.options);
                        r.options_count      = static_cast<int>(demo.options.size());
                        r.answer_index       = demo.answer_index;
                        r.question           = demo.question;
                        r.state              = demo.state;
                        r.option_labels      = candidate_labels(strategy, demo.type, demo.options.size());
                        TaskMapping dm;
                        std::string e2;
                        build_task(p.kind, p.question, p.state, p.options, dm, e2);
                        for (int oi : dm.candidate_to_option) r.option_texts.push_back(p.options[static_cast<size_t>(oi)]);
                        r.answer_label = r.option_labels[static_cast<size_t>(demo.answer_index)];
                        r.answer_text  = r.option_texts[static_cast<size_t>(demo.answer_index)];
                    }
                }
            }
        }

        // R1
        if (have_demo) {
            auto t1 = Clock::now();
            DecisionOutput r1_out = engine->decide_with_demonstration(input, demo);
            double r1_ms = ms_since(t1);
            fill_condition(r1_out, it.option_labels, it.option_texts, q.ground_truth, r1_ms, it.r1);
            second_passes++;
        }
        if (hooks.gate_enabled) {
            it.gate.enabled    = true;
            it.gate.threshold  = hooks.gate_threshold;
            it.gate.similarity = r.ok() ? r.estimated_cosine : 0.0;
            it.gate.decision   = r2::to_string(r2::gate_decide(r.ok(), it.gate.similarity, hooks.gate_threshold));
            if (it.gate.decision == std::string("activated")) gate_activated++;
            else if (r.ok())                                  gate_rejected++;
            else                                              gate_unavailable++;
        }
        r1_seconds += (r.embed_ms + r.search_ms + r.fetch_ms + r.decompress_ms + it.r1.total_ms) / 1000.0;

        // Parity: S1 after R1 must reproduce the first S1 decision exactly.
        if (parity_checked < parity_n && it.s1.ok()) {
            DecisionOutput again = engine->decide(input);
            parity_checked++;
            if (!again.ok || again.selected != s1_out.selected || again.raw_logits != s1_out.raw_logits ||
                again.prompt_token_count != s1_out.prompt_token_count) {
                parity_mismatch++;
                fprintf(stderr, "  PARITY MISMATCH on %s\n", it.query_id.c_str());
            }
        }

        if (save_prompts && prompts_out.is_open()) {
            nlohmann::ordered_json pj;
            pj["query_id"]  = it.query_id;
            pj["s1_prompt"] = render_s1_prompt(strategy, input);
            if (have_demo) pj["r1_prompt"] = render_r1_prompt(strategy, input, demo);
            prompts_out << pj.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << "\n";
        }

        it.paired = it.s1.ok() && it.r1.ok();
        if (it.paired) it.transition = classify(it.s1.correct, it.r1.correct);
        items.push_back(std::move(it));

        const R1Item& last = items.back();
        if (verbose || (qi + 1) % 50 == 0 || qi + 1 == queries.size()) {
            printf("  [%zu/%zu] %-6s s1=%s r1=%s ret=%s ham=%.0f %s  s1=%.0fms r1=%.0fms\n",
                   qi + 1, queries.size(), last.kind.c_str(),
                   last.s1.ok() ? (last.s1.correct ? "ok" : "x") : last.s1.status.c_str(),
                   last.r1.ok() ? (last.r1.correct ? "ok" : "x") : last.r1.status.c_str(),
                   last.retrieval.status.c_str(), last.retrieval.hamming_distance,
                   last.paired ? to_string(last.transition) : "-",
                   last.s1.total_ms, last.r1_total_ms());
            fflush(stdout);
        }
    }

    // --- Outputs ---
    R1RunInfo info;
    info.output_dir    = output_dir;
    info.corpus        = corpus;
    info.preliminary   = preliminary;
    info.n_ctx         = ctx_size;
    info.embedding_dim = emb_dim;
    info.wall_seconds  = ms_since(t_run) / 1000.0;
    info.s1_seconds    = s1_seconds;
    info.r1_seconds    = r1_seconds;
    info.notes.push_back("S1 parity: S1 re-run after R1 on the first " + std::to_string(parity_checked) +
                         " queries — " + std::to_string(parity_mismatch) + " mismatches.");
    info.notes.push_back("Exemplar DB holds " + std::to_string(db_non_train) +
                         " records outside the training split; they are never used as demonstrations.");
    info.notes.push_back(std::to_string(db_train_no_one_hot) +
                         " training records have no one-hot answer (soft target) and are not eligible exemplars.");

    if (hooks.gate_enabled) {
        info.notes.push_back("R2 gate at estimated cosine >= " + std::to_string(hooks.gate_threshold) +
                             ": " + std::to_string(gate_rejected) + " queries kept S1 without a 1-shot pass, "
                             "so the R1 columns of this report cover activated queries only.");
    }

    auto& c = info.config;
    c["experiment"]            = hooks.experiment;
    c["run_timestamp"]         = run_timestamp;
    c["corpus"]                = corpus;
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
    c["db_info"] = {
        {"schema_version",         reader.get_info("schema_version")},
        {"embedding_fingerprint",  reader.get_info("embedding_fingerprint")},
        {"embedding_text_version", reader.get_info("embedding_text_version")},
        {"distance_metric_label",  reader.get_info("distance_metric")},
        {"vector_storage",         "bit (vec_quantize_binary); KNN distance is Hamming"},
        {"compression_format",     reader.get_info("compression_format")},
        {"example_count",          db_count},
        {"records_in_train_split", db_train},
        {"records_outside_train_split", db_non_train},
        {"train_records_without_one_hot", db_train_no_one_hot}};
    c["query_prefix"]          = r0::kQueryPrefix;
    c["prompt_format_version"] = kPromptFormatVersion;
    c["prompt_format"]         = "S1 user turn for the exemplar, assistant turn = exemplar answer label, "
                                 "then the unchanged S1 query prompt";
    c["scheme"]                = "letters";
    c["layout"]                = "auto";
    c["ensemble"]              = "none (single order)";
    c["calibration"]           = "none (temperature 1.0, no prior correction)";
    c["n_ctx"]                 = ctx_size;
    c["top_k"]                 = top_k;
    c["kind_filter"]           = kind_filter.empty() ? "all" : kind_filter;
    c["limit"]                 = limit;
    c["offset"]                = offset;
    c["seed"]                  = has_seed ? nlohmann::ordered_json(seed) : nlohmann::ordered_json(nullptr);
    c["leakage_policy"]        = "demonstrations restricted to IDs whose split is 'train' in train_path; "
                                 "the query's own ID is rejected; query text never includes the target";
    c["max_fetch_k"]           = r0::R0Reader::kMaxFetchK;
    c["query_filter"] = {
        {"records_read", n_read}, {"parse_errors", n_parse_err}, {"non_english", n_lang},
        {"kind_skipped", n_kind}, {"train_split_rejected", n_train_split},
        {"in_train_allowlist_rejected", n_in_allowlist}, {"no_one_hot_target", n_no_gt},
        {"task_mapping_errors", n_task_err}, {"eligible", n_eligible_queries},
        {"evaluated", queries.size()}};
    c["parity_check"] = {{"checked", parity_checked}, {"mismatches", parity_mismatch}};
    if (hooks.gate_enabled) {
        c["gate"] = {
            {"metric", "estimated_cosine = cos(pi * hamming / embedding_dim)"},
            {"rule", "similarity >= threshold activates the 1-shot pass"},
            {"threshold", hooks.gate_threshold},
            {"activated", gate_activated},
            {"rejected", gate_rejected},
            {"no_exemplar_or_retrieval_error", gate_unavailable},
            {"second_pass_calls", second_passes}};
    }

    printf("\nWriting outputs to %s\n", output_dir.c_str());
    bool ok = true;
    // One failing writer must not lose the others after a multi-hour run.
    auto run_writer = [&](const char* name, const std::function<bool()>& fn) {
        try {
            if (!fn()) { ok = false; fprintf(stderr, "error: failed to write %s\n", name); }
        } catch (const std::exception& e) {
            ok = false;
            fprintf(stderr, "error: exception while writing %s: %s\n", name, e.what());
        }
    };
    run_writer("config.json",          [&] { return write_config(info); });
    run_writer("predictions.jsonl",    [&] { return write_predictions_jsonl(output_dir, items); });
    run_writer("transitions.csv",      [&] { return write_transitions_csv(output_dir, items); });
    run_writer("retrieval_effect.csv", [&] { return write_retrieval_effect_csv(output_dir, items); });
    run_writer("latency.csv",          [&] { return write_latency_csv(output_dir, items); });
    run_writer("summary.md",           [&] { return write_summary_md(info, items); });
    run_writer("examples.md",          [&] { return write_examples_md(output_dir, items); });
    if (!ok) {
        fprintf(stderr, "error: some outputs could not be written\n");
        return 1;
    }

    TransitionCounts all;
    for (const auto& it : items) if (it.paired) all.add(it.transition);
    printf("Paired %d: S1 %.2f%%  R1 %.2f%%  diff %+.2f pp  rescue %d  harm %d  (parity mismatches %d)\n",
           all.n(), all.s1_accuracy() * 100, all.r1_accuracy() * 100, all.accuracy_diff() * 100,
           all.rescue, all.harm, parity_mismatch);
    if (hooks.gate_enabled) {
        printf("Gate: activated %d  rejected %d  unavailable %d  second passes %d\n",
               gate_activated, gate_rejected, gate_unavailable, second_passes);
    }
    return parity_mismatch == 0 ? 0 : 2;
}

#ifndef PJEV_R1_NO_MAIN
int main(int argc, char** argv)
{
    return pjev::r1::run_harness(argc, argv, pjev::r1::HarnessHooks{});
}
#endif
