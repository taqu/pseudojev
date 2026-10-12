#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "embedding/embedding.h"
#include "inference/llama_instance.h"
#include "model/model_identity.h"

#include "exemplar_record.h"
#include "embedding_text.h"
#include "vec_autoext.h"
#include "r0_retrieval.h"
#include "r0_output.h"

#include "sqlite/sqlite3.h"

#include "nlohmann/json.hpp"

using namespace pjev;
using namespace pjev::r0;
using namespace pjev::filldb;

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

static double ms_since(std::chrono::steady_clock::time_point t0)
{
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static std::string iso_timestamp()
{
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
    return buf;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

static void print_usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "Required:\n"
        "  --db PATH          Exemplar database (default: models/exsamplar.db)\n"
        "  --input PATH       Input JSONL file   (default: datasets/train.jsonl)\n"
        "  --model PATH       Embedding model    (default: models/nomic-embed-text-v1.5.Q8_0.gguf)\n"
        "  --output DIR       Output directory   (default: results/r0)\n"
        "\n"
        "Optional:\n"
        "  --top-k N              Number of neighbors to retrieve (default: 5)\n"
        "  --limit N              Process at most N queries\n"
        "  --kind KIND            Filter queries: choice | noul | score | all (default: all)\n"
        "  --split SPLIT          Filter records by split value (default: all splits)\n"
        "  --skip-db-records      Skip records whose ID is present in the exemplar DB\n"
        "  --exclude-same-group   Exclude candidates sharing the query's group_id\n"
        "  --verbose              Print per-query progress\n"
        "  --help                 Show this message\n",
        prog);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static int g_n_processed = 0;

int main(int argc, char** argv)
{
    signal(SIGABRT, [](int) {
        fprintf(stderr, "\nSIGABRT after %d queries\n", g_n_processed);
        fflush(stderr);
        signal(SIGABRT, SIG_DFL);
        raise(SIGABRT);
    });

    // --- Defaults ---
    R0RunConfig cfg;
    cfg.db_path    = "models/exsamplar.db";
    cfg.input_path = "datasets/train.jsonl";
    cfg.model_path = "models/nomic-embed-text-v1.5.Q8_0.gguf";
    cfg.output_dir = "results/r0";
    cfg.top_k      = 5;
    cfg.limit      = -1;

    bool verbose         = false;
    bool skip_db_records = false;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* name) {
            return strcmp(argv[i], name) == 0;
        };
        auto next_str = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires an argument\n", flag);
                exit(1);
            }
            return argv[++i];
        };

        if (arg("--help") || arg("-h")) {
            print_usage(argv[0]);
            return 0;
        } else if (arg("--db")) {
            cfg.db_path = next_str("--db");
        } else if (arg("--input")) {
            cfg.input_path = next_str("--input");
        } else if (arg("--model")) {
            cfg.model_path = next_str("--model");
        } else if (arg("--output")) {
            cfg.output_dir = next_str("--output");
        } else if (arg("--top-k")) {
            cfg.top_k = atoi(next_str("--top-k"));
        } else if (arg("--limit")) {
            cfg.limit = atoi(next_str("--limit"));
        } else if (arg("--kind")) {
            cfg.kind_filter = next_str("--kind");
            if (cfg.kind_filter == "all") cfg.kind_filter.clear();
        } else if (arg("--split")) {
            cfg.split_filter = next_str("--split");
        } else if (arg("--skip-db-records")) {
            skip_db_records = true;
            cfg.skip_db_records = true;
        } else if (arg("--exclude-same-group")) {
            cfg.exclude_same_group = true;
        } else if (arg("--verbose")) {
            verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (cfg.top_k < 1 || cfg.top_k > 100) {
        fprintf(stderr, "error: --top-k must be between 1 and 100\n");
        return 1;
    }

    cfg.run_timestamp = iso_timestamp();

    // --- Open database ---
    R0Reader reader(cfg.db_path);
    if (!reader.open()) {
        fprintf(stderr, "error: cannot open database: %s\n", cfg.db_path.c_str());
        return 1;
    }

    cfg.db_example_count  = static_cast<int>(reader.example_count());
    cfg.db_embedding_model  = reader.get_info("embedding_model");
    cfg.db_distance_metric  = reader.get_info("distance_metric");
    cfg.db_normalization    = reader.get_info("normalization");
    std::string db_dim_str  = reader.get_info("embedding_dim");
    cfg.db_embedding_dim    = db_dim_str.empty() ? 0 : std::stoi(db_dim_str);

    printf("Database: %s (%d examples)\n", cfg.db_path.c_str(), cfg.db_example_count);

    // --- Load embedding model ---
    LlamaInstance llama_guard;

    EmbeddingConfig emb_cfg;
    emb_cfg.model_path = cfg.model_path;
    emb_cfg.verbose    = verbose;

    Embedding embedding(emb_cfg);
    int emb_dim = embedding.dimension();
    if (emb_dim <= 0) {
        fprintf(stderr, "error: failed to load embedding model: %s\n", cfg.model_path.c_str());
        return 1;
    }
    printf("Embedding model: %s (dim=%d)\n", cfg.model_path.c_str(), emb_dim);

    // --- Verify model compatibility ---
    ModelIdentity mid = ModelIdentity::from_path(cfg.model_path, nullptr);
    std::string fingerprint = mid.sha256.size() >= 16 ? mid.sha256.substr(0, 16) : "";

    std::string compat_err;
    if (!reader.check_compatible(emb_dim, fingerprint, compat_err)) {
        fprintf(stderr, "error: model incompatibility — %s\n", compat_err.c_str());
        fprintf(stderr, "  The query embeddings would be incompatible with the stored index.\n");
        fprintf(stderr, "  Use the same model that was used to build the exemplar database.\n");
        return 1;
    }
    printf("Model compatibility: OK\n");

    // --- Optionally collect DB IDs to skip ---
    std::unordered_set<std::string> db_ids;
    if (skip_db_records) {
        // We can't directly enumerate all IDs without opening in RW mode.
        // Re-open via a second connection to enumerate IDs.
        // Actually R0Reader exposes get_info; we need to add an ID enumeration method.
        // For now, use a separate sqlite3 connection to collect IDs.
        {
            // register_sqlite_vec_autoext() already called by R0Reader::open()
            sqlite3* tmp_db = nullptr;
            if (sqlite3_open_v2(cfg.db_path.c_str(), &tmp_db,
                    SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK) {
                sqlite3_stmt* stmt = nullptr;
                sqlite3_prepare_v2(tmp_db,
                    "SELECT id FROM examples", -1, &stmt, nullptr);
                while (sqlite3_step(stmt) == SQLITE_ROW) {
                    const char* id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
                    if (id) db_ids.insert(id);
                }
                sqlite3_finalize(stmt);
                sqlite3_close(tmp_db);
            }
        }
        printf("Loaded %zu DB IDs to skip.\n", db_ids.size());
        fflush(stdout);
    }

    // --- Process input records ---
    std::ifstream fin(cfg.input_path);
    if (!fin.is_open()) {
        fprintf(stderr, "error: cannot open input file: %s\n", cfg.input_path.c_str());
        return 1;
    }
    printf("Input: %s\n", cfg.input_path.c_str());

    std::vector<QueryResult> results;
    int n_read = 0, n_skip_lang = 0, n_skip_split = 0, n_skip_kind = 0;
    int n_skip_db = 0, n_parse_err = 0, n_processed = 0;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        n_read++;

        if (cfg.limit >= 0 && n_processed >= cfg.limit) break;

        // Parse JSON
        nlohmann::ordered_json j;
        try {
            j = nlohmann::ordered_json::parse(line);
        } catch (...) {
            n_parse_err++;
            if (verbose) fprintf(stderr, "  parse error on line %d\n", n_read);
            continue;
        }

        ExemplarRecord rec;
        std::string err = parse_exemplar(j, rec);
        if (!err.empty()) {
            n_parse_err++;
            if (verbose) fprintf(stderr, "  exemplar parse error on line %d: %s\n", n_read, err.c_str());
            continue;
        }

        // Language filter
        if (rec.language != "en") { n_skip_lang++; continue; }

        // Split filter
        if (!cfg.split_filter.empty() && rec.split != cfg.split_filter) {
            n_skip_split++;
            continue;
        }

        // Kind filter
        if (!cfg.kind_filter.empty() && rec.kind != cfg.kind_filter) {
            n_skip_kind++;
            continue;
        }

        // Skip records already in DB
        if (skip_db_records && db_ids.count(rec.id)) {
            n_skip_db++;
            continue;
        }

        // --- Build query embedding text ---
        std::string query_body = filldb::build_embedding_text(
            rec.kind, rec.domain, rec.family, rec.question, rec.state, rec.options);
        std::string query_text = std::string(kQueryPrefix) + query_body;

        // --- Generate embedding ---
        std::vector<float> emb_vec;
        auto t_emb_start = std::chrono::steady_clock::now();
        if (!embedding.encode(emb_vec, query_text)) {
            if (verbose) fprintf(stderr, "  embedding failed for: %s\n", rec.id.c_str());
            continue;
        }
        double embed_ms = ms_since(t_emb_start);

        // --- KNN search ---
        QueryResult qr;
        auto t_search_start = std::chrono::steady_clock::now();
        reader.search(rec, emb_vec, cfg.top_k, cfg.exclude_same_group, qr);
        double search_ms = ms_since(t_search_start);

        qr.embed_ms  = embed_ms;
        qr.search_ms = search_ms;

        results.push_back(std::move(qr));
        n_processed++;
        g_n_processed = n_processed;

        if (verbose || (n_processed % 100 == 0)) {
            printf("  [%d] %s  embed=%.0fms search=%.0fms retrieved=%d\n",
                   n_processed,
                   results.back().query_id.c_str(),
                   embed_ms, search_ms,
                   results.back().retrieved_count);
        }
        fflush(stdout);
    }
    fin.close();

    printf("\n");
    printf("Records read:          %d\n", n_read);
    printf("Parse errors:          %d\n", n_parse_err);
    printf("Skipped (non-English): %d\n", n_skip_lang);
    printf("Skipped (split):       %d\n", n_skip_split);
    printf("Skipped (kind):        %d\n", n_skip_kind);
    printf("Skipped (in DB):       %d\n", n_skip_db);
    printf("Queries processed:     %d\n", n_processed);

    if (n_processed == 0) {
        fprintf(stderr, "warning: no queries were processed — check input file and filters.\n");
        if (!skip_db_records) {
            fprintf(stderr,
                "hint: if the input file only contains records that are already in the\n"
                "exemplar database, use --skip-db-records to exclude them.\n");
        }
        return 1;
    }

    if (!cfg.split_filter.empty() && n_read > 0 && n_processed == 0) {
        fprintf(stderr,
            "hint: no records matched --split=%s\n"
            "  Available splits may differ from the requested filter.\n",
            cfg.split_filter.c_str());
    }

    cfg.query_count = n_processed;
    if (cfg.note.empty() && cfg.split_filter.empty()) {
        cfg.note = "No dedicated validation split found in input; "
                   "all English records used as queries. "
                   "Records in the exemplar DB were " +
                   std::string(skip_db_records ? "excluded (--skip-db-records)."
                                               : "NOT excluded.");
    }

    // --- Write outputs ---
    printf("\nWriting outputs to: %s\n", cfg.output_dir.c_str());

    bool ok = true;
    ok &= write_config(cfg);
    ok &= write_retrieval_jsonl(cfg.output_dir, results);
    ok &= write_summary_md(cfg.output_dir, cfg, results);
    ok &= write_examples_md(cfg.output_dir, results);
    ok &= write_histogram_csv(cfg.output_dir, results);

    if (ok) {
        printf("Done. Outputs:\n");
        printf("  %s/config.json\n",             cfg.output_dir.c_str());
        printf("  %s/retrieval.jsonl\n",          cfg.output_dir.c_str());
        printf("  %s/summary.md\n",               cfg.output_dir.c_str());
        printf("  %s/examples.md\n",              cfg.output_dir.c_str());
        printf("  %s/similarity_histogram.csv\n", cfg.output_dir.c_str());
    } else {
        fprintf(stderr, "error: some output files could not be written.\n");
        return 1;
    }

    return 0;
}
