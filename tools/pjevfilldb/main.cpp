#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>

#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"

#include "embedding/embedding.h"
#include "inference/llama_instance.h"
#include "model/model_identity.h"

#include "exemplar_record.h"
#include "embedding_text.h"
#include "exemplar_database.h"

using namespace pjev;
using namespace pjev::filldb;
using json = nlohmann::ordered_json;

struct Options {
    std::string input;
    std::string output;
    std::string model    = "models/nomic-embed-text-v1.5.Q8_0.gguf";
    bool        rebuild  = false;
    int         limit    = 0;  // 0 = no limit
    int         batch_sz = 100;
    bool        verbose  = false;
};

static void usage(const char* prog)
{
    printf("Usage: %s --input FILE --output FILE [options]\n"
           "\n"
           "Options:\n"
           "  --model PATH      Embedding model (default: models/nomic-embed-text-v1.5.gguf)\n"
           "  --rebuild         Recreate the database from scratch\n"
           "  --limit N         Process at most N eligible examples\n"
           "  --batch-size N    Insert batch size (default: 100)\n"
           "  --verbose         Enable detailed progress logging\n"
           "  --help            Show this message\n",
           prog);
}

static bool parse_args(int argc, char** argv, Options& opts)
{
    for(int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* {
            if(i + 1 < argc) return argv[++i];
            fprintf(stderr, "error: %s requires an argument\n", argv[i]);
            return nullptr;
        };
        if(!strcmp(argv[i], "--input"))       { auto v = next(); if(!v) return false; opts.input   = v; }
        else if(!strcmp(argv[i], "--output")) { auto v = next(); if(!v) return false; opts.output  = v; }
        else if(!strcmp(argv[i], "--model"))  { auto v = next(); if(!v) return false; opts.model   = v; }
        else if(!strcmp(argv[i], "--limit"))  { auto v = next(); if(!v) return false; opts.limit   = std::atoi(v); }
        else if(!strcmp(argv[i], "--batch-size")){ auto v = next(); if(!v) return false; opts.batch_sz = std::atoi(v); }
        else if(!strcmp(argv[i], "--rebuild"))   { opts.rebuild = true; }
        else if(!strcmp(argv[i], "--verbose"))   { opts.verbose = true; }
        else if(!strcmp(argv[i], "--help"))      { usage(argv[0]); std::exit(0); }
        else { fprintf(stderr, "error: unknown option: %s\n", argv[i]); return false; }
    }
    if(opts.input.empty())  { fprintf(stderr, "error: --input is required\n");  return false; }
    if(opts.output.empty()) { fprintf(stderr, "error: --output is required\n"); return false; }
    return true;
}

int main(int argc, char** argv)
{
    Options opts;
    if(!parse_args(argc, argv, opts)) return 1;

    if(opts.verbose) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::warn);
    }

    auto t_start = std::chrono::steady_clock::now();

    // Open input
    std::ifstream infile(opts.input);
    if(!infile.is_open()) {
        fprintf(stderr, "error: cannot open input file: %s\n", opts.input.c_str());
        return 1;
    }

    // Create output parent directory if needed
    {
        auto parent = std::filesystem::path(opts.output).parent_path();
        if(!parent.empty() && !std::filesystem::exists(parent)) {
            std::error_code ec;
            std::filesystem::create_directories(parent, ec);
            if(ec) {
                fprintf(stderr, "error: cannot create directory %s: %s\n",
                        parent.string().c_str(), ec.message().c_str());
                return 1;
            }
        }
    }

    // Initialise llama backend
    LlamaInstance llama;

    // Load embedding model
    EmbeddingConfig emb_cfg;
    emb_cfg.model_path = opts.model;
    emb_cfg.verbose    = opts.verbose;

    std::unique_ptr<Embedding> embedding;
    try {
        embedding = std::make_unique<Embedding>(emb_cfg);
    } catch(const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    const int emb_dim = embedding->dimension();
    if(emb_dim <= 0) {
        fprintf(stderr, "error: embedding dimension is %d\n", emb_dim);
        return 1;
    }

    // Model fingerprint
    const std::string model_fp = pjev::sha256_file(opts.model);
    // Use first 16 hex chars as compact fingerprint
    const std::string fp_short = model_fp.size() >= 16 ? model_fp.substr(0, 16) : model_fp;

    // Open database
    ExemplarDatabase db(opts.output);
    if(!db.open(opts.rebuild)) {
        fprintf(stderr, "error: cannot open database: %s\n", opts.output.c_str());
        return 1;
    }

    if(db.has_schema()) {
        std::string compat_err;
        if(!db.check_compatible(emb_dim, fp_short, compat_err)) {
            fprintf(stderr, "error: incompatible database — %s\n"
                            "       Use --rebuild to recreate.\n",
                    compat_err.c_str());
            return 1;
        }
        // Resume: prepare statements by re-running create_schema (idempotent via IF NOT EXISTS)
        if(!db.create_schema(emb_dim, opts.model, fp_short)) {
            fprintf(stderr, "error: failed to prepare statements\n");
            return 1;
        }
    } else {
        if(!db.create_schema(emb_dim, opts.model, fp_short)) {
            fprintf(stderr, "error: failed to create schema\n");
            return 1;
        }
    }

    // Statistics
    int64_t n_read      = 0;
    int64_t n_eligible  = 0;
    int64_t n_inserted  = 0;
    int64_t n_duplicate = 0;
    int64_t n_conflict  = 0;
    int64_t n_emb_err   = 0;
    int64_t n_parse_err = 0;
    int64_t n_ineligible= 0;
    std::map<std::string, int64_t> by_kind;

    // Process JSONL
    std::string line;
    int64_t batch_pending = 0;
    db.begin_transaction();
    std::vector<float> emb_vec;
    while(std::getline(infile, line)) {
        if(line.empty()) continue;
        ++n_read;

        // Parse JSON
        json j;
        try {
            j = json::parse(line);
        } catch(...) {
            if(opts.verbose)
                printf("[line %lld] WARN: malformed JSON — skipped\n", (long long)n_read);
            ++n_parse_err;
            continue;
        }

        ExemplarRecord rec;
        std::string err = parse_exemplar(j, rec);
        if(!err.empty()) {
            if(opts.verbose)
                printf("[line %lld] WARN: %s — skipped\n", (long long)n_read, err.c_str());
            ++n_parse_err;
            continue;
        }

        if(!is_eligible(rec)) {
            ++n_ineligible;
            continue;
        }
        ++n_eligible;

        // Check duplicate
        bool is_conflict = false;
        if(db.id_exists(rec.id, rec, is_conflict)) {
            if(is_conflict) {
                printf("[line %lld] CONFLICT: id=%s already exists with different content\n",
                       (long long)n_read, rec.id.c_str());
                ++n_conflict;
            } else {
                ++n_duplicate;
            }
            continue;
        }

        // Build embedding text
        rec.embedding_text = build_document_embedding_text(
            rec.kind, rec.domain, rec.family,
            rec.question, rec.state, rec.options);

        // Generate embedding
        emb_vec.clear();
        if(!embedding->encode(emb_vec, rec.embedding_text)) {
            if(opts.verbose)
                printf("[line %lld] WARN: embedding failed for id=%s\n",
                       (long long)n_read, rec.id.c_str());
            ++n_emb_err;
            continue;
        }
        if((int)emb_vec.size() != emb_dim) {
            ++n_emb_err;
            continue;
        }

        // Insert
        int64_t rowid = 0;
        bool ok = db.insert_group(rec) &&
                  db.insert_example(rec, rowid) &&
                  db.insert_vector(rowid, emb_vec, rec.kind, rec.number_of_levels);
        if(!ok) {
            if(opts.verbose)
                printf("[line %lld] WARN: db insert failed for id=%s\n",
                       (long long)n_read, rec.id.c_str());
            db.rollback_transaction();
            db.begin_transaction();
            batch_pending = 0;
            continue;
        }

        ++n_inserted;
        ++by_kind[rec.kind];
        ++batch_pending;

        if(opts.verbose && n_inserted % 100 == 0) {
            printf("  inserted %lld records...\n", (long long)n_inserted);
        }

        if(batch_pending >= opts.batch_sz) {
            db.commit_transaction();
            db.begin_transaction();
            batch_pending = 0;
        }

        if(opts.limit > 0 && n_inserted >= opts.limit) break;
    }

    db.commit_transaction();

    if(opts.rebuild) {
        if(!db.finalize_rebuild()) {
            fprintf(stderr, "error: failed to finalise rebuild\n");
            return 1;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();

    int64_t db_size = db.file_size();
    double  db_mb   = (double)db_size / (1024.0 * 1024.0);

    printf("\nInput file:  %s\n", opts.input.c_str());
    printf("Output DB:   %s\n\n", opts.output.c_str());

    printf("Records read:           %6lld\n", (long long)n_read);
    printf("Parse errors:           %6lld\n", (long long)n_parse_err);
    printf("Ineligible:             %6lld\n", (long long)n_ineligible);
    printf("Inserted:               %6lld\n", (long long)n_inserted);
    printf("Duplicates skipped:     %6lld\n", (long long)n_duplicate);
    printf("Conflicts:              %6lld\n", (long long)n_conflict);
    printf("Embed errors:           %6lld\n", (long long)n_emb_err);

    if(!by_kind.empty()) {
        printf("\nBy kind:\n");
        for(auto& kv : by_kind) {
            printf("  %-20s %6lld\n", kv.first.c_str(), (long long)kv.second);
        }
    }

    printf("\nEmbedding dimension:    %6d\n", emb_dim);
    printf("Embedding model:  %s\n", opts.model.c_str());
    printf("Embedding text ver.:    %6d\n", ExemplarRecord::kEmbeddingFormatVersion);
    printf("\nElapsed: %.0f ms\n", elapsed_s * 1000.0);
    printf("Database size: %.2f MB\n", db_mb);

    return (n_conflict > 0 || n_emb_err > 0) ? 2 : 0;
}
