#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "exemplar_record.h"
#include "sqlite/sqlite3.h"

namespace pjev
{
namespace r0
{

// Prefix for query embeddings — must differ from kDocumentPrefix ("search_document: ").
static constexpr const char* kQueryPrefix = "search_query: ";

struct RetrievedExample {
    int         rank             = 0;
    int64_t     example_id       = 0;
    double      distance         = 0.0;
    double      similarity       = 0.0;  // 1.0 - distance (cosine)
    std::string kind;
    int         number_of_levels = 0;
    int         answer_index     = -1;
    std::string ground_truth;            // reconstructed from options[answer_index]
    int64_t     group_id         = 0;
    bool        same_group       = false;
};

struct QueryResult {
    int64_t     query_id               = 0;
    std::string kind;
    int         number_of_levels       = 0;
    int64_t     query_group_id         = 0;
    std::string query_ground_truth;
    int         query_answer_index     = -1;
    double      embed_ms               = 0.0;
    double      search_ms              = 0.0;
    int         retrieved_count        = 0;
    std::vector<RetrievedExample> retrieved;
};

// Read-only access to the exemplar database for KNN retrieval.
class R0Reader
{
public:
    explicit R0Reader(std::string db_path);
    ~R0Reader();

    R0Reader(const R0Reader&)            = delete;
    R0Reader& operator=(const R0Reader&) = delete;

    // Open the database in read-only mode.
    bool open();

    // Read a value from db_info. Returns "" if the key is absent.
    std::string get_info(const char* key) const;

    // Verify embedding dimension and model fingerprint match the stored metadata.
    bool check_compatible(int emb_dim, const std::string& fingerprint, std::string& err) const;

    int64_t example_count() const;

    // Perform a KNN search for one query.
    // embedding: unit-norm vector matching the DB dimension.
    // top_k: desired result count.
    // exclude_same_group: if true, candidates sharing query.group_id are excluded.
    // Returns true even when retrieved_count < top_k (documented in out).
    bool search(
        const filldb::ExemplarRecord& query,
        const std::vector<float>&     embedding,
        int                           top_k,
        bool                          exclude_same_group,
        QueryResult&                  out) const;

private:
    bool resolve_rowid(int64_t rowid, RetrievedExample& ex) const;

    std::string  db_path_;
    sqlite3*     db_          = nullptr;
    sqlite3_stmt* stmt_resolve_ = nullptr;
};

} // namespace r0
} // namespace pjev
