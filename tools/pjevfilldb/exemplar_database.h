#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "exemplar_record.h"
#include "sqlite/sqlite3.h"

namespace pjev
{
namespace filldb
{

class ExemplarDatabase
{
public:
    explicit ExemplarDatabase(std::string path);
    ~ExemplarDatabase();

    ExemplarDatabase(const ExemplarDatabase&)            = delete;
    ExemplarDatabase& operator=(const ExemplarDatabase&) = delete;

    // Open the database.  If rebuild=true, all writes go to "{path}.tmp".
    bool open(bool rebuild);

    // Create tables and vec0 virtual table for a fresh database.
    bool create_schema(int emb_dim,
                       const std::string& model_path,
                       const std::string& model_fingerprint);

    // True if db_info and exemplar_vectors tables exist with a valid schema.
    bool has_schema() const;

    // Check that the existing schema is compatible with the given model.
    // Fills err on mismatch.
    bool check_compatible(int                emb_dim,
                          const std::string& model_fingerprint,
                          std::string&       err) const;

    // True if an example with this id already exists.
    // Sets is_conflict=true when the id exists but content differs.
    bool id_exists(int64_t id,
                   const ExemplarRecord& rec,
                   bool& is_conflict) const;

    bool begin_transaction();
    bool commit_transaction();
    bool rollback_transaction();

    bool insert_group(const ExemplarRecord& rec);
    bool insert_example(const ExemplarRecord& rec, int64_t& out_rowid);
    bool insert_vector(int64_t             rowid,
                       const std::vector<float>& embedding,
                       const std::string&  kind,
                       int                 number_of_levels);

    // For rebuild mode: rename .tmp → .db.  No-op in normal mode.
    bool finalize_rebuild();

    int64_t example_count() const;
    int64_t file_size() const;

private:
    bool        exec(const char* sql) const;
    int64_t     query_int64(const char* sql, int64_t fallback = 0) const;
    std::string query_text(const char* sql, const char* fallback = "") const;

    std::string path_;
    std::string active_path_; // path_ or path_+".tmp" when rebuilding
    bool        rebuild_ = false;
    sqlite3*    db_      = nullptr;

    sqlite3_stmt* stmt_insert_group_   = nullptr;
    sqlite3_stmt* stmt_insert_example_ = nullptr;
    sqlite3_stmt* stmt_insert_vector_  = nullptr;
    sqlite3_stmt* stmt_id_exists_      = nullptr;
};

} // namespace filldb
} // namespace pjev
