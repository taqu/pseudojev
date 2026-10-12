#include "exemplar_database.h"

#include <cassert>
#include <cstdio>
#include <string>

#include "vec_autoext.h"

namespace pjev
{
namespace filldb
{

ExemplarDatabase::ExemplarDatabase(std::string path)
    : path_(std::move(path))
{
}

ExemplarDatabase::~ExemplarDatabase()
{
    if(stmt_insert_group_)   { sqlite3_finalize(stmt_insert_group_);   stmt_insert_group_   = nullptr; }
    if(stmt_insert_example_) { sqlite3_finalize(stmt_insert_example_); stmt_insert_example_ = nullptr; }
    if(stmt_insert_vector_)  { sqlite3_finalize(stmt_insert_vector_);  stmt_insert_vector_  = nullptr; }
    if(stmt_id_exists_)      { sqlite3_finalize(stmt_id_exists_);      stmt_id_exists_      = nullptr; }

    if(db_) {
        sqlite3_close(db_);
        db_ = nullptr;
        // Remove abandoned temp file if rebuild was never finalised
        if(rebuild_ && active_path_ != path_) {
            std::remove(active_path_.c_str());
        }
    }
}

bool ExemplarDatabase::open(bool rebuild)
{
    register_sqlite_vec_autoext();

    rebuild_     = rebuild;
    active_path_ = rebuild ? path_ + ".tmp" : path_;

    if(rebuild) {
        std::remove(active_path_.c_str());
    }

    int flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
    if(sqlite3_open_v2(active_path_.c_str(), &db_, flags, nullptr) != SQLITE_OK) {
        return false;
    }

    exec("PRAGMA journal_mode=WAL");
    exec("PRAGMA synchronous=NORMAL");
    exec("PRAGMA foreign_keys=ON");
    return true;
}

bool ExemplarDatabase::create_schema(
    int emb_dim, const std::string& model_path, const std::string& model_fingerprint)
{
    assert(db_ != nullptr);

    if(!exec("CREATE TABLE IF NOT EXISTS db_info ("
             "  key   TEXT PRIMARY KEY,"
             "  value TEXT NOT NULL"
             ")")) {
        return false;
    }

    if(!exec("CREATE TABLE IF NOT EXISTS groups ("
             "  group_id        TEXT PRIMARY KEY,"
             "  source          TEXT,"
             "  scenario_family TEXT"
             ")")) {
        return false;
    }

    if(!exec("CREATE TABLE IF NOT EXISTS examples ("
             "  rowid                    INTEGER PRIMARY KEY,"
             "  id                       TEXT UNIQUE NOT NULL,"
             "  group_id                 TEXT NOT NULL,"
             "  kind                     TEXT NOT NULL,"
             "  state_json               TEXT NOT NULL,"
             "  question                 TEXT NOT NULL,"
             "  options_json             TEXT NOT NULL,"
             "  target_json              TEXT NOT NULL,"
             "  answer_index             INTEGER,"
             "  answer_text              TEXT,"
             "  number_of_levels         INTEGER,"
             "  score_values_json        TEXT,"
             "  domain                   TEXT,"
             "  family                   TEXT,"
             "  language                 TEXT,"
             "  scenario_family          TEXT,"
             "  template_id              TEXT,"
             "  source_instance_id       TEXT,"
             "  semantic_context_sha256  TEXT,"
             "  counterfactual_json      TEXT,"
             "  metadata_json            TEXT,"
             "  source                   TEXT,"
             "  split                    TEXT NOT NULL,"
             "  embedding_text           TEXT NOT NULL,"
             "  embedding_format_version INTEGER NOT NULL,"
             "  FOREIGN KEY (group_id) REFERENCES groups(group_id)"
             ")")) {
        return false;
    }

    // vec0 virtual table: kind and number_of_levels are metadata columns
    // that participate in KNN filtering
    std::string vec_sql =
        "CREATE VIRTUAL TABLE IF NOT EXISTS exemplar_vectors USING vec0("
        "  embedding float[" + std::to_string(emb_dim) + "],"
        "  kind text,"
        "  number_of_levels integer"
        ")";
    if(!exec(vec_sql.c_str())) {
        return false;
    }

    // Store db_info
    auto insert_info = [&](const char* key, const std::string& val) {
        std::string sql = "INSERT OR REPLACE INTO db_info(key,value) VALUES('" +
                          std::string(key) + "','" + val + "')";
        return exec(sql.c_str());
    };
    insert_info("schema_version",        "1");
    insert_info("embedding_model",       model_path);
    insert_info("embedding_fingerprint", model_fingerprint);
    insert_info("embedding_dim",         std::to_string(emb_dim));
    insert_info("embedding_text_version",std::to_string(ExemplarRecord::kEmbeddingFormatVersion));
    insert_info("distance_metric",       "cosine");
    insert_info("normalization",         "L2");

    // Prepare reusable statements
    if(sqlite3_prepare_v2(db_,
        "INSERT OR IGNORE INTO groups(group_id,source,scenario_family) VALUES(?,?,?)",
        -1, &stmt_insert_group_, nullptr) != SQLITE_OK) return false;

    if(sqlite3_prepare_v2(db_,
        "INSERT INTO examples("
        "  id,group_id,kind,state_json,question,options_json,target_json,"
        "  answer_index,answer_text,number_of_levels,score_values_json,"
        "  domain,family,language,scenario_family,template_id,"
        "  source_instance_id,semantic_context_sha256,counterfactual_json,"
        "  metadata_json,source,split,embedding_text,embedding_format_version"
        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        -1, &stmt_insert_example_, nullptr) != SQLITE_OK) return false;

    if(sqlite3_prepare_v2(db_,
        "INSERT INTO exemplar_vectors(rowid,embedding,kind,number_of_levels) VALUES(?,?,?,?)",
        -1, &stmt_insert_vector_, nullptr) != SQLITE_OK) return false;

    if(sqlite3_prepare_v2(db_,
        "SELECT state_json,options_json,target_json,question FROM examples WHERE id=?",
        -1, &stmt_id_exists_, nullptr) != SQLITE_OK) return false;

    return true;
}

bool ExemplarDatabase::has_schema() const
{
    if(!db_) return false;
    const char* sql =
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' "
        "AND name IN ('db_info','groups','examples')";
    return query_int64(sql) == 3;
}

bool ExemplarDatabase::check_compatible(
    int emb_dim, const std::string& model_fingerprint, std::string& err) const
{
    int64_t stored_dim = query_int64(
        "SELECT value FROM db_info WHERE key='embedding_dim'");
    if(stored_dim != emb_dim) {
        err = "embedding dimension mismatch: stored=" + std::to_string(stored_dim) +
              " requested=" + std::to_string(emb_dim);
        return false;
    }
    std::string stored_fp = query_text(
        "SELECT value FROM db_info WHERE key='embedding_fingerprint'");
    if(stored_fp != model_fingerprint) {
        err = "embedding model fingerprint mismatch";
        return false;
    }
    return true;
}

bool ExemplarDatabase::id_exists(
    const std::string& id, const ExemplarRecord& rec, bool& is_conflict) const
{
    is_conflict = false;
    if(!stmt_id_exists_) return false;

    sqlite3_reset(stmt_id_exists_);
    sqlite3_bind_text(stmt_id_exists_, 1, id.c_str(), -1, SQLITE_STATIC);
    int rc = sqlite3_step(stmt_id_exists_);
    if(rc != SQLITE_ROW) return false;

    // Row exists — check for conflict by comparing key fields
    const char* state_json   = (const char*)sqlite3_column_text(stmt_id_exists_, 0);
    const char* options_json = (const char*)sqlite3_column_text(stmt_id_exists_, 1);
    const char* target_json  = (const char*)sqlite3_column_text(stmt_id_exists_, 2);
    const char* question_txt = (const char*)sqlite3_column_text(stmt_id_exists_, 3);

    std::string stored_state    = state_json   ? state_json   : "";
    std::string stored_options  = options_json ? options_json : "";
    std::string stored_target   = target_json  ? target_json  : "";
    std::string stored_question = question_txt ? question_txt : "";

    std::string cur_state   = rec.state.dump();
    std::string cur_options = nlohmann::ordered_json(rec.options).dump();
    std::string cur_target  = nlohmann::ordered_json(rec.target).dump();

    is_conflict = (stored_state    != cur_state    ||
                   stored_options  != cur_options  ||
                   stored_target   != cur_target   ||
                   stored_question != rec.question);
    return true;
}

bool ExemplarDatabase::begin_transaction()
{
    return exec("BEGIN");
}

bool ExemplarDatabase::commit_transaction()
{
    return exec("COMMIT");
}

bool ExemplarDatabase::rollback_transaction()
{
    return exec("ROLLBACK");
}

bool ExemplarDatabase::insert_group(const ExemplarRecord& rec)
{
    if(!stmt_insert_group_) return false;
    sqlite3_reset(stmt_insert_group_);
    sqlite3_bind_text(stmt_insert_group_, 1, rec.group_id.c_str(),       -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_group_, 2, rec.source.c_str(),         -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_group_, 3, rec.scenario_family.c_str(),-1, SQLITE_STATIC);
    return sqlite3_step(stmt_insert_group_) == SQLITE_DONE;
}

bool ExemplarDatabase::insert_example(const ExemplarRecord& rec, int64_t& out_rowid)
{
    if(!stmt_insert_example_) return false;
    sqlite3_reset(stmt_insert_example_);

    std::string state_json   = rec.state.dump();
    std::string options_json = nlohmann::ordered_json(rec.options).dump();
    std::string target_json  = nlohmann::ordered_json(rec.target).dump();
    std::string sv_json      = rec.score_values.empty()
                                   ? ""
                                   : nlohmann::ordered_json(rec.score_values).dump();
    std::string cf_json      = rec.counterfactual.is_null() ? "" : rec.counterfactual.dump();
    std::string meta_json    = rec.metadata.is_null() ? "{}" : rec.metadata.dump();

    int col = 1;
    sqlite3_bind_text(stmt_insert_example_, col++, rec.id.c_str(),         -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.group_id.c_str(),   -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.kind.c_str(),       -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, state_json.c_str(),     -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.question.c_str(),   -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, options_json.c_str(),   -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_insert_example_, col++, target_json.c_str(),    -1, SQLITE_TRANSIENT);

    if(rec.answer_index >= 0) sqlite3_bind_int(stmt_insert_example_, col++, rec.answer_index);
    else                      sqlite3_bind_null(stmt_insert_example_, col++);

    if(!rec.answer_text.empty())
        sqlite3_bind_text(stmt_insert_example_, col++, rec.answer_text.c_str(), -1, SQLITE_STATIC);
    else
        sqlite3_bind_null(stmt_insert_example_, col++);

    if(rec.number_of_levels > 0)
        sqlite3_bind_int(stmt_insert_example_, col++, rec.number_of_levels);
    else
        sqlite3_bind_null(stmt_insert_example_, col++);

    if(!sv_json.empty())
        sqlite3_bind_text(stmt_insert_example_, col++, sv_json.c_str(), -1, SQLITE_TRANSIENT);
    else
        sqlite3_bind_null(stmt_insert_example_, col++);

    sqlite3_bind_text(stmt_insert_example_, col++, rec.domain.c_str(),           -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.family.c_str(),           -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.language.c_str(),         -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.scenario_family.c_str(),  -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.template_id.c_str(),      -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.source_instance_id.c_str(),-1,SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.semantic_context_sha256.c_str(),-1,SQLITE_STATIC);

    if(!cf_json.empty())
        sqlite3_bind_text(stmt_insert_example_, col++, cf_json.c_str(), -1, SQLITE_TRANSIENT);
    else
        sqlite3_bind_null(stmt_insert_example_, col++);

    sqlite3_bind_text(stmt_insert_example_, col++, meta_json.c_str(),         -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.source.c_str(),        -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.split.c_str(),         -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt_insert_example_, col++, rec.embedding_text.c_str(),-1, SQLITE_STATIC);
    sqlite3_bind_int (stmt_insert_example_, col++, ExemplarRecord::kEmbeddingFormatVersion);

    if(sqlite3_step(stmt_insert_example_) != SQLITE_DONE) return false;
    out_rowid = sqlite3_last_insert_rowid(db_);
    return true;
}

bool ExemplarDatabase::insert_vector(
    int64_t rowid, const std::vector<float>& embedding,
    const std::string& kind, int number_of_levels)
{
    if(!stmt_insert_vector_) return false;
    sqlite3_reset(stmt_insert_vector_);
    sqlite3_bind_int64(stmt_insert_vector_, 1, rowid);
    sqlite3_bind_blob (stmt_insert_vector_, 2,
                       embedding.data(),
                       (int)(embedding.size() * sizeof(float)),
                       SQLITE_STATIC);
    sqlite3_bind_text (stmt_insert_vector_, 3, kind.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int  (stmt_insert_vector_, 4, number_of_levels);
    return sqlite3_step(stmt_insert_vector_) == SQLITE_DONE;
}

bool ExemplarDatabase::finalize_rebuild()
{
    if(!rebuild_ || active_path_ == path_) return true;

    // Flush and close the temp database
    if(stmt_insert_group_)   { sqlite3_finalize(stmt_insert_group_);   stmt_insert_group_   = nullptr; }
    if(stmt_insert_example_) { sqlite3_finalize(stmt_insert_example_); stmt_insert_example_ = nullptr; }
    if(stmt_insert_vector_)  { sqlite3_finalize(stmt_insert_vector_);  stmt_insert_vector_  = nullptr; }
    if(stmt_id_exists_)      { sqlite3_finalize(stmt_id_exists_);      stmt_id_exists_      = nullptr; }
    if(db_) { sqlite3_close(db_); db_ = nullptr; }

    std::remove(path_.c_str());
    if(std::rename(active_path_.c_str(), path_.c_str()) != 0) return false;

    // Prevent destructor from deleting the newly renamed file
    active_path_ = path_;
    return true;
}

int64_t ExemplarDatabase::example_count() const
{
    return query_int64("SELECT COUNT(*) FROM examples");
}

int64_t ExemplarDatabase::file_size() const
{
    return query_int64(
        "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()");
}

bool ExemplarDatabase::exec(const char* sql) const
{
    return sqlite3_exec(db_, sql, nullptr, nullptr, nullptr) == SQLITE_OK;
}

int64_t ExemplarDatabase::query_int64(const char* sql, int64_t fallback) const
{
    sqlite3_stmt* stmt = nullptr;
    if(sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return fallback;
    int64_t result = fallback;
    if(sqlite3_step(stmt) == SQLITE_ROW) result = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
    return result;
}

std::string ExemplarDatabase::query_text(const char* sql, const char* fallback) const
{
    sqlite3_stmt* stmt = nullptr;
    if(sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) return fallback;
    std::string result = fallback;
    if(sqlite3_step(stmt) == SQLITE_ROW) {
        const char* text = (const char*)sqlite3_column_text(stmt, 0);
        if(text) result = text;
    }
    sqlite3_finalize(stmt);
    return result;
}

} // namespace filldb
} // namespace pjev
