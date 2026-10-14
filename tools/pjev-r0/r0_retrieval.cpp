#include "r0_retrieval.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <sstream>

#include "lz4_utils.h"
#include "vec_autoext.h"
#include "nlohmann/json.hpp"

namespace pjev
{
namespace r0
{

R0Reader::R0Reader(std::string db_path)
    : db_path_(std::move(db_path))
{}

R0Reader::~R0Reader()
{
    if (stmt_resolve_) { sqlite3_finalize(stmt_resolve_); stmt_resolve_ = nullptr; }
    if (db_)           { sqlite3_close(db_);               db_           = nullptr; }
}

bool R0Reader::open()
{
    register_sqlite_vec_autoext();

    int flags = SQLITE_OPEN_READONLY;
    if (sqlite3_open_v2(db_path_.c_str(), &db_, flags, nullptr) != SQLITE_OK) {
        return false;
    }

    // v2 schema: no answer_text column; options_lz4 is decompressed to get ground_truth
    const char* resolve_sql =
        "SELECT id, group_id, kind, answer_index, options_lz4, "
        "       number_of_levels, template_id, source_instance_id, scenario_family "
        "FROM examples WHERE rowid=?";
    if (sqlite3_prepare_v2(db_, resolve_sql, -1, &stmt_resolve_, nullptr) != SQLITE_OK) {
        return false;
    }

    return true;
}

std::string R0Reader::get_info(const char* key) const
{
    assert(db_ != nullptr);
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_,
            "SELECT value FROM db_info WHERE key=?", -1, &stmt, nullptr) != SQLITE_OK) {
        return {};
    }
    sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
    std::string result;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (v) result = v;
    }
    sqlite3_finalize(stmt);
    return result;
}

bool R0Reader::check_compatible(int emb_dim, const std::string& fingerprint, std::string& err) const
{
    // Require schema version 2
    std::string db_schema_ver = get_info("schema_version");
    if (db_schema_ver != "2") {
        err = "schema version '" + db_schema_ver + "' is not supported "
              "(expected '2'); rebuild the database with: pjev-filldb --rebuild";
        return false;
    }

    std::string db_dim_str = get_info("embedding_dim");
    std::string db_fp      = get_info("embedding_fingerprint");
    std::string db_metric  = get_info("distance_metric");
    std::string db_norm    = get_info("normalization");

    if (db_dim_str.empty()) {
        err = "database has no embedding_dim metadata — is this a valid exemplar database?";
        return false;
    }

    int db_dim = std::stoi(db_dim_str);
    if (db_dim != emb_dim) {
        err = "embedding dimension mismatch: model=" + std::to_string(emb_dim)
            + " db=" + std::to_string(db_dim);
        return false;
    }

    if (!fingerprint.empty() && !db_fp.empty() && fingerprint != db_fp) {
        err = "model fingerprint mismatch: query model=" + fingerprint
            + " db stored=" + db_fp
            + " — embeddings may be incompatible";
        return false;
    }

    if (db_metric != "cosine") {
        err = "unexpected distance metric in database: '" + db_metric
            + "' (expected 'cosine')";
        return false;
    }

    if (db_norm != "L2") {
        err = "unexpected normalization in database: '" + db_norm
            + "' (expected 'L2')";
        return false;
    }

    return true;
}

int64_t R0Reader::example_count() const
{
    assert(db_ != nullptr);
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_,
            "SELECT COUNT(*) FROM examples", -1, &stmt, nullptr) != SQLITE_OK) {
        return 0;
    }
    int64_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return count;
}

bool R0Reader::resolve_rowid(int64_t rowid, RetrievedExample& ex) const
{
    sqlite3_reset(stmt_resolve_);
    sqlite3_bind_int64(stmt_resolve_, 1, rowid);
    if (sqlite3_step(stmt_resolve_) != SQLITE_ROW) return false;

    auto col_text = [&](int c) -> std::string {
        const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt_resolve_, c));
        return v ? v : "";
    };

    ex.example_id  = col_text(0);
    ex.group_id    = col_text(1);
    ex.kind        = col_text(2);
    ex.answer_index = (sqlite3_column_type(stmt_resolve_, 3) == SQLITE_NULL)
                       ? -1 : sqlite3_column_int(stmt_resolve_, 3);

    // Decompress options_lz4 and reconstruct ground_truth from answer_index
    const void* opts_blob  = sqlite3_column_blob(stmt_resolve_, 4);
    int         opts_bytes = sqlite3_column_bytes(stmt_resolve_, 4);
    std::string decomp_err;
    std::string options_json = pjev::lz4::decompress(opts_blob, (size_t)opts_bytes, decomp_err);
    if (decomp_err.empty() && !options_json.empty() && ex.answer_index >= 0) {
        try {
            auto opts = nlohmann::json::parse(options_json);
            if (opts.is_array() &&
                ex.answer_index < static_cast<int>(opts.size()) &&
                opts[ex.answer_index].is_string()) {
                ex.ground_truth = opts[ex.answer_index].get<std::string>();
            }
        } catch (...) {}
    }

    ex.number_of_levels   = sqlite3_column_int(stmt_resolve_, 5);
    ex.template_id        = col_text(6);
    ex.source_instance_id = col_text(7);
    ex.scenario_family    = col_text(8);
    return true;
}

bool R0Reader::search(
    const filldb::ExemplarRecord& query,
    const std::vector<float>&     embedding,
    int                           top_k,
    bool                          exclude_same_group,
    QueryResult&                  out) const
{
    assert(db_ != nullptr);
    assert(!embedding.empty());

    out.query_id                  = query.id;
    out.kind                      = query.kind;
    out.number_of_levels          = query.number_of_levels;
    out.query_group_id            = query.group_id;
    out.query_template_id         = query.template_id;
    out.query_source_instance_id  = query.source_instance_id;
    out.query_answer_index        = query.answer_index;
    out.query_ground_truth        = query.answer_text;
    out.retrieved.clear();
    out.retrieved_count = 0;

    // Overfetch when excluding same group so we can still return true top-k.
    int fetch_k = exclude_same_group
                  ? std::max(top_k * 4, top_k + 20)
                  : top_k;

    std::string sql;
    if (query.kind == "score") {
        sql = "SELECT rowid, distance FROM exemplar_vectors "
              "WHERE embedding MATCH vec_quantize_binary(?) AND kind='score' AND number_of_levels=? AND k="
              + std::to_string(fetch_k) + " ORDER BY distance";
    } else {
        sql = "SELECT rowid, distance FROM exemplar_vectors "
              "WHERE embedding MATCH vec_quantize_binary(?) AND kind=? AND k="
              + std::to_string(fetch_k) + " ORDER BY distance";
    }

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    int blob_bytes = static_cast<int>(embedding.size() * sizeof(float));
    sqlite3_bind_blob(stmt, 1, embedding.data(), blob_bytes, SQLITE_STATIC);
    if (query.kind == "score") {
        sqlite3_bind_int(stmt, 2, query.number_of_levels);
    } else {
        sqlite3_bind_text(stmt, 2, query.kind.c_str(), -1, SQLITE_STATIC);
    }

    struct Candidate { int64_t rowid; double distance; };
    std::vector<Candidate> candidates;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Candidate c;
        c.rowid    = sqlite3_column_int64(stmt, 0);
        c.distance = sqlite3_column_double(stmt, 1);
        candidates.push_back(c);
    }
    sqlite3_finalize(stmt);

    // Sort by distance for deterministic ordering (tie-break by rowid).
    std::stable_sort(candidates.begin(), candidates.end(),
        [](const Candidate& a, const Candidate& b) {
            if (a.distance != b.distance) return a.distance < b.distance;
            return a.rowid < b.rowid;
        });

    int rank = 1;
    for (const auto& c : candidates) {
        if (out.retrieved_count >= top_k) break;

        RetrievedExample ex;
        ex.distance  = c.distance;
        ex.similarity = std::max(0.0, 1.0 - c.distance);

        if (!resolve_rowid(c.rowid, ex)) continue;

        if (exclude_same_group && ex.group_id == query.group_id) continue;

        ex.rank                 = rank++;
        ex.same_group           = (ex.group_id == query.group_id);
        ex.same_template        = (!query.template_id.empty() &&
                                   ex.template_id == query.template_id);
        ex.same_source_instance = (!query.source_instance_id.empty() &&
                                   ex.source_instance_id == query.source_instance_id);

        out.retrieved.push_back(std::move(ex));
        out.retrieved_count++;
    }

    return true;
}

} // namespace r0
} // namespace pjev
