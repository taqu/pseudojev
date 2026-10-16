#include "r0_retrieval.h"

#include <cassert>
#include <chrono>
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
    if (stmt_id_)      { sqlite3_finalize(stmt_id_);      stmt_id_      = nullptr; }
    if (stmt_payload_) { sqlite3_finalize(stmt_payload_); stmt_payload_ = nullptr; }
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
        "SELECT id, group_id, kind, answer_index, options_lz4, number_of_levels "
        "FROM examples WHERE rowid=?";
    if (sqlite3_prepare_v2(db_, resolve_sql, -1, &stmt_resolve_, nullptr) != SQLITE_OK) {
        return false;
    }
    if (sqlite3_prepare_v2(db_, "SELECT id FROM examples WHERE rowid=?",
                           -1, &stmt_id_, nullptr) != SQLITE_OK) {
        return false;
    }
    const char* payload_sql =
        "SELECT id, group_id, kind, question, answer_index, number_of_levels, "
        "       domain, family, state_lz4, options_lz4 "
        "FROM examples WHERE id=?";
    if (sqlite3_prepare_v2(db_, payload_sql, -1, &stmt_payload_, nullptr) != SQLITE_OK) {
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
    // Require schema version 4
    std::string db_schema_ver = get_info("schema_version");
    if (db_schema_ver != "4") {
        err = "schema version '" + db_schema_ver + "' is not supported "
              "(expected '4'); rebuild the database with: pjev-filldb --rebuild";
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

    ex.example_id  = sqlite3_column_int64(stmt_resolve_, 0);
    ex.group_id    = sqlite3_column_int64(stmt_resolve_, 1);
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
    return true;
}

int64_t R0Reader::rowid_to_id(int64_t rowid) const
{
    sqlite3_reset(stmt_id_);
    sqlite3_bind_int64(stmt_id_, 1, rowid);
    if (sqlite3_step(stmt_id_) != SQLITE_ROW) return 0;
    return sqlite3_column_int64(stmt_id_, 0);
}

void R0Reader::begin_result(const filldb::ExemplarRecord& query, QueryResult& out) const
{
    out.query_id                  = query.id;
    out.kind                      = query.kind;
    out.number_of_levels          = query.number_of_levels;
    out.query_group_id            = query.group_id;
    out.query_answer_index        = query.answer_index;
    out.query_ground_truth        = query.answer_text;
    out.retrieved.clear();
    out.retrieved_count = 0;
}

void R0Reader::set_relation_flags(const filldb::ExemplarRecord& query, RetrievedExample& ex) const
{
    ex.same_group = (ex.group_id == query.group_id);
}

bool R0Reader::knn(
    const filldb::ExemplarRecord& query,
    const std::vector<float>&     embedding,
    int                           fetch_k,
    std::vector<Candidate>&       candidates) const
{
    candidates.clear();
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

    begin_result(query, out);

    // Overfetch when excluding same group so we can still return true top-k.
    int fetch_k = exclude_same_group
                  ? std::max(top_k * 4, top_k + 20)
                  : top_k;

    std::vector<Candidate> candidates;
    if (!knn(query, embedding, fetch_k, candidates)) return false;

    int rank = 1;
    for (const auto& c : candidates) {
        if (out.retrieved_count >= top_k) break;

        RetrievedExample ex;
        ex.distance  = c.distance;
        ex.similarity = std::max(0.0, 1.0 - c.distance);

        if (!resolve_rowid(c.rowid, ex)) continue;

        if (exclude_same_group && ex.group_id == query.group_id) continue;

        ex.rank = rank++;
        set_relation_flags(query, ex);

        out.retrieved.push_back(std::move(ex));
        out.retrieved_count++;
    }

    return true;
}

bool R0Reader::search_filtered(
    const filldb::ExemplarRecord&         query,
    const std::vector<float>&             embedding,
    int                                   top_k,
    const std::function<bool(int64_t)>&   eligible,
    int                                   max_fetch_k,
    QueryResult&                          out,
    int*                                  fetched_k_out) const
{
    assert(db_ != nullptr);
    assert(!embedding.empty());

    max_fetch_k = std::min(max_fetch_k, kMaxFetchK);
    int fetch_k = std::min(std::max(top_k, 1), max_fetch_k);
    std::vector<Candidate> candidates;
    for (;;) {
        begin_result(query, out);
        if (!knn(query, embedding, fetch_k, candidates)) return false;

        int rank = 1;
        for (const auto& c : candidates) {
            if (out.retrieved_count >= top_k) break;
            // Cheap id lookup first; payload columns are only read for accepted rows.
            if (!eligible(rowid_to_id(c.rowid))) continue;

            RetrievedExample ex;
            ex.distance   = c.distance;
            ex.similarity = std::max(0.0, 1.0 - c.distance);
            if (!resolve_rowid(c.rowid, ex)) continue;

            ex.rank = rank++;
            set_relation_flags(query, ex);
            out.retrieved.push_back(std::move(ex));
            out.retrieved_count++;
        }

        // Done when filled, when the index has no more candidates, or at the cap.
        if (out.retrieved_count >= top_k ||
            static_cast<int>(candidates.size()) < fetch_k ||
            fetch_k >= max_fetch_k) {
            break;
        }
        fetch_k = std::min(fetch_k * 4, max_fetch_k);
    }
    if (fetched_k_out) *fetched_k_out = fetch_k;
    return true;
}

bool R0Reader::load_payload(int64_t example_id, ExemplarPayload& out, std::string& err) const
{
    assert(db_ != nullptr);
    err.clear();
    sqlite3_reset(stmt_payload_);
    sqlite3_bind_int64(stmt_payload_, 1, example_id);
    if (sqlite3_step(stmt_payload_) != SQLITE_ROW) {
        err = "exemplar not found";
        return false;
    }

    auto col_text = [&](int c) -> std::string {
        const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt_payload_, c));
        return v ? v : "";
    };

    out.example_id         = sqlite3_column_int64(stmt_payload_, 0);
    out.group_id           = sqlite3_column_int64(stmt_payload_, 1);
    out.kind               = col_text(2);
    out.question           = col_text(3);
    out.answer_index       = (sqlite3_column_type(stmt_payload_, 4) == SQLITE_NULL)
                             ? -1 : sqlite3_column_int(stmt_payload_, 4);
    out.number_of_levels   = sqlite3_column_int(stmt_payload_, 5);
    out.domain             = col_text(6);
    out.family             = col_text(7);

    auto t0 = std::chrono::steady_clock::now();
    std::string state_json = pjev::lz4::decompress(
        sqlite3_column_blob(stmt_payload_, 8),
        static_cast<size_t>(sqlite3_column_bytes(stmt_payload_, 8)), err);
    if (!err.empty()) { err = "state_lz4: " + err; return false; }
    std::string options_json = pjev::lz4::decompress(
        sqlite3_column_blob(stmt_payload_, 9),
        static_cast<size_t>(sqlite3_column_bytes(stmt_payload_, 9)), err);
    if (!err.empty()) { err = "options_lz4: " + err; return false; }

    try {
        out.state = nlohmann::ordered_json::parse(state_json);
        auto opts = nlohmann::ordered_json::parse(options_json);
        if (!opts.is_array()) { err = "options payload is not an array"; return false; }
        out.options.clear();
        for (const auto& o : opts) {
            if (!o.is_string()) { err = "options payload contains a non-string"; return false; }
            out.options.push_back(o.get<std::string>());
        }
    } catch (const std::exception& e) {
        err = std::string("payload JSON: ") + e.what();
        return false;
    }
    out.decompress_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    if (out.answer_index >= static_cast<int>(out.options.size())) {
        err = "answer_index out of range for stored options";
        return false;
    }
    return true;
}

} // namespace r0
} // namespace pjev
