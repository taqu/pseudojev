#include "exemplar_store.h"

#include <algorithm>

#include "lz4_utils.h"
#include "sqlite/sqlite3.h"
#include "vec_autoext.h"

namespace pjev
{
namespace retrieval
{

ExemplarStore::~ExemplarStore()
{
    close();
}

void ExemplarStore::close()
{
    if (stmt_id_)      { sqlite3_finalize(stmt_id_);      stmt_id_      = nullptr; }
    if (stmt_payload_) { sqlite3_finalize(stmt_payload_); stmt_payload_ = nullptr; }
    if (db_)           { sqlite3_close(db_);               db_           = nullptr; }
}

bool ExemplarStore::open(const std::string& db_path, const std::string& allowlist_path, std::string& err)
{
    close();
    register_sqlite_vec_autoext();
    if (sqlite3_open_v2(db_path.c_str(), &db_, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        err = "cannot open exemplar database read-only: " + db_path;
        if (db_) err += std::string(" (") + sqlite3_errmsg(db_) + ")";
        close();
        return false;
    }
    // Preparing these touches the schema, so a missing table or a file that is not a SQLite database
    // fails here rather than at the first query.
    if (sqlite3_prepare_v2(db_, "SELECT id FROM examples WHERE rowid=?", -1, &stmt_id_, nullptr) != SQLITE_OK ||
        sqlite3_prepare_v2(db_,
                           "SELECT id, kind, question, answer_index, number_of_levels, state_lz4, options_lz4 "
                           "FROM examples WHERE id=?",
                           -1, &stmt_payload_, nullptr) != SQLITE_OK) {
        err = "exemplar database " + db_path + " is not usable: " + sqlite3_errmsg(db_);
        close();
        return false;
    }
    if (!load_allowlist(allowlist_path, allow_, err)) {
        close();
        return false;
    }
    return true;
}

std::string ExemplarStore::info(const char* key) const
{
    if (!db_) return {};
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT value FROM db_info WHERE key=?", -1, &stmt, nullptr) != SQLITE_OK) {
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

bool ExemplarStore::check_compatible(int emb_dim, const std::string& fingerprint, std::string& err) const
{
    const std::string schema = info("schema_version");
    if (schema != "4") {
        err = "exemplar database schema version '" + schema + "' is not supported (expected '4')";
        return false;
    }
    const std::string dim = info("embedding_dim");
    if (dim.empty()) {
        err = "exemplar database has no embedding_dim metadata";
        return false;
    }
    if (dim != std::to_string(emb_dim)) {
        err = "embedding dimension mismatch: model=" + std::to_string(emb_dim) + " database=" + dim;
        return false;
    }
    const std::string db_fp = info("embedding_fingerprint");
    if (!fingerprint.empty() && !db_fp.empty() && fingerprint != db_fp) {
        err = "embedding model fingerprint mismatch: model=" + fingerprint + " database=" + db_fp +
              " (the database was built with a different embedding model)";
        return false;
    }
    if (info("distance_metric") != "cosine") {
        err = "unexpected distance metric in exemplar database: '" + info("distance_metric") + "'";
        return false;
    }
    if (info("normalization") != "L2") {
        err = "unexpected normalization in exemplar database: '" + info("normalization") + "'";
        return false;
    }
    return true;
}

int64_t ExemplarStore::example_count() const
{
    if (!db_) return 0;
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM examples", -1, &stmt, nullptr) != SQLITE_OK) return 0;
    int64_t n = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) n = sqlite3_column_int64(stmt, 0);
    sqlite3_finalize(stmt);
    return n;
}

int64_t ExemplarStore::rowid_to_id(int64_t rowid) const
{
    sqlite3_reset(stmt_id_);
    sqlite3_bind_int64(stmt_id_, 1, rowid);
    if (sqlite3_step(stmt_id_) != SQLITE_ROW) return 0;
    return sqlite3_column_int64(stmt_id_, 0);
}

bool ExemplarStore::knn(const std::string& kind, int number_of_levels, const std::vector<float>& embedding,
                        int fetch_k, std::vector<Candidate>& out, std::string& err) const
{
    out.clear();
    const bool score = kind == "score";
    const std::string sql =
        std::string("SELECT rowid, distance FROM exemplar_vectors WHERE embedding MATCH vec_quantize_binary(?) AND ") +
        (score ? "kind='score' AND number_of_levels=?" : "kind=?") + " AND k=" + std::to_string(fetch_k) +
        " ORDER BY distance";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        err = std::string("exemplar search failed: ") + sqlite3_errmsg(db_);
        return false;
    }
    sqlite3_bind_blob(stmt, 1, embedding.data(), static_cast<int>(embedding.size() * sizeof(float)), SQLITE_STATIC);
    if (score) sqlite3_bind_int(stmt, 2, number_of_levels);
    else       sqlite3_bind_text(stmt, 2, kind.c_str(), -1, SQLITE_STATIC);
    int rc;
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        out.push_back({sqlite3_column_int64(stmt, 0), sqlite3_column_double(stmt, 1)});
    }
    if (rc != SQLITE_DONE) {
        err = std::string("exemplar search failed: ") + sqlite3_errmsg(db_);
        sqlite3_finalize(stmt);
        return false;
    }
    sqlite3_finalize(stmt);
    // Deterministic order: distance, then rowid (sqlite-vec does not order ties).
    std::stable_sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
        if (a.distance != b.distance) return a.distance < b.distance;
        return a.rowid < b.rowid;
    });
    return true;
}

bool ExemplarStore::search(const std::string&        kind,
                           int                       number_of_levels,
                           const std::vector<float>& embedding,
                           int                       top_k,
                           std::vector<SearchHit>&   out,
                           std::string&              err) const
{
    out.clear();
    if (!db_) { err = "exemplar database is not open"; return false; }
    if (embedding.empty()) { err = "empty query embedding"; return false; }
    // Ineligible rows are skipped, so the KNN window grows (x4, capped) until top_k eligible rows are
    // found, the index has no more candidates, or the cap is reached.
    int fetch_k = std::min(std::max(top_k, 1), kMaxFetchK);
    std::vector<Candidate> candidates;
    for (;;) {
        out.clear();
        if (!knn(kind, number_of_levels, embedding, fetch_k, candidates, err)) return false;
        for (const auto& c : candidates) {
            if (static_cast<int>(out.size()) >= top_k) break;
            const int64_t id = rowid_to_id(c.rowid);
            if (!is_allowed(id)) continue;
            out.push_back({id, c.distance});
        }
        if (static_cast<int>(out.size()) >= top_k || static_cast<int>(candidates.size()) < fetch_k ||
            fetch_k >= kMaxFetchK) {
            break;
        }
        fetch_k = std::min(fetch_k * 4, kMaxFetchK);
    }
    return true;
}

bool ExemplarStore::load(int64_t id, ExemplarPayload& out, std::string& err) const
{
    if (!db_) { err = "exemplar database is not open"; return false; }
    sqlite3_reset(stmt_payload_);
    sqlite3_bind_int64(stmt_payload_, 1, id);
    const int rc = sqlite3_step(stmt_payload_);
    if (rc != SQLITE_ROW) {
        err = rc == SQLITE_DONE ? "exemplar " + format_exemplar_id(id) + " not found"
                                : std::string("exemplar read failed: ") + sqlite3_errmsg(db_);
        return false;
    }
    auto text = [&](int c) -> std::string {
        const char* v = reinterpret_cast<const char*>(sqlite3_column_text(stmt_payload_, c));
        return v ? v : "";
    };
    out                  = ExemplarPayload{};
    out.id               = sqlite3_column_int64(stmt_payload_, 0);
    out.kind             = text(1);
    out.question         = text(2);
    out.answer_index     = sqlite3_column_type(stmt_payload_, 3) == SQLITE_NULL ? -1 : sqlite3_column_int(stmt_payload_, 3);
    out.number_of_levels = sqlite3_column_int(stmt_payload_, 4);

    std::string lz4_err;
    const std::string state_json = lz4::decompress(sqlite3_column_blob(stmt_payload_, 5),
        static_cast<size_t>(sqlite3_column_bytes(stmt_payload_, 5)), lz4_err);
    if (!lz4_err.empty()) { err = "exemplar " + format_exemplar_id(id) + " state_lz4: " + lz4_err; return false; }
    const std::string options_json = lz4::decompress(sqlite3_column_blob(stmt_payload_, 6),
        static_cast<size_t>(sqlite3_column_bytes(stmt_payload_, 6)), lz4_err);
    if (!lz4_err.empty()) { err = "exemplar " + format_exemplar_id(id) + " options_lz4: " + lz4_err; return false; }
    try {
        out.state = nlohmann::ordered_json::parse(state_json);
        const auto opts = nlohmann::ordered_json::parse(options_json);
        if (!opts.is_array()) { err = "exemplar " + format_exemplar_id(id) + ": options payload is not an array"; return false; }
        for (const auto& o : opts) {
            if (!o.is_string()) { err = "exemplar " + format_exemplar_id(id) + ": options payload contains a non-string"; return false; }
            out.options.push_back(o.get<std::string>());
        }
    } catch (const std::exception& e) {
        err = "exemplar " + format_exemplar_id(id) + " payload JSON: " + e.what();
        return false;
    }
    if (out.answer_index >= static_cast<int>(out.options.size())) {
        err = "exemplar " + format_exemplar_id(id) + ": answer_index out of range for stored options";
        return false;
    }
    return true;
}

} // namespace retrieval
} // namespace pjev
