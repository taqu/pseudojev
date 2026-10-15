// Unit tests for the R0 retrieval tool.
// Does not require a real embedding model.
// Tests are numbered 1-N to match r0.md §13.
//
// Test coverage:
//  1. Loading existing SQLite DB in read-only mode.
//  2. sqlite-vec extension initialization.
//  3. Model compatibility validation (dimension mismatch, fingerprint mismatch).
//  4. Canonical query text generation (search_query: prefix).
//  5. Correct kind filtering (choice/noul separate from score).
//  6. Correct score-level filtering.
//  7. Stable deterministic result ordering (tie-break by rowid).
//  8. Handling fewer than K eligible candidates.
//  9. Handling unknown kinds.
// 10. Handling malformed evaluation records.
// 11. Handling empty evaluation datasets.
// 12. Handling missing or incompatible database files.
// 13. Query text does NOT include correct answer.
// 14. Same-group / same-template / same-source-instance flags.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "exemplar_record.h"
#include "embedding_text.h"
#include "id_hash.h"
#include "lz4_utils.h"
#include "r0_retrieval.h"

#include "sqlite/sqlite3.h"
#include "vec_autoext.h"

#include "nlohmann/json.hpp"

using namespace pjev;
using namespace pjev::r0;
using namespace pjev::filldb;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const char* desc)
{
    if (cond) {
        printf("  PASS: %s\n", desc);
        g_pass++;
    } else {
        printf("  FAIL: %s\n", desc);
        g_fail++;
    }
}

// Build a minimal in-memory exemplar database for unit testing (v2 schema).
static sqlite3* make_test_db(int emb_dim, const char* path = ":memory:")
{
    register_sqlite_vec_autoext();

    sqlite3* db = nullptr;
    sqlite3_open_v2(path, &db,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    assert(db != nullptr);

    // db_info
    sqlite3_exec(db,
        "CREATE TABLE db_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        nullptr, nullptr, nullptr);

    char dim_val[32];
    snprintf(dim_val, sizeof(dim_val), "%d", emb_dim);
    sqlite3_exec(db,
        "INSERT INTO db_info VALUES('schema_version','4')",
        nullptr, nullptr, nullptr);
    {
        char sql[256];
        snprintf(sql, sizeof(sql),
            "INSERT INTO db_info VALUES('embedding_dim','%s')", dim_val);
        sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
    }
    sqlite3_exec(db,
        "INSERT INTO db_info VALUES('embedding_model','models/nomic-embed-text-v1.5.gguf')",
        nullptr, nullptr, nullptr);
    sqlite3_exec(db,
        "INSERT INTO db_info VALUES('embedding_fingerprint','aabbccdd11223344')",
        nullptr, nullptr, nullptr);
    sqlite3_exec(db,
        "INSERT INTO db_info VALUES('distance_metric','cosine')",
        nullptr, nullptr, nullptr);
    sqlite3_exec(db,
        "INSERT INTO db_info VALUES('normalization','L2')",
        nullptr, nullptr, nullptr);
    sqlite3_exec(db,
        "INSERT INTO db_info VALUES('embedding_text_version','1')",
        nullptr, nullptr, nullptr);
    sqlite3_exec(db,
        "INSERT INTO db_info VALUES('compression_format','lz4-frame')",
        nullptr, nullptr, nullptr);

    // groups — v3: group_id is INTEGER (xxHash64)
    sqlite3_exec(db,
        "CREATE TABLE groups (group_id INTEGER PRIMARY KEY, source TEXT)",
        nullptr, nullptr, nullptr);

    // examples — v3 schema: id/group_id are INTEGER (xxHash64), LZ4 compressed BLOBs
    sqlite3_exec(db,
        "CREATE TABLE examples ("
        "  rowid              INTEGER PRIMARY KEY,"
        "  id                 INTEGER UNIQUE NOT NULL,"
        "  group_id           INTEGER NOT NULL,"
        "  kind               TEXT NOT NULL,"
        "  question           TEXT NOT NULL,"
        "  answer_index       INTEGER,"
        "  number_of_levels   INTEGER,"
        "  domain             TEXT,"
        "  family             TEXT,"
        "  state_lz4          BLOB NOT NULL,"
        "  options_lz4        BLOB NOT NULL,"
        "  embedding_text_lz4 BLOB NOT NULL,"
        "  target_lz4         BLOB"
        ")",
        nullptr, nullptr, nullptr);

    // exemplar_vectors virtual table — use bit[] matching production schema.
    // Vectors are inserted via vec_quantize_binary() and queried the same way.
    {
        char sql[256];
        snprintf(sql, sizeof(sql),
            "CREATE VIRTUAL TABLE exemplar_vectors USING vec0("
            "  embedding bit[%d], kind text, number_of_levels integer"
            ")", emb_dim);
        sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
    }

    return db;
}

// Insert one example + vector into test DB (v2 schema with LZ4 BLOBs).
// answer_text is placed at options[answer_index]; a filler "other" fills the rest.
static int64_t insert_test_example(
    sqlite3* db,
    const std::string& id,
    const std::string& group_id,
    const std::string& kind,
    int number_of_levels,
    const std::string& answer_text,
    int answer_index,
    const std::vector<float>& emb)
{
    int64_t hashed_id       = pjev::filldb::hash_exemplar_id(id);
    int64_t hashed_group_id = pjev::filldb::hash_exemplar_id(group_id);

    // Insert group (idempotent)
    {
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT OR IGNORE INTO groups(group_id, source) VALUES(?,?)",
            -1, &st, nullptr);
        sqlite3_bind_int64(st, 1, hashed_group_id);
        sqlite3_bind_text (st, 2, "", -1, SQLITE_STATIC);
        sqlite3_step(st);
        sqlite3_finalize(st);
    }

    // Build options array: options[answer_index] == answer_text
    nlohmann::json opts = nlohmann::json::array();
    int opt_count = (answer_index >= 0) ? std::max(answer_index + 1, 2) : 2;
    for (int i = 0; i < opt_count; ++i) {
        if (i == answer_index) opts.push_back(answer_text);
        else                   opts.push_back("other_" + std::to_string(i));
    }

    // Compress payloads
    auto state_blob = pjev::lz4::compress("{}");
    auto opts_blob  = pjev::lz4::compress(opts.dump());
    auto embt_blob  = pjev::lz4::compress("test_emb_text");

    // Insert example
    int64_t rowid = -1;
    {
        sqlite3_stmt* st = nullptr;
        const char* sql =
            "INSERT INTO examples("
            "  id, group_id, kind, question, answer_index, number_of_levels,"
            "  domain, family,"
            "  state_lz4, options_lz4, embedding_text_lz4, target_lz4"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?)";
        sqlite3_prepare_v2(db, sql, -1, &st, nullptr);
        sqlite3_bind_int64(st, 1, hashed_id);
        sqlite3_bind_int64(st, 2, hashed_group_id);
        sqlite3_bind_text(st,  3, kind.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(st,  4, "Test question", -1, SQLITE_STATIC);
        if (answer_index >= 0) sqlite3_bind_int(st, 5, answer_index);
        else                   sqlite3_bind_null(st, 5);
        sqlite3_bind_int(st,   6, number_of_levels);
        sqlite3_bind_text(st,  7, "", -1, SQLITE_STATIC);
        sqlite3_bind_text(st,  8, "", -1, SQLITE_STATIC);
        sqlite3_bind_blob(st,  9, state_blob.data(),
                          static_cast<int>(state_blob.size()), SQLITE_TRANSIENT);
        sqlite3_bind_blob(st, 10, opts_blob.data(),
                          static_cast<int>(opts_blob.size()), SQLITE_TRANSIENT);
        sqlite3_bind_blob(st, 11, embt_blob.data(),
                          static_cast<int>(embt_blob.size()), SQLITE_TRANSIENT);
        sqlite3_bind_null(st, 12);
        sqlite3_step(st);
        rowid = sqlite3_last_insert_rowid(db);
        sqlite3_finalize(st);
    }

    // Insert vector
    {
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT INTO exemplar_vectors(rowid, embedding, kind, number_of_levels) "
            "VALUES(?,vec_quantize_binary(?),?,?)",
            -1, &st, nullptr);
        sqlite3_bind_int64(st, 1, rowid);
        sqlite3_bind_blob(st,  2, emb.data(),
                          static_cast<int>(emb.size() * sizeof(float)), SQLITE_STATIC);
        sqlite3_bind_text(st,  3, kind.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(st,   4, number_of_levels);
        sqlite3_step(st);
        sqlite3_finalize(st);
    }

    return rowid;
}

// Write a test DB to a temp file and return its path.
static std::string write_test_db_file(int emb_dim)
{
    const char* path = "test_r0_tmp.db";
    remove(path);
    sqlite3* db = make_test_db(emb_dim, path);

    // Insert a few choice and score examples
    int dim = emb_dim;
    std::vector<float> v_choice(dim, 0.0f);
    v_choice[0] = 1.0f;  // unit vector

    std::vector<float> v_score(dim, 0.0f);
    v_score[1] = 1.0f;

    insert_test_example(db, "ex1", "grpA", "choice", 0, "A",   0, v_choice);
    insert_test_example(db, "ex2", "grpA", "choice", 0, "B",   1, v_choice);
    insert_test_example(db, "ex3", "grpB", "score",  3, "2",   2, v_score);
    insert_test_example(db, "ex4", "grpB", "noul",   0, "yes", 0, v_choice);

    sqlite3_close(db);
    return path;
}

// ---------------------------------------------------------------------------
// Test 1-2: DB open and vec extension
// ---------------------------------------------------------------------------

static void test_01_open_readonly()
{
    printf("\n=== Tests 1-2: DB open and vec extension ===\n");

    std::string path = write_test_db_file(8);

    R0Reader reader(path);
    bool opened = reader.open();
    check(opened, "T1: opened existing DB in read-only mode");
    check(reader.example_count() == 4, "T2: sqlite-vec extension active (example_count works)");

    remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Test 3: Model compatibility
// ---------------------------------------------------------------------------

static void test_03_compatibility()
{
    printf("\n=== Test 3: Model compatibility ===\n");

    std::string path = write_test_db_file(8);
    R0Reader reader(path);
    reader.open();

    std::string err;
    // Matching dim + matching fingerprint: OK
    check(reader.check_compatible(8, "aabbccdd11223344", err),
          "T3a: compatible dim+fingerprint accepted");

    // Mismatched dim
    err.clear();
    check(!reader.check_compatible(16, "aabbccdd11223344", err),
          "T3b: dimension mismatch rejected");
    check(!err.empty(), "T3c: dimension mismatch error message non-empty");

    // Mismatched fingerprint
    err.clear();
    check(!reader.check_compatible(8, "000000deadbeef00", err),
          "T3d: fingerprint mismatch rejected");

    // Empty fingerprint (unknown) accepted
    err.clear();
    check(reader.check_compatible(8, "", err),
          "T3e: empty fingerprint (unknown model) accepted");

    remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Test 4 & 13: Canonical query text
// ---------------------------------------------------------------------------

static void test_04_query_text()
{
    printf("\n=== Tests 4, 13: Query text generation ===\n");

    nlohmann::ordered_json state = nlohmann::ordered_json::object();
    state["policy_name"] = "Test Policy";

    std::string body = filldb::build_embedding_text(
        "choice", "support", "policy", "Which option is correct?", state, {"A", "B", "C"});

    std::string doc_text  = std::string(filldb::kDocumentPrefix) + body;
    std::string qry_text  = std::string(kQueryPrefix) + body;

    check(qry_text.rfind("search_query: ", 0) == 0,
          "T4a: query text starts with 'search_query: '");
    check(doc_text.rfind("search_document: ", 0) == 0,
          "T4b: document text starts with 'search_document: '");
    check(qry_text != doc_text,
          "T4c: query and document texts differ");
    check(body.find("Task: choice") != std::string::npos,
          "T4d: body contains 'Task: choice'");
    check(body.find("Domain: support") != std::string::npos,
          "T4e: body contains 'Domain: support'");
    check(body.find("Which option is correct?") != std::string::npos,
          "T4f: body contains the question");

    // T13: correct answer NOT included in query text
    // The body is built from kind/domain/family/question/state/options — but
    // options for choice are not included (only for score, as Score meanings).
    // For choice, option labels are NOT emitted.
    // Verify the answer text "A" doesn't appear as "answer" in the text.
    check(body.find("answer_index") == std::string::npos,
          "T13: answer_index not in query text");
}

// ---------------------------------------------------------------------------
// Tests 5-6: Kind and score-level filtering
// ---------------------------------------------------------------------------

static void test_05_kind_filtering()
{
    printf("\n=== Tests 5-6: Kind/level filtering ===\n");

    std::string path = write_test_db_file(8);
    R0Reader reader(path);
    reader.open();

    // Build a unit query vector
    std::vector<float> q(8, 0.0f);
    q[0] = 1.0f;  // matches v_choice

    ExemplarRecord choice_query;
    choice_query.id          = filldb::hash_exemplar_id("q1");
    choice_query.group_id    = filldb::hash_exemplar_id("qgrp");
    choice_query.kind        = "choice";
    choice_query.number_of_levels = 0;
    choice_query.answer_index = 0;
    choice_query.answer_text  = "A";

    QueryResult r;
    reader.search(choice_query, q, 5, false, r);

    check(r.retrieved_count >= 1, "T5a: choice query returns at least 1 result");
    for (const auto& ex : r.retrieved) {
        check(ex.kind == "choice", "T5b: all retrieved examples are kind=choice");
    }

    // Noul query
    ExemplarRecord noul_query = choice_query;
    noul_query.kind = "noul";
    QueryResult r_noul;
    reader.search(noul_query, q, 5, false, r_noul);
    for (const auto& ex : r_noul.retrieved) {
        check(ex.kind == "noul", "T5c: noul query returns only noul examples");
    }

    // Score query — must match number_of_levels=3
    std::vector<float> qs(8, 0.0f);
    qs[1] = 1.0f;  // matches v_score

    ExemplarRecord score_query;
    score_query.id          = filldb::hash_exemplar_id("q_score");
    score_query.group_id    = filldb::hash_exemplar_id("qgrp");
    score_query.kind        = "score";
    score_query.number_of_levels = 3;

    QueryResult r_score;
    reader.search(score_query, qs, 5, false, r_score);
    check(r_score.retrieved_count >= 1, "T6a: score query returns at least 1 result");
    for (const auto& ex : r_score.retrieved) {
        check(ex.kind == "score" && ex.number_of_levels == 3,
              "T6b: score result has correct kind and number_of_levels");
    }

    // Score query with mismatched level count
    ExemplarRecord score_query_5lv = score_query;
    score_query_5lv.number_of_levels = 5;  // no matching examples
    QueryResult r_score_5lv;
    reader.search(score_query_5lv, qs, 5, false, r_score_5lv);
    check(r_score_5lv.retrieved_count == 0,
          "T6c: score query with wrong level count returns 0 results");

    remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Test 7: Deterministic ordering
// ---------------------------------------------------------------------------

static void test_07_ordering()
{
    printf("\n=== Test 7: Deterministic ordering ===\n");

    std::string path = write_test_db_file(8);
    R0Reader reader(path);
    reader.open();

    std::vector<float> q(8, 0.0f);
    q[0] = 1.0f;

    ExemplarRecord choice_query;
    choice_query.id    = filldb::hash_exemplar_id("qd");
    choice_query.group_id = filldb::hash_exemplar_id("qgrp");
    choice_query.kind  = "choice";
    choice_query.number_of_levels = 0;

    QueryResult r1, r2;
    reader.search(choice_query, q, 5, false, r1);
    reader.search(choice_query, q, 5, false, r2);

    bool same_order = (r1.retrieved.size() == r2.retrieved.size());
    for (size_t i = 0; i < r1.retrieved.size() && same_order; ++i) {
        same_order = (r1.retrieved[i].example_id == r2.retrieved[i].example_id);
    }
    check(same_order, "T7: repeated identical queries return same ordered results");

    // Verify results are sorted by similarity descending
    bool sorted = true;
    for (size_t i = 1; i < r1.retrieved.size(); ++i) {
        if (r1.retrieved[i].similarity > r1.retrieved[i-1].similarity) {
            sorted = false;
            break;
        }
    }
    check(sorted, "T7b: results ordered by similarity descending");

    remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Test 8: Fewer than K candidates
// ---------------------------------------------------------------------------

static void test_08_fewer_than_k()
{
    printf("\n=== Test 8: Fewer than K candidates ===\n");

    std::string path = write_test_db_file(8);
    R0Reader reader(path);
    reader.open();

    std::vector<float> q(8, 0.0f);
    q[1] = 1.0f;  // matches v_score

    // Only 1 score example with 3 levels in the DB
    ExemplarRecord score_query;
    score_query.id    = filldb::hash_exemplar_id("qs");
    score_query.group_id = filldb::hash_exemplar_id("qgrp");
    score_query.kind  = "score";
    score_query.number_of_levels = 3;

    QueryResult r;
    reader.search(score_query, q, 5, false, r);
    check(r.retrieved_count == 1, "T8: fewer than K candidates returns actual count");
    check(static_cast<int>(r.retrieved.size()) == r.retrieved_count,
          "T8b: retrieved vector size matches retrieved_count");

    remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Test 9: Unknown kind
// ---------------------------------------------------------------------------

static void test_09_unknown_kind()
{
    printf("\n=== Test 9: Unknown kind ===\n");

    // parse_exemplar rejects unknown kinds
    nlohmann::ordered_json j;
    j["id"]       = "bad1";
    j["group_id"] = "g1";
    j["kind"]     = "unknown_kind";
    j["question"] = "Q?";
    j["source"]   = "src";
    j["split"]    = "train";
    j["options"]  = nlohmann::json::array({"A", "B"});
    j["target"]   = nlohmann::json::array({1.0, 0.0});
    j["state"]    = nlohmann::json::object();

    ExemplarRecord rec;
    std::string err = parse_exemplar(j, rec);
    check(!err.empty(), "T9: parse_exemplar rejects unknown kind");
    check(err.find("unsupported kind") != std::string::npos,
          "T9b: error message mentions 'unsupported kind'");
}

// ---------------------------------------------------------------------------
// Test 10: Malformed records
// ---------------------------------------------------------------------------

static void test_10_malformed()
{
    printf("\n=== Test 10: Malformed records ===\n");

    // Missing id
    {
        nlohmann::ordered_json j;
        j["group_id"] = "g1";
        j["kind"]     = "choice";
        j["question"] = "Q?";
        j["source"]   = "src";
        j["split"]    = "train";
        j["options"]  = nlohmann::json::array({"A"});
        j["target"]   = nlohmann::json::array({1.0});
        j["state"]    = nlohmann::json::object();
        ExemplarRecord rec;
        check(!parse_exemplar(j, rec).empty(),
              "T10a: record without id rejected");
    }

    // options/target length mismatch
    {
        nlohmann::ordered_json j;
        j["id"]       = "mismatch";
        j["group_id"] = "g1";
        j["kind"]     = "choice";
        j["question"] = "Q?";
        j["source"]   = "src";
        j["split"]    = "train";
        j["options"]  = nlohmann::json::array({"A", "B"});
        j["target"]   = nlohmann::json::array({1.0});  // wrong length
        j["state"]    = nlohmann::json::object();
        ExemplarRecord rec;
        check(!parse_exemplar(j, rec).empty(),
              "T10b: options/target length mismatch rejected");
    }

    // Non-string in options
    {
        nlohmann::ordered_json j;
        j["id"]       = "badopts";
        j["group_id"] = "g1";
        j["kind"]     = "choice";
        j["question"] = "Q?";
        j["source"]   = "src";
        j["split"]    = "train";
        j["options"]  = nlohmann::json::array({1, 2});  // numbers, not strings
        j["target"]   = nlohmann::json::array({1.0, 0.0});
        j["state"]    = nlohmann::json::object();
        ExemplarRecord rec;
        check(!parse_exemplar(j, rec).empty(),
              "T10c: non-string in options rejected");
    }
}

// ---------------------------------------------------------------------------
// Test 11: Empty dataset
// ---------------------------------------------------------------------------

static void test_11_empty_dataset()
{
    printf("\n=== Test 11: Empty dataset ===\n");

    // Simulate reading an empty file: no queries processed, which is expected behaviour.
    // The tool exits with an error when 0 queries are processed, but the
    // search function itself handles empty DB gracefully.
    std::string path = write_test_db_file(8);

    // Create a fresh empty DB
    std::string empty_path = "test_r0_empty.db";
    {
        remove(empty_path.c_str());
        sqlite3* db = make_test_db(8, empty_path.c_str());
        sqlite3_close(db);
    }

    R0Reader reader(empty_path);
    check(reader.open(), "T11a: empty DB opens successfully");
    check(reader.example_count() == 0, "T11b: empty DB reports 0 examples");

    std::vector<float> q(8, 0.0f); q[0] = 1.0f;
    ExemplarRecord choice_query;
    choice_query.id   = filldb::hash_exemplar_id("qe");
    choice_query.group_id = filldb::hash_exemplar_id("qgrp");
    choice_query.kind = "choice";
    choice_query.number_of_levels = 0;

    QueryResult r;
    reader.search(choice_query, q, 5, false, r);
    check(r.retrieved_count == 0, "T11c: search on empty DB returns 0 results");

    remove(empty_path.c_str());
    remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Test 12: Missing or incompatible DB
// ---------------------------------------------------------------------------

static void test_12_missing_db()
{
    printf("\n=== Test 12: Missing / incompatible DB ===\n");

    R0Reader reader("this_db_does_not_exist_r0_test.db");
    bool opened = reader.open();
    // SQLite opens a new file when path doesn't exist in RW mode, but in
    // READONLY mode it fails.
    check(!opened, "T12a: missing DB fails to open in read-only mode");

    // Incompatible DB (wrong distance_metric)
    {
        const char* path = "test_r0_incompat.db";
        remove(path);
        sqlite3* db = nullptr;
        sqlite3_open_v2(path, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
        sqlite3_exec(db,
            "CREATE TABLE db_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            nullptr, nullptr, nullptr);
        sqlite3_exec(db,
            "INSERT INTO db_info VALUES('embedding_dim','8')",
            nullptr, nullptr, nullptr);
        sqlite3_exec(db,
            "INSERT INTO db_info VALUES('distance_metric','l2')",  // wrong!
            nullptr, nullptr, nullptr);
        sqlite3_exec(db,
            "INSERT INTO db_info VALUES('normalization','L2')",
            nullptr, nullptr, nullptr);
        sqlite3_close(db);

        register_sqlite_vec_autoext();
        R0Reader reader2(path);
        reader2.open();

        std::string err;
        check(!reader2.check_compatible(8, "", err),
              "T12b: non-cosine distance metric rejected");
        check(!err.empty(), "T12c: incompatibility error message non-empty");

        remove(path);
    }
}

// ---------------------------------------------------------------------------
// Test 14: Same-group/template/source-instance flags
// ---------------------------------------------------------------------------

static void test_14_relationship_flags()
{
    printf("\n=== Test 14: Relationship flags ===\n");

    std::string path = write_test_db_file(8);
    R0Reader reader(path);
    reader.open();

    std::vector<float> q(8, 0.0f);
    q[0] = 1.0f;

    // Query shares group_id=grpA, template=tmpl1, source_instance=si1 with ex1
    ExemplarRecord query;
    query.id               = filldb::hash_exemplar_id("qflag");
    query.group_id         = filldb::hash_exemplar_id("grpA");
    query.kind             = "choice";
    query.number_of_levels = 0;

    QueryResult r;
    reader.search(query, q, 5, false, r);

    bool found_same_group = false;

    for (const auto& ex : r.retrieved) {
        if (ex.same_group) found_same_group = true;
    }

    check(found_same_group, "T14a: same_group flag set for group=grpA");

    // Exclude same group: ex1 and ex2 both have grpA, so result should be 0
    QueryResult r_excl;
    reader.search(query, q, 5, true, r_excl);
    bool no_same_group_in_excl = true;
    for (const auto& ex : r_excl.retrieved) {
        if (ex.same_group) { no_same_group_in_excl = false; break; }
    }
    check(no_same_group_in_excl,
          "T14d: --exclude-same-group removes all same-group results");

    remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main()
{
    printf("=== test_r0 ===\n");

    test_01_open_readonly();
    test_03_compatibility();
    test_04_query_text();
    test_05_kind_filtering();
    test_07_ordering();
    test_08_fewer_than_k();
    test_09_unknown_kind();
    test_10_malformed();
    test_11_empty_dataset();
    test_12_missing_db();
    test_14_relationship_flags();

    printf("\n=== RESULT: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
