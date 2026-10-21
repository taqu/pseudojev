// test_filldb.cpp — unit tests for pjevfilldb components (no llama model required)

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "lz4/xxhash.h"

#include "exemplar_record.h"
#include "retrieval/embedding_text.h"
#include "exemplar_database.h"
#include "retrieval/id_hash.h"
#include "retrieval/lz4_utils.h"
#include "retrieval/vec_autoext.h"
#include "nlohmann/json.hpp"
#include "sqlite/sqlite3.h"

using namespace pjev::filldb;
using namespace pjev::retrieval;
using json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
static int g_fails = 0;

static void check(bool cond, const char* what)
{
    if(cond) {
        printf("pass: %s\n", what);
    } else {
        printf("FAIL: %s\n", what);
        ++g_fails;
    }
}

// ---------------------------------------------------------------------------
// Helper JSON factories
// ---------------------------------------------------------------------------
static json make_choice_json()
{
    return json::parse(R"({
        "id": "c1", "group_id": "g1", "kind": "choice",
        "question": "Which option?", "options": ["A","B","C"],
        "target": [0.0, 1.0, 0.0], "split": "train", "source": "test",
        "state": {"policy_name": "Test policy"},
        "metadata": {"language": "en", "domain": "test", "family": "basic"}
    })");
}

static json make_noul_json()
{
    return json::parse(R"({
        "id": "n1", "group_id": "g1", "kind": "noul",
        "question": "Is it valid?", "options": ["yes","no"],
        "target": [1.0, 0.0], "split": "train", "source": "test",
        "state": {},
        "metadata": {"language": "en", "domain": "logic", "family": "basic"}
    })");
}

static json make_score_json(int levels = 5)
{
    json opts = json::array();
    json tgt  = json::array();
    for(int i = 0; i < levels; ++i) {
        opts.push_back("Level " + std::to_string(i));
        tgt.push_back(i == 2 ? 1.0 : 0.0);
    }
    json sv = json::array();
    for(int i = 0; i < levels; ++i) sv.push_back(i);

    return json{
        {"id",       "s1"},
        {"group_id", "g1"},
        {"kind",     "score"},
        {"question", "Rate this."},
        {"options",  opts},
        {"target",   tgt},
        {"split",    "train"},
        {"source",   "test"},
        {"state",    json::object()},
        {"metadata", json{{"language","en"},{"domain","eval"},{"family","rating"},
                          {"score_values", sv}}}
    };
}

// ---------------------------------------------------------------------------
// Temp-file helper
// ---------------------------------------------------------------------------
static std::string tmp_db_path(const char* suffix = "")
{
    return std::string("test_filldb_unit") + suffix + ".db";
}

static void remove_file(const std::string& p)
{
    std::remove(p.c_str());
    std::remove((p + ".tmp").c_str());
    std::remove((p + "-wal").c_str());
    std::remove((p + "-shm").c_str());
}

// ---------------------------------------------------------------------------
// Helper: build a minimal valid ExemplarRecord for DB tests
// ---------------------------------------------------------------------------
static ExemplarRecord make_example_record(
    const std::string& id   = "ex1",
    const std::string& kind = "choice",
    int                nol  = 0)
{
    ExemplarRecord r;
    r.id           = hash_exemplar_id(id);
    r.group_id     = hash_exemplar_id("grp1");
    r.kind         = kind;
    r.source       = "unit";
    r.split        = "train";
    r.question     = "Test question for " + id;
    r.options      = {"A", "B"};
    r.target       = {1.0f, 0.0f};
    r.state        = json::object();
    r.answer_index = 0;
    r.answer_text  = "A";
    r.number_of_levels = nol;
    r.language     = "en";
    r.domain       = "test";
    r.family       = "basic";
    r.embedding_text = "Task: " + kind + "\n\nQuestion:\n" + r.question;
    return r;
}

// Unit-norm 8-float vectors used for KNN smoke tests.
// vec_quantize_binary requires length divisible by 8.
static std::vector<float> vec_choice() { return {1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}; }
static std::vector<float> vec_noul()   { return {0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}; }
static std::vector<float> vec_score5() { return {0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f}; }
static std::vector<float> vec_score4() { return {0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,0.0f}; }

// ---------------------------------------------------------------------------
// ===== SECTION 1 – JSON parsing (parse_exemplar) =====
// ---------------------------------------------------------------------------
static void test_parse_exemplar()
{
    // 1. Valid choice record
    {
        ExemplarRecord r;
        json j = make_choice_json();
        std::string err = parse_exemplar(j, r);
        check(err.empty(),         "1. valid choice: no error");
        check(r.id == hash_exemplar_id("c1"),       "1. valid choice: id");
        check(r.group_id == hash_exemplar_id("g1"), "1. valid choice: group_id");
        check(r.kind == "choice",  "1. valid choice: kind");
        check(r.question == "Which option?", "1. valid choice: question");
        check(r.options.size() == 3, "1. valid choice: options size");
        check(r.language == "en",  "1. valid choice: language");
        check(r.domain == "test",  "1. valid choice: domain");
        check(r.family == "basic", "1. valid choice: family");
    }

    // 2. Valid noul record
    {
        ExemplarRecord r;
        json j = make_noul_json();
        std::string err = parse_exemplar(j, r);
        check(err.empty(),           "2. valid noul: no error");
        check(r.kind == "noul",      "2. valid noul: kind");
        check(r.answer_index == 0,   "2. valid noul: answer_index=0");
    }

    // 3. Valid score record
    {
        ExemplarRecord r;
        json j = make_score_json(5);
        std::string err = parse_exemplar(j, r);
        check(err.empty(),              "3. valid score: no error");
        check(r.number_of_levels == 5, "3. valid score: number_of_levels=5");
        check(!r.score_values.empty(), "3. valid score: score_values present");
    }

    // 4. Missing required field "id"
    {
        ExemplarRecord r;
        json j = make_choice_json();
        j.erase("id");
        std::string err = parse_exemplar(j, r);
        check(!err.empty(), "4. missing id: returns error");
    }

    // 5. Missing required field "question"
    {
        ExemplarRecord r;
        json j = make_choice_json();
        j.erase("question");
        std::string err = parse_exemplar(j, r);
        check(!err.empty(), "5. missing question: returns error");
    }

    // 6. Missing required field "options"
    {
        ExemplarRecord r;
        json j = make_choice_json();
        j.erase("options");
        std::string err = parse_exemplar(j, r);
        check(!err.empty(), "6. missing options: returns error");
    }

    // 7. Optional metadata absent
    {
        ExemplarRecord r;
        json j = make_choice_json();
        j.erase("metadata");
        std::string err = parse_exemplar(j, r);
        check(err.empty(),       "7. missing metadata: parses OK");
        check(r.language.empty(), "7. missing metadata: language empty");
        check(r.domain.empty(),   "7. missing metadata: domain empty");
        check(r.family.empty(),   "7. missing metadata: family empty");
    }

    // 8. Malformed JSON is caught at caller level (not parse_exemplar's job)
    //    — we just verify that parse_exemplar itself doesn't crash on structured-but-odd input
    {
        ExemplarRecord r;
        json j = json::object(); // empty object — parse_exemplar should return error
        std::string err = parse_exemplar(j, r);
        check(!err.empty(), "8. empty object: returns error (caller would handle malformed JSON)");
    }

    // 9. Unknown kind returns error
    {
        ExemplarRecord r;
        json j = make_choice_json();
        j["kind"] = "weird_kind";
        std::string err = parse_exemplar(j, r);
        check(!err.empty(), "9. unknown kind: returns error");
    }

    // 10. One-hot target with index=1
    {
        ExemplarRecord r;
        json j = make_choice_json(); // target=[0,1,0], options=["A","B","C"]
        std::string err = parse_exemplar(j, r);
        check(err.empty(),          "10. one-hot index=1: no error");
        check(r.answer_index == 1,  "10. one-hot index=1: answer_index=1");
        check(r.answer_text == "B", "10. one-hot index=1: answer_text=B");
    }

    // 11. Non-one-hot target → answer_index=-1, answer_text=""
    {
        ExemplarRecord r;
        json j = make_choice_json();
        j["target"] = json::array({0.5, 0.5, 0.0});
        std::string err = parse_exemplar(j, r);
        check(err.empty(),          "11. non-one-hot: no error");
        check(r.answer_index == -1, "11. non-one-hot: answer_index=-1");
        check(r.answer_text.empty(),"11. non-one-hot: answer_text empty");
    }

    // 12. Score record gets number_of_levels = options.size()
    {
        ExemplarRecord r;
        json j = make_score_json(4);
        std::string err = parse_exemplar(j, r);
        check(err.empty(),              "12. score levels = options.size(): no error");
        check(r.number_of_levels == 4, "12. score levels = options.size(): value=4");
    }

    // 13. Non-score record gets number_of_levels = 0
    {
        ExemplarRecord r;
        json j = make_choice_json();
        std::string err = parse_exemplar(j, r);
        check(err.empty(),              "13. non-score: no error");
        check(r.number_of_levels == 0, "13. non-score: number_of_levels=0");
    }
}

// ---------------------------------------------------------------------------
// ===== SECTION 2 – Eligibility (is_eligible) =====
// ---------------------------------------------------------------------------
static void test_eligibility()
{
    // 14. split=train, language=en → eligible
    {
        ExemplarRecord r;
        r.split    = "train";
        r.language = "en";
        check(is_eligible(r), "14. train+en: eligible");
    }

    // 15. split=val, language=en → not eligible
    {
        ExemplarRecord r;
        r.split    = "val";
        r.language = "en";
        // Every English record enters the database; the runtime allowlist restricts demonstrations
        // to the training split.
        check(is_eligible(r), "15. val+en: stored (excluded at query time by the allowlist)");
    }

    // 16. split=train, language=fr → not eligible
    {
        ExemplarRecord r;
        r.split    = "train";
        r.language = "fr";
        check(!is_eligible(r), "16. train+fr: not eligible");
    }

    // 17. split=test, language=en → not eligible
    {
        ExemplarRecord r;
        r.split    = "test";
        r.language = "en";
        check(is_eligible(r), "17. test+en: stored (excluded at query time by the allowlist)");
    }
}

// ---------------------------------------------------------------------------
// ===== SECTION 3 – Canonical text =====
// ---------------------------------------------------------------------------
static void test_embedding_text()
{
    json state_with_policy = json::parse(R"({
        "policy_name": "My scenario",
        "numbered_requirements": ["Req A", "Req B"],
        "evidence_policy": "strict"
    })");

    std::vector<std::string> opts_choice = {"A", "B", "C"};
    std::vector<std::string> opts_score  = {"Poor","Fair","Good","VGood","Excellent"};

    // 18. Deterministic
    {
        std::string t1 = build_embedding_text("choice","test","basic","What?",state_with_policy,opts_choice);
        std::string t2 = build_embedding_text("choice","test","basic","What?",state_with_policy,opts_choice);
        check(t1 == t2, "18. deterministic output");
    }

    // 19. Document version prepends kDocumentPrefix
    {
        std::string body = build_embedding_text("choice","test","basic","Q",json::object(),opts_choice);
        std::string doc  = build_document_embedding_text("choice","test","basic","Q",json::object(),opts_choice);
        check(doc.substr(0, std::strlen(kDocumentPrefix)) == kDocumentPrefix,
              "19. document prefix present");
        check(doc.size() > body.size(), "19. document longer than body");
    }

    // 20. Kind appears in text
    {
        std::string t = build_embedding_text("score","","","Q",json::object(),opts_score);
        check(t.find("Task: score") != std::string::npos, "20. kind in text");
    }

    // 21. Domain appears in text when provided
    {
        std::string t = build_embedding_text("choice","ethics","basic","Q",json::object(),opts_choice);
        check(t.find("Domain: ethics") != std::string::npos, "21. domain in text");
    }

    // 22. Family appears in text when provided
    {
        std::string t = build_embedding_text("choice","","myfamily","Q",json::object(),opts_choice);
        check(t.find("Family: myfamily") != std::string::npos, "22. family in text");
    }

    // 23. Question appears in text
    {
        std::string q = "Is the sky blue?";
        std::string t = build_embedding_text("choice","","",q,json::object(),opts_choice);
        check(t.find(q) != std::string::npos, "23. question in text");
    }

    // 24. policy_name from state appears as Scenario
    {
        std::string t = build_embedding_text("choice","","","Q",state_with_policy,opts_choice);
        check(t.find("Scenario:") != std::string::npos,        "24. Scenario heading present");
        check(t.find("My scenario") != std::string::npos,      "24. policy_name value present");
    }

    // 25. numbered_requirements appear numbered
    {
        std::string t = build_embedding_text("choice","","","Q",state_with_policy,opts_choice);
        check(t.find("1. Req A") != std::string::npos, "25. requirement 1 numbered");
        check(t.find("2. Req B") != std::string::npos, "25. requirement 2 numbered");
    }

    // 26. evidence_policy appears
    {
        std::string t = build_embedding_text("choice","","","Q",state_with_policy,opts_choice);
        check(t.find("strict") != std::string::npos, "26. evidence_policy value in text");
    }

    // 27. Score kind includes "Score meanings:" with options listed by index
    {
        std::string t = build_embedding_text("score","","","Rate it.",json::object(),opts_score);
        check(t.find("Score meanings:") != std::string::npos, "27. score meanings heading");
        check(t.find("0: Poor") != std::string::npos,         "27. score level 0 listed");
        check(t.find("4: Excellent") != std::string::npos,    "27. score level 4 listed");
    }

    // 28. Choice kind does NOT include "Score meanings:"
    {
        std::string t = build_embedding_text("choice","","","Q",json::object(),opts_choice);
        check(t.find("Score meanings:") == std::string::npos, "28. choice: no Score meanings");
    }

    // 29. Different kinds produce different texts from same question
    {
        std::string tc = build_embedding_text("choice","","","Q",json::object(),opts_choice);
        std::string tn = build_embedding_text("noul","","","Q",json::object(),opts_choice);
        check(tc != tn, "29. different kinds produce different texts");
    }

    // 30. Original JSON not mutated
    {
        json state_orig = json::parse(R"({"policy_name": "P1"})");
        std::string dump_before = state_orig.dump();
        build_embedding_text("choice","d","f","Q",state_orig,opts_choice);
        std::string dump_after = state_orig.dump();
        check(dump_before == dump_after, "30. state JSON not mutated");
    }
}

// ---------------------------------------------------------------------------
// ===== SECTION 4 – ExemplarDatabase =====
// ---------------------------------------------------------------------------
static void test_exemplar_database()
{
    const std::string db_path = tmp_db_path();
    remove_file(db_path);

    // 31. open creates a database file
    {
        ExemplarDatabase db(db_path);
        bool ok = db.open(false);
        check(ok, "31. open returns true");
        // file should now exist (we just opened it)
        FILE* f = std::fopen(db_path.c_str(), "rb");
        check(f != nullptr, "31. database file created");
        if(f) std::fclose(f);
    }
    remove_file(db_path);

    // 32–33. create_schema creates tables; has_schema returns true
    {
        ExemplarDatabase db(db_path);
        db.open(false);
        bool cs = db.create_schema(8, "test_model", "fp1234");
        check(cs, "32. create_schema succeeds");
        check(db.has_schema(), "33. has_schema true after create_schema");
    }
    remove_file(db_path);

    // 34–36. check_compatible
    {
        ExemplarDatabase db(db_path);
        db.open(false);
        db.create_schema(8, "m", "fp_abc");

        std::string err;
        check(db.check_compatible(8, "fp_abc", err), "34. check_compatible passes exact match");

        err.clear();
        check(!db.check_compatible(4, "fp_abc", err), "35. check_compatible fails on dim mismatch");
        check(err.find("dimension") != std::string::npos, "35. dim mismatch err contains 'dimension'");

        err.clear();
        check(!db.check_compatible(8, "wrong_fp", err), "36. check_compatible fails on fingerprint mismatch");
    }
    remove_file(db_path);

    // 37–40. insert_group, insert_example, insert_vector, example_count
    {
        ExemplarDatabase db(db_path);
        db.open(false);
        db.create_schema(8, "m", "fp");

        ExemplarRecord rec = make_example_record("ex1","choice",0);

        check(db.insert_group(rec), "37. insert_group succeeds");

        int64_t rowid = 0;
        check(db.insert_example(rec, rowid), "38. insert_example succeeds");
        check(rowid > 0,                     "38. insert_example returns rowid > 0");

        std::vector<float> emb = {1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        check(db.insert_vector(rowid, emb, "choice", 0), "39. insert_vector succeeds");

        check(db.example_count() == 1, "40. example_count == 1 after one insert");
    }
    remove_file(db_path);

    // 41–43. id_exists, conflict detection
    {
        ExemplarDatabase db(db_path);
        db.open(false);
        db.create_schema(8, "m", "fp");

        ExemplarRecord rec = make_example_record("ex_dup","choice",0);
        db.insert_group(rec);
        int64_t rowid = 0;
        db.insert_example(rec, rowid);

        bool conflict = false;
        bool exists = db.id_exists(rec.id, rec, conflict);
        check(exists,    "41. id_exists returns true for inserted id");
        check(!conflict, "42. id_exists is_conflict=false for exact duplicate");

        // Conflict: same id, different question
        ExemplarRecord rec2 = rec;
        rec2.question = "Different question";
        conflict = false;
        exists = db.id_exists(rec.id, rec2, conflict);
        check(exists,   "43. id_exists true for same id");
        check(conflict, "43. id_exists is_conflict=true when question differs");
    }
    remove_file(db_path);

    // 44. Rerun (same id) is skipped without conflict
    {
        ExemplarDatabase db(db_path);
        db.open(false);
        db.create_schema(4, "m", "fp");

        ExemplarRecord rec = make_example_record("skip_me","noul",0);
        db.insert_group(rec);
        int64_t rowid = 0;
        db.insert_example(rec, rowid);

        bool conflict = false;
        bool exists = db.id_exists(rec.id, rec, conflict);
        check(exists && !conflict, "44. rerun same id: exists=true, no conflict => skip");
    }
    remove_file(db_path);

    // 45. begin/commit transaction
    {
        ExemplarDatabase db(db_path);
        db.open(false);
        db.create_schema(8, "m", "fp");

        check(db.begin_transaction(), "45. begin_transaction ok");
        ExemplarRecord rec = make_example_record("txn1","choice",0);
        db.insert_group(rec);
        int64_t rowid = 0;
        db.insert_example(rec, rowid);
        std::vector<float> emb = {1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
        db.insert_vector(rowid, emb, "choice", 0);
        check(db.commit_transaction(), "45. commit_transaction ok");
        check(db.example_count() == 1, "45. record present after commit");
    }
    remove_file(db_path);

    // 46. begin/rollback leaves no record
    {
        ExemplarDatabase db(db_path);
        db.open(false);
        db.create_schema(8, "m", "fp");

        db.begin_transaction();
        ExemplarRecord rec = make_example_record("rolled","choice",0);
        db.insert_group(rec);
        int64_t rowid = 0;
        db.insert_example(rec, rowid);
        check(db.rollback_transaction(), "46. rollback_transaction ok");
        check(db.example_count() == 0,   "46. no record after rollback");
    }
    remove_file(db_path);

    // 47. Rebuild mode: open(true) creates .tmp file, finalize_rebuild renames it
    {
        const std::string rbpath = tmp_db_path("_rebuild");
        remove_file(rbpath);

        ExemplarDatabase db(rbpath);
        bool ok = db.open(true);
        check(ok, "47. rebuild open returns true");

        std::string tmp = rbpath + ".tmp";
        FILE* f = std::fopen(tmp.c_str(), "rb");
        check(f != nullptr, "47. rebuild creates .tmp file");
        if(f) std::fclose(f);

        db.create_schema(8, "m", "fp");
        ExemplarRecord rec = make_example_record("rbex","choice",0);
        db.insert_group(rec);
        int64_t rowid = 0;
        db.insert_example(rec, rowid);

        check(db.finalize_rebuild(), "47. finalize_rebuild succeeds");

        FILE* g = std::fopen(rbpath.c_str(), "rb");
        check(g != nullptr, "47. final db file exists after finalize_rebuild");
        if(g) std::fclose(g);

        FILE* h = std::fopen(tmp.c_str(), "rb");
        check(h == nullptr, "47. .tmp file removed after finalize_rebuild");
        if(h) std::fclose(h);

        remove_file(rbpath);
    }
}

// ---------------------------------------------------------------------------
// ===== SECTION 5 – Vector retrieval smoke test (raw sqlite3 + sqlite-vec) =====
// ---------------------------------------------------------------------------
static void test_knn_smoke()
{
    const std::string knn_path = tmp_db_path("_knn");
    remove_file(knn_path);

    // Build a database with 4 records using ExemplarDatabase
    {
        ExemplarDatabase db(knn_path);
        db.open(false); // open() calls register_sqlite_vec_autoext internally
        db.create_schema(8, "knn_model", "knn_fp");

        struct Row { std::string id; std::string kind; int nol; std::vector<float> emb; };
        std::vector<Row> rows = {
            {"knn_choice", "choice", 0, vec_choice()},
            {"knn_noul",   "noul",   0, vec_noul()},
            {"knn_score5", "score",  5, vec_score5()},
            {"knn_score4", "score",  4, vec_score4()},
        };

        db.begin_transaction();
        for(const auto& row : rows) {
            ExemplarRecord rec = make_example_record(row.id, row.kind, row.nol);
            db.insert_group(rec);
            int64_t rowid = 0;
            db.insert_example(rec, rowid);
            db.insert_vector(rowid, row.emb, row.kind, row.nol);
        }
        db.commit_transaction();
    }

    // Now open with raw sqlite3 for KNN queries.
    // register_sqlite_vec_autoext() was already called by ExemplarDatabase::open(),
    // so sqlite-vec is already registered for this process.
    sqlite3* raw = nullptr;
    int rc = sqlite3_open(knn_path.c_str(), &raw);
    check(rc == SQLITE_OK, "KNN smoke: open raw db");

    auto query_knn = [&](const std::string& kind_filter,
                         int nol_filter,
                         bool filter_nol,
                         std::vector<float> query_vec,
                         int k) -> std::vector<int64_t>
    {
        // Stored vectors are bit[N] via vec_quantize_binary; query must be quantized too.
        std::string sql;
        if(filter_nol) {
            sql = "SELECT rowid, distance FROM exemplar_vectors "
                  "WHERE embedding MATCH vec_quantize_binary(?) AND kind=? AND number_of_levels=? AND k=" + std::to_string(k) +
                  " ORDER BY distance";
        } else {
            sql = "SELECT rowid, distance FROM exemplar_vectors "
                  "WHERE embedding MATCH vec_quantize_binary(?) AND kind=? AND k=" + std::to_string(k) +
                  " ORDER BY distance";
        }

        sqlite3_stmt* stmt = nullptr;
        if(sqlite3_prepare_v2(raw, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            return {};
        }

        sqlite3_bind_blob(stmt, 1,
                          query_vec.data(),
                          (int)(query_vec.size() * sizeof(float)),
                          SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, kind_filter.c_str(), -1, SQLITE_STATIC);
        if(filter_nol) {
            sqlite3_bind_int(stmt, 3, nol_filter);
        }

        std::vector<int64_t> rowids;
        while(sqlite3_step(stmt) == SQLITE_ROW) {
            rowids.push_back(sqlite3_column_int64(stmt, 0));
        }
        sqlite3_finalize(stmt);
        return rowids;
    };

    // Helper: get id (INTEGER) from examples table by rowid
    auto get_id = [&](int64_t rowid) -> int64_t {
        std::string sql = "SELECT id FROM examples WHERE rowid=?";
        sqlite3_stmt* stmt = nullptr;
        if(sqlite3_prepare_v2(raw, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) return 0;
        sqlite3_bind_int64(stmt, 1, rowid);
        int64_t id = 0;
        if(sqlite3_step(stmt) == SQLITE_ROW) {
            id = sqlite3_column_int64(stmt, 0);
        }
        sqlite3_finalize(stmt);
        return id;
    };

    // Helper: get distance from a KNN query result
    auto query_knn_distance = [&](const std::string& kind_filter,
                                  int nol_filter,
                                  bool filter_nol,
                                  std::vector<float> query_vec,
                                  int k) -> std::vector<double>
    {
        std::string sql;
        if(filter_nol) {
            sql = "SELECT rowid, distance FROM exemplar_vectors "
                  "WHERE embedding MATCH vec_quantize_binary(?) AND kind=? AND number_of_levels=? AND k=" + std::to_string(k) +
                  " ORDER BY distance";
        } else {
            sql = "SELECT rowid, distance FROM exemplar_vectors "
                  "WHERE embedding MATCH vec_quantize_binary(?) AND kind=? AND k=" + std::to_string(k) +
                  " ORDER BY distance";
        }

        sqlite3_stmt* stmt = nullptr;
        if(sqlite3_prepare_v2(raw, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) return {};

        sqlite3_bind_blob(stmt, 1, query_vec.data(), (int)(query_vec.size() * sizeof(float)), SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, kind_filter.c_str(), -1, SQLITE_STATIC);
        if(filter_nol) sqlite3_bind_int(stmt, 3, nol_filter);

        std::vector<double> dists;
        while(sqlite3_step(stmt) == SQLITE_ROW) {
            dists.push_back(sqlite3_column_double(stmt, 1));
        }
        sqlite3_finalize(stmt);
        return dists;
    };

    // 48. KNN with kind='choice' returns 1 result, not noul or score
    {
        auto rows = query_knn("choice", 0, false, vec_choice(), 1);
        check(rows.size() == 1, "48. KNN choice: 1 result");
        if(!rows.empty()) {
            int64_t id = get_id(rows[0]);
            check(id == hash_exemplar_id("knn_choice"), "48. KNN choice: correct record");
        }
    }

    // 49. KNN with kind='noul' returns 1 result
    {
        auto rows = query_knn("noul", 0, false, vec_noul(), 1);
        check(rows.size() == 1, "49. KNN noul: 1 result");
        if(!rows.empty()) {
            int64_t id = get_id(rows[0]);
            check(id == hash_exemplar_id("knn_noul"), "49. KNN noul: correct record");
        }
    }

    // 50. KNN with kind='score' AND number_of_levels=5 returns 1 result (score-5)
    {
        auto rows = query_knn("score", 5, true, vec_score5(), 1);
        check(rows.size() == 1, "50. KNN score nol=5: 1 result");
        if(!rows.empty()) {
            int64_t id = get_id(rows[0]);
            check(id == hash_exemplar_id("knn_score5"), "50. KNN score nol=5: returns score-5 record");
        }
    }

    // 51. KNN with kind='score' AND number_of_levels=4 returns 1 result (score-4)
    {
        auto rows = query_knn("score", 4, true, vec_score4(), 1);
        check(rows.size() == 1, "51. KNN score nol=4: 1 result");
        if(!rows.empty()) {
            int64_t id = get_id(rows[0]);
            check(id == hash_exemplar_id("knn_score4"), "51. KNN score nol=4: returns score-4 record");
        }
    }

    // 52. KNN with kind='score' AND number_of_levels=5 does NOT return score-4
    {
        auto rows = query_knn("score", 5, true, vec_score5(), 10);
        bool found_score4 = false;
        for(auto rid : rows) {
            if(get_id(rid) == hash_exemplar_id("knn_score4")) { found_score4 = true; break; }
        }
        check(!found_score4, "52. KNN score nol=5 does not return score-4");
    }

    // 53. Returned rowid resolves to correct row in examples table
    {
        auto rows = query_knn("choice", 0, false, vec_choice(), 1);
        check(!rows.empty() && get_id(rows[0]) == hash_exemplar_id("knn_choice"),
              "53. rowid resolves to correct examples row");
    }

    // 54. Distance values are finite
    {
        auto dists = query_knn_distance("score", 5, true, vec_score5(), 1);
        check(!dists.empty() && std::isfinite(dists[0]),
              "54. distance values are finite");
    }

    // 55. Stored vectors have expected dimension (4)
    {
        sqlite3_stmt* stmt = nullptr;
        const char* sql = "SELECT vec_length(embedding) FROM exemplar_vectors LIMIT 1";
        int rc2 = sqlite3_prepare_v2(raw, sql, -1, &stmt, nullptr);
        if(rc2 == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
            int dim = sqlite3_column_int(stmt, 0);
            check(dim == 8, "55. stored vector dimension == 8");
        } else {
            check(false, "55. could not query vector dimension");
        }
        if(stmt) sqlite3_finalize(stmt);
    }

    sqlite3_close(raw);
    remove_file(knn_path);
}

// ---------------------------------------------------------------------------
// ===== SECTION 6 – LZ4 compress/decompress =====
// ---------------------------------------------------------------------------
static void test_lz4()
{
    using namespace pjev::lz4;

    // 56. Round-trip small text
    {
        std::string input = "Hello, world!";
        auto compressed = compress(input);
        std::string err;
        std::string result = decompress(compressed.data(), compressed.size(), err);
        check(err.empty() && result == input, "56. lz4 round-trip small text");
    }

    // 57. Round-trip large text (256 KB of repeated pattern)
    {
        std::string input(256 * 1024, 'A');
        for(size_t i = 0; i < input.size(); ++i) input[i] = (char)('A' + (i % 26));
        auto compressed = compress(input);
        std::string err;
        std::string result = decompress(compressed.data(), compressed.size(), err);
        check(err.empty() && result == input, "57. lz4 round-trip large text (256 KB)");
    }

    // 58. Round-trip empty string
    {
        std::string input;
        auto compressed = compress(input);
        std::string err;
        std::string result = decompress(compressed.data(), compressed.size(), err);
        check(err.empty() && result == input, "58. lz4 round-trip empty string");
    }

    // 59. Round-trip UTF-8 string
    {
        std::string input = "こんにちは世界 — naïve résumé 🎉";
        auto compressed = compress(input);
        std::string err;
        std::string result = decompress(compressed.data(), compressed.size(), err);
        check(err.empty() && result == input, "59. lz4 round-trip UTF-8 string");
    }

    // 60. Round-trip structured JSON
    {
        nlohmann::ordered_json j;
        j["policy_name"] = "Test";
        j["requirements"] = {"A", "B", "C"};
        j["nested"] = {{"x", 1}, {"y", 2}};
        std::string input = j.dump();
        auto compressed = compress(input);
        std::string err;
        std::string result = decompress(compressed.data(), compressed.size(), err);
        check(err.empty() && result == input, "60. lz4 round-trip JSON");
    }

    // 61. Truncated LZ4 data returns error
    {
        // Use a large enough input so truncating to 1/4 cuts into the payload.
        std::string input(4096, 'X');
        for(size_t i = 0; i < input.size(); ++i) input[i] = (char)('A' + (i % 26));
        auto compressed = compress(input);
        // Remove the last 32 bytes to truncate the frame
        if(compressed.size() > 32) {
            compressed.resize(compressed.size() - 32);
            std::string err;
            std::string result = decompress(compressed.data(), compressed.size(), err);
            check(!err.empty(), "61. truncated frame returns error");
        } else {
            check(true, "61. (skipped: compressed too small to truncate meaningfully)");
        }
    }

    // 62. Corrupted LZ4 data returns error
    {
        std::string input = "Corrupt test data for lz4";
        auto compressed = compress(input);
        if(compressed.size() > 12) {
            // Flip bytes in the middle of the frame (past the header)
            for(size_t i = 8; i < std::min(compressed.size(), (size_t)16); ++i)
                compressed[i] ^= 0xFF;
            std::string err;
            std::string result = decompress(compressed.data(), compressed.size(), err);
            check(!err.empty(), "62. corrupted frame returns error");
        } else {
            check(true, "62. (skipped: compressed too small to corrupt meaningfully)");
        }
    }

    // 63. Zero-byte input to decompress returns empty string (not an error)
    {
        std::string err;
        std::string result = decompress(nullptr, 0, err);
        check(err.empty() && result.empty(), "63. zero-byte decompress returns empty, no error");
    }

    // 64. Compressed size < input size for compressible input
    {
        // Highly compressible: 10 KB of the same byte
        std::string input(10 * 1024, 'Z');
        auto compressed = compress(input);
        check(compressed.size() < input.size(), "64. compressed size < input for compressible data");
    }

    // 65. Exact byte preservation
    {
        // All 256 byte values
        std::string input(256, '\0');
        for(int i = 0; i < 256; ++i) input[i] = (char)i;
        auto compressed = compress(input);
        std::string err;
        std::string result = decompress(compressed.data(), compressed.size(), err);
        check(err.empty() && result.size() == 256, "65. all byte values preserved");
        bool exact = (err.empty() && result == input);
        check(exact, "65. exact bytes match after round-trip");
    }
}

// ---------------------------------------------------------------------------
// ===== SECTION 7 – hash_exemplar_id =====
// ---------------------------------------------------------------------------
static void test_id_hash()
{
    // 66. Known xxHash64 test vector: XXH64("", 0, seed=0) = 0xEF46DB3751D8E999
    {
        constexpr XXH64_hash_t expected_u = 0xEF46DB3751D8E999ULL;
        int64_t expected_s;
        static_assert(sizeof(expected_u) == sizeof(expected_s));
        std::memcpy(&expected_s, &expected_u, sizeof(expected_s));
        check(hash_exemplar_id("") == expected_s,
              "66. empty string: known xxHash64 vector");
    }

    // 67. Short identifier: deterministic
    {
        int64_t h1 = hash_exemplar_id("c1");
        int64_t h2 = hash_exemplar_id("c1");
        check(h1 == h2, "67. short id: deterministic");
    }

    // 68. Long Open Jev identifier: deterministic
    {
        const char* long_id = "community-diversity-v2/20260921/support/refund_window/0/1/score";
        int64_t h1 = hash_exemplar_id(long_id);
        int64_t h2 = hash_exemplar_id(long_id);
        check(h1 == h2, "68. long Open Jev id: deterministic");
    }

    // 69. Different strings produce different hashes
    {
        int64_t ha = hash_exemplar_id("id_a");
        int64_t hb = hash_exemplar_id("id_b");
        check(ha != hb, "69. different strings: different hashes");
        check(hash_exemplar_id("grpA") != hash_exemplar_id("grpB"),
              "69b. different group ids: different hashes");
    }

    // 70. Same group_id string always produces the same numeric group ID
    {
        int64_t g1 = hash_exemplar_id("grp_shared");
        int64_t g2 = hash_exemplar_id("grp_shared");
        int64_t g3 = hash_exemplar_id("grp_shared");
        check(g1 == g2 && g2 == g3, "70. same group_id: always same hash");
    }

    // 71. UTF-8 input hashed by raw bytes (no normalization)
    {
        std::string utf8_a = "こんにちは";
        std::string utf8_b = "こんにちは";
        check(hash_exemplar_id(utf8_a) == hash_exemplar_id(utf8_b),
              "71. UTF-8 input: same bytes same hash");
        check(hash_exemplar_id(utf8_a) != hash_exemplar_id("konnichiha"),
              "71b. UTF-8 bytes != ASCII bytes");
    }

    // 72. Result type is signed int64_t (negative values are valid)
    {
        // The empty string hash 0xEF46DB3751D8E999 has high bit set → negative
        int64_t h = hash_exemplar_id("");
        check(h < 0, "72. high bit set hash is negative (valid SQLite INTEGER)");
    }

    // 73. Signed 64-bit SQLite round-trip
    {
        const std::string db_path = tmp_db_path("_hash");
        remove_file(db_path);

        ExemplarRecord rec = make_example_record("hash_rt_test", "choice", 0);
        int64_t rowid = 0;
        {
            // Closed before the file is removed (an open database cannot be deleted on Windows).
            ExemplarDatabase db(db_path);
            db.open(false);
            db.create_schema(8, "m", "fp");
            db.insert_group(rec);
            db.insert_example(rec, rowid);
        }

        // Read back id via raw sqlite3
        sqlite3* raw = nullptr;
        sqlite3_open(db_path.c_str(), &raw);
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(raw, "SELECT id FROM examples WHERE rowid=?", -1, &stmt, nullptr);
        sqlite3_bind_int64(stmt, 1, rowid);
        bool got_row = (sqlite3_step(stmt) == SQLITE_ROW);
        int64_t stored_id = got_row ? sqlite3_column_int64(stmt, 0) : 0;
        sqlite3_finalize(stmt);
        sqlite3_close(raw);

        check(got_row, "73. id round-trip: row found");
        check(stored_id == rec.id, "73. id round-trip: stored value matches hash");
        remove_file(db_path);
    }

    // 74. Zero seed: call multiple times → always same result
    {
        const char* s = "test_seed_stability";
        int64_t v0 = hash_exemplar_id(s);
        int64_t v1 = hash_exemplar_id(s);
        int64_t v2 = hash_exemplar_id(s);
        check(v0 == v1 && v1 == v2, "74. zero-seed: multiple calls same result");
    }

    // 75. Empty string differs from single-space string
    {
        check(hash_exemplar_id("") != hash_exemplar_id(" "),
              "75. empty string hash differs from single space");
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    printf("=== parse_exemplar ===\n");
    test_parse_exemplar();

    printf("\n=== is_eligible ===\n");
    test_eligibility();

    printf("\n=== build_embedding_text / build_document_embedding_text ===\n");
    test_embedding_text();

    printf("\n=== ExemplarDatabase ===\n");
    test_exemplar_database();

    printf("\n=== KNN smoke test (sqlite-vec) ===\n");
    test_knn_smoke();

    printf("\n=== LZ4 compress/decompress ===\n");
    test_lz4();

    printf("\n=== hash_exemplar_id ===\n");
    test_id_hash();

    printf("\n");
    if(g_fails == 0) {
        printf("All tests passed.\n");
    } else {
        printf("%d test(s) FAILED.\n", g_fails);
    }
    return g_fails;
}
