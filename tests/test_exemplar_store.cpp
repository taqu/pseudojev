// Unit tests for the read-only exemplar store: database loading, compatibility checks, the allowlist,
// eligibility-filtered search (kind / level filters, (distance, rowid) order, window growth), LZ4
// payloads and corrupt-input handling. Builds a small database with pjev-filldb's writer; no model.

#include "exemplar_database.h"
#include "exemplar_record.h"
#include "retrieval/exemplar_allowlist.h"
#include "retrieval/exemplar_store.h"
#include "retrieval/id_hash.h"
#include "sqlite/sqlite3.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace pjev;
using namespace pjev::retrieval;
using pjev::filldb::ExemplarDatabase;
using pjev::filldb::ExemplarRecord;
using json = nlohmann::ordered_json;

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); g_failures++; } \
    else         { printf("pass: %s\n", msg); } \
} while (0)

static const int kDim = 16;
static const char* kFingerprint = "0123456789abcdef";

static void remove_db(const std::string& p)
{
    for (const char* s : {"", ".tmp", "-wal", "-shm"}) std::remove((p + s).c_str());
}

// Sign pattern with `neg` leading negative components: Hamming distance `neg` to the all-positive query.
static std::vector<float> vec(int neg)
{
    std::vector<float> v(kDim, 1.0f);
    for (int i = 0; i < neg; ++i) v[static_cast<size_t>(i)] = -1.0f;
    return v;
}

struct Row {
    std::string id;
    std::string kind;
    std::vector<std::string> options;
    int answer;    // -1 = soft target
    int neg;       // distance to the query
    bool allowed;
};

static const std::vector<Row> kRows = {
    {"c4-not-allowed", "choice", {"a", "b"}, 0, 0, false},
    {"c1",             "choice", {"a", "b"}, 1, 1, true},
    {"c2",             "choice", {"a", "b"}, 0, 1, true},
    {"c3",             "choice", {"a", "b", "c"}, 2, 2, true},
    {"c5",             "choice", {"a", "b"}, 0, 3, true},
    {"c6",             "choice", {"a", "b"}, 1, 4, true},
    {"n1",             "noul",   {"no", "yes"}, 1, 0, true},
    {"s5",             "score",  {"0", "1", "2", "3", "4"}, 3, 0, true},
    {"s3",             "score",  {"lo", "mid", "hi"}, 1, 0, true},
    {"soft",           "choice", {"a", "b"}, -1, 5, true},
};

static std::string build_db(const std::string& path)
{
    remove_db(path);
    ExemplarDatabase db(path);
    db.open(false);
    db.create_schema(kDim, "test-embedding.gguf", kFingerprint);
    db.begin_transaction();
    for (const auto& r : kRows) {
        ExemplarRecord rec;
        rec.id       = hash_exemplar_id(r.id);
        rec.group_id = hash_exemplar_id("g-" + r.id);
        rec.kind     = r.kind;
        rec.source   = "unit";
        rec.split    = "train";
        rec.question = "Question " + r.id;
        rec.options  = r.options;
        rec.target.assign(r.options.size(), 0.0f);
        if (r.answer >= 0) rec.target[static_cast<size_t>(r.answer)] = 1.0f;
        else rec.target.assign(r.options.size(), 1.0f / r.options.size());
        rec.answer_index = r.answer;
        rec.answer_text  = r.answer >= 0 ? r.options[static_cast<size_t>(r.answer)] : "";
        rec.state = r.id == "c3" ? json("plain text state") : json{{"policy_name", "policy " + r.id}, {"limit", 3}};
        rec.number_of_levels = r.kind == "score" ? static_cast<int>(r.options.size()) : 0;
        rec.language = "en";
        rec.embedding_text = "Task: " + r.kind;
        int64_t rowid = 0;
        db.insert_group(rec);
        db.insert_example(rec, rowid);
        db.insert_vector(rowid, vec(r.neg), r.kind, rec.number_of_levels);
    }
    db.commit_transaction();
    return path;
}

static std::string write_text(const std::string& path, const std::string& text)
{
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f << text;
    return path;
}

static std::string build_allowlist(const std::string& path)
{
    std::vector<int64_t> ids;
    for (const auto& r : kRows)
        if (r.allowed && r.answer >= 0) ids.push_back(hash_exemplar_id(r.id));
    std::string err;
    write_allowlist(path, ids, {"unit test allowlist"}, err);
    return path;
}

static std::vector<std::string> names(const std::vector<SearchHit>& hits)
{
    std::vector<std::string> out;
    for (const auto& h : hits)
        for (const auto& r : kRows)
            if (hash_exemplar_id(r.id) == h.id) out.push_back(r.id);
    return out;
}

static void test_allowlist_format()
{
    int64_t id = 0;
    CHECK(format_exemplar_id(static_cast<int64_t>(0x95ba7a22fb1c447dULL)) == "95ba7a22fb1c447d", "allowlist: hex of a negative ID");
    CHECK(parse_exemplar_id("95ba7a22fb1c447d", id) && id == static_cast<int64_t>(0x95ba7a22fb1c447dULL), "allowlist: parse hex");
    CHECK(!parse_exemplar_id("95ba7a22fb1c447", id) && !parse_exemplar_id("95ba7a22fb1c447g", id), "allowlist: reject malformed IDs");

    std::string err;
    CHECK(write_allowlist("test_store_allow_rt.txt", {5, -1, 5, 3}, {"header line"}, err), "allowlist: write");
    std::string text;
    {
        std::ifstream f("test_store_allow_rt.txt");
        text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    CHECK(text == "# header line\n0000000000000003\n0000000000000005\nffffffffffffffff\n", "allowlist: sorted, unique, header");
    ExemplarAllowlist a;
    CHECK(load_allowlist("test_store_allow_rt.txt", a, err) && a.size() == 3 && a.count(-1), "allowlist: load");
    write_text("test_store_allow_bad.txt", "# x\n0000000000000003\nnot-an-id\n");
    CHECK(!load_allowlist("test_store_allow_bad.txt", a, err) && err.find("line 3") != std::string::npos,
          "allowlist: malformed line reported");
    write_text("test_store_allow_empty.txt", "# nothing\n\n");
    CHECK(!load_allowlist("test_store_allow_empty.txt", a, err), "allowlist: empty list rejected");
    CHECK(!load_allowlist("does-not-exist.txt", a, err), "allowlist: missing file rejected");
    for (const char* p : {"test_store_allow_rt.txt", "test_store_allow_bad.txt", "test_store_allow_empty.txt"}) std::remove(p);
}

static void test_open_and_compat(const std::string& db, const std::string& allow)
{
    ExemplarStore s;
    std::string err;
    CHECK(s.open(db, allow, err), "open: database and allowlist");
    CHECK(s.example_count() == static_cast<int64_t>(kRows.size()), "open: example count");
    CHECK(s.allowlist_size() == 8, "open: allowlist size");
    CHECK(s.info("schema_version") == "4" && s.info("missing_key").empty(), "open: db_info");
    CHECK(s.check_compatible(kDim, kFingerprint, err), "compat: matching model");
    CHECK(s.check_compatible(kDim, "", err), "compat: unknown fingerprint is not checked");
    CHECK(!s.check_compatible(768, kFingerprint, err) && err.find("dimension") != std::string::npos, "compat: dimension mismatch");
    CHECK(!s.check_compatible(kDim, "ffffffffffffffff", err) && err.find("fingerprint") != std::string::npos,
          "compat: fingerprint mismatch");

    ExemplarStore m;
    CHECK(!m.open("does-not-exist.db", allow, err), "open: missing database fails");
    CHECK(!m.open(db, "does-not-exist.txt", err), "open: missing allowlist fails");
    write_text("test_store_garbage.db", std::string(4096, 'x'));
    CHECK(!m.open("test_store_garbage.db", allow, err), "open: corrupt database fails");
    std::vector<SearchHit> hits;
    CHECK(!m.search("choice", 0, vec(0), 1, hits, err), "search: not open fails");
    std::remove("test_store_garbage.db");
}

static void test_search(const std::string& db, const std::string& allow)
{
    ExemplarStore s;
    std::string err;
    s.open(db, allow, err);
    const std::vector<float> q = vec(0);
    std::vector<SearchHit> hits;

    CHECK(s.search("choice", 0, q, 1, hits, err) && names(hits) == std::vector<std::string>({"c1"}),
          "search: nearest eligible (ineligible nearer row skipped, tie by rowid)");
    CHECK(hits[0].distance == 1.0, "search: Hamming distance");
    CHECK(s.search("choice", 0, q, 3, hits, err) && names(hits) == std::vector<std::string>({"c1", "c2", "c3"}),
          "search: (distance, rowid) order");
    CHECK(s.search("choice", 0, q, 50, hits, err) && names(hits) == std::vector<std::string>({"c1", "c2", "c3", "c5", "c6"}),
          "search: window grows until the index is exhausted; soft-target row not listed");
    CHECK(s.search("noul", 0, q, 5, hits, err) && names(hits) == std::vector<std::string>({"n1"}), "search: kind filter");
    CHECK(s.search("score", 5, q, 5, hits, err) && names(hits) == std::vector<std::string>({"s5"}), "search: score level filter (5)");
    CHECK(s.search("score", 3, q, 5, hits, err) && names(hits) == std::vector<std::string>({"s3"}), "search: score level filter (3)");
    CHECK(s.search("score", 4, q, 5, hits, err) && hits.empty(), "search: no eligible exemplar is not an error");
    CHECK(s.is_allowed(hash_exemplar_id("c1")) && !s.is_allowed(hash_exemplar_id("c4-not-allowed")), "is_allowed");
    CHECK(!s.search("choice", 0, {}, 1, hits, err), "search: empty embedding rejected");
}

static void test_load(const std::string& db, const std::string& allow)
{
    ExemplarStore s;
    std::string err;
    s.open(db, allow, err);
    ExemplarPayload p;
    CHECK(s.load(hash_exemplar_id("c1"), p, err) == LoadResult::Ok, "load: ok");
    CHECK(p.kind == "choice" && p.question == "Question c1" && p.options == std::vector<std::string>({"a", "b"}) &&
          p.answer_index == 1, "load: fields");
    CHECK(p.state.is_object() && p.state.begin().key() == "policy_name" && p.state["limit"] == 3,
          "load: LZ4 state decompressed, key order kept");
    CHECK(s.load(hash_exemplar_id("c3"), p, err) == LoadResult::Ok && p.state.is_string() && p.state.get<std::string>() == "plain text state",
          "load: string state");
    CHECK(s.load(hash_exemplar_id("s5"), p, err) == LoadResult::Ok && p.number_of_levels == 5, "load: level count");
    CHECK(s.load(hash_exemplar_id("soft"), p, err) == LoadResult::Ok && p.answer_index == -1, "load: soft target has no answer index");
    CHECK(s.load(12345, p, err) == LoadResult::Invalid && err.find("not found") != std::string::npos, "load: missing row is invalid");

    // Corrupt one payload in a copy of the database.
    const std::string bad = "test_store_corrupt.db";
    remove_db(bad);
    {
        std::ifstream in(db, std::ios::binary);
        std::ofstream out(bad, std::ios::binary);
        out << in.rdbuf();
    }
    sqlite3* raw = nullptr;
    sqlite3_open(bad.c_str(), &raw);
    const std::string sql = "UPDATE examples SET state_lz4 = x'04224d18deadbeef' WHERE id = " +
                            std::to_string(hash_exemplar_id("c2"));
    sqlite3_exec(raw, sql.c_str(), nullptr, nullptr, nullptr);
    sqlite3_close(raw);
    {
        ExemplarStore c;  // closed before the file is removed
        CHECK(c.open(bad, allow, err), "corrupt payload: database still opens");
        CHECK(c.load(hash_exemplar_id("c2"), p, err) == LoadResult::Invalid && err.find("state_lz4") != std::string::npos,
              "corrupt payload: LZ4 error reported");
        CHECK(c.load(hash_exemplar_id("c1"), p, err) == LoadResult::Ok, "corrupt payload: other rows still load");
    }
    remove_db(bad);
}

int main()
{
    const std::string db = build_db("test_exemplar_store.db");
    const std::string allow = build_allowlist("test_exemplar_store_allow.txt");

    test_allowlist_format();
    test_open_and_compat(db, allow);
    test_search(db, allow);
    test_load(db, allow);

    remove_db(db);
    std::remove(allow.c_str());
    if (g_failures == 0) {
        printf("\nPASS (0 failure(s))\n");
        return 0;
    }
    fprintf(stderr, "\nFAIL (%d failure(s))\n", g_failures);
    return 1;
}
