// Tests for distribution: ReleaseConfig (pjev.json), runtime resource path resolution, get_executable_dir
#include "distribution/release_config.h"
#include "util/platform.h"
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); fails++; } \
    else         { printf("pass: %s\n", msg); } \
} while(0)

// Write a temp file, return path
static std::string write_temp(const std::string& name, const std::string& content) {
    // Use system temp or current dir
    std::string path = std::string(".") + "/" + name;
    std::ofstream f(path);
    f << content;
    return path;
}
static void remove_temp(const std::string& path) { std::remove(path.c_str()); }

// ---------------------------------------------------------------------------

static void test_exe_dir() {
    using namespace pjev;
    std::string dir = get_executable_dir();
    // Should be non-empty (we're running this executable right now)
    CHECK(!dir.empty(), "get_executable_dir: non-empty");
    printf("  exe dir: %s\n", dir.c_str());
}

static void test_release_config_parse() {
    using namespace pjev;
    std::string path = write_temp("test_pjev.json", R"({
        "schema_version": 1,
        "pjev_version": "0.8.0",
        "default_model": "model.gguf",
        "threads": 4,
        "context_size": 2048,
        "embedding_model": "emb.gguf",
        "exemplar_db": "data/ex.db",
        "exemplar_allowlist": "allow.txt",
        "kv_reuse": false,
        "batch_questions": true
    })");

    ReleaseConfig rc;
    std::string err;
    bool ok = ReleaseConfig::from_file(path, rc, err);
    remove_temp(path);

    CHECK(ok, "from_file: parses successfully");
    CHECK(rc.schema_version == 1, "schema_version=1");
    CHECK(rc.pjev_version == "0.8.0", "pjev_version");
    CHECK(rc.default_model == "model.gguf", "default_model");
    CHECK(rc.threads == 4, "threads");
    CHECK(rc.context_size == 2048, "context_size");
    CHECK(rc.embedding_model == "emb.gguf", "embedding_model");
    CHECK(rc.exemplar_db == "data/ex.db", "exemplar_db");
    CHECK(rc.exemplar_allowlist == "allow.txt", "exemplar_allowlist");
    // Keys of earlier releases (kv_reuse, batch_questions, calibration) are ignored, not errors.
}

static void test_release_config_bad_type() {
    using namespace pjev;
    std::string path = write_temp("test_pjev_bad.json", R"({"threads": "eight"})");
    ReleaseConfig rc;
    std::string err;
    CHECK(!ReleaseConfig::from_file(path, rc, err) && !err.empty(), "from_file: wrong value type is an error");
    remove_temp(path);
}

static void test_resolve_runtime_paths() {
    using namespace pjev;
    namespace fs = std::filesystem;
    // No executable directory: explicit values win, the rest falls back to models/<default>.
    RuntimePaths flags;
    flags.exemplar_db = "/explicit/ex.db";
    RuntimePaths p = ReleaseConfig::resolve_runtime_paths(flags, "");
    CHECK(p.exemplar_db == "/explicit/ex.db", "runtime paths: explicit flag wins");
    CHECK(p.model == "models/bonsai2_2b.gguf", "runtime paths: development model fallback");
    CHECK(p.embedding_model == "models/nomic-embed-text-v1.5.gguf", "runtime paths: development embedding fallback");
    CHECK(p.exemplar_allowlist == "models/examplar-allowlist.txt", "runtime paths: development allowlist fallback");

    // Files next to the executable.
    const fs::path dir = fs::absolute("test_dist_exe_dir");
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::ofstream((dir / "nomic-embed-text-v1.5.gguf").string()) << "x";
    std::ofstream((dir / "examplar.db").string()) << "x";
    p = ReleaseConfig::resolve_runtime_paths(RuntimePaths{}, dir.string());
    CHECK(p.embedding_model == (dir / "nomic-embed-text-v1.5.gguf").string(), "runtime paths: adjacent embedding model");
    CHECK(p.exemplar_db == (dir / "examplar.db").string(), "runtime paths: adjacent database");
    CHECK(p.exemplar_allowlist == "models/examplar-allowlist.txt", "runtime paths: missing adjacent file falls back");

    // pjev.json next to the executable: bare names are relative to it, paths are kept.
    std::ofstream((dir / "pjev.json").string()) << R"({"embedding_model": "e.gguf", "exemplar_allowlist": "sub/a.txt"})";
    p = ReleaseConfig::resolve_runtime_paths(RuntimePaths{}, dir.string());
    CHECK(p.embedding_model == (dir / "e.gguf").string(), "runtime paths: pjev.json bare name relative to exe dir");
    CHECK(p.exemplar_allowlist == "sub/a.txt", "runtime paths: pjev.json path kept");
    CHECK(p.exemplar_db == (dir / "examplar.db").string(), "runtime paths: unset key falls through to adjacent file");
    fs::remove_all(dir);
}

static void test_release_config_missing() {
    using namespace pjev;
    ReleaseConfig rc;
    std::string err;
    bool ok = ReleaseConfig::from_file("/nonexistent/path/pjev.json", rc, err);
    CHECK(!ok, "from_file: returns false for missing file");
    CHECK(!err.empty(), "from_file: sets error message for missing file");
}

static void test_resolve_model_explicit() {
    using namespace pjev;
    // Explicit path always wins regardless of exe_dir
    std::string result = ReleaseConfig::resolve_model("/explicit/model.gguf", "/some/dir");
    CHECK(result == "/explicit/model.gguf", "resolve_model: explicit path returned unchanged");
}

static void test_resolve_model_no_exe_dir() {
    using namespace pjev;
    // Empty exe_dir with no explicit model -> returns ""
    std::string result = ReleaseConfig::resolve_model("", "");
    CHECK(result.empty(), "resolve_model: empty when no exe_dir and no explicit");
}

static void test_to_json() {
    using namespace pjev;
    ReleaseConfig rc;
    rc.default_model = "test.gguf";
    rc.threads = 8;
    auto j = rc.to_json();
    CHECK(j["default_model"].get<std::string>() == "test.gguf", "to_json: default_model");
    CHECK(j["threads"].get<int>() == 8, "to_json: threads");
    CHECK(j["schema_version"].get<int>() == 1, "to_json: schema_version");
}

int main() {
    test_exe_dir();
    test_release_config_parse();
    test_release_config_missing();
    test_release_config_bad_type();
    test_resolve_runtime_paths();
    test_resolve_model_explicit();
    test_resolve_model_no_exe_dir();
    test_to_json();

    if (fails == 0) {
        printf("\nAll distribution tests passed.\n");
        return 0;
    }
    fprintf(stderr, "\n%d distribution test(s) FAILED.\n", fails);
    return 1;
}
