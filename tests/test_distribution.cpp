// Tests for Phase 8 distribution: ReleaseConfig, path resolution, get_executable_dir
#include "distribution/release_config.h"
#include "util/platform.h"
#include <cassert>
#include <cstdio>
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
    CHECK(!rc.kv_reuse, "kv_reuse=false");
    CHECK(rc.batch_questions, "batch_questions=true");
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
