// Unit tests for model identity: SHA-256 and ModelIdentity fields. No real model files are used.

#include "model/model_identity.h"

#include <cstdio>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Test framework
// ---------------------------------------------------------------------------
static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_failures++; } \
    else         { printf("pass: %s\n", msg); } \
} while(0)

// ---------------------------------------------------------------------------
// SHA-256 tests
// ---------------------------------------------------------------------------
static void test_sha256()
{
    using pjev::sha256_data;

    CHECK(sha256_data("").size() == 64,
          "sha256 empty: 64 hex chars");
    CHECK(sha256_data("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "sha256 empty: known value");
    CHECK(sha256_data("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "sha256 'abc': known value");
    // Same input → same output
    CHECK(sha256_data("hello") == sha256_data("hello"),
          "sha256 deterministic");
    // Different inputs → different output
    CHECK(sha256_data("a") != sha256_data("b"),
          "sha256 distinct inputs produce distinct hashes");
}

// ---------------------------------------------------------------------------
// ModelIdentity tests
// ---------------------------------------------------------------------------
static void test_model_identity()
{
    // from_path without a backend: only file-level fields
    pjev::ModelIdentity id = pjev::ModelIdentity::from_path("/nonexistent/model.gguf");
    CHECK(id.gguf_path == "/nonexistent/model.gguf", "identity: gguf_path set");
    CHECK(id.name == "model",                         "identity: name from filename stem");
    CHECK(id.size_bytes == 0,                         "identity: size_bytes=0 for missing file");
    CHECK(id.sha256.empty(),                          "identity: sha256 empty for missing file");

    // to_json includes all required fields
    auto j = id.to_json();
    CHECK(j.contains("name"),         "identity json: name");
    CHECK(j.contains("sha256"),       "identity json: sha256");
    CHECK(j.contains("size_bytes"),   "identity json: size_bytes");
    CHECK(j.contains("quantization"), "identity json: quantization");
    CHECK(j.contains("architecture"), "identity json: architecture");
    CHECK(j.contains("model_desc"),   "identity json: model_desc");
    CHECK(j.contains("n_params"),     "identity json: n_params");
}

int main()
{
    test_sha256();
    test_model_identity();

    if (g_failures == 0) {
        printf("\nPASS (0 failure(s))\n");
        return 0;
    }
    fprintf(stderr, "\nFAIL (%d failure(s))\n", g_failures);
    return 1;
}
