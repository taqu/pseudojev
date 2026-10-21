// Unit tests for model identity: SHA-256, candidate token probing, ModelIdentity fields.
// No real model files are used; a mock backend is constructed directly.

#include "model/model_identity.h"
#include "inference/backend.h"

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
// Mock backend: each candidate character tokenises to its ASCII code (single token)
// except multi-char strings -> multi-token
// ---------------------------------------------------------------------------
struct MockBackend : public pjev::ILlamaBackend {
    std::vector<int> tokenize(const std::string& text,
                              bool, bool) const override {
        // 1-char strings → single token (ascii code)
        // multi-char strings → one token per character
        std::vector<int> r;
        for (char c : text) r.push_back((int)(unsigned char)c);
        return r;
    }
    int vocab_size() const override { return 256; }
    std::string token_to_piece(int id) const override { return std::string(1, (char)id); }
    const float* eval_tokens(const std::vector<int>&) override {
        static float logits[256] = {};
        return logits;
    }
    // metadata defaults are inherited (return "" / 0)
};

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
// CandidateCompatibility tests
// ---------------------------------------------------------------------------
static void test_candidate_compat()
{
    MockBackend mb;

    // Single-char candidates → single token (length 1 string → 1 token)
    auto cc1 = pjev::probe_candidates(mb, {"A", "B", "C", "D"});
    CHECK(cc1.all_single_token,   "ABCD: all single-token");
    CHECK(cc1.records.size() == 4, "ABCD: 4 records");
    CHECK(cc1.records[0].single_token, "A: single_token=true");
    CHECK(cc1.records[0].token_count == 1, "A: token_count=1");
    CHECK(cc1.records[0].token_id == (int)'A', "A: token_id == ascii('A')");
    CHECK(cc1.issues.empty(), "ABCD: no issues");

    // Multi-char string → multi-token
    auto cc2 = pjev::probe_candidates(mb, {"Yes", "No"});
    CHECK(!cc2.all_single_token, "Yes/No: not all single-token");
    CHECK(!cc2.issues.empty(),   "Yes/No: issues reported");
    CHECK(!cc2.records[0].single_token, "Yes: single_token=false");
    CHECK(cc2.records[0].token_count == 3, "Yes: token_count=3");

    // to_json round-trip
    auto j = cc1.to_json();
    CHECK(j.contains("all_single_token"), "compat json: all_single_token field");
    CHECK(j.contains("records"),          "compat json: records field");
    CHECK(j["all_single_token"].get<bool>(), "compat json: all_single_token=true");
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
    test_candidate_compat();
    test_model_identity();

    if (g_failures == 0) {
        printf("\nPASS (0 failure(s))\n");
        return 0;
    }
    fprintf(stderr, "\nFAIL (%d failure(s))\n", g_failures);
    return 1;
}
