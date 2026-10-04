// tests for Phase 6 hardening: structured errors, request IDs, field limits, exception boundary
#include "server/jev_api.h"
#include "decision/prompt_strategy.h"
#include "decision/decision_engine.h"
#include "mock_backend.h"
#include <nlohmann/json.hpp>
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); fails++; } \
    else         { printf("pass: %s\n", msg); } \
} while(0)

// Helper: parse error from response
static bool parse_error(const std::string& body, std::string& code, std::string& msg) {
    try {
        auto j = json::parse(body);
        if (!j.contains("error")) return false;
        code = j["error"]["code"].get<std::string>();
        msg  = j["error"]["message"].get<std::string>();
        return true;
    } catch (...) { return false; }
}

// Test 1: structured error format on invalid JSON
static void test_structured_errors() {
    MockBackend mock;
    pjev::DecisionEngine eng(mock);
    pjev::JevApiHandler handler(eng);

    // invalid JSON → {"error": {"code": "invalid_json", ...}}
    {
        int32_t status = 0;
        std::string resp;
        handler.handle("{{{bad", status, resp);
        CHECK(status == 400, "invalid_json: status 400");
        std::string code, msg;
        CHECK(parse_error(resp, code, msg), "invalid_json: error field present");
        CHECK(code == "invalid_json", "invalid_json: code == invalid_json");
    }

    // missing state → invalid_request
    {
        json req;
        req["questions"] = json::object();
        int32_t status = 0; std::string resp;
        handler.handle(req.dump(), status, resp);
        CHECK(status == 400, "missing_state: status 400");
        std::string code, msg;
        CHECK(parse_error(resp, code, msg), "missing_state: has error field");
        CHECK(code == "invalid_request", "missing_state: code == invalid_request");
    }

    // unknown primitive → unsupported_primitive
    {
        json req;
        req["state"] = "S";
        req["questions"]["q"]["type"] = "frobnicator";
        req["questions"]["q"]["instructions"] = "What?";
        int32_t status = 0; std::string resp;
        handler.handle(req.dump(), status, resp);
        CHECK(status == 422, "unknown_prim: status 422");
        std::string code, msg;
        CHECK(parse_error(resp, code, msg), "unknown_prim: has error");
        CHECK(code == "unsupported_primitive", "unknown_prim: code == unsupported_primitive");
    }

    // choice missing criteria → invalid_request
    {
        json req;
        req["state"] = "S";
        req["questions"]["q"]["type"] = "choice";
        req["questions"]["q"]["instructions"] = "Which?";
        // no criteria
        int32_t status = 0; std::string resp;
        handler.handle(req.dump(), status, resp);
        CHECK(status == 400, "choice_no_criteria: status 400");
        std::string code, msg;
        parse_error(resp, code, msg);
        CHECK(code == "invalid_request", "choice_no_criteria: code == invalid_request");
    }

    // score missing criteria → invalid_request
    {
        json req;
        req["state"] = "S";
        req["questions"]["q"]["type"] = "score";
        req["questions"]["q"]["instructions"] = "Score?";
        // criteria should be array, provide object
        req["questions"]["q"]["criteria"] = json::object();
        int32_t status = 0; std::string resp;
        handler.handle(req.dump(), status, resp);
        CHECK(status == 400, "score_bad_criteria: status 400");
        std::string code, msg;
        parse_error(resp, code, msg);
        CHECK(code == "invalid_request", "score_bad_criteria: invalid_request");
    }
}

// Test 2: request ID in responses
static void test_request_ids() {
    MockBackend mock;
    mock.set_winner(65); // A wins
    pjev::DecisionEngine eng(mock, pjev::PromptConfig{pjev::Layout::AUTO, pjev::Scheme::LETTERS});
    pjev::JevApiHandler handler(eng);

    // Successful response includes request_id
    {
        json req;
        req["state"] = "S";
        req["questions"]["q"]["type"] = "noul";
        req["questions"]["q"]["instructions"] = "?";
        req["questions"]["q"]["criteria"]["false"] = "";
        req["questions"]["q"]["criteria"]["true"]  = "";
        int32_t status = 0; std::string resp;
        handler.handle(req.dump(), status, resp);
        CHECK(status == 200, "req_id: successful status 200");
        auto j = json::parse(resp);
        CHECK(j.contains("request_id"), "req_id: response has request_id");
        std::string rid = j["request_id"].get<std::string>();
        CHECK(rid.substr(0,4) == "req-", "req_id: starts with req-");
    }

    // Error response also includes request_id
    {
        int32_t status = 0; std::string resp;
        handler.handle("bad json", status, resp);
        CHECK(status == 400, "req_id_error: status 400");
        auto j = json::parse(resp);
        CHECK(j.contains("request_id"), "req_id_error: error has request_id");
    }

    // IDs are monotonically increasing
    {
        // Send two requests and check IDs differ
        int32_t s1 = 0, s2 = 0;
        std::string r1, r2;
        handler.handle("bad", s1, r1);
        handler.handle("bad", s2, r2);
        auto j1 = json::parse(r1);
        auto j2 = json::parse(r2);
        std::string id1 = j1["request_id"].get<std::string>();
        std::string id2 = j2["request_id"].get<std::string>();
        CHECK(id1 != id2, "req_id_monotone: consecutive IDs differ");
    }
}

// Test 3: field size limits
static void test_field_limits() {
    MockBackend mock;
    pjev::DecisionEngine eng(mock);
    pjev::JevApiHandler handler(eng);

    // State too long (> 32 KiB)
    {
        json req;
        req["state"] = std::string(33 * 1024, 'x');
        json q;
        q["type"] = "noul";
        q["instructions"] = "?";
        q["criteria"]["false"] = "";
        q["criteria"]["true"]  = "";
        req["questions"]["q"] = q;
        int32_t status = 0; std::string resp;
        handler.handle(req.dump(), status, resp);
        CHECK(status == 400, "state_too_long: status 400");
        std::string code, msg;
        CHECK(parse_error(resp, code, msg), "state_too_long: error field");
        CHECK(code == "invalid_request", "state_too_long: invalid_request");
    }

    // Instructions too long (> 8 KiB)
    {
        json req;
        req["state"] = "S";
        json q;
        q["type"] = "noul";
        q["instructions"] = std::string(9 * 1024, 'x');
        q["criteria"]["false"] = "";
        q["criteria"]["true"]  = "";
        req["questions"]["q"] = q;
        int32_t status = 0; std::string resp;
        handler.handle(req.dump(), status, resp);
        CHECK(status == 400, "instructions_too_long: status 400");
        std::string code, msg;
        parse_error(resp, code, msg);
        CHECK(code == "invalid_request", "instructions_too_long: invalid_request");
    }

    // Too many choice options (> 20)
    {
        json req;
        req["state"] = "S";
        json q;
        q["type"] = "choice";
        q["instructions"] = "?";
        json criteria;
        for (int i = 0; i < 25; i++) criteria[std::to_string(i)] = "opt";
        q["criteria"] = criteria;
        req["questions"]["q"] = q;
        int32_t status = 0; std::string resp;
        handler.handle(req.dump(), status, resp);
        CHECK(status == 400, "too_many_options: status 400");
        std::string code, msg;
        parse_error(resp, code, msg);
        CHECK(code == "invalid_request", "too_many_options: invalid_request");
    }
}

// Test 4: inference error → structured error, no crash
static void test_inference_error() {
    // Create a failing backend: eval_tokens throws
    struct ThrowingBackend : public pjev::ILlamaBackend {
        std::vector<int> tokenize(const std::string& t, bool, bool) const override {
            // Single-char tokens OK for candidate assignment
            if (t.size() == 1) return {(int)(unsigned char)t[0]};
            return {1, 2}; // multi-token
        }
        int vocab_size() const override { return 300; }
        std::string token_to_piece(int id) const override {
            if (id >= 65 && id <= 90) return std::string(1, (char)id);
            return "?";
        }
        const float* eval_tokens(const std::vector<int>&) override {
            throw std::runtime_error("simulated backend failure");
        }
    };

    ThrowingBackend tb;
    pjev::DecisionEngine eng(tb, pjev::PromptConfig{pjev::Layout::AUTO, pjev::Scheme::LETTERS});
    pjev::JevApiHandler handler(eng);

    {
        json req;
        req["state"] = "S";
        req["questions"]["q"]["type"] = "noul";
        req["questions"]["q"]["instructions"] = "?";
        req["questions"]["q"]["criteria"]["false"] = "";
        req["questions"]["q"]["criteria"]["true"]  = "";
        int32_t status = 0; std::string resp;
        // Must not crash or throw
        handler.handle(req.dump(), status, resp);
        CHECK(status == 500 || status == 400, "inference_error: returns 4xx or 5xx");
        // Response must be valid JSON
        bool valid_json = true;
        try { (void)json::parse(resp); } catch (...) { valid_json = false; }
        CHECK(valid_json, "inference_error: response is valid JSON");
        std::string code, msg;
        bool has_error = parse_error(resp, code, msg);
        CHECK(has_error, "inference_error: response has error field");
        printf("  inference_error: code=%s\n", code.c_str());
    }
}

// Test 5: overload (max_queued=1) → server_busy
static void test_overload() {
    MockBackend mock;
    mock.set_winner(65);
    pjev::DecisionEngine eng(mock, pjev::PromptConfig{pjev::Layout::AUTO, pjev::Scheme::LETTERS});
    // max_queued=1: only 1 concurrent request allowed
    pjev::JevApiHandler handler(eng, "test", 1);

    // First request occupies the slot (we simulate by launching it in a thread
    // that blocks on mutex_ — but that requires inference to be slow).
    // Instead, test the simpler case: max_queued=0 → always busy
    pjev::JevApiHandler busy_handler(eng, "test", 0);
    {
        json req;
        req["state"] = "S";
        req["questions"]["q"]["type"] = "noul";
        req["questions"]["q"]["instructions"] = "?";
        req["questions"]["q"]["criteria"]["false"] = "";
        req["questions"]["q"]["criteria"]["true"]  = "";
        int32_t status = 0; std::string resp;
        busy_handler.handle(req.dump(), status, resp);
        CHECK(status == 503, "overload: status 503");
        std::string code, msg;
        parse_error(resp, code, msg);
        CHECK(code == "server_busy", "overload: code == server_busy");
    }
}

// Test 6: backward compat — existing test_jev_parsing patterns still work
static void test_backward_compat() {
    MockBackend mock;
    mock.set_winner(65);
    pjev::DecisionEngine eng(mock, pjev::PromptConfig{pjev::Layout::AUTO, pjev::Scheme::LETTERS});
    pjev::JevApiHandler handler(eng, "pseudojev-test");

    // Valid choice request still returns 200 and has answers/model/usage
    json req;
    req["state"] = "S";
    req["questions"]["q"]["type"] = "choice";
    req["questions"]["q"]["instructions"] = "Which?";
    json crit;
    crit["A"] = "Option A";
    crit["B"] = "Option B";
    req["questions"]["q"]["criteria"] = crit;
    int32_t status = 0; std::string resp;
    handler.handle(req.dump(), status, resp);
    CHECK(status == 200, "compat: choice 200");
    auto j = json::parse(resp);
    CHECK(j.contains("answers"), "compat: has answers");
    CHECK(j.contains("model"),   "compat: has model");
    CHECK(j.contains("usage"),   "compat: has usage");
    CHECK(j["model"] == "pseudojev-test", "compat: model name");

    // Valid noul request
    {
        json r;
        r["state"] = "S";
        r["questions"]["q"]["type"] = "noul";
        r["questions"]["q"]["instructions"] = "?";
        r["questions"]["q"]["criteria"]["false"] = "";
        r["questions"]["q"]["criteria"]["true"] = "";
        int32_t s2 = 0; std::string r2;
        handler.handle(r.dump(), s2, r2);
        CHECK(s2 == 200, "compat: noul 200");
    }
}

int main() {
    test_structured_errors();
    test_request_ids();
    test_field_limits();
    test_inference_error();
    test_overload();
    test_backward_compat();

    printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
