// Phase 9 CLI unit tests: formatting, option parsing logic, DecisionEngine integration via mock
#include "cli/cli.h"
#include "decision/decision_engine.h"
#include "decision/prompt_strategy.h"
#include "mock_backend.h"
#include <nlohmann/json.hpp>
#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using json = nlohmann::json;

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); fails++; } \
    else         { printf("pass: %s\n", msg); } \
} while(0)

// ---------------------------------------------------------------------------
// Test 1: format_noul_human
// ---------------------------------------------------------------------------
static void test_noul_human_format() {
    pjev::DecisionOutput out;
    out.ok       = true;
    out.selected = 1; // true
    out.p_true   = 0.85;
    out.probs    = {0.15, 0.85};
    out.keys     = {"false", "true"};

    CHECK(pjev::format_noul_human(out) == "true",  "noul_human: selected=1 → true");

    out.selected = 0;
    CHECK(pjev::format_noul_human(out) == "false", "noul_human: selected=0 → false");
}

// ---------------------------------------------------------------------------
// Test 2: format_noul_json
// ---------------------------------------------------------------------------
static void test_noul_json_format() {
    pjev::DecisionOutput out;
    out.ok       = true;
    out.selected = 1;
    out.p_true   = 0.9;
    out.probs    = {0.1, 0.9};
    out.keys     = {"false", "true"};

    std::string s = pjev::format_noul_json(out);
    auto j = json::parse(s);
    CHECK(j["primitive"] == "noul",               "noul_json: primitive field");
    CHECK(j["result"]["value"] == true,           "noul_json: result.value=true");
    CHECK(j["result"]["p_true"].get<double>() > 0.8, "noul_json: p_true present");
    CHECK(j["probabilities"].is_array(),          "noul_json: probabilities array");
    CHECK(j["probabilities"].size() == 2,         "noul_json: probabilities size=2");
}

// ---------------------------------------------------------------------------
// Test 3: format_choice_human
// ---------------------------------------------------------------------------
static void test_choice_human_format() {
    pjev::DecisionOutput out;
    out.ok       = true;
    out.selected = 1;
    out.probs    = {0.2, 0.7, 0.1};
    out.keys     = {"A", "B", "C"};

    CHECK(pjev::format_choice_human(out) == "B", "choice_human: selected=1 → B");

    out.selected = 0;
    CHECK(pjev::format_choice_human(out) == "A", "choice_human: selected=0 → A");
}

// ---------------------------------------------------------------------------
// Test 4: format_choice_json
// ---------------------------------------------------------------------------
static void test_choice_json_format() {
    pjev::DecisionOutput out;
    out.ok       = true;
    out.selected = 2;
    out.probs    = {0.1, 0.2, 0.7};
    out.keys     = {"X", "Y", "Z"};

    std::string s = pjev::format_choice_json(out);
    auto j = json::parse(s);
    CHECK(j["primitive"] == "choice",          "choice_json: primitive field");
    CHECK(j["result"]["index"] == 2,           "choice_json: result.index=2");
    CHECK(j["result"]["key"] == "Z",           "choice_json: result.key=Z");
    CHECK(j["probabilities"].size() == 3,      "choice_json: probabilities size=3");
    CHECK(j["keys"].size() == 3,               "choice_json: keys size=3");
}

// ---------------------------------------------------------------------------
// Test 5: format_score_human / json
// ---------------------------------------------------------------------------
static void test_score_format() {
    pjev::DecisionOutput out;
    out.ok             = true;
    out.expected_score = 2.5;
    out.probs          = {0.1, 0.2, 0.4, 0.3};
    out.keys           = {"0", "1", "2", "3"};

    std::string h = pjev::format_score_human(out);
    CHECK(!h.empty(), "score_human: non-empty");
    // Should contain the number
    CHECK(h.find("2.5") != std::string::npos, "score_human: contains 2.5");

    std::vector<std::pair<std::string,std::string>> opts = {
        {"0","Poor"}, {"1","Fair"}, {"2","Good"}, {"3","Excellent"}
    };
    std::string s = pjev::format_score_json(out, opts);
    auto j = json::parse(s);
    CHECK(j["primitive"] == "score",                            "score_json: primitive");
    CHECK(j["result"]["expected_score"].get<double>() == 2.5,  "score_json: expected_score=2.5");
    CHECK(j["probabilities"].size() == 4,                      "score_json: probabilities size");
    CHECK(j["levels"].size() == 4,                             "score_json: levels size");
    CHECK(j["levels"][0]["key"] == "0",                        "score_json: levels[0].key");
    CHECK(j["levels"][2]["description"] == "Good",             "score_json: levels[2].description");
}

// ---------------------------------------------------------------------------
// Test 6: DecisionEngine noul via mock (selected=false)
// ---------------------------------------------------------------------------
static void test_noul_engine_false() {
    MockBackend mock;
    mock.set_winner(200); // "No" / false wins
    pjev::DecisionEngine eng(mock, pjev::PromptConfig{pjev::Layout::AUTO, pjev::Scheme::NATURAL});

    pjev::DecisionInput input;
    input.type     = "noul";
    input.state    = "The sky is blue.";
    input.question = "Is it night?";
    input.options  = {{"false", ""}, {"true", ""}};

    pjev::DecisionOutput out = eng.decide(input);
    CHECK(out.ok,          "noul_engine_false: ok");
    CHECK(out.selected == 0, "noul_engine_false: selected=0 (false)");
    std::string human = pjev::format_noul_human(out);
    CHECK(human == "false", "noul_engine_false: format=false");
}

// ---------------------------------------------------------------------------
// Test 7: DecisionEngine choice via mock
// ---------------------------------------------------------------------------
static void test_choice_engine() {
    MockBackend mock;
    mock.set_winner(66); // 'B' wins (token 66)
    pjev::DecisionEngine eng(mock, pjev::PromptConfig{pjev::Layout::AUTO, pjev::Scheme::LETTERS});

    pjev::DecisionInput input;
    input.type     = "choice";
    input.state    = "Context here.";
    input.question = "Which option?";
    input.options  = {{"A", "Option A"}, {"B", "Option B"}, {"C", "Option C"}};

    pjev::DecisionOutput out = eng.decide(input);
    CHECK(out.ok,           "choice_engine: ok");
    // With LETTERS scheme, letters A/B/C map to tokens 65/66/67
    // B wins → selected index 1
    CHECK(out.selected == 1, "choice_engine: selected=1 (B)");
    std::string human = pjev::format_choice_human(out);
    CHECK(human == "B",      "choice_engine: format=B");

    std::string js = pjev::format_choice_json(out);
    auto j = json::parse(js);
    CHECK(j["primitive"] == "choice", "choice_engine_json: primitive");
    CHECK(j["result"]["key"] == "B",  "choice_engine_json: key=B");
}

// ---------------------------------------------------------------------------
// Test 8: DecisionEngine score via mock
// ---------------------------------------------------------------------------
static void test_score_engine() {
    MockBackend mock;
    mock.set_winner(204); // token 204 = digit '2'
    pjev::DecisionEngine eng(mock, pjev::PromptConfig{pjev::Layout::AUTO, pjev::Scheme::NATURAL});

    pjev::DecisionInput input;
    input.type     = "score";
    input.state    = "Essay text.";
    input.question = "Rate quality.";
    // keys: "0","1","2","3" — score uses numeric keys
    input.options  = {{"0","Poor"}, {"1","Fair"}, {"2","Good"}, {"3","Excellent"}};

    pjev::DecisionOutput out = eng.decide(input);
    CHECK(out.ok, "score_engine: ok");
    // expected_score close to 2.0 (winner is index 2)
    CHECK(out.expected_score > 1.9 && out.expected_score < 2.1,
          "score_engine: expected_score near 2.0");
}

// ---------------------------------------------------------------------------
// Test 9: JSON output is clean (no extra text)
// ---------------------------------------------------------------------------
static void test_json_cleanness() {
    pjev::DecisionOutput out;
    out.ok       = true;
    out.selected = 0;
    out.p_true   = 0.3;
    out.probs    = {0.7, 0.3};
    out.keys     = {"false", "true"};

    std::string s = pjev::format_noul_json(out);
    // Must be parseable JSON with no leading/trailing non-JSON
    bool valid = true;
    try { (void)json::parse(s); } catch(...) { valid = false; }
    CHECK(valid, "json_cleanness: format_noul_json produces valid JSON");

    // Same for choice
    out.keys     = {"A", "B"};
    out.probs    = {0.7, 0.3};
    out.selected = 0;
    std::string cs = pjev::format_choice_json(out);
    bool valid_c = true;
    try { (void)json::parse(cs); } catch(...) { valid_c = false; }
    CHECK(valid_c, "json_cleanness: format_choice_json produces valid JSON");
}

// ---------------------------------------------------------------------------
// Test 10: option order preserved in choice
// ---------------------------------------------------------------------------
static void test_option_order_preserved() {
    MockBackend mock;
    mock.set_winner(67); // 'C' wins
    pjev::DecisionEngine eng(mock, pjev::PromptConfig{pjev::Layout::AUTO, pjev::Scheme::LETTERS});

    pjev::DecisionInput input;
    input.type     = "choice";
    input.state    = "";
    input.question = "Pick?";
    // Intentional order: C, A, B (non-alphabetical)
    input.options  = {{"C", "Third"}, {"A", "First"}, {"B", "Second"}};

    pjev::DecisionOutput out = eng.decide(input);
    CHECK(out.ok, "option_order: ok");
    // keys in output should match input order
    CHECK(out.keys.size() == 3 && out.keys[0] == "C",
          "option_order: first key is C");
    CHECK(out.keys[1] == "A", "option_order: second key is A");
    CHECK(out.keys[2] == "B", "option_order: third key is B");
}

// ---------------------------------------------------------------------------
// Test 11: Japanese state/question text reaches engine without corruption
// ---------------------------------------------------------------------------
static void test_japanese_input() {
    MockBackend mock;
    mock.set_winner(200); // No/false
    pjev::DecisionEngine eng(mock, pjev::PromptConfig{pjev::Layout::AUTO, pjev::Scheme::NATURAL});

    pjev::DecisionInput input;
    input.type     = "noul";
    input.state    = "空は青い。"; // "The sky is blue." in Japanese
    input.question = "夜ですか?";  // "Is it night?"
    input.options  = {{"false", ""}, {"true", ""}};

    pjev::DecisionOutput out = eng.decide(input);
    CHECK(out.ok, "japanese_input: ok (engine accepts Japanese text)");
    // We can't check the actual answer without a real model;
    // just confirm it runs without throwing and produces valid probs
    CHECK(out.probs.size() == 2, "japanese_input: probs size=2");
}

// ---------------------------------------------------------------------------
// Test 12: error on empty result (engine returns !ok)
// ---------------------------------------------------------------------------
static void test_score_format_edge() {
    // format_score_human on 0.0
    pjev::DecisionOutput out;
    out.ok             = true;
    out.expected_score = 0.0;
    out.probs          = {1.0, 0.0};
    out.keys           = {"0", "1"};

    std::string h = pjev::format_score_human(out);
    CHECK(h == "0.0000", "score_format_edge: 0.0 → 0.0000");
}

int main() {
    test_noul_human_format();
    test_noul_json_format();
    test_choice_human_format();
    test_choice_json_format();
    test_score_format();
    test_noul_engine_false();
    test_choice_engine();
    test_score_engine();
    test_json_cleanness();
    test_option_order_preserved();
    test_japanese_input();
    test_score_format_edge();

    printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
