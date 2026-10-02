#include "decision/decision_engine.h"
#include "mock_backend.h"
#include <cstdio>
#include <cmath>
#include <cassert>

static int fails = 0;
static void check(bool cond, const char* what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
    else { printf("pass: %s\n", what); }
}

int main() {
    using namespace pjev;
    MockBackend mock;

    // Test 1: choice with LETTERS scheme — A wins
    {
        DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::LETTERS});
        mock.set_winner(65); // "A"
        DecisionInput inp;
        inp.type     = "choice";
        inp.state    = "Customer complaint about billing.";
        inp.question = "Which queue?";
        inp.options  = {{"billing","Billing"}, {"technical","Technical"}, {"sales","Sales"}};
        auto out = eng.decide(inp);
        check(out.ok, "choice LETTERS ok");
        check(out.selected == 0, "choice LETTERS: A (billing) selected");
        check(out.keys[0] == "billing", "choice LETTERS: key 0 is billing");
        check(out.probs.size() == 3, "choice LETTERS: 3 probs");
        check(out.probs[0] > 0.99, "choice LETTERS: billing prob > 0.99");
    }

    // Test 2: choice — B wins
    {
        DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::LETTERS});
        mock.set_winner(66); // "B"
        DecisionInput inp;
        inp.type     = "choice";
        inp.state    = "API returns 500 error.";
        inp.question = "Which queue?";
        inp.options  = {{"billing","Billing"}, {"technical","Technical"}};
        auto out = eng.decide(inp);
        check(out.ok, "choice B wins ok");
        check(out.selected == 1, "choice: B (technical) selected");
        check(out.keys[out.selected] == "technical", "choice B wins: key is technical");
    }

    // Test 3: noul with NATURAL scheme — Yes wins -> p_true high
    {
        DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::NATURAL});
        mock.set_winner(201); // "Yes"
        DecisionInput inp;
        inp.type     = "noul";
        inp.state    = "Please refund the duplicate charge.";
        inp.question = "Is the user asking for a refund?";
        inp.options  = {{"false","No"}, {"true","Yes"}};
        auto out = eng.decide(inp);
        check(out.ok, "noul NATURAL ok");
        check(out.p_true > 0.99, "noul: p_true high when Yes wins");
        check(out.selected == 1, "noul: selected index 1 (true)");
    }

    // Test 4: noul — No wins -> p_true low
    {
        DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::NATURAL});
        mock.set_winner(200); // "No"
        DecisionInput inp;
        inp.type     = "noul";
        inp.state    = "Thanks, working now.";
        inp.question = "Is user reporting a problem?";
        inp.options  = {{"false","No"}, {"true","Yes"}};
        auto out = eng.decide(inp);
        check(out.ok, "noul No wins ok");
        check(out.p_true < 0.01, "noul: p_true low when No wins");
    }

    // Test 5: score with NATURAL scheme — level 2 wins
    {
        DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::NATURAL});
        mock.set_winner(204); // "2"
        DecisionInput inp;
        inp.type     = "score";
        inp.state    = "I am furious!";
        inp.question = "How frustrated?";
        inp.options  = {{"0","Calm"},{"1","Mild"},{"2","Frustrated"},{"3","Furious"}};
        auto out = eng.decide(inp);
        check(out.ok, "score ok");
        check(out.selected == 2, "score: level 2 selected");
        check(out.probs[2] > 0.99, "score: prob[2] high");
        check(out.expected_score > 1.9 && out.expected_score < 2.1,
              "score: expected_score ~2");
    }

    // Test 6: error — unknown type
    {
        DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::LETTERS});
        DecisionInput inp;
        inp.type     = "invalid";
        inp.state    = "S";
        inp.question = "Q?";
        inp.options  = {{"a","A"},{"b","B"}};
        auto out = eng.decide(inp);
        check(!out.ok, "unknown type returns error");
        check(!out.error.empty(), "unknown type has error message");
    }

    // Test 7: error — too few candidates
    {
        DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::LETTERS});
        DecisionInput inp;
        inp.type     = "choice";
        inp.state    = "S";
        inp.question = "Q?";
        inp.options  = {{"a","A"}};  // only 1 candidate
        auto out = eng.decide(inp);
        check(!out.ok, "1 candidate returns error");
    }

    printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
