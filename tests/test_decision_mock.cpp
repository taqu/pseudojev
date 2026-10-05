#include "decision/decision_engine.h"
#include "mock_backend.h"
#include <cstdio>
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

    // Test 3: noul — B wins (semantic true) -> p_true high
    {
        DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::LETTERS});
        mock.set_winner(66); // "B" = semantic true (index 1)
        DecisionInput inp;
        inp.type     = "noul";
        inp.state    = "Please refund the duplicate charge.";
        inp.question = "Is the user asking for a refund?";
        inp.options  = {{"false",""}, {"true",""}};
        auto out = eng.decide(inp);
        check(out.ok, "noul B wins ok");
        check(out.p_true > 0.99, "noul: p_true high when B (true) wins");
        check(out.selected == 1, "noul: selected index 1 (true)");
    }

    // Test 4: noul — A wins (semantic false) -> p_true low
    {
        DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::LETTERS});
        mock.set_winner(65); // "A" = semantic false (index 0)
        DecisionInput inp;
        inp.type     = "noul";
        inp.state    = "Thanks, working now.";
        inp.question = "Is user reporting a problem?";
        inp.options  = {{"false",""}, {"true",""}};
        auto out = eng.decide(inp);
        check(out.ok, "noul A wins ok");
        check(out.p_true < 0.01, "noul: p_true low when A (false) wins");
        check(out.selected == 0, "noul: selected index 0 (false)");
    }

    // Test 4b: noul reversed order — true is first (A), false is second (B)
    // Same winner token A (65) should now yield HIGH p_true because A=true
    {
        DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::LETTERS});
        mock.set_winner(65); // "A" = semantic true (reversed order)
        DecisionInput inp;
        inp.type     = "noul";
        inp.state    = "The sky is blue.";
        inp.question = "Is it daytime?";
        inp.options  = {{"true",""}, {"false",""}};  // reversed
        auto out = eng.decide(inp);
        check(out.ok, "noul reversed ok");
        check(out.p_true > 0.99, "noul reversed: p_true high when A=true wins");
        check(out.selected == 0, "noul reversed: selected index 0 (true)");
    }

    // Test 4c: same logits, different option order -> different p_true, same semantic meaning
    // Normal order: A=false, B=true; logit(A)>>logit(B) -> semantic false, p_true low
    // Reversed order: A=true, B=false; same logit(A)>>logit(B) -> semantic true, p_true high
    {
        DecisionEngine eng_normal (mock, PromptConfig{Layout::AUTO, Scheme::LETTERS});
        DecisionEngine eng_reversed(mock, PromptConfig{Layout::AUTO, Scheme::LETTERS});
        mock.set_winner(65); // A dominates

        DecisionInput normal_inp;
        normal_inp.type    = "noul";
        normal_inp.state   = "Test state.";
        normal_inp.question = "Test question?";
        normal_inp.options = {{"false",""}, {"true",""}};   // A=false, B=true
        auto normal_out = eng_normal.decide(normal_inp);

        DecisionInput rev_inp;
        rev_inp.type    = "noul";
        rev_inp.state   = "Test state.";
        rev_inp.question = "Test question?";
        rev_inp.options = {{"true",""}, {"false",""}};   // A=true, B=false
        auto rev_out = eng_reversed.decide(rev_inp);

        check(normal_out.ok && rev_out.ok, "noul synthetic logits: both ok");
        check(normal_out.p_true < 0.01,    "noul synthetic: normal order, A wins -> p_true low");
        check(rev_out.p_true   > 0.99,     "noul synthetic: reversed order, A wins -> p_true high");
        // Candidate token A wins in both cases — semantic interpretation differs
        check(normal_out.selected == 0 && rev_out.selected == 0,
              "noul synthetic: selected=0 (A) in both cases");
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
