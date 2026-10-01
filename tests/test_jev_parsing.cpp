#include "server/jev_api.h"
#include "mock_backend.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <string>

using json = nlohmann::ordered_json;

static int fails = 0;
static void check(bool cond, const char* what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
    else { printf("pass: %s\n", what); }
}

int main() {
    MockBackend mock;
    DecisionEngine eng(mock, PromptConfig{Layout::AUTO, Scheme::LETTERS});
    JevApiHandler handler(eng, "pseudojev-test");

    // Test 1: valid choice request
    {
        mock.set_winner(65); // A wins
        json req;
        req["model"] = "pseudojev";
        req["state"] = "Customer billing issue.";
        json decision;
        decision["type"] = "choice";
        decision["instructions"] = "Which queue?";
        json criteria;
        criteria["billing"]   = "Billing dept";
        criteria["technical"] = "Tech dept";
        decision["criteria"] = criteria;
        req["questions"]["decision"] = decision;
        int status = 0;
        std::string resp;
        handler.handle(req.dump(), status, resp);
        check(status == 200, "choice request: status 200");
        auto r = json::parse(resp);
        check(r.contains("answers"), "choice response has answers");
        check(r["answers"]["decision"]["type"] == "choice", "answer type is choice");
        check(r["answers"]["decision"].contains("choice"), "answer has choice field");
        check(r["answers"]["decision"].contains("probabilities"), "answer has probabilities");
        check(r["model"] == "pseudojev-test", "response has model name");
    }

    // Test 2: valid noul request
    {
        mock.set_winner(65); // A (which is "false" in LETTERS scheme for noul)
        json req;
        req["model"] = "pseudojev";
        req["state"] = "Thanks, all good.";
        json decision;
        decision["type"] = "noul";
        decision["instructions"] = "Is user reporting a problem?";
        json criteria;
        criteria["false"] = "No problem";
        criteria["true"]  = "Has problem";
        decision["criteria"] = criteria;
        req["questions"]["decision"] = decision;
        int status = 0;
        std::string resp;
        handler.handle(req.dump(), status, resp);
        check(status == 200, "noul request: status 200");
        auto r = json::parse(resp);
        check(r["answers"]["decision"]["type"] == "noul", "noul answer type");
        check(r["answers"]["decision"].contains("noul"), "noul answer has noul field");
        double p = r["answers"]["decision"]["noul"].get<double>();
        check(p >= 0.0 && p <= 1.0, "noul value in [0,1]");
    }

    // Test 3: valid score request
    {
        mock.set_winner(66); // B wins (level 1 in LETTERS scheme)
        json req;
        req["model"] = "pseudojev";
        req["state"] = "Slightly annoyed.";
        json decision;
        decision["type"] = "score";
        decision["instructions"] = "How frustrated?";
        json criteria = json::array();
        criteria.push_back("Calm");
        criteria.push_back("Annoyed");
        criteria.push_back("Furious");
        decision["criteria"] = criteria;
        req["questions"]["decision"] = decision;
        int status = 0;
        std::string resp;
        handler.handle(req.dump(), status, resp);
        check(status == 200, "score request: status 200");
        auto r = json::parse(resp);
        check(r["answers"]["decision"]["type"] == "score", "score answer type");
        check(r["answers"]["decision"].contains("probabilities"), "score has probabilities");
    }

    // Test 4: invalid JSON -> 400
    {
        int status = 0;
        std::string resp;
        handler.handle("not json at all {{{", status, resp);
        check(status == 400, "invalid JSON -> 400");
    }

    // Test 5: missing required fields -> 400
    {
        json req;
        req["model"] = "pseudojev";
        // missing state and questions
        int status = 0;
        std::string resp;
        handler.handle(req.dump(), status, resp);
        check(status == 400, "missing state -> 400");
    }

    // Test 6: unknown type -> 422
    {
        json req;
        req["model"] = "pseudojev";
        req["state"] = "S";
        json decision;
        decision["type"] = "unknown";
        decision["instructions"] = "Q?";
        req["questions"]["decision"] = decision;
        int status = 0;
        std::string resp;
        handler.handle(req.dump(), status, resp);
        check(status == 422, "unknown type -> 422");
    }

    // Test 7: empty body -> 400
    {
        int status = 0;
        std::string resp;
        handler.handle("", status, resp);
        check(status == 400, "empty body -> 400");
    }

    // Test 8: response has usage field
    {
        mock.set_winner(65);
        json req;
        req["model"] = "pseudojev";
        req["state"] = "S";
        json decision;
        decision["type"] = "noul";
        decision["instructions"] = "Q?";
        json criteria;
        criteria["false"] = "";
        criteria["true"]  = "";
        decision["criteria"] = criteria;
        req["questions"]["decision"] = decision;
        int status = 0;
        std::string resp;
        handler.handle(req.dump(), status, resp);
        auto r = json::parse(resp);
        check(r.contains("usage"), "response has usage field");
    }

    printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
