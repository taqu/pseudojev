#include "decision/prompt_strategy.h"
#include <cassert>
#include <cstdio>
#include <algorithm>

static int fails = 0;
static void check(bool cond, const char* what) {
    if (!cond) { printf("FAIL: %s\n", what); fails++; }
    else { printf("pass: %s\n", what); }
}

int main() {
    // Test 1: NATURAL scheme label assignment for all types
    {
        PromptStrategy ps(PromptConfig{Layout::AUTO, Scheme::NATURAL});
        std::vector<Candidate> cands;
        Candidate c0; c0.key = "false"; c0.description = ""; c0.internal = "";
        Candidate c1; c1.key = "true";  c1.description = ""; c1.internal = "";
        cands.push_back(c0);
        cands.push_back(c1);
        ps.assign_labels("noul", cands);
        check(cands[0].internal == "No",  "noul NATURAL: false=No");
        check(cands[1].internal == "Yes", "noul NATURAL: true=Yes");

        std::vector<Candidate> scands;
        for (int i = 0; i < 3; i++) {
            Candidate c; c.key = std::to_string(i); c.description = ""; c.internal = "";
            scands.push_back(c);
        }
        ps.assign_labels("score", scands);
        check(scands[0].internal == "0", "score NATURAL: 0=0");
        check(scands[1].internal == "1", "score NATURAL: 1=1");
        check(scands[2].internal == "2", "score NATURAL: 2=2");

        std::vector<Candidate> ccands;
        Candidate ca; ca.key = "billing";   ca.description = ""; ca.internal = "";
        Candidate cb; cb.key = "technical"; cb.description = ""; cb.internal = "";
        ccands.push_back(ca);
        ccands.push_back(cb);
        ps.assign_labels("choice", ccands);
        check(ccands[0].internal == "A", "choice NATURAL: 0=A");
        check(ccands[1].internal == "B", "choice NATURAL: 1=B");
    }

    // Test 2: LETTERS scheme uses A/B/C for all types
    {
        PromptStrategy ps(PromptConfig{Layout::AUTO, Scheme::LETTERS});
        std::vector<Candidate> cands;
        Candidate c0; c0.key = "false"; c0.description = ""; c0.internal = "";
        Candidate c1; c1.key = "true";  c1.description = ""; c1.internal = "";
        cands.push_back(c0);
        cands.push_back(c1);
        ps.assign_labels("noul", cands);
        check(cands[0].internal == "A", "noul LETTERS: false=A");
        check(cands[1].internal == "B", "noul LETTERS: true=B");

        std::vector<Candidate> scands;
        for (int i = 0; i < 3; i++) {
            Candidate c; c.key = std::to_string(i); c.description = ""; c.internal = "";
            scands.push_back(c);
        }
        ps.assign_labels("score", scands);
        check(scands[0].internal == "A", "score LETTERS: 0=A");
    }

    // Test 3: AUTO layout — score uses state-last
    {
        PromptStrategy ps(PromptConfig{Layout::AUTO, Scheme::NATURAL});
        std::vector<Candidate> cands;
        Candidate c0; c0.key = "0"; c0.description = "Calm";    c0.internal = "";
        Candidate c1; c1.key = "1"; c1.description = "Annoyed"; c1.internal = "";
        Candidate c2; c2.key = "2"; c2.description = "Angry";   c2.internal = "";
        cands.push_back(c0); cands.push_back(c1); cands.push_back(c2);
        ps.assign_labels("score", cands);
        auto segs = ps.build_prompt_segments("score", "Hello!", "How frustrated?", cands);
        // Segment 1 (untrusted) should have "State:" AFTER question
        const std::string& content = segs[1].text;
        auto q_pos = content.find("Question:");
        auto s_pos = content.find("State:");
        check(q_pos != std::string::npos && s_pos != std::string::npos, "AUTO score has both blocks");
        check(q_pos < s_pos, "AUTO score: question before state (state-last)");
    }

    // Test 4: AUTO layout — choice uses state-first
    {
        PromptStrategy ps(PromptConfig{Layout::AUTO, Scheme::NATURAL});
        std::vector<Candidate> cands;
        Candidate ca; ca.key = "a"; ca.description = "Alpha"; ca.internal = "";
        Candidate cb; cb.key = "b"; cb.description = "Beta";  cb.internal = "";
        cands.push_back(ca); cands.push_back(cb);
        ps.assign_labels("choice", cands);
        auto segs = ps.build_prompt_segments("choice", "Customer complaint", "Which dept?", cands);
        const std::string& content = segs[1].text;
        auto q_pos = content.find("Question:");
        auto s_pos = content.find("State:");
        check(s_pos < q_pos, "AUTO choice: state before question (state-first)");
    }

    // Test 5: STATE_FIRST layout
    {
        PromptStrategy ps(PromptConfig{Layout::STATE_FIRST, Scheme::NATURAL});
        std::vector<Candidate> cands;
        Candidate c0; c0.key = "0"; c0.description = "Low";  c0.internal = "";
        Candidate c1; c1.key = "1"; c1.description = "High"; c1.internal = "";
        cands.push_back(c0); cands.push_back(c1);
        ps.assign_labels("score", cands);
        auto segs = ps.build_prompt_segments("score", "S", "Q", cands);
        auto s_pos = segs[1].text.find("State:");
        auto q_pos = segs[1].text.find("Question:");
        check(s_pos < q_pos, "STATE_FIRST: state before question");
    }

    // Test 6: STATE_LAST layout
    {
        PromptStrategy ps(PromptConfig{Layout::STATE_LAST, Scheme::NATURAL});
        std::vector<Candidate> cands;
        Candidate ca; ca.key = "a"; ca.description = "A"; ca.internal = "";
        Candidate cb; cb.key = "b"; cb.description = "B"; cb.internal = "";
        cands.push_back(ca); cands.push_back(cb);
        ps.assign_labels("choice", cands);
        auto segs = ps.build_prompt_segments("choice", "S", "Q", cands);
        auto s_pos = segs[1].text.find("State:");
        auto q_pos = segs[1].text.find("Question:");
        check(q_pos < s_pos, "STATE_LAST: question before state");
    }

    // Test 7: Prompt segments structure (3 segments, trusted/untrusted/trusted)
    {
        PromptStrategy ps;
        std::vector<Candidate> cands;
        Candidate c0; c0.key = "false"; c0.description = ""; c0.internal = "";
        Candidate c1; c1.key = "true";  c1.description = ""; c1.internal = "";
        cands.push_back(c0); cands.push_back(c1);
        ps.assign_labels("noul", cands);
        auto segs = ps.build_prompt_segments("noul", "State text", "Question text", cands);
        check(segs.size() == 3, "build_prompt_segments returns 3 segments");
        check(segs[0].trusted,  "segment 0 is trusted (template prefix)");
        check(!segs[1].trusted, "segment 1 is untrusted (user content)");
        check(segs[2].trusted,  "segment 2 is trusted (template suffix)");
        check(segs[0].text.find("<|im_start|>") != std::string::npos,
              "template prefix contains im_start");
        check(segs[2].text.find("<|im_end|>") != std::string::npos,
              "template suffix contains im_end");
    }

    // Test 8: all_internal_texts for NATURAL has A-J, 0-9, No, Yes
    {
        PromptStrategy ps(PromptConfig{Layout::AUTO, Scheme::NATURAL});
        auto texts = ps.all_internal_texts();
        bool has_A  = std::find(texts.begin(), texts.end(), "A")   != texts.end();
        bool has_No = std::find(texts.begin(), texts.end(), "No")  != texts.end();
        bool has_0  = std::find(texts.begin(), texts.end(), "0")   != texts.end();
        check(has_A,  "NATURAL all_internal_texts has A");
        check(has_No, "NATURAL all_internal_texts has No");
        check(has_0,  "NATURAL all_internal_texts has 0");
    }

    printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
