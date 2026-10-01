// phase0 — experimental decision core for pseudojev.
//
// Evaluates noul / choice / score decisions with Bonsai 1.7B through llama.cpp by
// reading the logits of a fixed set of single-token internal candidates (A, B, C, ...)
// at the answer position and normalizing them with a restricted softmax.
//
// This is an experiment, not product code. See phase0/README.md.

#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Phase 0 limit: internal candidates are the letters A..J.
static const int MAX_CANDIDATES = 10;

// ---------------------------------------------------------------------------
// Data model

struct Candidate {
    std::string key;          // semantic label ("billing", "true", "2", ...)
    std::string description;  // meaning shown to the model
    std::string internal;     // internal answer token text ("A", "B", ...)
    llama_token token_id = -1;
    float       logit    = 0.0f;
    double      prob     = 0.0;
};

struct DecisionRequest {
    std::string            id;
    std::string            type;  // noul | choice | score
    std::string            state;
    std::string            question;
    std::vector<Candidate> candidates;
    json                   expected;  // optional: expected key (string) or level (int)
    json                   check;     // optional smoke-test expectations
};

struct DecisionResult {
    bool        ok = false;
    std::string error;
    int         selected = -1;
    double      expected_score = 0.0;  // score only
    double      p_true = 0.0;          // noul only
    double      candidate_mass = 0.0;  // share of full-vocab softmax held by the candidates (diagnostic)
    llama_token gen_token = -1;        // constrained one-token generation
    bool        gen_matches = false;
    llama_token free_token = -1;       // unconstrained argmax over the whole vocabulary (diagnostic)
    int         n_prompt_tokens = 0;
    double      t_prompt_ms = 0, t_decision_ms = 0, t_gen_ms = 0, t_total_ms = 0;
    std::string prompt;
};

// ---------------------------------------------------------------------------
// Numerics

// Restricted, numerically stable softmax over candidate logits only.
// Returns an error string on invalid input or output.
static std::string restricted_softmax(const std::vector<float> & logits, std::vector<double> & probs) {
    probs.clear();
    if (logits.empty()) {
        return "empty candidate set";
    }
    for (float l : logits) {
        if (!std::isfinite(l)) {
            return "non-finite candidate logit";
        }
    }
    const double m = *std::max_element(logits.begin(), logits.end());
    double       sum = 0.0;
    for (float l : logits) {
        probs.push_back(std::exp((double) l - m));
        sum += probs.back();
    }
    double check = 0.0;
    for (double & p : probs) {
        p /= sum;
        if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
            return "probability out of range";
        }
        check += p;
    }
    if (std::fabs(check - 1.0) > 1e-9) {
        return "probabilities do not sum to 1";
    }
    return "";
}

static double expected_level(const std::vector<double> & probs) {
    double s = 0.0;
    for (size_t i = 0; i < probs.size(); i++) {
        s += (double) i * probs[i];
    }
    return s;
}

static int argmax(const std::vector<double> & v) {
    return (int) (std::max_element(v.begin(), v.end()) - v.begin());
}

// Model-free checks of the numeric core (run with --selftest).
static int selftest() {
    int  fails = 0;
    auto expect = [&](bool cond, const char * what) {
        printf("%s  %s\n", cond ? "PASS" : "FAIL", what);
        if (!cond) fails++;
    };
    std::vector<double> p;

    expect(restricted_softmax({ 0.0f, 0.0f }, p).empty() && std::fabs(p[0] - 0.5) < 1e-12, "uniform softmax");
    expect(restricted_softmax({ 1000.0f, 0.0f }, p).empty() && p[0] > 0.999999, "large logits stay finite");
    expect(restricted_softmax({ -1000.0f, -1001.0f }, p).empty() && std::fabs(p[0] + p[1] - 1.0) < 1e-12,
           "very negative logits stay finite");
    expect(!restricted_softmax({}, p).empty(), "empty candidate set rejected");
    expect(!restricted_softmax({ NAN, 0.0f }, p).empty(), "NaN logit rejected");
    expect(!restricted_softmax({ INFINITY, 0.0f }, p).empty(), "inf logit rejected");
    expect(std::fabs(expected_level({ 0.05, 0.20, 0.65, 0.10 }) - 1.80) < 1e-12, "expected score example = 1.80");

    // softmax of log-probs must reproduce the original distribution
    std::vector<float> lp = { std::log(0.05f), std::log(0.20f), std::log(0.65f), std::log(0.10f) };
    expect(restricted_softmax(lp, p).empty() && std::fabs(p[2] - 0.65) < 1e-6 && argmax(p) == 2,
           "softmax(log p) == p");

    printf("%s\n", fails ? "SELFTEST FAILED" : "SELFTEST OK");
    return fails ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Input parsing
//
// Two input shapes are accepted, one JSON object per line (JSONL), a JSON array, or a single object:
//
// simple:   {"type":"noul","state":..,"question":..}
//           {"type":"choice","state":..,"question":..,"choices":{"key":"description",...}}
//           {"type":"score","state":..,"question":..,"levels":["lowest",...,"highest"]}
// jevbench: raw JevBench dataset rows ({"state":..,"labels":[..],"question":{"type":..,"criteria":..}})

static std::string str_or(const json & j, const char * k, const std::string & def = "") {
    return j.contains(k) && j[k].is_string() ? j[k].get<std::string>() : def;
}

static std::string parse_request(const json & j, DecisionRequest & r) {
    r.id       = j.contains("id") ? (j["id"].is_string() ? j["id"].get<std::string>() : j["id"].dump()) : "";
    r.state    = str_or(j, "state");
    r.expected = j.contains("expected") ? j["expected"] : json();
    r.check    = j.contains("check") ? j["check"] : json();

    const bool jevbench = j.contains("question") && j["question"].is_object();
    if (jevbench) {
        const json & q = j["question"];
        r.type         = str_or(q, "type");
        r.question     = str_or(q, "instructions");
        const json & c = q.contains("criteria") ? q["criteria"] : json::object();
        if (r.type == "noul") {
            r.candidates.push_back({ "false", str_or(c, "false"), "", -1 });
            r.candidates.push_back({ "true", str_or(c, "true"), "", -1 });
            if (r.expected.is_string()) {
                r.expected = r.expected == "yes" ? "true" : "false";
            }
        } else if (r.type == "choice") {
            for (const auto & lab : j["labels"]) {
                const std::string k = lab.get<std::string>();
                r.candidates.push_back({ k, str_or(c, k.c_str(), k), "", -1 });
            }
        } else if (r.type == "score") {
            for (const auto & d : c) {
                r.candidates.push_back({ std::to_string(r.candidates.size()), d.get<std::string>(), "", -1 });
            }
        }
    } else {
        r.type     = str_or(j, "type");
        r.question = str_or(j, "question");
        if (r.type == "noul") {
            r.candidates.push_back({ "false", "", "", -1 });
            r.candidates.push_back({ "true", "", "", -1 });
        } else if (r.type == "choice" && j.contains("choices")) {
            for (const auto & [k, v] : j["choices"].items()) {
                r.candidates.push_back({ k, v.get<std::string>(), "", -1 });
            }
        } else if (r.type == "score" && j.contains("levels")) {
            for (const auto & d : j["levels"]) {
                r.candidates.push_back({ std::to_string(r.candidates.size()), d.get<std::string>(), "", -1 });
            }
        }
    }

    if (r.type != "noul" && r.type != "choice" && r.type != "score") {
        return "unknown decision type '" + r.type + "'";
    }
    if (r.candidates.size() < 2) {
        return "need at least 2 candidates";
    }
    if ((int) r.candidates.size() > MAX_CANDIDATES) {
        return "too many candidates (max " + std::to_string(MAX_CANDIDATES) + ")";
    }
    return "";
}

// Candidate scheme: which internal token stands for each candidate.
//   letters: A, B, C, ... for every type (noul: A = No, B = Yes)
//   natural: noul -> No / Yes, score -> 0, 1, 2, ..., choice -> A, B, C, ...
enum class Scheme { LETTERS, NATURAL };

static void assign_internal(DecisionRequest & r, Scheme scheme) {
    for (size_t i = 0; i < r.candidates.size(); i++) {
        std::string s = std::string(1, (char) ('A' + i));
        if (scheme == Scheme::NATURAL && r.type == "noul") s = i == 0 ? "No" : "Yes";
        if (scheme == Scheme::NATURAL && r.type == "score") s = std::to_string(i);
        r.candidates[i].internal = s;
    }
}

// Every internal candidate text the scheme can use; all are verified at startup.
static std::vector<std::string> all_internal_texts(Scheme scheme) {
    std::vector<std::string> v;
    for (int i = 0; i < MAX_CANDIDATES; i++) v.push_back(std::string(1, (char) ('A' + i)));
    if (scheme == Scheme::LETTERS) return v;
    for (int i = 0; i < MAX_CANDIDATES; i++) v.push_back(std::to_string(i));
    v.push_back("No");
    v.push_back("Yes");
    return v;
}

static bool read_inputs(const std::string & path, std::vector<json> & out) {
    std::stringstream ss;
    if (path == "-") {
        ss << std::cin.rdbuf();
    } else {
        std::ifstream f(path);
        if (!f) {
            fprintf(stderr, "cannot open %s\n", path.c_str());
            return false;
        }
        ss << f.rdbuf();
    }
    const std::string text = ss.str();
    // Whole document first (single object or array), then fall back to JSONL.
    try {
        json doc = json::parse(text);
        if (doc.is_array()) {
            for (auto & e : doc) out.push_back(e);
        } else {
            out.push_back(doc);
        }
        return true;
    } catch (...) {
    }
    std::istringstream lines(text);
    std::string        line;
    int                n = 0;
    while (std::getline(lines, line)) {
        n++;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        try {
            out.push_back(json::parse(line));
        } catch (const std::exception & e) {
            fprintf(stderr, "%s:%d: %s\n", path.c_str(), n, e.what());
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Prompt construction — deliberately plain and easy to edit.

// Prompt layout: "state-first" (state, question, options) or "state-last" (options, question,
// state). "auto" uses state-last for score only: on the JevBench public splitsauto (default: score=state-last, others=state-first) | state-first | state-last made
// score collapse onto the last level (recency bias), while state-last hurt choice.
enum class Layout { AUTO, STATE_FIRST, STATE_LAST };
static Layout g_layout = Layout::AUTO;

static std::string build_user_prompt(const DecisionRequest & r) {
    std::string answers;
    for (size_t i = 0; i < r.candidates.size(); i++) {
        answers += (i ? ", " : "") + r.candidates[i].internal;
    }
    std::string opts = r.type == "score" ? "Possible answers (ordered from lowest to highest):\n" : "Possible answers:\n";
    for (size_t i = 0; i < r.candidates.size(); i++) {
        const auto & c    = r.candidates[i];
        std::string  desc = c.description;
        if (r.type == "noul") {
            // noul candidates carry only an optional criterion; the polarity word is added here
            // unless the internal token already is that word.
            const std::string word = i == 0 ? "No" : "Yes";
            if (c.internal != word) desc = desc.empty() ? word : word + ". " + desc;
        }
        opts += c.internal + (desc.empty() ? "" : ": " + desc) + "\n";
    }
    std::string p = "You are performing a classification task.\n\n";
    const bool state_last = g_layout == Layout::STATE_LAST || (g_layout == Layout::AUTO && r.type == "score");
    if (state_last) {
        p += opts + "\nQuestion:\n" + r.question + "\n\nState:\n" + r.state + "\n";
    } else {
        p += "State:\n" + r.state + "\n\nQuestion:\n" + r.question + "\n\n" + opts;
    }
    p += "\nReply with exactly one of " + answers + " and nothing else.";
    return p;
}

enum class ChatFormat { RAW, CHATML, CHATML_NOTHINK };

static std::string wrap_prompt(const std::string & user, ChatFormat fmt) {
    switch (fmt) {
        case ChatFormat::RAW:
            // No trailing space: the tokenizer would merge it with the answer (" A" is one token),
            // so in raw mode candidates are scored as space-prefixed tokens instead.
            return user + "\n\nAnswer:";
        case ChatFormat::CHATML:
            return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
        case ChatFormat::CHATML_NOTHINK:
            // Qwen3-style template with thinking disabled: an empty think block is prefilled.
            return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }
    return user;
}

// ---------------------------------------------------------------------------
// Engine

struct Engine {
    llama_model *       model = nullptr;
    llama_context *     ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    int                 n_vocab = 0;
    ChatFormat          fmt = ChatFormat::CHATML_NOTHINK;
    bool                verbose = false;
    bool                do_gen = true;
    Scheme              scheme = Scheme::NATURAL;
    std::map<std::string, llama_token> cand_tokens;  // internal candidate text -> verified token

    // Text the model must emit for a candidate at the answer position. In chat formats the answer
    // starts right after "\n", so it is the bare text; in raw format it follows "Answer:" and
    // carries a leading space.
    std::string emitted(const std::string & internal) const { return fmt == ChatFormat::RAW ? " " + internal : internal; }

    std::vector<llama_token> tokenize(const std::string & text, bool add_special, bool parse_special) const {
        int                      n = -llama_tokenize(vocab, text.c_str(), (int) text.size(), nullptr, 0, add_special, parse_special);
        std::vector<llama_token> toks(n);
        llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), n, add_special, parse_special);
        return toks;
    }

    std::string piece(llama_token t) const {
        char buf[256];
        int  n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
        return n > 0 ? std::string(buf, n) : "";
    }

    // Verify that every internal candidate text maps to exactly one token and that no two texts
    // share a token. Only verified texts can ever be used as candidates.
    bool verify_candidate_tokens() {
        std::set<llama_token> seen;
        bool                  ok = true;
        fprintf(stderr, "candidate token map (%s):\n", fmt == ChatFormat::RAW ? "space-prefixed" : "bare");
        for (const auto & s : all_internal_texts(scheme)) {
            auto t = tokenize(emitted(s), false, false);
            fprintf(stderr, "  %-4s -> %s\n", json(emitted(s)).dump().c_str(),
                    t.size() == 1 ? std::to_string(t[0]).c_str() : ("NOT SINGLE TOKEN (" + std::to_string(t.size()) + ")").c_str());
            if (t.size() != 1 || seen.count(t[0])) {
                ok = false;
                continue;
            }
            seen.insert(t[0]);
            cand_tokens[s] = t[0];
        }
        return ok;
    }

    // Context-sensitivity check: tokenizing the whole prompt + letter must end in exactly the
    // verified letter token, i.e. the tokenizer does not merge the letter with preceding text.
    bool check_boundary(const std::string & prompt, const Candidate & c) const {
        auto a = tokenize(prompt, true, true);
        auto b = tokenize(prompt + emitted(c.internal), true, true);
        return b.size() == a.size() + 1 && std::equal(a.begin(), a.end(), b.begin()) && b.back() == c.token_id;
    }

    DecisionResult decide(DecisionRequest & r) {
        using clk = std::chrono::steady_clock;
        DecisionResult res;
        auto           t0 = clk::now();

        // 1. candidate -> token
        assign_internal(r, scheme);
        std::set<llama_token> ids;
        for (auto & c : r.candidates) {
            auto it = cand_tokens.find(c.internal);
            if (it == cand_tokens.end()) {
                res.error = "candidate " + c.internal + " is not a verified single token";
                return res;
            }
            c.token_id = it->second;
            if (!ids.insert(c.token_id).second) {
                res.error = "duplicate candidate token id";
                return res;
            }
        }

        // 2. prompt
        res.prompt = wrap_prompt(build_user_prompt(r), fmt);
        for (const auto & c : r.candidates) {
            if (!check_boundary(res.prompt, c)) {
                res.error = "candidate " + c.internal + " merges with prompt tail (tokenizer boundary problem)";
                return res;
            }
        }
        auto toks = tokenize(res.prompt, true, true);
        res.n_prompt_tokens = (int) toks.size();
        if (res.n_prompt_tokens >= (int) llama_n_ctx(ctx)) {
            res.error = "prompt too long";
            return res;
        }

        // 3. prompt evaluation, logits requested for the last position only
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); i++) {
            batch.token[i]     = toks[i];
            batch.pos[i]       = (llama_pos) i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = i + 1 == toks.size();
        }
        batch.n_tokens = (int) toks.size();
        auto t1 = clk::now();
        int  rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        auto t2 = clk::now();
        res.t_prompt_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        if (rc != 0) {
            res.error = "llama_decode failed: " + std::to_string(rc);
            return res;
        }

        // 4. candidate logits -> restricted softmax
        const float *      logits = llama_get_logits_ith(ctx, -1);
        std::vector<float> cl;
        for (auto & c : r.candidates) {
            c.logit = logits[c.token_id];
            cl.push_back(c.logit);
        }
        std::vector<double> probs;
        res.error = restricted_softmax(cl, probs);
        if (!res.error.empty()) {
            return res;
        }
        for (size_t i = 0; i < probs.size(); i++) {
            r.candidates[i].prob = probs[i];
        }
        res.selected = argmax(probs);
        if (r.type == "score") res.expected_score = expected_level(probs);
        if (r.type == "noul") res.p_true = probs[1];
        auto t3 = clk::now();
        res.t_decision_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // Diagnostics over the full vocabulary (not used for the decision).
        {
            float  m = -INFINITY;
            int    am = 0;
            for (int i = 0; i < n_vocab; i++) {
                if (logits[i] > m) { m = logits[i]; am = i; }
            }
            double z = 0.0, zc = 0.0;
            for (int i = 0; i < n_vocab; i++) z += std::exp((double) logits[i] - m);
            for (auto & c : r.candidates) zc += std::exp((double) c.logit - m);
            res.candidate_mass = zc / z;
            res.free_token     = am;
        }

        // 5. constrained one-token generation (validation only): every non-candidate token is
        //    biased to -inf, then a greedy sampler picks exactly one token.
        if (do_gen) {
            std::vector<llama_logit_bias> bias;
            bias.reserve(n_vocab);
            for (int i = 0; i < n_vocab; i++) {
                if (!ids.count(i)) bias.push_back({ i, -INFINITY });
            }
            auto sp = llama_sampler_chain_default_params();
            sp.no_perf = true;
            llama_sampler * chain = llama_sampler_chain_init(sp);
            llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, (int) bias.size(), bias.data()));
            llama_sampler_chain_add(chain, llama_sampler_init_greedy());
            res.gen_token = llama_sampler_sample(chain, ctx, -1);
            llama_sampler_free(chain);
            res.gen_matches = res.gen_token == r.candidates[res.selected].token_id;
            res.t_gen_ms    = std::chrono::duration<double, std::milli>(clk::now() - t3).count();
        }

        res.t_total_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        res.ok         = true;
        return res;
    }
};

static long rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string   line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) return std::stol(line.substr(6));
    }
    return -1;
}

static json result_json(const Engine & E, const DecisionRequest & r, const DecisionResult & res) {
    json o;
    o["id"]   = r.id;
    o["type"] = r.type;
    if (!res.ok) {
        o["error"] = res.error;
        return o;
    }
    const Candidate & sel = r.candidates[res.selected];
    if (r.type == "score") {
        json pa = json::array();
        for (auto & c : r.candidates) pa.push_back(c.prob);
        o["probabilities"]  = pa;
        o["score"]          = res.expected_score;
        o["selected_level"] = res.selected;
    } else {
        json pm = json::object();
        for (auto & c : r.candidates) pm[c.key] = c.prob;
        o["probabilities"] = pm;
        if (r.type == "noul") {
            o["value"]    = res.p_true;
            o["selected"] = sel.key == "true";
        } else {
            o["selected"] = sel.key;
        }
    }
    json lg = json::object();
    for (auto & c : r.candidates) lg[c.internal] = c.logit;
    o["raw_logits"] = lg;
    if (!r.expected.is_null()) {
        o["expected"] = r.expected;
        o["correct"]  = r.type == "score" ? (r.expected.is_number() && r.expected.get<int>() == res.selected)
                                          : (r.expected.is_string() && r.expected.get<std::string>() == sel.key);
    }
    o["candidate_mass"] = res.candidate_mass;
    o["free_argmax"]    = E.piece(res.free_token);
    if (E.do_gen) {
        o["generated"]   = E.piece(res.gen_token);
        o["gen_matches"] = res.gen_matches;
    }
    o["n_prompt_tokens"] = res.n_prompt_tokens;
    o["t_prompt_ms"]     = res.t_prompt_ms;
    o["t_decision_ms"]   = res.t_decision_ms;
    o["t_gen_ms"]        = res.t_gen_ms;
    o["t_total_ms"]      = res.t_total_ms;
    return o;
}

static void print_debug(const Engine & E, const DecisionRequest & r, const DecisionResult & res) {
    fprintf(stderr, "==================== %s [%s]\n", r.id.c_str(), r.type.c_str());
    fprintf(stderr, "---- prompt (%d tokens)\n%s\n----\n", res.n_prompt_tokens, res.prompt.c_str());
    if (!res.ok) {
        fprintf(stderr, "ERROR: %s\n", res.error.c_str());
        return;
    }
    for (size_t i = 0; i < r.candidates.size(); i++) {
        const auto & c = r.candidates[i];
        fprintf(stderr, "  %s %s -> %-16s token_id=%-6d logit=%8.3f  p=%.4f\n", (int) i == res.selected ? "*" : " ",
                c.internal.c_str(), c.key.c_str(), c.token_id, c.logit, c.prob);
    }
    if (r.type == "score") fprintf(stderr, "  expected score = %.3f\n", res.expected_score);
    if (r.type == "noul") fprintf(stderr, "  p_true = %.4f\n", res.p_true);
    fprintf(stderr, "  candidate mass (full-vocab) = %.4f, unconstrained argmax = %s\n", res.candidate_mass,
            json(E.piece(res.free_token)).dump().c_str());
    if (E.do_gen) {
        fprintf(stderr, "  constrained generation = %s (%s)\n", E.piece(res.gen_token).c_str(),
                res.gen_matches ? "matches argmax" : "MISMATCH");
    }
    fprintf(stderr, "  time: prompt %.1f ms, decision %.3f ms, gen %.1f ms, total %.1f ms\n", res.t_prompt_ms,
            res.t_decision_ms, res.t_gen_ms, res.t_total_ms);
}

static void usage() {
    fprintf(stderr,
            "usage: phase0 -m MODEL [options] INPUT...\n"
            "  INPUT            .json / .jsonl file (simple or JevBench rows), '-' for stdin\n"
            "  -v               debug output (prompt, token map, logits, timings) on stderr\n"
            "  -t N             CPU threads (default 8)\n"
            "  --format F       nothink (default) | chatml | raw\n"
            "  --scheme S       natural (default: noul=No/Yes, score=0..N-1, choice=A..) | letters\n"
            "  --layout L       auto (default: score=state-last, others=state-first) | state-first | state-last\n"
            "  --no-gen         skip constrained one-token generation check\n"
            "  --repeat N       evaluate every case N times and check results are identical\n"
            "  --per-type N     use only the first N cases of each decision type\n"
            "  --check          evaluate 'check' expectations in the input (smoke tests)\n"
            "  --selftest       run model-free numeric tests and exit\n");
}

int main(int argc, char ** argv) {
    std::string              model_path;
    std::vector<std::string> inputs;
    Engine                   E;
    int                      threads = 8, repeat = 1, per_type = 0;
    bool                     check = false;

    for (int i = 1; i < argc; i++) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); exit(2); }
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "-v") E.verbose = true;
        else if (a == "-t") threads = std::stoi(next());
        else if (a == "--repeat") repeat = std::stoi(next());
        else if (a == "--per-type") per_type = std::stoi(next());
        else if (a == "--no-gen") E.do_gen = false;
        else if (a == "--layout") {
            std::string l = next();
            g_layout = l == "state-first" ? Layout::STATE_FIRST : l == "state-last" ? Layout::STATE_LAST : Layout::AUTO;
        }
        else if (a == "--scheme") E.scheme = next() == "letters" ? Scheme::LETTERS : Scheme::NATURAL;
        else if (a == "--check") check = true;
        else if (a == "--selftest") return selftest();
        else if (a == "--format") {
            std::string f = next();
            E.fmt = f == "raw" ? ChatFormat::RAW : f == "chatml" ? ChatFormat::CHATML : ChatFormat::CHATML_NOTHINK;
        } else if (a == "-h" || a == "--help") { usage(); return 0; }
        else inputs.push_back(a);
    }
    if (model_path.empty() || inputs.empty()) {
        usage();
        return 2;
    }

    // Parse all requests up front so input errors surface before the model loads.
    std::vector<DecisionRequest> reqs;
    std::map<std::string, int>   type_count;
    for (auto & in : inputs) {
        std::vector<json> rows;
        if (!read_inputs(in, rows)) return 1;
        for (auto & j : rows) {
            DecisionRequest r;
            std::string     err = parse_request(j, r);
            if (!err.empty()) {
                fprintf(stderr, "skip %s: %s\n", r.id.c_str(), err.c_str());
                continue;
            }
            if (per_type > 0 && type_count[r.type]++ >= per_type) continue;
            reqs.push_back(std::move(r));
        }
    }

    // Model load. Single sequence, greedy, CPU — deterministic by construction.
    if (!E.verbose) {
        llama_log_set([](ggml_log_level lvl, const char * text, void *) {
            if (lvl >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
        }, nullptr);
    }
    llama_backend_init();
    auto t_load0 = std::chrono::steady_clock::now();
    auto mp      = llama_model_default_params();
    mp.n_gpu_layers = 0;
    E.model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!E.model) {
        fprintf(stderr, "failed to load model %s\n", model_path.c_str());
        return 1;
    }
    auto cp            = llama_context_default_params();
    cp.n_ctx           = 4096;
    cp.n_batch         = 4096;
    cp.n_threads       = threads;
    cp.n_threads_batch = threads;
    cp.no_perf         = true;
    E.ctx = llama_init_from_model(E.model, cp);
    if (!E.ctx) {
        fprintf(stderr, "failed to create context\n");
        return 1;
    }
    E.vocab   = llama_model_get_vocab(E.model);
    E.n_vocab = llama_vocab_n_tokens(E.vocab);
    double t_load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();
    {
        char desc[256];
        llama_model_desc(E.model, desc, sizeof(desc));
        fprintf(stderr, "model: %s, n_vocab=%d, add_bos=%d, load %.1f ms, RSS %ld MiB\n", desc, E.n_vocab,
                (int) llama_vocab_get_add_bos(E.vocab), t_load_ms, rss_kb() / 1024);
    }
    if (!E.verify_candidate_tokens()) {
        fprintf(stderr, "candidate token verification failed\n");
        return 1;
    }

    // Evaluate.
    struct Agg { int n = 0, labeled = 0, correct = 0, gen_ok = 0; double abs_err = 0; };
    std::map<std::string, Agg>                     agg;
    std::map<std::string, std::vector<std::pair<int, double>>> score_groups;  // check.group -> (rank, score)
    int    n_err = 0, n_check_fail = 0, n_nondeterministic = 0;
    double sum_ms = 0;
    for (auto & r : reqs) {
        DecisionResult res = E.decide(r);
        if (E.verbose) print_debug(E, r, res);
        json out = result_json(E, r, res);

        if (res.ok && repeat > 1) {
            // Re-run and compare raw logits bit-for-bit.
            std::vector<float> first;
            for (auto & c : r.candidates) first.push_back(c.logit);
            double max_diff = 0;
            for (int k = 1; k < repeat; k++) {
                DecisionRequest r2  = r;
                DecisionResult  re2 = E.decide(r2);
                for (size_t i = 0; i < first.size(); i++) {
                    max_diff = std::max(max_diff, (double) std::fabs(r2.candidates[i].logit - first[i]));
                }
            }
            out["repeat_max_logit_diff"] = max_diff;
            if (max_diff > 0) n_nondeterministic++;
        }

        if (!res.ok) {
            n_err++;
        } else {
            sum_ms += res.t_total_ms;
            Agg & a = agg[r.type];
            a.n++;
            if (out.contains("correct")) a.labeled++;
            if (out.contains("correct") && out["correct"].get<bool>()) a.correct++;
            if (res.gen_matches) a.gen_ok++;
            if (r.type == "score" && r.expected.is_number()) a.abs_err += std::fabs(res.expected_score - r.expected.get<double>());
        }

        if (check && res.ok && r.check.is_object()) {
            bool pass = true;
            if (r.check.contains("selected")) {
                const json & want = r.check["selected"];
                pass = want == out["selected"];
            }
            if (r.check.contains("group")) {
                score_groups[r.check["group"].get<std::string>()].push_back({ r.check["rank"].get<int>(), res.expected_score });
            }
            out["check_pass"] = pass;
            if (!pass) n_check_fail++;
        }
        std::cout << out.dump() << std::endl;
    }

    // Summary.
    fprintf(stderr, "\n==== summary: %zu cases, %d errors, model load %.1f ms, RSS %ld MiB\n", reqs.size(), n_err,
            t_load_ms, rss_kb() / 1024);
    for (auto & [t, a] : agg) {
        fprintf(stderr, "  %-6s n=%-3d", t.c_str(), a.n);
        if (E.do_gen) fprintf(stderr, " constrained-gen-matches=%d/%d", a.gen_ok, a.n);
        if (a.labeled) {
            fprintf(stderr, " accuracy=%d/%d (%.3f)", a.correct, a.labeled, (double) a.correct / a.labeled);
            if (t == "score") fprintf(stderr, " MAE(expected score)=%.3f", a.abs_err / a.labeled);
        }
        fprintf(stderr, "\n");
    }
    if (!agg.empty()) {
        int n = 0;
        for (auto & [t, a] : agg) n += a.n;
        fprintf(stderr, "  mean total latency %.1f ms/case\n", n ? sum_ms / n : 0.0);
    }
    if (repeat > 1) fprintf(stderr, "  determinism: %d case(s) with differing logits over %d repeats\n", n_nondeterministic, repeat);
    if (check) {
        for (auto & [g, v] : score_groups) {
            std::sort(v.begin(), v.end());
            bool ordered = true;
            for (size_t i = 1; i < v.size(); i++) ordered &= v[i].second > v[i - 1].second;
            fprintf(stderr, "  score order group '%s': %s\n", g.c_str(), ordered ? "PASS" : "FAIL");
            if (!ordered) n_check_fail++;
        }
        fprintf(stderr, "  checks: %s (%d failure(s))\n", n_check_fail ? "FAIL" : "PASS", n_check_fail);
    }

    llama_free(E.ctx);
    llama_model_free(E.model);
    llama_backend_free();
    return (n_err || n_check_fail || n_nondeterministic) ? 1 : 0;
}
