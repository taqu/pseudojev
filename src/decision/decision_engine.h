#pragma once
#include "../inference/backend.h"
#include "prompt_strategy.h"
#include <map>
#include <string>
#include <vector>

struct DecisionInput {
    std::string type;     // "noul" | "choice" | "score"
    std::string state;
    std::string question; // from "instructions"
    // ordered list of (key, description) — order matters for candidate assignment
    std::vector<std::pair<std::string, std::string>> options;
    bool prior_correction = false;
    std::vector<float> prior_logits; // per-candidate prior logits (size == options.size()); used when prior_correction=true
};

struct DecisionOutput {
    bool        ok      = false;
    std::string error;
    int         selected = -1;   // index into options
    double      p_true   = 0.0;  // noul: probability of "true"
    double      expected_score = 0.0; // score: expected value
    std::vector<double> probs;   // per-candidate probabilities (same order as options)
    std::vector<std::string> keys; // candidate keys in order
};

class DecisionEngine {
public:
    // Throws std::runtime_error if candidate token verification fails.
    DecisionEngine(ILlamaBackend& backend, const PromptConfig& cfg = {});

    // Not thread-safe. External callers must serialize access.
    DecisionOutput decide(const DecisionInput& input);

    // Compute candidate logits for a blank (empty state/question) prompt of the given type and n_options.
    // Returns per-candidate raw logits (size = n_options), or empty vector on failure.
    // Used by the experiment framework for prior correction.
    std::vector<float> compute_blank_logits(const std::string& type, int n_options);

private:
    ILlamaBackend& backend_;
    PromptStrategy strategy_;
    std::map<std::string, int> cand_token_map_; // internal_text -> token_id

    static std::string restricted_softmax(const std::vector<float>& logits,
                                          std::vector<double>& probs);
    static double expected_level(const std::vector<double>& probs);
    static int argmax(const std::vector<double>& v);
};
