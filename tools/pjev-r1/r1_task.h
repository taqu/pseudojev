#pragma once

#include <string>
#include <vector>

#include "decision/decision_engine.h"
#include "r0_retrieval.h"
#include "r1_output.h"

#include "nlohmann/json.hpp"

namespace pjev
{
namespace r1
{

// Version tag for the R1 prompt layout, recorded in config.json.
static constexpr const char* kPromptFormatVersion = "r1-demo-turn-v1";

// State as a pjev client sends it: string states verbatim, structured states as compact JSON
// (key order preserved).
std::string state_text(const nlohmann::ordered_json& state);

// An Open Jev task (kind / question / state / options) mapped onto a pjev DecisionInput.
// Open Jev options are indexed in dataset order; decision candidates in pjev order.
// They coincide for choice and score; noul always maps to pjev's (false, true) order.
struct TaskMapping {
    DecisionInput    input;
    std::vector<int> option_to_candidate; // Open Jev option index -> candidate index
    std::vector<int> candidate_to_option; // candidate index -> Open Jev option index
};

// Build the S1 decision input. Never reads the target.
// Returns false and fills err when the task cannot be expressed as a pjev decision.
bool build_task(const std::string&              kind,
                const std::string&              question,
                const nlohmann::ordered_json&   state,
                const std::vector<std::string>& options,
                TaskMapping&                    out,
                std::string&                    err);

// Build the demonstration for a retrieved exemplar, using its own option order and its own
// stored answer. Fails when the exemplar has no one-hot answer.
bool build_demonstration(const r0::ExemplarPayload& payload,
                         Demonstration&             demo,
                         std::string&               err);

// Record an engine decision as a condition result. S1 and R1 both go through this, so they are
// scored identically: prediction = engine argmax (candidate index), correctness by index.
// ground_truth is a candidate index and is used only after the decision.
void fill_condition(const DecisionOutput&           out,
                    const std::vector<std::string>& labels,
                    const std::vector<std::string>& texts,
                    int                             ground_truth,
                    double                          wall_ms,
                    ConditionResult&                res);

// Candidate labels shown in the prompt for n candidates (same assignment as PromptStrategy).
std::vector<std::string> candidate_labels(const PromptStrategy& strategy, Type type, size_t n);

// Full prompt text (segments concatenated) for diagnostics; never used for inference.
std::string render_s1_prompt(const PromptStrategy& strategy, const DecisionInput& input);
std::string render_r1_prompt(const PromptStrategy& strategy, const DecisionInput& input,
                             const Demonstration& demo);

} // namespace r1
} // namespace pjev
