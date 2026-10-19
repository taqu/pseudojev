#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "r1_metrics.h"

#include "nlohmann/json.hpp"

namespace pjev
{
namespace r1
{

// Result of one inference condition (S1 or R1) for one query.
struct ConditionResult {
    // ok | context_limit | inference_error | not_run
    std::string         status = "not_run";
    std::string         error;
    int                 prediction       = -1;  // candidate index
    std::string         prediction_label;
    std::string         prediction_text;
    bool                correct          = false;
    std::vector<float>  logits;                 // candidate order
    std::vector<double> probs;
    double              margin           = 0.0;  // top-1 minus top-2 logit
    double              signed_margin    = 0.0;  // ground-truth minus best incorrect logit
    int                 prompt_tokens    = 0;
    double              prompt_ms        = 0.0;  // prompt construction + tokenization
    double              eval_ms          = 0.0;  // prompt evaluation
    double              scoring_ms       = 0.0;  // candidate logit extraction + softmax
    double              total_ms         = 0.0;  // wall time of the decision call

    bool ok() const { return status == "ok"; }
};

struct RetrievalInfo {
    // ok | no_eligible_exemplar | embedding_error | search_error | payload_error | not_run
    std::string status = "not_run";
    std::string error;
    int64_t     example_id         = 0;
    std::string kind;
    int         number_of_levels   = 0;
    double      hamming_distance   = 0.0;
    double      hamming_similarity = 0.0;
    double      estimated_cosine   = 0.0;
    double      estimated_cosine_distance = 0.0;
    std::string domain;
    std::string family;
    bool        same_group           = false;
    bool        same_domain          = false;
    bool        same_family          = false;
    bool        same_option_texts    = false;  // score: identical level descriptions
    int         answer_index         = -1;     // candidate index of the demonstration answer
    std::string answer_label;
    std::string answer_text;
    int         options_count        = 0;
    int         fetched_k            = 0;
    double      embed_ms             = 0.0;
    double      search_ms            = 0.0;
    double      fetch_ms             = 0.0;    // exemplar row read
    double      decompress_ms        = 0.0;    // LZ4 + payload JSON parse
    // Demonstration content, kept for examples.md
    std::string              question;
    std::string              state;
    std::vector<std::string> option_texts;     // candidate order
    std::vector<std::string> option_labels;

    bool ok() const { return status == "ok"; }
};

// R2 similarity gate outcome; only populated when the harness runs with a gate.
struct GateInfo {
    bool        enabled    = false;
    double      threshold  = 0.0;
    double      similarity = 0.0;  // estimated cosine of the Top-1 exemplar
    std::string decision;          // activated | rejected | no_exemplar | invalid_similarity
};

struct R1Item {
    std::string              query_id;
    int64_t                  query_hash       = 0;
    std::string              kind;
    int                      number_of_levels = 0;
    int                      options_count    = 0;
    std::string              question;
    std::string              state;
    std::vector<std::string> option_texts;     // candidate order
    std::vector<std::string> option_labels;
    int                      ground_truth     = -1;  // candidate index
    ConditionResult          s1;
    ConditionResult          r1;
    RetrievalInfo            retrieval;
    GateInfo                 gate;
    bool                     paired           = false;
    Transition               transition       = Transition::BothIncorrect;

    // R1 production-style latency: retrieval overhead plus the 1-shot decision.
    double r1_total_ms() const
    {
        return retrieval.embed_ms + retrieval.search_ms + retrieval.fetch_ms +
               retrieval.decompress_ms + r1.total_ms;
    }
};

struct R1RunInfo {
    nlohmann::ordered_json config;   // written verbatim to config.json
    std::string            output_dir;
    std::string            corpus;
    bool                   preliminary = false;  // --limit / --kind / --seed subset
    int                    n_ctx       = 0;
    int                    embedding_dim = 0;
    double                 wall_seconds = 0.0;
    double                 s1_seconds   = 0.0;
    double                 r1_seconds   = 0.0;
    std::vector<std::string> notes;
};

bool write_config(const R1RunInfo& info);
bool write_predictions_jsonl(const std::string& dir, const std::vector<R1Item>& items);
bool write_transitions_csv(const std::string& dir, const std::vector<R1Item>& items);
bool write_retrieval_effect_csv(const std::string& dir, const std::vector<R1Item>& items);
bool write_latency_csv(const std::string& dir, const std::vector<R1Item>& items);
bool write_summary_md(const R1RunInfo& info, const std::vector<R1Item>& items);
bool write_examples_md(const std::string& dir, const std::vector<R1Item>& items);

nlohmann::ordered_json item_to_json(const R1Item& item);

// Similarity bins over paired items: quantile edges on Hamming distance (closest first).
struct SimilarityBin {
    double           hamming_lo = 0.0;
    double           hamming_hi = 0.0;
    TransitionCounts counts;
};
std::vector<SimilarityBin> similarity_bins(const std::vector<const R1Item*>& paired, int n_bins);

} // namespace r1
} // namespace pjev
