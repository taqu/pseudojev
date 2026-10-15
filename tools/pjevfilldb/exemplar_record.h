#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace pjev
{
namespace filldb
{

struct ExemplarRecord {
    // Core identity (stored as xxHash64 integers in the database)
    int64_t     id       = 0;
    int64_t     group_id = 0;
    std::string kind;       // "choice" | "noul" | "score"
    std::string source;
    std::string split;

    // Task content
    std::string              question;
    std::vector<std::string> options;
    std::vector<float>       target;
    nlohmann::ordered_json   state;

    // Derived answer (one-hot targets only)
    int         answer_index = -1;  // -1 if not one-hot
    std::string answer_text;

    // Score-specific
    int              number_of_levels = 0;  // options.size() for score, 0 otherwise
    std::vector<int> score_values;

    // Metadata
    std::string            language;
    std::string            domain;
    std::string            family;
    std::string            semantic_context_sha256;
    nlohmann::ordered_json counterfactual;
    nlohmann::ordered_json metadata;

    // Embedding
    std::string embedding_text;

    static constexpr int kEmbeddingFormatVersion = 1;
};

// Populate rec from a parsed JSON object.
// Returns "" on success, or a human-readable error string.
std::string parse_exemplar(const nlohmann::ordered_json& j, ExemplarRecord& rec);

// Returns true iff the record should enter the exemplar bank
// (split == "train" && language == "en").
bool is_eligible(const ExemplarRecord& rec);

} // namespace filldb
} // namespace pjev
