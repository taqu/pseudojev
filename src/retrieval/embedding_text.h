#pragma once

#include <string>

#include "nlohmann/json.hpp"

namespace pjev
{
namespace retrieval
{

// Version incremented whenever the canonical text format changes.
// Stored in db_info so incompatible databases can be detected.
static constexpr int kEmbeddingTextFormatVersion = 1;

// Prefix used for document vectors (matching nomic-embed-text conventions).
static constexpr const char* kDocumentPrefix = "search_document: ";

// Build the canonical embedding text for a stored document.
// Prepends kDocumentPrefix automatically.
std::string build_document_embedding_text(
    const std::string&          kind,
    const std::string&          domain,
    const std::string&          family,
    const std::string&          question,
    const nlohmann::ordered_json& state,
    const std::vector<std::string>& options);

// Build the canonical text body (without prefix) — exposed for testing.
std::string build_embedding_text(
    const std::string&          kind,
    const std::string&          domain,
    const std::string&          family,
    const std::string&          question,
    const nlohmann::ordered_json& state,
    const std::vector<std::string>& options);

} // namespace retrieval
} // namespace pjev
