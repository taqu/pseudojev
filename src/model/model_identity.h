#ifndef INC_PJEV_MODEL_IDENTITY_H_
#define INC_PJEV_MODEL_IDENTITY_H_
#include "../inference/backend.h"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace pjev
{

// SHA-256 of arbitrary data
std::string sha256_data(const std::string& data);

// SHA-256 of a file; returns empty string on read failure
std::string sha256_file(const std::string& path);

struct ModelIdentity
{
    std::string name;           // display name (from metadata or filename)
    std::string source;         // provenance URL or empty
    std::string gguf_path;
    std::string sha256;
    int64_t     size_bytes    = 0;
    std::string quantization; // e.g. "Q4_K_M" — extracted from model_desc
    std::string architecture; // e.g. "llama" — from general.architecture metadata
    std::string model_desc;   // full description from llama_model_desc
    int64_t     n_params      = 0;
    std::string license;
    std::string pjev_commit;
    std::string llamacpp_revision;

    // Populate from the filesystem and an optionally-loaded backend.
    // If backend is nullptr only the file-level fields (sha256, size_bytes, gguf_path) are set.
    static ModelIdentity from_path(const std::string& path,
                                    ILlamaBackend* backend = nullptr);

    nlohmann::json to_json() const;
};

struct CandidateTokenRecord
{
    std::string text;
    int         token_id    = -1;
    int         token_count = 0;
    bool        single_token = false;
};

struct CandidateCompatibility
{
    bool all_single_token = false;
    std::vector<CandidateTokenRecord> records;
    std::vector<std::string> issues;

    nlohmann::json to_json() const;
};

// Check whether each candidate text tokenises to a single token.
CandidateCompatibility probe_candidates(ILlamaBackend& backend,
                                         const std::vector<std::string>& candidates);

} // namespace pjev
#endif // INC_PJEV_MODEL_IDENTITY_H_
