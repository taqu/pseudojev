#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "exemplar_allowlist.h"
#include "nlohmann/json.hpp"

struct sqlite3;
struct sqlite3_stmt;

namespace pjev
{
namespace retrieval
{

// One accepted search result. distance is the sqlite-vec KNN distance over binary-quantized vectors,
// i.e. a Hamming distance in bits.
struct SearchHit {
    int64_t id       = 0;
    double  distance = 0.0;
};

// Full content of one stored exemplar, reconstructed from its LZ4 payloads.
struct ExemplarPayload {
    int64_t                  id               = 0;
    std::string              kind;
    std::string              question;
    nlohmann::ordered_json   state;
    std::vector<std::string> options;              // dataset order
    int                      answer_index     = -1; // -1 when the stored target is not one-hot
    int                      number_of_levels = 0;  // score only
};

// What task routing needs from the exemplar bank (an interface so routing can be tested without a
// database).
class IExemplarIndex
{
public:
    virtual ~IExemplarIndex() = default;

    // Up to top_k eligible exemplars of the task's kind (score: and level count), nearest first in
    // (distance, rowid) order. Fewer (or none) is not an error. False only on a database failure.
    virtual bool search(const std::string&        kind,
                        int                       number_of_levels,
                        const std::vector<float>& embedding,
                        int                       top_k,
                        std::vector<SearchHit>&   out,
                        std::string&              err) const = 0;

    // Load and decompress one exemplar. False on a missing row, corrupt LZ4 frame or malformed payload.
    virtual bool load(int64_t id, ExemplarPayload& out, std::string& err) const = 0;

    // Whether the exemplar is in the allowlist.
    virtual bool is_allowed(int64_t id) const = 0;
};

// Read-only access to the exemplar database (schema version 4) with the allowlist applied to search.
// Not thread-safe; callers serialize access.
class ExemplarStore : public IExemplarIndex
{
public:
    // Maximum k accepted by the sqlite-vec KNN query.
    static constexpr int kMaxFetchK = 4096;

    ExemplarStore() = default;
    ~ExemplarStore() override;

    ExemplarStore(const ExemplarStore&)            = delete;
    ExemplarStore& operator=(const ExemplarStore&) = delete;

    // Opens the database read-only and loads the allowlist.
    bool open(const std::string& db_path, const std::string& allowlist_path, std::string& err);

    // Value from db_info, or "" when absent.
    std::string info(const char* key) const;

    // Verifies schema version, embedding dimension, embedding model fingerprint (when both are known),
    // distance metric and normalization against the stored metadata.
    bool check_compatible(int emb_dim, const std::string& fingerprint, std::string& err) const;

    int64_t example_count() const;
    size_t  allowlist_size() const { return allow_.size(); }

    bool search(const std::string&        kind,
                int                       number_of_levels,
                const std::vector<float>& embedding,
                int                       top_k,
                std::vector<SearchHit>&   out,
                std::string&              err) const override;
    bool load(int64_t id, ExemplarPayload& out, std::string& err) const override;
    bool is_allowed(int64_t id) const override { return allow_.count(id) > 0; }

private:
    struct Candidate {
        int64_t rowid;
        double  distance;
    };
    bool    knn(const std::string& kind, int number_of_levels, const std::vector<float>& embedding,
                int fetch_k, std::vector<Candidate>& out, std::string& err) const;
    int64_t rowid_to_id(int64_t rowid) const;
    void    close();

    sqlite3*          db_           = nullptr;
    sqlite3_stmt*     stmt_id_      = nullptr;
    sqlite3_stmt*     stmt_payload_ = nullptr;
    ExemplarAllowlist allow_;
};

} // namespace retrieval
} // namespace pjev
