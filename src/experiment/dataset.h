#ifndef INC_PJEV_DATASET_H_
#define INC_PJEV_DATASET_H_
#include "../decision/decision_engine.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace pjev
{
struct DatasetRow {
    std::string id;
    DecisionInput input;
    nlohmann::ordered_json expected; // null, string (choice/noul key), or int (score level index)
    // Optional multilingual metadata (empty when absent)
    std::string language;               // "en", "ja", etc.
    std::string pair_id;                // shared ID across language versions of the same item
    std::string source_language;        // "en" for translated items
    std::string translation_provenance; // "human", "machine", "manual_revision", etc.
    std::string difficulty;             // "easy", "original", "hard"
};

// Parse one JSON row (supports simple format and JevBench format).
// Returns "" on success, error string on failure.
std::string parse_row(const nlohmann::ordered_json& j, DatasetRow& row);

// Load JSONL/JSON file into rows. Returns false on I/O or parse error.
bool load_dataset(const std::string& path, std::vector<DatasetRow>& rows, std::string& err);
}
#endif //INC_PJEV_DATASET_H_

