#ifndef INC_PJEV_RELEASE_CONFIG_H_
#define INC_PJEV_RELEASE_CONFIG_H_
#include <string>
#include <nlohmann/json.hpp>

namespace pjev {

// Runtime resources of the task-aware decision path.
struct RuntimePaths {
    std::string model;               // inference model (GGUF)
    std::string embedding_model;     // query embedding model (GGUF); must match the exemplar database
    std::string exemplar_db;         // exemplar database (SQLite, schema 4), opened read-only
    std::string exemplar_allowlist;  // eligible exemplar IDs (see retrieval/exemplar_allowlist.h)
    std::string calibration;         // calibration artifact (JSON); empty = calibration disabled
};

// Default file names of the runtime resources, looked up next to the executable and, as a development
// fallback, under models/ relative to the working directory.
constexpr const char* kDefaultModelFile              = "bonsai2_2b.gguf";
constexpr const char* kDefaultDevModelFile           = "bonsai2_2b.gguf";
constexpr const char* kDefaultEmbeddingModelFile     = "nomic-embed-text-v1.5.gguf";
constexpr const char* kDefaultExemplarDbFile         = "examplar.db";
constexpr const char* kDefaultExemplarAllowlistFile  = "examplar-allowlist.txt";
constexpr const char* kDefaultCalibrationFile        = "calibration.json";

// Optional pjev.json next to the executable. Unknown keys are ignored.
struct ReleaseConfig {
    int         schema_version  = 1;
    std::string pjev_version;          // informational
    std::string default_model  = kDefaultModelFile;
    std::string embedding_model;       // empty = not set
    std::string exemplar_db;           // empty = not set
    std::string exemplar_allowlist;    // empty = not set
    std::string calibration;           // empty = not set
    int         threads        = -1;   // -1 = not set (use CLI default)
    int         context_size   = -1;   // -1 = not set

    // Load from a JSON file. Returns false and sets err on failure.
    static bool from_file(const std::string& path, ReleaseConfig& out, std::string& err);
    // pjev.json in exe_dir; false when absent or unreadable.
    static bool load_adjacent(const std::string& exe_dir, ReleaseConfig& out);
    nlohmann::json to_json() const;

    // Model path resolution.
    // Precedence:
    //   1. explicit_model (non-empty CLI flag) — returned unchanged
    //   2. pjev.json found in exe_dir — use default_model resolved relative to exe_dir
    //   3. model.gguf adjacent to executable
    //   4. "" (empty — caller should fall back to CWD-relative default)
    //
    // exe_dir: result of get_executable_dir()
    // explicit_model: value of --model flag, or "" if not given
    static std::string resolve_model(const std::string& explicit_model,
                                      const std::string& exe_dir);

    // All runtime resources. Per resource:
    //   1. the explicit value (CLI flag), unchanged
    //   2. pjev.json in exe_dir (a bare file name is relative to exe_dir)
    //   3. the default file name next to the executable, when that file exists
    //   4. models/<default file name> relative to the working directory (development layout)
    static RuntimePaths resolve_runtime_paths(const RuntimePaths& explicit_paths,
                                              const std::string& exe_dir);
};

} // namespace pjev
#endif
